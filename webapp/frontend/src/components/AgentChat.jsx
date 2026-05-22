/**
 * webapp/frontend/src/components/AgentChat.jsx
 *
 * « Agent IA Live » — tableau de bord d'exécution du pipeline LangGraph.
 *
 * Se connecte à l'endpoint SSE /api/agent/stream/?map_id=...&georeference=...
 * et restitue le déroulé en direct sous une forme « chat / dashboard » :
 *
 *   ┌──────────────────────────────────────────────┐
 *   │ 🤖 Agent IA Live — Suivi CartoVec   ● En direct│  ← en-tête
 *   ├──────────────────────────────────────────────┤
 *   │ [Prétraitement] ➔ [Segmentation] ➔ [Vecto…]   │  ← stepper de nœud actif
 *   ├──────────────────────────────────────────────┤
 *   │   bulle utilisateur (droite)                   │
 *   │ bulle agent (gauche, markdown léger)           │  ← fil de discussion auto-scroll
 *   │ ┌── System Event (terminal vert sur ardoise)─┐ │
 *   │ │ → Executing: preprocess                     │ │  ← logs SSE en flux
 *   │ │ [preprocess] Recadrage offset (10, 20)…     │ │
 *   │ └─────────────────────────────────────────────┘ │
 *   │ ⚡ Charger les vecteurs sur la carte            │  ← action injectée au succès
 *   └──────────────────────────────────────────────┘
 *
 * Notre agent n'est pas un LLM conversationnel : le « prompt » = lancer le
 * traitement d'une carte déjà téléversée (map_id), avec ou sans SIG.
 *
 * Props :
 *   mapId          {number}   PK de la carte à traiter (déclenche la connexion).
 *   georeference   {boolean}  Mode SIG (true) ou pixel (false, défaut).
 *   apiBase        {string}   Base de l'API REST (défaut import.meta.env).
 *   onGeoJsonReady {func}     Callback(url) appelé quand l'agent renvoie une
 *                             couche GeoJSON valide. Sert à remonter l'URL vers
 *                             App.jsx (état activeGeoJsonUrl) pour que MapViewer
 *                             recharge automatiquement les vecteurs.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

// ── Stepper : 3 phases macro affichées, chacune adossée aux nœuds réels ──────
// Le backend émet des node_start pour perceive/preprocess/vectorize/qa_check/
// self_correct/georef/export. On les regroupe en trois étapes lisibles.
const NODE_PHASES = [
  { id: "pre", label: "Prétraitement",          nodes: ["perceive", "preprocess"] },
  { id: "seg", label: "Segmentation Sémantique", nodes: ["vectorize"] },
  { id: "vec", label: "Vectorisation Space-Image", nodes: ["qa_check", "self_correct", "georef", "export"] },
];

function phaseIndexForNode(node) {
  for (let i = 0; i < NODE_PHASES.length; i += 1) {
    if (NODE_PHASES[i].nodes.includes(node)) return i;
  }
  return -1;
}

function makeThreadId() {
  return `sess-${Math.random().toString(36).slice(2, 12)}`;
}

// ── Rendu markdown minimal (pas de dépendance) ───────────────────────────────
// Gère **gras**, `code` et les retours à la ligne. Construit des nœuds React
// (pas de dangerouslySetInnerHTML : sûr vis-à-vis du contenu agent).
function renderInline(text, keyPrefix) {
  const out = [];
  const regex = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m;
  let i = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(<strong key={`${keyPrefix}-b${i}`}>{tok.slice(2, -2)}</strong>);
    } else {
      out.push(
        <code key={`${keyPrefix}-c${i}`} style={styles.inlineCode}>{tok.slice(1, -1)}</code>,
      );
    }
    last = m.index + tok.length;
    i += 1;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function renderMarkdown(text) {
  return String(text || "")
    .split("\n")
    .map((line, idx) => (
      <span key={idx} style={{ display: "block" }}>
        {renderInline(line, idx) }
      </span>
    ));
}

export default function AgentChat({
  mapId,
  georeference = false,
  apiBase = API_BASE,
  onGeoJsonReady,
}) {
  const [connState, setConnState] = useState("idle"); // idle|connecting|open|done|error
  const [activeNode, setActiveNode] = useState(null);
  const [maxPhaseReached, setMaxPhaseReached] = useState(-1);
  const [messages, setMessages] = useState([]);      // bulles chat {id, role, text}
  const [terminal, setTerminal] = useState([]);      // lignes terminal {id, level, text}
  const [finalMsg, setFinalMsg] = useState(null);    // paquet agent_response
  const [runId, setRunId] = useState(0);             // bump => relance

  const esRef = useRef(null);
  const threadRef = useRef(makeThreadId());
  const lineSeq = useRef(0);
  const msgSeq = useRef(0);
  const scrollRef = useRef(null);
  const termRef = useRef(null);

  // ── Helpers d'ajout (id stable, pas de collision de clés) ──────────────────
  const pushTerminal = useCallback((text, level = "log") => {
    lineSeq.current += 1;
    setTerminal((prev) => [...prev, { id: lineSeq.current, level, text }]);
  }, []);

  const pushMessage = useCallback((role, text) => {
    msgSeq.current += 1;
    setMessages((prev) => [...prev, { id: msgSeq.current, role, text }]);
  }, []);

  const resetState = useCallback(() => {
    setActiveNode(null);
    setMaxPhaseReached(-1);
    setMessages([]);
    setTerminal([]);
    setFinalMsg(null);
  }, []);

  // ── Auto-scroll : on colle le fil et le terminal au bas à chaque update ────
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, finalMsg, connState]);

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight;
  }, [terminal]);

  // ── Connexion SSE ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapId) return undefined;

    resetState();
    setConnState("connecting");

    // Bulle « utilisateur » : le prompt = lancer le traitement de la carte.
    msgSeq.current += 1;
    const launchMsg = {
      id: msgSeq.current,
      role: "user",
      text: `▶ Lancer le traitement — Carte #${mapId} · ${georeference ? "mode SIG (WGS84)" : "mode pixel"}`,
    };
    setMessages([launchMsg]);

    const url = `${apiBase}/agent/stream/`
      + `?map_id=${encodeURIComponent(mapId)}`
      + `&georeference=${georeference ? "true" : "false"}`
      + `&thread_id=${encodeURIComponent(threadRef.current)}`;

    let closedCleanly = false;
    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => setConnState("open");

    es.onmessage = (event) => {
      let pkt;
      try {
        pkt = JSON.parse(event.data);
      } catch {
        return;
      }
      switch (pkt.type) {
        case "open":
          setConnState("open");
          pushTerminal(`→ Connexion établie (thread ${pkt.thread_id || threadRef.current})`, "sys");
          break;

        case "node_start": {
          setActiveNode(pkt.node);
          const idx = phaseIndexForNode(pkt.node);
          if (idx >= 0) setMaxPhaseReached((p) => Math.max(p, idx));
          pushTerminal(`→ Executing: ${pkt.node}`, "node");
          break;
        }

        case "log": {
          const prefix = pkt.node ? `[${pkt.node}] ` : "";
          pushTerminal(`${prefix}${pkt.message || ""}`, "log");
          break;
        }

        case "agent_response":
          setFinalMsg(pkt);
          setActiveNode(null);
          setMaxPhaseReached(NODE_PHASES.length - 1);
          pushTerminal(
            `[SUCCESS] Sortie générée : ${pkt.geojson_url || "(aucune couche)"}`,
            "ok",
          );
          if (pkt.text) pushMessage("agent", pkt.text);
          // Part 2 : remonter l'URL vers App pour auto-charger la carte.
          if (pkt.geojson_url && typeof onGeoJsonReady === "function") {
            onGeoJsonReady(pkt.geojson_url);
          }
          break;

        case "error":
          pushTerminal(`[ERROR] ${pkt.message || "erreur inconnue"}`, "err");
          pushMessage("system", `⚠ ${pkt.message || "Erreur de l'agent."}`);
          setConnState("error");
          break;

        case "done":
          closedCleanly = true;
          pushTerminal("✓ done — flux clos par le serveur.", "sys");
          setConnState((s) => (s === "error" ? "error" : "done"));
          es.close();
          break;

        default:
          break;
      }
    };

    // Coupure réseau. Si le run n'est pas terminé proprement, on signale
    // [CONNECTION LOST] dans le terminal sans figer l'UI.
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        if (!closedCleanly) {
          pushTerminal("[CONNECTION LOST] flux interrompu par le serveur ou le réseau.", "err");
          setConnState((s) => (s === "done" ? "done" : "error"));
        }
        return;
      }
      // Sinon EventSource retente seul : on l'indique sans bloquer.
      pushTerminal("… reconnexion en cours (coupure transitoire).", "sys");
      setConnState("connecting");
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [mapId, georeference, apiBase, runId, resetState, pushTerminal, pushMessage, onGeoJsonReady]);

  const handleRetry = useCallback(() => {
    threadRef.current = makeThreadId(); // nouvelle session isolée
    setRunId((n) => n + 1);
  }, []);

  const handleLoadVectors = useCallback(() => {
    if (finalMsg?.geojson_url && typeof onGeoJsonReady === "function") {
      onGeoJsonReady(finalMsg.geojson_url);
    }
  }, [finalMsg, onGeoJsonReady]);

  if (!mapId) {
    return (
      <div style={styles.wrap}>
        <Header connState="idle" />
        <div style={styles.empty}>Sélectionne une carte pour lancer l'agent.</div>
      </div>
    );
  }

  const activePhase = activeNode != null ? phaseIndexForNode(activeNode) : -1;

  return (
    <div style={styles.wrap}>
      <Header connState={connState} />

      {/* ── Stepper de nœud actif (3 phases) ───────────────────────────── */}
      <div style={styles.stepper}>
        {NODE_PHASES.map((phase, idx) => {
          const isActive = idx === activePhase;
          const isDone = !isActive && idx <= maxPhaseReached;
          const bg = isActive ? "#f59e0b" : isDone ? "#16a34a" : "#1e293b";
          const fg = isActive || isDone ? "#0b1120" : "#94a3b8";
          return (
            <React.Fragment key={phase.id}>
              <div
                style={{
                  ...styles.stepPill,
                  background: bg,
                  color: fg,
                  boxShadow: isActive ? "0 0 0 3px rgba(245,158,11,0.30)" : "none",
                }}
                title={`Nœuds : ${phase.nodes.join(", ")}`}
              >
                {isDone ? "✓ " : isActive ? "• " : ""}{phase.label}
              </div>
              {idx < NODE_PHASES.length - 1 && (
                <span style={styles.arrow}>➔</span>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* ── Fil de discussion auto-scroll ──────────────────────────────── */}
      <div ref={scrollRef} style={styles.feed}>
        {messages.map((msg) => (
          <Bubble key={msg.id} role={msg.role} text={msg.text} />
        ))}

        {/* ── Terminal « System Event » ─────────────────────────────────
            Classes Tailwind demandées rendues en styles inline :
            bg-slate-900 text-green-400 font-mono text-xs p-3 rounded shadow-inner */}
        {terminal.length > 0 && (
          <div style={styles.terminalShell}>
            <div style={styles.terminalLabel}>System Event</div>
            <div ref={termRef} style={styles.terminal}>
              {terminal.map((line) => (
                <div key={line.id} style={{ ...styles.termLine, color: termColor(line.level) }}>
                  {line.text}
                </div>
              ))}
              {connState === "connecting" && (
                <div style={{ ...styles.termLine, color: "#fbbf24" }}>
                  <span style={styles.caret}>▌</span> en attente du flux…
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Action de succès : charger les vecteurs ──────────────────── */}
        {finalMsg && finalMsg.geojson_url && (
          <div style={styles.actionRow}>
            <button type="button" style={styles.loadBtn} onClick={handleLoadVectors}>
              ⚡ Charger les vecteurs sur la carte
            </button>
            {finalMsg.qgis_bundle && (
              <a href={finalMsg.qgis_bundle} download style={styles.qgisBtn}>
                🖥️ Projet QGIS
              </a>
            )}
          </div>
        )}

        {/* ── Erreur : bouton réessayer ────────────────────────────────── */}
        {connState === "error" && (
          <div style={styles.actionRow}>
            <button type="button" style={styles.retryBtn} onClick={handleRetry}>
              ↻ Réessayer
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sous-composants ──────────────────────────────────────────────────────────

function Header({ connState }) {
  const badge = connBadge(connState);
  return (
    <div style={styles.header}>
      <span style={styles.title}>🤖 Agent IA Live — Suivi CartoVec</span>
      <span style={{ ...styles.badge, background: badge.bg }}>{badge.text}</span>
    </div>
  );
}

function Bubble({ role, text }) {
  if (role === "user") {
    return (
      <div style={styles.bubbleRowRight}>
        <div style={{ ...styles.bubble, ...styles.bubbleUser }}>{text}</div>
      </div>
    );
  }
  if (role === "system") {
    return (
      <div style={styles.bubbleRowLeft}>
        <div style={{ ...styles.bubble, ...styles.bubbleSystem }}>{text}</div>
      </div>
    );
  }
  // agent : markdown léger, aligné à gauche
  return (
    <div style={styles.bubbleRowLeft}>
      <div style={{ ...styles.bubble, ...styles.bubbleAgent }}>{renderMarkdown(text)}</div>
    </div>
  );
}

function connBadge(state) {
  switch (state) {
    case "open":       return { text: "● En direct", bg: "#16a34a" };
    case "connecting": return { text: "○ Connexion", bg: "#f59e0b" };
    case "done":       return { text: "✓ Terminé",   bg: "#334155" };
    case "error":      return { text: "✕ Erreur",    bg: "#b91c1c" };
    default:           return { text: "Inactif",     bg: "#475569" };
  }
}

function termColor(level) {
  switch (level) {
    case "err":  return "#f87171"; // rouge clair
    case "ok":   return "#4ade80"; // vert (succès)
    case "node": return "#38bdf8"; // cyan (nœud)
    case "sys":  return "#a3a3a3"; // gris (système)
    default:     return "#4ade80"; // text-green-400 (log)
  }
}

// ── Styles inline (le projet n'utilise PAS Tailwind) ─────────────────────────
const styles = {
  wrap: {
    display: "flex", flexDirection: "column", height: "100%", minHeight: 560,
    fontFamily: "'Inter', system-ui, sans-serif", color: "#e2e8f0",
    background: "#0b1120",
  },
  header: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    gap: 8, padding: "12px 14px", borderBottom: "1px solid rgba(255,255,255,0.08)",
    background: "#0f172a",
  },
  title: { fontSize: 14, fontWeight: 700, letterSpacing: 0.2 },
  badge: {
    padding: "3px 10px", borderRadius: 999, fontSize: 11, fontWeight: 700,
    color: "#fff", whiteSpace: "nowrap",
  },
  empty: { padding: 24, color: "#64748b", textAlign: "center", fontSize: 13 },

  stepper: {
    display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap",
    padding: "10px 14px", borderBottom: "1px solid rgba(255,255,255,0.06)",
  },
  stepPill: {
    padding: "5px 10px", borderRadius: 8, fontSize: 11, fontWeight: 700,
    transition: "all 0.2s", whiteSpace: "nowrap",
  },
  arrow: { color: "#475569", fontSize: 13 },

  feed: {
    flex: 1, overflowY: "auto", padding: 14, display: "flex",
    flexDirection: "column", gap: 10,
  },
  bubbleRowLeft:  { display: "flex", justifyContent: "flex-start" },
  bubbleRowRight: { display: "flex", justifyContent: "flex-end" },
  bubble: {
    maxWidth: "85%", padding: "8px 12px", borderRadius: 12, fontSize: 13,
    lineHeight: 1.45, wordBreak: "break-word",
  },
  bubbleUser: {
    background: "#2563eb", color: "#fff", borderBottomRightRadius: 3,
  },
  bubbleAgent: {
    background: "#1e293b", color: "#e2e8f0", borderBottomLeftRadius: 3,
    border: "1px solid rgba(255,255,255,0.06)",
  },
  bubbleSystem: {
    background: "rgba(185,28,28,0.15)", color: "#fca5a5", borderBottomLeftRadius: 3,
    border: "1px solid rgba(185,28,28,0.45)",
  },
  inlineCode: {
    background: "rgba(148,163,184,0.18)", padding: "1px 5px", borderRadius: 4,
    fontFamily: "ui-monospace, 'JetBrains Mono', monospace", fontSize: 12,
  },

  // Terminal « System Event » :
  // bg-slate-900 #0f172a / text-green-400 #4ade80 / font-mono / text-xs 12px
  // p-3 12px / rounded 6px / shadow-inner (boxShadow inset)
  terminalShell: { display: "flex", flexDirection: "column", gap: 4 },
  terminalLabel: {
    fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase",
    color: "#64748b",
  },
  terminal: {
    background: "#0f172a", color: "#4ade80",
    fontFamily: "ui-monospace, 'JetBrains Mono', 'Courier New', monospace",
    fontSize: 12, padding: 12, borderRadius: 6,
    boxShadow: "inset 0 2px 6px rgba(0,0,0,0.6)",
    maxHeight: 220, overflowY: "auto",
    border: "1px solid rgba(255,255,255,0.05)",
  },
  termLine: { padding: "1px 0", whiteSpace: "pre-wrap", lineHeight: 1.5 },
  caret: { opacity: 0.7 },

  actionRow: { display: "flex", gap: 8, flexWrap: "wrap", marginTop: 2 },
  loadBtn: {
    padding: "9px 16px", background: "#f59e0b", color: "#0b1120", border: "none",
    borderRadius: 8, cursor: "pointer", fontWeight: 800, fontSize: 13,
    boxShadow: "0 2px 10px rgba(245,158,11,0.35)",
  },
  qgisBtn: {
    padding: "9px 14px", background: "#7c3aed", color: "#fff", borderRadius: 8,
    textDecoration: "none", fontWeight: 700, fontSize: 13,
    display: "inline-flex", alignItems: "center",
  },
  retryBtn: {
    padding: "8px 16px", background: "#b91c1c", color: "#fff", border: "none",
    borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 13,
  },
};
