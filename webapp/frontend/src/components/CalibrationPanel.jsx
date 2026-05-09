/**
 * webapp/frontend/src/components/CalibrationPanel.jsx
 *
 * Active Learning calibration status panel for the CartoVec MapViewer.
 *
 * PURPOSE:
 *   Shows the operator in real-time how their polygon corrections are
 *   improving the AI's HSV detection thresholds. This closes the feedback
 *   loop visually: "You corrected 5 polygons → the model now detects
 *   red roads 23% more accurately."
 *
 * FEATURES:
 *   - Per-layer correction count + "ACTIVE" badge (≥3 corrections)
 *   - Live HSV range visualization (colored bars for H, S, V)
 *   - Correction history timeline
 *   - Reset button (admin only)
 *   - Auto-refresh every 10s after corrections are saved
 *
 * USAGE (inside MapViewer.jsx):
 *   import CalibrationPanel from './CalibrationPanel';
 *
 *   <CalibrationPanel
 *     mapSeries="ams_tunisia"
 *     apiBaseUrl="/api"
 *     correctionCount={queue.length + savedCount}
 *     isAdmin={user?.is_staff}
 *   />
 */

import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const LAYER_COLORS = {
  red_roads:  "#e74c3c",
  buildings:  "#7f8c8d",
  vegetation: "#27ae60",
  contours:   "#8B4513",
  water:      "#3498db",
  default:    "#9b59b6",
};

const HSV_BAR_COLORS = {
  H: "linear-gradient(to right, #e74c3c, #f39c12, #2ecc71, #3498db, #9b59b6, #e74c3c)",
  S: "linear-gradient(to right, #bdc3c7, #e74c3c)",
  V: "linear-gradient(to right, #2c3e50, #ecf0f1)",
};

// ─────────────────────────────────────────────────────────────────────────────
// Sub: HSV Range Bar
// ─────────────────────────────────────────────────────────────────────────────

function HSVBar({ channel, min, max, maxVal }) {
  const left  = (min / maxVal) * 100;
  const width = ((max - min) / maxVal) * 100;

  return (
    <div style={barStyles.wrapper} title={`${channel}: [${min}, ${max}]`}>
      <span style={barStyles.label}>{channel}</span>
      <div style={{ ...barStyles.track, background: HSV_BAR_COLORS[channel] }}>
        <div
          style={{
            ...barStyles.range,
            left:  `${left}%`,
            width: `${width}%`,
          }}
        />
      </div>
      <span style={barStyles.values}>{min}–{max}</span>
    </div>
  );
}

const barStyles = {
  wrapper: {
    display: "flex", alignItems: "center", gap: 6,
    marginBottom: 4,
  },
  label: {
    width: 14, fontSize: 10, fontWeight: 700,
    color: "#95a5a6", fontFamily: "monospace",
  },
  track: {
    flex: 1, height: 6, borderRadius: 3, position: "relative",
    opacity: 0.6,
  },
  range: {
    position: "absolute", top: 0, height: "100%",
    background: "rgba(255,255,255,0.85)",
    borderRadius: 3, boxShadow: "0 0 4px rgba(255,255,255,0.6)",
  },
  values: {
    width: 54, fontSize: 9, color: "#bdc3c7",
    fontFamily: "monospace", textAlign: "right",
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Sub: Layer Card
// ─────────────────────────────────────────────────────────────────────────────

function LayerCard({ name, data, minCorrections }) {
  const color     = LAYER_COLORS[name] || LAYER_COLORS.default;
  const isActive  = data.corrections >= minCorrections;
  const progress  = Math.min(data.corrections / minCorrections, 1);

  return (
    <div style={{ ...cardStyles.card, borderLeft: `3px solid ${color}` }}>
      {/* Header */}
      <div style={cardStyles.header}>
        <span style={{ ...cardStyles.name, color }}>{name}</span>
        <span style={{
          ...cardStyles.badge,
          background: isActive ? "#27ae60" : "#7f8c8d",
        }}>
          {isActive ? "✓ ACTIVE" : `${data.corrections}/${minCorrections}`}
        </span>
      </div>

      {/* Progress bar toward activation */}
      {!isActive && (
        <div style={cardStyles.progressTrack}>
          <div style={{ ...cardStyles.progressFill, width: `${progress * 100}%`, background: color }} />
        </div>
      )}

      {/* HSV ranges (only if active) */}
      {isActive && (
        <div style={{ marginTop: 6 }}>
          <HSVBar channel="H" min={data.H[0]} max={data.H[1]} maxVal={179} />
          <HSVBar channel="S" min={data.S[0]} max={data.S[1]} maxVal={255} />
          <HSVBar channel="V" min={data.V[0]} max={data.V[1]} maxVal={255} />
        </div>
      )}

      {/* Last updated */}
      {data.last_updated && (
        <div style={cardStyles.updated}>
          Updated: {new Date(data.last_updated).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

const cardStyles = {
  card: {
    background: "rgba(255,255,255,0.04)",
    borderRadius: 6, padding: "8px 10px",
    marginBottom: 8,
  },
  header: {
    display: "flex", justifyContent: "space-between",
    alignItems: "center", marginBottom: 4,
  },
  name: {
    fontWeight: 700, fontSize: 11,
    textTransform: "uppercase", letterSpacing: 0.5,
  },
  badge: {
    fontSize: 9, fontWeight: 700, color: "#fff",
    padding: "2px 6px", borderRadius: 10,
    letterSpacing: 0.3,
  },
  progressTrack: {
    height: 3, borderRadius: 2,
    background: "rgba(255,255,255,0.1)", overflow: "hidden",
    marginTop: 4,
  },
  progressFill: {
    height: "100%", borderRadius: 2,
    transition: "width 0.5s ease",
  },
  updated: {
    marginTop: 4, fontSize: 9,
    color: "#57606f", fontFamily: "monospace",
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Sub: Correction History Timeline
// ─────────────────────────────────────────────────────────────────────────────

function HistoryTimeline({ history }) {
  if (!history.length) {
    return (
      <div style={timelineStyles.empty}>
        No corrections yet. Edit or delete a polygon to start calibrating.
      </div>
    );
  }

  return (
    <div style={timelineStyles.container}>
      {history.slice(0, 10).map((item) => (
        <div key={item.id} style={timelineStyles.item}>
          <span style={{
            ...timelineStyles.typeBadge,
            background: item.type === "edit" ? "#2980b9" : "#c0392b",
          }}>
            {item.type === "edit" ? "✏" : "🗑"} {item.type}
          </span>
          <span style={timelineStyles.layer}>
            {item.layer}
          </span>
          <span style={timelineStyles.time}>
            {new Date(item.created_at).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
}

const timelineStyles = {
  container: { maxHeight: 160, overflowY: "auto" },
  item: {
    display: "flex", alignItems: "center", gap: 8,
    padding: "4px 0",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
    fontSize: 11,
  },
  typeBadge: {
    fontSize: 9, color: "#fff", padding: "1px 6px",
    borderRadius: 8, fontWeight: 700, flexShrink: 0,
  },
  layer: {
    flex: 1, color: "#bdc3c7", fontFamily: "monospace",
  },
  time: {
    color: "#57606f", fontSize: 9, fontFamily: "monospace",
  },
  empty: {
    color: "#57606f", fontSize: 11, fontStyle: "italic",
    textAlign: "center", padding: "12px 0",
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Main CalibrationPanel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CalibrationPanel — Active Learning status panel.
 *
 * Props:
 *   mapSeries       {string}   Map series key: "ams_tunisia" | "ams_algeria"
 *   mapId           {number}   MapUpload ID (for history fetch)
 *   apiBaseUrl      {string}   API base URL
 *   correctionCount {number}   Total corrections saved (triggers refresh)
 *   isAdmin         {boolean}  Show reset button if true
 */
export default function CalibrationPanel({
  mapSeries = "ams_tunisia",
  mapId,
  apiBaseUrl = "/api",
  correctionCount = 0,
  isAdmin = false,
}) {
  const [calibStatus, setCalibStatus]   = useState(null);
  const [history, setHistory]           = useState([]);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);
  const [tab, setTab]                   = useState("ranges");   // "ranges" | "history"
  const [resetting, setResetting]       = useState(false);
  const [collapsed, setCollapsed]       = useState(false);

  // ── Fetch calibration status ──────────────────────────────────────────────
  const fetchStatus = useCallback(async () => {
    try {
      const [statusRes, histRes] = await Promise.all([
        axios.get(`${apiBaseUrl}/calibration/${mapSeries}/`),
        mapId
          ? axios.get(`${apiBaseUrl}/calibration/history/?map_id=${mapId}`)
          : Promise.resolve({ data: { history: [] } }),
      ]);
      setCalibStatus(statusRes.data);
      setHistory(histRes.data.history || []);
      setError(null);
    } catch (err) {
      setError(err.response?.status === 503
        ? "Active Learning module not available"
        : err.message);
    } finally {
      setLoading(false);
    }
  }, [mapSeries, mapId, apiBaseUrl]);

  // Fetch on mount and whenever correctionCount changes
  useEffect(() => { fetchStatus(); }, [fetchStatus, correctionCount]);

  // Auto-refresh every 10s when there are recent corrections
  useEffect(() => {
    if (correctionCount === 0) return;
    const interval = setInterval(fetchStatus, 10_000);
    return () => clearInterval(interval);
  }, [fetchStatus, correctionCount]);

  // ── Reset calibration ──────────────────────────────────────────────────────
  const handleReset = async () => {
    if (!window.confirm(`Reset calibration for "${mapSeries}" to defaults?`)) return;
    setResetting(true);
    try {
      await axios.post(`${apiBaseUrl}/calibration/${mapSeries}/reset/`);
      await fetchStatus();
    } catch (err) {
      setError("Reset failed: " + err.message);
    } finally {
      setResetting(false);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  const totalCorrections = history.length;
  const activeLayerCount = calibStatus
    ? Object.values(calibStatus.layers || {}).filter(l => l.active).length
    : 0;

  return (
    <div style={panelStyles.container}>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div style={panelStyles.header} onClick={() => setCollapsed(c => !c)}>
        <div style={panelStyles.headerLeft}>
          <span style={panelStyles.icon}>🧠</span>
          <span style={panelStyles.title}>Active Learning</span>
          {calibStatus?.active && (
            <span style={panelStyles.activePill}>CALIBRATING</span>
          )}
        </div>
        <div style={panelStyles.headerRight}>
          <span style={panelStyles.stat}>{totalCorrections} corrections</span>
          <span style={panelStyles.chevron}>{collapsed ? "▶" : "▼"}</span>
        </div>
      </div>

      {/* ── Body ───────────────────────────────────────────────────────── */}
      {!collapsed && (
        <div style={panelStyles.body}>

          {loading && <div style={panelStyles.loading}>Loading calibration data…</div>}
          {error   && <div style={panelStyles.error}>{error}</div>}

          {!loading && !error && calibStatus && (
            <>
              {/* Summary row */}
              <div style={panelStyles.summary}>
                <div style={panelStyles.summaryItem}>
                  <span style={panelStyles.summaryNum}>{activeLayerCount}</span>
                  <span style={panelStyles.summaryLabel}>active layers</span>
                </div>
                <div style={panelStyles.summaryItem}>
                  <span style={panelStyles.summaryNum}>{totalCorrections}</span>
                  <span style={panelStyles.summaryLabel}>total corrections</span>
                </div>
                <div style={panelStyles.summaryItem}>
                  <span style={panelStyles.summaryNum}>{calibStatus.min_corrections_needed}</span>
                  <span style={panelStyles.summaryLabel}>needed to activate</span>
                </div>
              </div>

              {/* Tab bar */}
              <div style={panelStyles.tabs}>
                {["ranges", "history"].map(t => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    style={{
                      ...panelStyles.tab,
                      background: tab === t ? "rgba(243,156,18,0.2)" : "transparent",
                      color: tab === t ? "#f39c12" : "#7f8c8d",
                      borderBottom: tab === t ? "2px solid #f39c12" : "2px solid transparent",
                    }}
                  >
                    {t === "ranges" ? "📊 Ranges" : "📋 History"}
                  </button>
                ))}
              </div>

              {/* Tab content */}
              {tab === "ranges" && (
                <div>
                  {Object.entries(calibStatus.layers).map(([name, data]) => (
                    <LayerCard
                      key={name}
                      name={name}
                      data={data}
                      minCorrections={calibStatus.min_corrections_needed}
                    />
                  ))}
                </div>
              )}

              {tab === "history" && (
                <HistoryTimeline history={history} />
              )}

              {/* Info banner */}
              <div style={panelStyles.infoBanner}>
                <strong>How it works:</strong> Each polygon edit updates the HSV
                detection thresholds for the next run. After {calibStatus.min_corrections_needed} corrections
                per layer, the AI uses your calibrated ranges automatically.
              </div>

              {/* Admin reset */}
              {isAdmin && (
                <button
                  onClick={handleReset}
                  disabled={resetting}
                  style={panelStyles.resetBtn}
                >
                  {resetting ? "Resetting…" : "↺ Reset to Defaults"}
                </button>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────────

const panelStyles = {
  container: {
    position: "absolute", bottom: 20, left: 10, zIndex: 1000,
    width: 280,
    background: "rgba(15, 20, 30, 0.93)",
    color: "#ecf0f1",
    borderRadius: 10,
    border: "1px solid rgba(243,156,18,0.3)",
    fontFamily: "'JetBrains Mono', 'Courier New', monospace",
    fontSize: 11,
    boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
    overflow: "hidden",
  },
  header: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "10px 14px",
    background: "rgba(243,156,18,0.08)",
    borderBottom: "1px solid rgba(243,156,18,0.15)",
    cursor: "pointer",
    userSelect: "none",
  },
  headerLeft:  { display: "flex", alignItems: "center", gap: 8 },
  headerRight: { display: "flex", alignItems: "center", gap: 10 },
  icon:        { fontSize: 14 },
  title:       { fontWeight: 700, color: "#f39c12", letterSpacing: 0.5 },
  activePill: {
    fontSize: 8, fontWeight: 700, color: "#27ae60",
    background: "rgba(39,174,96,0.15)",
    padding: "1px 6px", borderRadius: 8,
    border: "1px solid rgba(39,174,96,0.4)",
    letterSpacing: 1,
  },
  stat:    { color: "#7f8c8d", fontSize: 10 },
  chevron: { color: "#7f8c8d", fontSize: 10 },
  body:    { padding: "10px 14px", maxHeight: 420, overflowY: "auto" },
  loading: { color: "#7f8c8d", fontStyle: "italic", textAlign: "center", padding: 12 },
  error:   { color: "#e74c3c", fontSize: 10, padding: 8,
             background: "rgba(231,76,60,0.1)", borderRadius: 4 },
  summary: {
    display: "flex", gap: 0, marginBottom: 12,
    background: "rgba(255,255,255,0.03)",
    borderRadius: 6, overflow: "hidden",
  },
  summaryItem: {
    flex: 1, display: "flex", flexDirection: "column",
    alignItems: "center", padding: "8px 4px",
    borderRight: "1px solid rgba(255,255,255,0.05)",
  },
  summaryNum:   { fontSize: 20, fontWeight: 700, color: "#f39c12", lineHeight: 1.1 },
  summaryLabel: { fontSize: 8, color: "#7f8c8d", textAlign: "center",
                  textTransform: "uppercase", letterSpacing: 0.3, marginTop: 2 },
  tabs: {
    display: "flex", marginBottom: 10,
    borderBottom: "1px solid rgba(255,255,255,0.08)",
  },
  tab: {
    flex: 1, padding: "6px 0", border: "none",
    cursor: "pointer", fontSize: 10, fontWeight: 700,
    fontFamily: "inherit", letterSpacing: 0.5,
    transition: "all 0.15s",
  },
  infoBanner: {
    marginTop: 10,
    background: "rgba(243,156,18,0.07)",
    border: "1px solid rgba(243,156,18,0.2)",
    borderRadius: 6, padding: "8px 10px",
    fontSize: 10, color: "#bdc3c7", lineHeight: 1.6,
  },
  resetBtn: {
    width: "100%", marginTop: 10, padding: "6px 0",
    background: "transparent", border: "1px solid rgba(231,76,60,0.4)",
    color: "#e74c3c", borderRadius: 5, cursor: "pointer",
    fontSize: 10, fontWeight: 700, fontFamily: "inherit",
    letterSpacing: 0.5,
  },
};
