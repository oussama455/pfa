/**
 * TrainingPanel.jsx
 *
 * Panneau d'entrainement U-Net :
 *  - Formulaire : dataset (soduco|semap), epochs, batch_size, target_size, lr, encoder, flags
 *  - Bouton "Lancer" -> POST /api/training/
 *  - Liste des jobs avec polling toutes les 5 s
 *  - Pour chaque job done : bouton "Telecharger .pth" + bouton "Utiliser pour l'inference"
 *  - Pour chaque job running : affichage des dernieres lignes de log (collapsable)
 *
 * Props :
 *   apiBase        : prefixe API (defaut "/api")
 *   onWeightsSelected : callback(weights_path: string) appele quand
 *                       l'utilisateur clique "Utiliser pour l'inference"
 */
import React, { useEffect, useState } from "react";

const POLL_MS = 5000;

const DEFAULT_FORM = {
  dataset: "semap",
  epochs: 20,
  batch_size: 8,
  target_size: 512,
  learning_rate: 0.0001,
  encoder: "resnet34",
  no_synthetic: false,
  no_augment: false,
};

function fmtDate(s) {
  if (!s) return "—";
  return new Date(s).toLocaleString("fr-TN", {
    dateStyle: "short", timeStyle: "short",
  });
}

function fmtMiou(v) {
  if (v == null) return "—";
  return v.toFixed(4);
}

function statusColor(s) {
  return {
    queued:  "#888",
    running: "#1976d2",
    done:    "#2e7d32",
    failed:  "#c62828",
    aborted: "#ef6c00",
  }[s] || "#555";
}

export default function TrainingPanel({ apiBase = "/api", onWeightsSelected }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [jobs, setJobs] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [expandedLog, setExpandedLog] = useState(null);  // id du job dont on affiche le log
  const [logContent, setLogContent] = useState("");

  const refresh = async () => {
    try {
      const res = await fetch(`${apiBase}/training/`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setJobs(await res.json());
    } catch (e) {
      setError(`Erreur chargement /api/training/ : ${e.message}`);
    }
  };

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [apiBase]);

  useEffect(() => {
    if (!expandedLog) return;
    let stopped = false;
    const fetchLog = async () => {
      try {
        const res = await fetch(`${apiBase}/training/${expandedLog}/log/`);
        if (!res.ok) return;
        const txt = await res.text();
        if (!stopped) setLogContent(txt);
      } catch {}
    };
    fetchLog();
    const t = setInterval(fetchLog, 3000);
    return () => { stopped = true; clearInterval(t); };
  }, [expandedLog, apiBase]);

  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;
    setForm((f) => ({
      ...f,
      [name]: type === "checkbox" ? checked
            : type === "number"   ? Number(value) : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const res = await fetch(`${apiBase}/training/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`HTTP ${res.status} : ${t}`);
      }
      await refresh();
    } catch (e) {
      setError(`Erreur creation job : ${e.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="training-panel" style={{ border: "1px solid #ccc",
                                              borderRadius: 8, padding: 16,
                                              background: "#fafafa" }}>
      <h2 style={{ marginTop: 0 }}>Entrainement U-Net</h2>

      <form onSubmit={handleSubmit} style={{ display: "grid",
                                              gridTemplateColumns: "repeat(2, 1fr)",
                                              gap: 12, marginBottom: 16 }}>
        <label>Dataset
          <select name="dataset" value={form.dataset} onChange={handleChange}
                  style={{ width: "100%", padding: 6 }}>
            <option value="semap">SEMAP (6 classes, 10 703 train)</option>
            <option value="soduco">SODUCO (5 classes, 256 train)</option>
          </select>
        </label>
        <label>Epochs
          <input type="number" name="epochs" value={form.epochs}
                  min={1} max={500} onChange={handleChange}
                  style={{ width: "100%", padding: 6 }} />
        </label>
        <label>Batch size
          <input type="number" name="batch_size" value={form.batch_size}
                  min={1} max={64} onChange={handleChange}
                  style={{ width: "100%", padding: 6 }} />
        </label>
        <label>Target size (px)
          <input type="number" name="target_size" value={form.target_size}
                  min={128} max={1024} step={32} onChange={handleChange}
                  style={{ width: "100%", padding: 6 }} />
        </label>
        <label>Learning rate
          <input type="number" name="learning_rate" value={form.learning_rate}
                  step={1e-5} onChange={handleChange}
                  style={{ width: "100%", padding: 6 }} />
        </label>
        <label>Encoder
          <select name="encoder" value={form.encoder} onChange={handleChange}
                  style={{ width: "100%", padding: 6 }}>
            <option value="resnet18">resnet18 (leger)</option>
            <option value="resnet34">resnet34 (defaut)</option>
            <option value="resnet50">resnet50 (precis)</option>
            <option value="efficientnet-b0">efficientnet-b0</option>
          </select>
        </label>
        <label>
          <input type="checkbox" name="no_synthetic" checked={form.no_synthetic}
                  onChange={handleChange} />
          {" "}Pas de synthetic (SEMAP only)
        </label>
        <label>
          <input type="checkbox" name="no_augment" checked={form.no_augment}
                  onChange={handleChange} />
          {" "}Pas d'augmentation
        </label>

        <button type="submit" disabled={submitting}
                style={{ gridColumn: "1 / -1", padding: "10px",
                          background: "#1976d2", color: "white",
                          border: 0, borderRadius: 4, cursor: "pointer",
                          fontWeight: 600 }}>
          {submitting ? "Lancement…" : "▶ Lancer l'entrainement"}
        </button>
      </form>

      {error && <p style={{ color: "#c00", fontSize: 13 }}>{error}</p>}

      <h3>Historique ({jobs.length} jobs)</h3>
      {jobs.length === 0 && <p style={{ color: "#888" }}>(aucun entrainement encore)</p>}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#eee" }}>
            <th style={{ padding: 6, textAlign: "left" }}>ID</th>
            <th style={{ padding: 6, textAlign: "left" }}>Dataset</th>
            <th style={{ padding: 6, textAlign: "left" }}>Params</th>
            <th style={{ padding: 6, textAlign: "left" }}>Statut</th>
            <th style={{ padding: 6, textAlign: "right" }}>mIoU</th>
            <th style={{ padding: 6, textAlign: "left" }}>Cree</th>
            <th style={{ padding: 6, textAlign: "left" }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((j) => (
            <tr key={j.id} style={{ borderBottom: "1px solid #eee" }}>
              <td style={{ padding: 6 }}>{j.id}</td>
              <td style={{ padding: 6 }}>{j.dataset}</td>
              <td style={{ padding: 6, fontSize: 11, color: "#666" }}>
                ep={j.epochs} bs={j.batch_size} ts={j.target_size}
                {j.no_synthetic && " (no-syn)"}
              </td>
              <td style={{ padding: 6, color: statusColor(j.status), fontWeight: 600 }}>
                {j.status}
              </td>
              <td style={{ padding: 6, textAlign: "right" }}>{fmtMiou(j.best_miou)}</td>
              <td style={{ padding: 6, fontSize: 11 }}>{fmtDate(j.created_at)}</td>
              <td style={{ padding: 6 }}>
                <button onClick={() => setExpandedLog(expandedLog === j.id ? null : j.id)}
                        style={{ fontSize: 11, marginRight: 4 }}>
                  {expandedLog === j.id ? "Cacher log" : "Voir log"}
                </button>
                {j.download_url && (
                  <a href={j.download_url} download
                      style={{ fontSize: 11, marginRight: 4 }}>
                    ⬇ .pth
                  </a>
                )}
                {j.output_weights_path && onWeightsSelected && (
                  <button onClick={() => onWeightsSelected(j.output_weights_path)}
                          style={{ fontSize: 11 }}>
                    Utiliser
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {expandedLog && (
        <div style={{ marginTop: 12 }}>
          <h4>Log du job #{expandedLog}</h4>
          <pre style={{ background: "#111", color: "#0f0", padding: 10,
                         maxHeight: 360, overflow: "auto", fontSize: 11,
                         fontFamily: "monospace" }}>
            {logContent || "(vide)"}
          </pre>
        </div>
      )}
    </div>
  );
}
