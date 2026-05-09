import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import MapViewer from "./components/MapViewer.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";
const STATUS_POLL_MS = 2500;

function statusLabel(status) {
  const labels = {
    pending: "En attente",
    processing: "En cours",
    done: "Termine",
    error: "Erreur",
    failed: "Echec",
  };
  return labels[status] || status || "Inconnu";
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("fr-TN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function defaultBounds(map) {
  return map?.raster_bounds || [[33.2, 8.0], [37.4, 11.8]];
}

function App() {
  const [maps, setMaps] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedMap, setSelectedMap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({ title: "", map_name: "", raster: null });

  const selected = useMemo(
    () => selectedMap || maps.find((item) => item.id === selectedId),
    [maps, selectedId, selectedMap],
  );

  const fetchMaps = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${API_BASE}/maps/`);
      setMaps(data);
      setError("");
      if (!selectedId && data.length) setSelectedId(data[0].id);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedId]);

  const fetchSelected = useCallback(async (id) => {
    if (!id) {
      setSelectedMap(null);
      return;
    }
    try {
      const { data } = await axios.get(`${API_BASE}/maps/${id}/`);
      setSelectedMap(data);
      setMaps((items) => items.map((item) => (item.id === data.id ? data : item)));
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  }, []);

  useEffect(() => {
    fetchMaps();
  }, [fetchMaps]);

  useEffect(() => {
    fetchSelected(selectedId);
  }, [fetchSelected, selectedId]);

  useEffect(() => {
    if (!selected || !["pending", "processing"].includes(selected.status)) return;
    const timer = setInterval(() => fetchSelected(selected.id), STATUS_POLL_MS);
    return () => clearInterval(timer);
  }, [fetchSelected, selected]);

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!form.raster) {
      setError("Choisis une image raster avant de lancer le traitement.");
      return;
    }

    const body = new FormData();
    body.append("title", form.title || form.raster.name);
    body.append("map_name", form.map_name);
    body.append("raster", form.raster);

    setUploading(true);
    try {
      const { data } = await axios.post(`${API_BASE}/maps/`, body);
      setMaps((items) => [data, ...items.filter((item) => item.id !== data.id)]);
      setSelectedId(data.id);
      setSelectedMap(data);
      setForm({ title: "", map_name: "", raster: null });
      event.target.reset();
      setError("");
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div>
            <h1>CartoVec</h1>
            <p>Vectorisation et correction de cartes historiques</p>
          </div>
          <span className="live-dot" title="Frontend actif" />
        </div>

        <form className="upload-panel" onSubmit={handleUpload}>
          <label>
            Titre
            <input
              value={form.title}
              onChange={(e) => setForm((state) => ({ ...state, title: e.target.value }))}
              placeholder="Tunis 1969"
            />
          </label>
          <label>
            Nom de feuille
            <input
              value={form.map_name}
              onChange={(e) => setForm((state) => ({ ...state, map_name: e.target.value }))}
              placeholder="tunis"
            />
          </label>
          <label>
            Raster
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.tif,.tiff"
              onChange={(e) => setForm((state) => ({ ...state, raster: e.target.files?.[0] || null }))}
            />
          </label>
          <button type="submit" disabled={uploading}>
            {uploading ? "Envoi..." : "Lancer"}
          </button>
        </form>

        {error && <div className="error-box">{error}</div>}

        <div className="section-title">Cartes recentes</div>
        <div className="map-list">
          {loading && <div className="muted-line">Chargement...</div>}
          {!loading && maps.length === 0 && (
            <div className="muted-line">Aucune carte traitee pour le moment.</div>
          )}
          {maps.map((item) => (
            <button
              key={item.id}
              className={`map-row ${item.id === selected?.id ? "active" : ""}`}
              onClick={() => setSelectedId(item.id)}
            >
              <span>
                <strong>{item.title}</strong>
                <small>{formatDate(item.created_at)}</small>
              </span>
              <em className={`status ${item.status}`}>{statusLabel(item.status)}</em>
            </button>
          ))}
        </div>
      </aside>

      <main className="workspace">
        {!selected && (
          <div className="empty-state">
            <h2>Ajoute une carte pour commencer</h2>
            <p>Les couches vectorielles et les outils de correction apparaitront ici.</p>
          </div>
        )}

        {selected && (
          <>
            <header className="workspace-header">
              <div>
                <h2>{selected.title}</h2>
                <p>
                  {formatDate(selected.created_at)}
                  {selected.confidence_score != null && (
                    <> · confiance {(selected.confidence_score * 100).toFixed(0)}%</>
                  )}
                </p>
              </div>
              <span className={`status ${selected.status}`}>{statusLabel(selected.status)}</span>
            </header>

            {selected.error_message && (
              <pre className="error-box wide">{selected.error_message}</pre>
            )}

            <section className="viewer-panel">
              <MapViewer
                mapId={selected.id}
                rasterUrl={selected.raster_url}
                rasterBounds={defaultBounds(selected)}
                apiBaseUrl={API_BASE}
              />
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
