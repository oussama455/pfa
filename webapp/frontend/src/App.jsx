import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import MapViewer from "./components/MapViewer.jsx";
import AgentChat from "./components/AgentChat.jsx";
import TrainingPanel from "./components/TrainingPanel.jsx";
import WeightsSelector from "./components/WeightsSelector.jsx";

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
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    title: "", map_name: "", raster: null, unet_weights: null,
    georeference: false,
  });
  const [showTraining, setShowTraining] = useState(false);

  // ── État partagé Agent ↔ Carte ─────────────────────────────────────────────
  // Quand l'agent termine (paquet agent_response avec geojson_url valide),
  // AgentChat remonte l'URL ici via onGeoJsonReady. On s'en sert comme jeton de
  // rechargement : MapViewer ré-interroge /maps/{id}/geojson/ et affiche les
  // vecteurs fraîchement produits, sans rechargement manuel.
  const [activeGeoJsonUrl, setActiveGeoJsonUrl] = useState(null);

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
    // Carte changée => on oublie les vecteurs de l'agent précédent.
    setActiveGeoJsonUrl(null);
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
    body.append("georeference", form.georeference ? "true" : "false");
    if (form.unet_weights) {
      body.append("unet_weights", form.unet_weights);
    }

    setUploading(true);
    try {
      const { data } = await axios.post(`${API_BASE}/maps/`, body);
      setMaps((items) => [data, ...items.filter((item) => item.id !== data.id)]);
      setSelectedId(data.id);
      setSelectedMap(data);
      setForm({ title: "", map_name: "", raster: null,
                unet_weights: form.unet_weights,
                georeference: form.georeference });
      event.target.reset();
      setError("");
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleShapefileDownload = async () => {
    if (!selected) return;

    setDownloading(true);
    try {
      const { data } = await axios.get(`${API_BASE}/maps/${selected.id}/shapefiles/`, {
        responseType: "blob",
      });
      const url = window.URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.download = `cartovec_export_${selected.id}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      setError("");
    } catch (err) {
      if (err.response?.data instanceof Blob) {
        try {
          const payload = JSON.parse(await err.response.data.text());
          setError(payload.detail || err.message);
        } catch {
          setError(err.message);
        }
      } else {
        setError(err.response?.data?.detail || err.message);
      }
    } finally {
      setDownloading(false);
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
          <div style={{ marginTop: 8 }}>
            <WeightsSelector
              apiBase={API_BASE}
              value={form.unet_weights}
              onChange={(v) => setForm((s) => ({ ...s, unet_weights: v }))}
            />
          </div>
          <label
            style={{
              display: "flex", alignItems: "center", gap: 6, marginTop: 10,
              cursor: "pointer", fontSize: 12,
            }}
            title="Décoché (défaut) : sortie en coordonnées pixel image, prête
pour un calque direct sur le raster. Coché : applique le géoréférencement
AMS/GSGS pour produire du WGS84/EPSG:4326."
          >
            <input
              type="checkbox"
              checked={form.georeference}
              onChange={(e) =>
                setForm((s) => ({ ...s, georeference: e.target.checked }))
              }
            />
            <span>Activer le géoréférencement (SIG)</span>
          </label>
          <button type="submit" disabled={uploading}>
            {uploading ? "Envoi..." : "Lancer"}
          </button>
        </form>

        {error && <div className="error-box">{error}</div>}

        <button
          type="button"
          className="training-toggle"
          onClick={() => setShowTraining((value) => !value)}
        >
          {showTraining ? "Masquer training" : "Training U-Net"}
        </button>

        {showTraining && (
          <TrainingPanel
            apiBase={API_BASE}
            onWeightsSelected={(weightsPath) => {
              setForm((state) => ({ ...state, unet_weights: weightsPath }));
              setShowTraining(false);
            }}
          />
        )}

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
              <div className="workspace-actions">
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
                  <button
                    type="button"
                    className="download-button"
                    onClick={handleShapefileDownload}
                    disabled={
                      selected.status !== "done"
                      || downloading
                      || !selected.has_georeference
                    }
                    title={
                      !selected.has_georeference
                        ? "Export QGIS indisponible en mode pixel. Relance le traitement en cochant « Activer le géoréférencement (SIG) »."
                        : selected.status === "done"
                          ? "Télécharger le projet QGIS (.qgs) + shapefiles lissés"
                          : "Le traitement doit être terminé avant l'export QGIS"
                    }
                  >
                    {downloading ? "Préparation..." : "Exporter le projet QGIS"}
                  </button>
                  {selected.has_georeference ? (
                    <small style={{ color: "#7f8c8d", fontSize: 11, maxWidth: 260, textAlign: "right" }}>
                      🖥️ Inclut un projet <code>.qgs</code> : ouvrez le fichier
                      extrait pour lancer directement QGIS avec vos calques
                      stylisés et lissés.
                    </small>
                  ) : (
                    <small style={{ color: "#b0883a", fontSize: 11, maxWidth: 260, textAlign: "right" }}>
                      ⚠️ Mode pixel : export QGIS désactivé (aucun CRS). Recoche
                      « Activer le géoréférencement (SIG) » à l'upload.
                    </small>
                  )}
                </div>
                <span className={`status ${selected.status}`}>{statusLabel(selected.status)}</span>
              </div>
            </header>

            {selected.error_message && (
              <pre className="error-box wide">{selected.error_message}</pre>
            )}

            {/* Espace de travail en deux colonnes :
                gauche = agent IA live (chat / suivi SSE),
                droite = carte WebGIS (Leaflet). Les deux sont des composants
                frères ; ils partagent l'URL GeoJSON via l'état du parent. */}
            <section className="viewer-panel" style={styles.workArea}>
              <div style={styles.agentColumn}>
                <AgentChat
                  mapId={selected.id}
                  georeference={!!selected.has_georeference}
                  apiBase={API_BASE}
                  onGeoJsonReady={(url) => setActiveGeoJsonUrl(url)}
                />
              </div>
              <div style={styles.mapColumn}>
                <MapViewer
                  mapId={selected.id}
                  rasterUrl={selected.original_image_url || selected.raster_url}
                  rasterBounds={defaultBounds(selected)}
                  rasterSize={selected.raster_size || null}
                  hasGeoreference={!!selected.has_georeference}
                  apiBaseUrl={API_BASE}
                  reloadToken={activeGeoJsonUrl}
                />
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

const styles = {
  // Conteneur deux colonnes. Sur écran étroit (flex-wrap), l'agent passe
  // au-dessus de la carte ; sinon il occupe une colonne fixe à gauche.
  workArea: {
    display: "flex",
    gap: 16,
    alignItems: "stretch",
    flexWrap: "wrap",
  },
  agentColumn: {
    flex: "1 1 360px",
    minWidth: 320,
    maxWidth: 460,
    background: "#11161f",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 10,
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
  },
  mapColumn: {
    flex: "2 1 520px",
    minWidth: 360,
    minHeight: 560,
    borderRadius: 10,
    overflow: "hidden",
  },
};

export default App;
