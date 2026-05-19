/**
 * webapp/frontend/src/components/MapViewer.jsx
 *
 * WebGIS interaction layer for CartoVec.
 *
 * FEATURES:
 *   1. Historical AMS raster map as base layer (ImageOverlay)
 *   2. GeoJSON vector overlays per semantic layer (buildings, roads, etc.)
 *   3. Human-in-the-loop (HITL) editing:
 *      - Click a polygon to select it
 *      - Delete false positives (removes from local state + queues API call)
 *      - Edit vertices via Leaflet.Draw (GeomanJS used here for React compat)
 *      - Save corrections back to Django REST API
 *   4. Layer visibility toggles
 *   5. Feature inspector panel
 *
 * DEPENDENCIES (install in webapp/frontend):
 *   npm install react-leaflet leaflet @geoman-io/leaflet-geoman-free axios
 *   npm install @types/leaflet  (TypeScript users)
 *
 * USAGE:
 *   <MapViewer
 *     mapId={42}
 *     rasterUrl="/media/uploads/tunis_sheet20.png"
 *     rasterBounds={[[36.7, 10.0], [37.0, 10.5]]}  // [[lat_sw, lon_sw], [lat_ne, lon_ne]]
 *     apiBaseUrl="/api"
 *   />
 *
 * API CONTRACT (Django REST endpoint expected):
 *   PATCH /api/maps/{mapId}/corrections/
 *   Body: { layer, action: "delete"|"edit", feature_id, geometry? }
 */

import React, {
  useState, useEffect, useCallback, useRef, useMemo
} from "react";
import {
  MapContainer, TileLayer, ImageOverlay, GeoJSON, useMap
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "@geoman-io/leaflet-geoman-free";
import "@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css";
import axios from "axios";
import CalibrationPanel from "./CalibrationPanel.jsx";

// ─────────────────────────────────────────────────────────────────────────────
// Constants & Configuration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * CRS personnalisé pour le mode pixel.
 *
 * L.CRS.Simple par défaut inverse l'axe Y (transformation (1,0,-1,0)),
 * ce qui mettrait notre image à l'envers. Ici on garde Y croissant vers
 * le bas, comme dans le GeoJSON sortie pipeline (origine top-left).
 *
 * Avec ce CRS, une feature GeoJSON [x, y] s'aligne pixel-à-pixel avec
 * l'image overlay dont les bounds sont [[0,0], [H, W]].
 */
const PIXEL_CRS = L.extend({}, L.CRS.Simple, {
  transformation: new L.Transformation(1, 0, 1, 0),
});

/** Visual style per semantic layer — military map conventions. */
const LAYER_STYLES = {
  buildings:  { color: "#2c3e50", weight: 1.5, fillColor: "#7f8c8d", fillOpacity: 0.55 },
  red_roads:  { color: "#c0392b", weight: 2.5, fillColor: "#e74c3c", fillOpacity: 0.3  },
  roads:      { color: "#c0392b", weight: 2.0, fillColor: "#e74c3c", fillOpacity: 0.3  },
  vegetation: { color: "#27ae60", weight: 1.0, fillColor: "#2ecc71", fillOpacity: 0.45 },
  contours:   { color: "#8B4513", weight: 1.2, fillColor: "none",    fillOpacity: 0    },
  water:      { color: "#2980b9", weight: 1.5, fillColor: "#3498db", fillOpacity: 0.4  },
  default:    { color: "#8e44ad", weight: 1.5, fillColor: "#9b59b6", fillOpacity: 0.35 },
};

/** Style applied to selected (clicked) feature for editing. */
const SELECTED_STYLE = {
  color: "#f39c12", weight: 3, fillColor: "#f1c40f", fillOpacity: 0.6,
  dashArray: "6, 3",
};

/** Style for features marked for deletion (pending confirmation). */
const PENDING_DELETE_STYLE = {
  color: "#e74c3c", weight: 2, fillColor: "#e74c3c", fillOpacity: 0.25,
  dashArray: "4, 4",
};

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Geoman edit toolbar initializer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GeomanControls — mounts the Leaflet-Geoman toolbar.
 *
 * Leaflet-Geoman is the maintained successor to Leaflet.Draw.
 * It provides polygon vertex editing compatible with react-leaflet v4.
 *
 * We only enable "Edit" and "Delete" modes (no drawing from scratch).
 * Drawing new polygons is out of scope for HITL correction; we correct
 * what the AI produced, not add entirely new features.
 */
function GeomanControls({ onFeatureEdited }) {
  const map = useMap();

  useEffect(() => {
    if (!map.pm) return; // geoman not loaded

    // Add toolbar (top-left, below zoom controls)
    map.pm.addControls({
      position: "topleft",
      drawCircle: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawRectangle: false,
      drawPolygon: false,   // disable — we only correct existing polygons
      drawMarker: false,
      drawText: false,
      editMode: true,       // ← enable vertex editing
      dragMode: true,       // ← enable polygon drag
      cutPolygon: false,
      removalMode: false,   // we handle deletion ourselves (with confirmation)
      rotateMode: false,
    });

    // Listen for edit completion
    map.on("pm:edit", (e) => {
      if (onFeatureEdited && e.layer) {
        onFeatureEdited(e.layer);
      }
    });

    return () => {
      map.pm.removeControls();
      map.off("pm:edit");
    };
  }, [map, onFeatureEdited]);

  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Feature Inspector Panel
// ─────────────────────────────────────────────────────────────────────────────

function FeatureInspector({ feature, layerName, onDelete, onSaveEdit, onClose }) {
  if (!feature) return null;

  const props = feature.properties || {};

  return (
    <div style={inspectorStyles.panel}>
      <div style={inspectorStyles.header}>
        <span style={inspectorStyles.title}>
          {layerName} · Feature #{props.label_id ?? "—"}
        </span>
        <button onClick={onClose} style={inspectorStyles.closeBtn} title="Close">✕</button>
      </div>

      <div style={inspectorStyles.body}>
        <div style={inspectorStyles.row}>
          <span style={inspectorStyles.label}>Layer</span>
          <span style={inspectorStyles.value}>{props.layer ?? layerName}</span>
        </div>
        {props.area_px != null && (
          <div style={inspectorStyles.row}>
            <span style={inspectorStyles.label}>Area (px²)</span>
            <span style={inspectorStyles.value}>{props.area_px.toLocaleString()}</span>
          </div>
        )}
        {props.centroid_x != null && (
          <div style={inspectorStyles.row}>
            <span style={inspectorStyles.label}>Centroid</span>
            <span style={inspectorStyles.value}>
              x={props.centroid_x?.toFixed(1)}, y={props.centroid_y?.toFixed(1)}
            </span>
          </div>
        )}
      </div>

      <div style={inspectorStyles.footer}>
        {/* Edit button — activates Geoman edit mode for this feature */}
        <button
          onClick={onSaveEdit}
          style={{ ...inspectorStyles.btn, background: "#2980b9" }}
          title="Edit vertices (Human-in-the-loop correction)"
        >
          ✏ Edit Vertices
        </button>

        {/* Delete button — marks feature as false positive */}
        <button
          onClick={onDelete}
          style={{ ...inspectorStyles.btn, background: "#c0392b" }}
          title="Mark as false positive and remove"
        >
          🗑 Delete
        </button>
      </div>

      <div style={inspectorStyles.hint}>
        Corrections are saved to the database for model retraining.
      </div>
    </div>
  );
}

const inspectorStyles = {
  panel: {
    position: "absolute", bottom: 20, right: 10, zIndex: 1000,
    background: "rgba(15, 20, 30, 0.92)", color: "#ecf0f1",
    borderRadius: 8, padding: "12px 16px", width: 260,
    fontFamily: "'JetBrains Mono', 'Courier New', monospace",
    fontSize: 12, boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
    border: "1px solid rgba(255,255,255,0.1)",
  },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center",
            marginBottom: 10, borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: 8 },
  title:  { fontWeight: 700, color: "#f39c12", letterSpacing: 0.5 },
  closeBtn: { background: "none", border: "none", color: "#bdc3c7", cursor: "pointer",
              fontSize: 14, padding: "0 4px" },
  body:   { marginBottom: 10 },
  row:    { display: "flex", justifyContent: "space-between", marginBottom: 4 },
  label:  { color: "#95a5a6", textTransform: "uppercase", fontSize: 10, letterSpacing: 0.5 },
  value:  { color: "#ecf0f1", fontWeight: 600 },
  footer: { display: "flex", gap: 8, marginBottom: 8 },
  btn:    { flex: 1, padding: "6px 0", border: "none", borderRadius: 5, color: "#fff",
            cursor: "pointer", fontSize: 11, fontWeight: 700, letterSpacing: 0.5 },
  hint:   { color: "#7f8c8d", fontSize: 10, textAlign: "center", fontStyle: "italic" },
};

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Layer Toggle Controls
// ─────────────────────────────────────────────────────────────────────────────

function LayerControls({ layers, visibility, onToggle }) {
  return (
    <div style={ctrlStyles.panel}>
      <div style={ctrlStyles.title}>LAYERS</div>
      {Object.entries(layers).map(([name, featureCount]) => {
        const style = LAYER_STYLES[name] || LAYER_STYLES.default;
        const isOn = visibility[name] !== false;
        return (
          <div
            key={name}
            onClick={() => onToggle(name)}
            style={{
              ...ctrlStyles.row,
              opacity: isOn ? 1 : 0.4,
              cursor: "pointer",
            }}
          >
            <span
              style={{
                ...ctrlStyles.swatch,
                background: style.fillColor === "none" ? "transparent" : style.fillColor,
                border: `2px solid ${style.color}`,
              }}
            />
            <span style={ctrlStyles.name}>{name}</span>
            <span style={ctrlStyles.count}>{featureCount}</span>
          </div>
        );
      })}
    </div>
  );
}

const ctrlStyles = {
  panel: {
    position: "absolute", top: 80, left: 10, zIndex: 1000,
    background: "rgba(15, 20, 30, 0.90)", color: "#ecf0f1",
    borderRadius: 8, padding: "10px 14px", minWidth: 160,
    fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
    boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  title: { fontWeight: 700, color: "#f39c12", letterSpacing: 1,
           textTransform: "uppercase", marginBottom: 8, fontSize: 10 },
  row:   { display: "flex", alignItems: "center", gap: 8, marginBottom: 6,
           padding: "3px 6px", borderRadius: 4,
           transition: "background 0.15s",
           "&:hover": { background: "rgba(255,255,255,0.05)" } },
  swatch: { width: 14, height: 14, borderRadius: 2, flexShrink: 0 },
  name:  { flex: 1, color: "#ecf0f1" },
  count: { color: "#7f8c8d", fontVariantNumeric: "tabular-nums" },
};

// ─────────────────────────────────────────────────────────────────────────────
// Correction Queue — manages unsaved HITL edits
// ─────────────────────────────────────────────────────────────────────────────

/**
 * useCorrections — manages the Human-in-the-loop correction queue.
 *
 * Architecture:
 *   - All corrections are stored in a local queue (pendingCorrections).
 *   - The UI reflects changes immediately (optimistic update).
 *   - Corrections are saved to Django REST API on demand ("Save All") or
 *     automatically after a debounce period.
 *   - On save, the API receives a list of correction objects; the backend
 *     stores them in a Correction model linked to MapUpload for retraining.
 */
function useCorrections(mapId, apiBaseUrl) {
  const [queue, setQueue] = useState([]);     // unsaved corrections
  const [savedCount, setSavedCount] = useState(0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const addDelete = useCallback((layerName, featureId) => {
    setQueue(q => [...q, {
      type: "delete",
      layer: layerName,
      feature_id: featureId,
      timestamp: new Date().toISOString(),
    }]);
  }, []);

  const addEdit = useCallback((layerName, featureId, newGeometry) => {
    setQueue(q => [...q, {
      type: "edit",
      layer: layerName,
      feature_id: featureId,
      geometry: newGeometry,
      timestamp: new Date().toISOString(),
    }]);
  }, []);

  const saveAll = useCallback(async () => {
    if (queue.length === 0) return;
    setSaving(true);
    setError(null);
    try {
      await axios.patch(
        `${apiBaseUrl}/maps/${mapId}/corrections/`,
        { corrections: queue },
        { headers: { "Content-Type": "application/json" } }
      );
      setSavedCount(c => c + queue.length);
      setQueue([]);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Save failed");
    } finally {
      setSaving(false);
    }
  }, [mapId, apiBaseUrl, queue]);

  return { queue, addDelete, addEdit, saveAll, saving, savedCount, error };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main MapViewer component
// ─────────────────────────────────────────────────────────────────────────────

/**
 * MapViewer — main WebGIS component.
 *
 * Props:
 *   mapId          {number}  Django MapUpload primary key
 *   rasterUrl      {string}  URL of the historical raster map image
 *   rasterBounds   {Array}   [[lat_sw, lon_sw], [lat_ne, lon_ne]] in WGS84
 *   apiBaseUrl     {string}  Base URL of the Django REST API (default "/api")
 *   geojsonLayers  {Object}  Optional: pre-loaded GeoJSON dict {layerName: FeatureCollection}
 *                            If omitted, fetched from apiBaseUrl/maps/{mapId}/geojson/
 */
export default function MapViewer({
  mapId,
  rasterUrl,
  rasterBounds,
  rasterSize = null,           // { width, height } en pixels — mode pixel
  hasGeoreference = false,     // false (défaut) = mode pixel, true = mode SIG
  apiBaseUrl = "/api",
  geojsonLayers: geojsonLayersProp = null,
  mapSeries = "ams_tunisia",
  isAdmin = false,
}) {
  // ── State ──────────────────────────────────────────────────────────────────
  const [geojsonLayers, setGeojsonLayers]   = useState(geojsonLayersProp || {});
  const [visibility, setVisibility]         = useState({});
  const [selectedFeature, setSelectedFeature] = useState(null);   // {feature, layerName, leafletLayer}
  const [deletedIds, setDeletedIds]         = useState(new Set()); // optimistic deletes
  const [loading, setLoading]               = useState(!geojsonLayersProp);
  const [fetchError, setFetchError]         = useState(null);

  // Correction queue
  const { queue, addDelete, addEdit, saveAll, saving, savedCount, error: saveError }
    = useCorrections(mapId, apiBaseUrl);

  // ── Mode pixel vs mode SIG ─────────────────────────────────────────────────
  // En mode pixel (défaut), Leaflet utilise PIXEL_CRS (Y descendant) :
  // bounds = [[0,0], [H, W]] et les coords GeoJSON [x, y] s'alignent
  // pixel-à-pixel avec l'image.
  const pixelBounds = useMemo(() => {
    if (hasGeoreference) return null;
    const w = rasterSize?.width  ?? 1000;
    const h = rasterSize?.height ?? 1000;
    return [[0, 0], [h, w]];   // [[lat_sw=0, lng_sw=0], [lat_ne=H, lng_ne=W]]
  }, [hasGeoreference, rasterSize]);

  // Map center: derived from bounds midpoint (pixel ou géo selon le mode)
  const mapCenter = useMemo(() => {
    if (!hasGeoreference) {
      if (!pixelBounds) return [0, 0];
      const [[lat0, lng0], [lat1, lng1]] = pixelBounds;
      return [(lat0 + lat1) / 2, (lng0 + lng1) / 2];
    }
    if (!rasterBounds) return [36.85, 10.25]; // Tunis default
    const [[lat0, lon0], [lat1, lon1]] = rasterBounds;
    return [(lat0 + lat1) / 2, (lon0 + lon1) / 2];
  }, [hasGeoreference, rasterBounds, pixelBounds]);

  // Bounds effectifs et CRS Leaflet à utiliser
  const effectiveBounds = hasGeoreference ? rasterBounds : pixelBounds;
  const effectiveCRS    = hasGeoreference ? L.CRS.EPSG3857 : PIXEL_CRS;
  // Niveau de zoom initial : -1 pour bien afficher toute l'image en mode pixel.
  const initialZoom     = hasGeoreference ? 12 : 0;

  // ── Fetch GeoJSON layers from API ──────────────────────────────────────────
  useEffect(() => {
    if (geojsonLayersProp) return; // already provided as prop

    setLoading(true);
    axios.get(`${apiBaseUrl}/maps/${mapId}/geojson/`)
      .then(res => {
        setGeojsonLayers(res.data?.layers || res.data || {});
        setLoading(false);
      })
      .catch(err => {
        setFetchError(err.message);
        setLoading(false);
      });
  }, [mapId, apiBaseUrl, geojsonLayersProp]);

  // ── Layer counts for controls ──────────────────────────────────────────────
  const layerCounts = useMemo(() =>
    Object.fromEntries(
      Object.entries(geojsonLayers).map(([name, gj]) => [
        name,
        gj?.features?.length ?? 0,
      ])
    ),
  [geojsonLayers]);

  // ── Toggle layer visibility ────────────────────────────────────────────────
  const toggleLayer = useCallback((name) => {
    setVisibility(v => ({ ...v, [name]: v[name] === false ? true : false }));
  }, []);

  // ── Feature click handler ──────────────────────────────────────────────────
  const handleFeatureClick = useCallback((feature, layerName, leafletLayer) => {
    // Deselect previous
    setSelectedFeature(prev => {
      if (prev?.leafletLayer) {
        prev.leafletLayer.setStyle(
          LAYER_STYLES[prev.layerName] || LAYER_STYLES.default
        );
      }
      return null;
    });
    // Select new
    leafletLayer.setStyle(SELECTED_STYLE);
    setSelectedFeature({ feature, layerName, leafletLayer });
  }, []);

  // ── Delete handler ────────────────────────────────────────────────────────
  const handleDelete = useCallback(() => {
    if (!selectedFeature) return;
    const { feature, layerName } = selectedFeature;
    const featureId = feature.properties?.label_id ?? feature.id;

    // Optimistic: add to local deleted set
    setDeletedIds(ids => new Set([...ids, `${layerName}::${featureId}`]));

    // Queue for API
    addDelete(layerName, featureId);
    setSelectedFeature(null);
  }, [selectedFeature, addDelete]);

  // ── Edit handler (called by Geoman after vertex drag) ────────────────────
  const handleFeatureEdited = useCallback((leafletLayer) => {
    if (!selectedFeature) return;
    const { feature, layerName } = selectedFeature;
    const featureId = feature.properties?.label_id ?? feature.id;
    const newGeoJSON = leafletLayer.toGeoJSON().geometry;
    addEdit(layerName, featureId, newGeoJSON);
  }, [selectedFeature, addEdit]);

  // ── GeoJSON style function (per feature) ─────────────────────────────────
  const styleFeature = useCallback((layerName) => (feature) => {
    const featureId = feature.properties?.label_id ?? feature.id;
    const key = `${layerName}::${featureId}`;
    if (deletedIds.has(key)) return PENDING_DELETE_STYLE;
    return LAYER_STYLES[layerName] || LAYER_STYLES.default;
  }, [deletedIds]);

  // ── Point-to-layer (for non-polygon layers like contours) ─────────────────
  const onEachFeature = useCallback((layerName) => (feature, leafletLayer) => {
    leafletLayer.on("click", (e) => {
      L.DomEvent.stopPropagation(e);
      handleFeatureClick(feature, layerName, leafletLayer);
    });

    // Tooltip on hover
    const label = feature.properties?.layer || layerName;
    const area  = feature.properties?.area_px;
    leafletLayer.bindTooltip(
      `<b>${label}</b>${area ? `<br/>Area: ${area.toLocaleString()} px²` : ""}`,
      { sticky: true, className: "cartovec-tooltip" }
    );
  }, [handleFeatureClick]);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ position: "relative", width: "100%", height: "100%", minHeight: 520 }}>

      {/* ── Loading / Error overlay ─────────────────────────────────────── */}
      {loading && (
        <div style={overlayStyles.loading}>
          <div style={overlayStyles.spinner} />
          <span style={{ marginTop: 12 }}>Loading GeoJSON layers…</span>
        </div>
      )}
      {fetchError && (
        <div style={overlayStyles.error}>
          ⚠ Failed to load layers: {fetchError}
        </div>
      )}

      {/* ── Save bar ───────────────────────────────────────────────────── */}
      {queue.length > 0 && (
        <div style={saveBarStyles.bar}>
          <span style={saveBarStyles.msg}>
            {queue.length} unsaved correction{queue.length !== 1 ? "s" : ""}
          </span>
          <button
            onClick={saveAll}
            disabled={saving}
            style={saveBarStyles.saveBtn}
          >
            {saving ? "Saving…" : "💾 Save Corrections"}
          </button>
          {saveError && <span style={saveBarStyles.err}>{saveError}</span>}
          {savedCount > 0 && <span style={saveBarStyles.ok}>✓ {savedCount} saved</span>}
        </div>
      )}

      {/* ── Map ────────────────────────────────────────────────────────── */}
      <MapContainer
        center={mapCenter}
        zoom={initialZoom}
        minZoom={hasGeoreference ? 3 : -4}
        maxZoom={hasGeoreference ? 18 : 4}
        crs={effectiveCRS}
        style={{ width: "100%", height: "100%", minHeight: 520,
                  background: hasGeoreference ? undefined : "#1a1f2e" }}
        zoomControl={true}
      >
        {/* Tuiles OSM uniquement en mode SIG — inutiles voire trompeuses
            en mode pixel où on n'a pas de référentiel géographique. */}
        {hasGeoreference && (
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://openstreetmap.org">OSM</a>'
            opacity={0.3}
          />
        )}

        {/* Raster historique : en mode pixel on étire l'image aux dimensions
            natives (bounds = [[0,0], [H, W]]). En mode SIG on utilise les
            bounds géographiques. */}
        {rasterUrl && effectiveBounds && (
          <ImageOverlay
            url={rasterUrl}
            bounds={effectiveBounds}
            opacity={hasGeoreference ? 0.85 : 1.0}
            zIndex={10}
          />
        )}

        {/* Vector layers */}
        {Object.entries(geojsonLayers).map(([layerName, geojson]) => {
          if (!geojson?.features?.length)           return null;
          if (visibility[layerName] === false)       return null;

          // Filter optimistically deleted features
          const filteredGJ = {
            ...geojson,
            features: geojson.features.filter(f => {
              const fid = f.properties?.label_id ?? f.id;
              return !deletedIds.has(`${layerName}::${fid}`);
            }),
          };

          return (
            <GeoJSON
              key={`${layerName}-${deletedIds.size}-${queue.length}`}
              data={filteredGJ}
              style={styleFeature(layerName)}
              onEachFeature={onEachFeature(layerName)}
              zIndex={20}
            />
          );
        })}

        {/* Geoman editing toolbar */}
        <GeomanControls onFeatureEdited={handleFeatureEdited} />

      </MapContainer>

      {/* ── Layer controls (top-left) ─────────────────────────────────── */}
      <LayerControls
        layers={layerCounts}
        visibility={visibility}
        onToggle={toggleLayer}
      />

      {/* ── Feature inspector (bottom-right) ─────────────────────────── */}
      <FeatureInspector
        feature={selectedFeature?.feature}
        layerName={selectedFeature?.layerName}
        onDelete={handleDelete}
        onSaveEdit={() => {
          // Enable Geoman edit mode on the selected layer
          const map = selectedFeature?.leafletLayer?._map;
          if (map?.pm && selectedFeature?.leafletLayer?.pm) {
            selectedFeature.leafletLayer.pm.enable({
              allowSelfIntersection: false,
            });
          }
        }}
        onClose={() => {
          if (selectedFeature?.leafletLayer) {
            selectedFeature.leafletLayer.setStyle(
              LAYER_STYLES[selectedFeature.layerName] || LAYER_STYLES.default
            );
            if (selectedFeature.leafletLayer.pm?.enabled()) {
              selectedFeature.leafletLayer.pm.disable();
            }
          }
          setSelectedFeature(null);
        }}
      />

      <CalibrationPanel
        mapId={mapId}
        mapSeries={mapSeries}
        apiBaseUrl={apiBaseUrl}
        correctionCount={queue.length + savedCount}
        isAdmin={isAdmin}
      />

    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────────

const overlayStyles = {
  loading: {
    position: "absolute", inset: 0, zIndex: 2000,
    background: "rgba(15, 20, 30, 0.7)",
    display: "flex", flexDirection: "column", alignItems: "center",
    justifyContent: "center", color: "#ecf0f1",
    fontFamily: "'JetBrains Mono', monospace", fontSize: 14,
  },
  spinner: {
    width: 36, height: 36, borderRadius: "50%",
    border: "3px solid rgba(255,255,255,0.15)",
    borderTopColor: "#f39c12",
    animation: "spin 0.8s linear infinite",
  },
  error: {
    position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
    zIndex: 2000, background: "#c0392b", color: "#fff",
    padding: "8px 16px", borderRadius: 6, fontSize: 13,
    fontFamily: "'JetBrains Mono', monospace",
  },
};

const saveBarStyles = {
  bar: {
    position: "absolute", top: 0, left: 0, right: 0, zIndex: 1500,
    background: "rgba(39, 55, 70, 0.95)", color: "#ecf0f1",
    display: "flex", alignItems: "center", gap: 12,
    padding: "8px 16px", fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
    borderBottom: "2px solid #f39c12",
  },
  msg:     { flex: 1, color: "#f39c12", fontWeight: 700 },
  saveBtn: {
    padding: "5px 14px", background: "#27ae60", color: "#fff", border: "none",
    borderRadius: 5, cursor: "pointer", fontWeight: 700, fontSize: 12,
  },
  err: { color: "#e74c3c" },
  ok:  { color: "#2ecc71" },
};
