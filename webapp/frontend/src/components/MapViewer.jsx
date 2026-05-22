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
import {
  sendMapCorrectionToServer,
  countVertices,
  DEFAULT_SIMPLIFY_TOLERANCE,
} from "../api/mapCorrections.js";

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

function getFeatureId(feature) {
  return feature?.properties?.label_id ?? feature?.properties?.id ?? feature?.id;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Geoman edit toolbar initializer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GeomanControls — mounts the Leaflet-Geoman toolbar.
 *
 * Leaflet-Geoman is the maintained successor to Leaflet.Draw, compatible with
 * react-leaflet v4. On active désormais le DESSIN de Polygons (bâtiments,
 * végétation, eau) et de LineStrings (routes, courbes de niveau), en plus de
 * l'édition de sommets et du drag.
 *
 *   pm:create → onShapeCreated(layer, shape)   tracé neuf terminé
 *   pm:edit   → onShapeEdited(layer)            sommets modifiés
 *
 * Le tracé du cercle/rectangle/marqueur/texte reste désactivé : hors périmètre
 * de la correction cartographique HITL.
 */
function GeomanControls({ onShapeCreated, onShapeEdited }) {
  const map = useMap();

  useEffect(() => {
    if (!map.pm) return undefined; // geoman not loaded

    map.pm.addControls({
      position: "topleft",
      drawCircle: false,
      drawCircleMarker: false,
      drawPolyline: true,    // ← LineStrings : routes, courbes de niveau
      drawRectangle: false,
      drawPolygon: true,     // ← Polygons : bâtiments, végétation, eau
      drawMarker: false,
      drawText: false,
      editMode: true,        // ← édition de sommets
      dragMode: true,        // ← drag de la forme
      cutPolygon: false,
      removalMode: false,    // suppression gérée nous-mêmes (avec confirmation)
      rotateMode: false,
    });

    // Les coordonnées doivent rester en espace pixel (CRS.Simple) : Geoman lit
    // le CRS de la carte, donc rien à projeter ici. On laisse les styles par
    // défaut de tracé (ré-appliqués ensuite selon la couche cible).
    const handleCreate = (e) => {
      if (onShapeCreated && e.layer) onShapeCreated(e.layer, e.shape);
    };
    const handleEdit = (e) => {
      if (onShapeEdited && e.layer) onShapeEdited(e.layer);
    };

    map.on("pm:create", handleCreate);
    map.on("pm:edit", handleEdit);

    return () => {
      map.off("pm:create", handleCreate);
      map.off("pm:edit", handleEdit);
      map.pm.removeControls();
    };
  }, [map, onShapeCreated, onShapeEdited]);

  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: Feature Inspector Panel
// ─────────────────────────────────────────────────────────────────────────────

function FeatureInspector({ feature, layerName, onDelete, onSaveEdit, onClose }) {
  if (!feature) return null;

  const props = feature.properties || {};
  const featureId = getFeatureId(feature);

  return (
    <div style={inspectorStyles.panel}>
      <div style={inspectorStyles.header}>
        <span style={inspectorStyles.title}>
          {layerName} · Feature #{featureId ?? "-"}
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

function LayerControls({
  layers, visibility, onToggle,
  calibLayers = {},          // { layerName: { active: bool, corrections: n } }
  selectedDrawLayer = null,  // couche cible du dessin
  onSelectDraw,              // (name) => void
  onCreateLayer,             // () => void  ouvre le dialogue de nouvelle couche
  colorOf,                   // (name) => couleur d'affichage
}) {
  return (
    <div style={ctrlStyles.panel}>
      <div style={ctrlStyles.title}>Couches</div>

      {Object.entries(layers).map(([name, featureCount]) => {
        const color = colorOf ? colorOf(name) : (LAYER_STYLES[name] || LAYER_STYLES.default).color;
        const isOn = visibility[name] !== false;
        const isTarget = selectedDrawLayer === name;
        const calib = calibLayers[name];
        const calibrated = !!calib?.active;
        return (
          <div
            key={name}
            style={{
              ...ctrlStyles.row,
              background: isTarget ? "rgba(243,156,18,0.14)" : "transparent",
              border: isTarget ? "1px solid rgba(243,156,18,0.5)" : "1px solid transparent",
            }}
          >
            {/* Œil : bascule la visibilité */}
            <button
              type="button"
              title={isOn ? "Masquer la couche" : "Afficher la couche"}
              onClick={() => onToggle(name)}
              style={{ ...ctrlStyles.eyeBtn, opacity: isOn ? 1 : 0.4 }}
            >
              {isOn ? "👁" : "🚫"}
            </button>

            <span
              style={{
                ...ctrlStyles.swatch,
                background: color === "none" ? "transparent" : color,
                border: `2px solid ${color === "none" ? "#888" : color}`,
              }}
            />

            {/* Nom : clic = sélectionne la couche cible du dessin */}
            <button
              type="button"
              title="Définir comme couche de dessin"
              onClick={() => onSelectDraw && onSelectDraw(name)}
              style={{ ...ctrlStyles.nameBtn, color: isTarget ? "#f39c12" : "#ecf0f1",
                       opacity: isOn ? 1 : 0.5 }}
            >
              {isTarget ? "✎ " : ""}{name}
            </button>

            <span style={ctrlStyles.count}>{featureCount}</span>

            {/* Badge calibration Active Learning */}
            <span
              style={{
                ...ctrlStyles.calibBadge,
                background: calibrated ? "rgba(39,174,96,0.18)" : "rgba(127,140,141,0.18)",
                color: calibrated ? "#2ecc71" : "#95a5a6",
                borderColor: calibrated ? "rgba(39,174,96,0.5)" : "rgba(127,140,141,0.4)",
              }}
              title={calibrated
                ? `Calibrée par l'Active Learning (${calib.corrections} corrections)`
                : "Plages HSV par défaut (pas encore assez de corrections)"}
            >
              {calibrated ? "Calibrée" : "Mode Défaut"}
            </span>
          </div>
        );
      })}

      <button type="button" style={ctrlStyles.newLayerBtn} onClick={onCreateLayer}>
        ＋ Créer une nouvelle couche
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-component: New Layer dialog (création client-side d'une couche custom)
// ─────────────────────────────────────────────────────────────────────────────

const NEW_LAYER_PRESETS = ["#e67e22", "#16a085", "#8e44ad", "#2c3e50", "#c0392b", "#2980b9"];

function NewLayerDialog({ existingNames, onCreate, onClose }) {
  const [name, setName] = useState("");
  const [color, setColor] = useState(NEW_LAYER_PRESETS[0]);
  const [geomType, setGeomType] = useState("polygon");
  const [err, setErr] = useState("");

  const submit = () => {
    const clean = name.trim();
    if (!clean) { setErr("Donne un nom à la couche."); return; }
    // Normalise vers une clé technique (registre de segmentation) : minuscules,
    // underscores. Ex: "Tranchées Militaires" → "tranchees_militaires".
    const key = clean
      .toLowerCase()
      .normalize("NFD").replace(/[̀-ͯ]/g, "")
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
    if (!key) { setErr("Nom invalide."); return; }
    if (existingNames.includes(key)) { setErr(`La couche « ${key} » existe déjà.`); return; }
    onCreate({ name: key, label: clean, color, geomType });
  };

  return (
    <div style={dlgStyles.backdrop} onClick={onClose}>
      <div style={dlgStyles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={dlgStyles.header}>Nouvelle couche personnalisée</div>

        <label style={dlgStyles.label}>Nom
          <input
            autoFocus
            value={name}
            onChange={(e) => { setName(e.target.value); setErr(""); }}
            onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
            placeholder="Tranchées_Militaires"
            style={dlgStyles.input}
          />
        </label>

        <label style={dlgStyles.label}>Type de géométrie
          <select
            value={geomType}
            onChange={(e) => setGeomType(e.target.value)}
            style={dlgStyles.input}
          >
            <option value="polygon">Polygone (surface)</option>
            <option value="line">Ligne (linéaire)</option>
          </select>
        </label>

        <div style={dlgStyles.label}>Couleur d'affichage
          <div style={dlgStyles.swatchRow}>
            {NEW_LAYER_PRESETS.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => setColor(c)}
                style={{
                  ...dlgStyles.colorChip, background: c,
                  outline: color === c ? "2px solid #fff" : "2px solid transparent",
                }}
              />
            ))}
            <input
              type="color"
              value={color}
              onChange={(e) => setColor(e.target.value)}
              style={dlgStyles.colorPicker}
              title="Couleur personnalisée"
            />
          </div>
        </div>

        {err && <div style={dlgStyles.err}>{err}</div>}

        <div style={dlgStyles.footer}>
          <button type="button" style={dlgStyles.cancelBtn} onClick={onClose}>Annuler</button>
          <button type="button" style={dlgStyles.okBtn} onClick={submit}>Créer la couche</button>
        </div>
      </div>
    </div>
  );
}

const dlgStyles = {
  backdrop: {
    position: "absolute", inset: 0, zIndex: 3000,
    background: "rgba(0,0,0,0.55)",
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  modal: {
    width: 320, background: "#0f1722", color: "#ecf0f1",
    borderRadius: 10, padding: 18, border: "1px solid rgba(255,255,255,0.12)",
    boxShadow: "0 12px 40px rgba(0,0,0,0.6)",
    fontFamily: "system-ui, sans-serif", fontSize: 13,
    display: "flex", flexDirection: "column", gap: 12,
  },
  header: { fontWeight: 700, fontSize: 15, color: "#f39c12" },
  label: { display: "flex", flexDirection: "column", gap: 5, fontSize: 12, color: "#bdc3c7" },
  input: {
    padding: "8px 10px", borderRadius: 6, border: "1px solid rgba(255,255,255,0.15)",
    background: "#1a2330", color: "#ecf0f1", fontSize: 13,
  },
  swatchRow: { display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" },
  colorChip: { width: 24, height: 24, borderRadius: 6, border: "none", cursor: "pointer" },
  colorPicker: { width: 28, height: 28, padding: 0, border: "none", background: "none", cursor: "pointer" },
  err: { color: "#e74c3c", fontSize: 12 },
  footer: { display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 4 },
  cancelBtn: {
    padding: "7px 14px", background: "transparent", color: "#bdc3c7",
    border: "1px solid rgba(255,255,255,0.2)", borderRadius: 6, cursor: "pointer", fontSize: 12,
  },
  okBtn: {
    padding: "7px 14px", background: "#f39c12", color: "#0b1120",
    border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 700, fontSize: 12,
  },
};

const ctrlStyles = {
  panel: {
    position: "absolute", top: 80, left: 10, zIndex: 1000,
    background: "rgba(15, 20, 30, 0.92)", color: "#ecf0f1",
    borderRadius: 8, padding: "10px 12px", minWidth: 240, maxWidth: 280,
    fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
    boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  title: { fontWeight: 700, color: "#f39c12", letterSpacing: 1,
           textTransform: "uppercase", marginBottom: 8, fontSize: 10 },
  row:   { display: "flex", alignItems: "center", gap: 6, marginBottom: 5,
           padding: "3px 5px", borderRadius: 5, transition: "background 0.15s" },
  eyeBtn: { background: "none", border: "none", cursor: "pointer", fontSize: 12,
            padding: 0, lineHeight: 1, flexShrink: 0 },
  swatch: { width: 12, height: 12, borderRadius: 2, flexShrink: 0 },
  nameBtn: { flex: 1, textAlign: "left", background: "none", border: "none",
             cursor: "pointer", fontFamily: "inherit", fontSize: 11,
             padding: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  count: { color: "#7f8c8d", fontVariantNumeric: "tabular-nums", flexShrink: 0 },
  calibBadge: { fontSize: 8, fontWeight: 700, padding: "1px 5px", borderRadius: 8,
                border: "1px solid", letterSpacing: 0.2, flexShrink: 0, whiteSpace: "nowrap" },
  newLayerBtn: {
    width: "100%", marginTop: 6, padding: "6px 0",
    background: "rgba(243,156,18,0.12)", color: "#f39c12",
    border: "1px dashed rgba(243,156,18,0.5)", borderRadius: 6,
    cursor: "pointer", fontSize: 11, fontWeight: 700, fontFamily: "inherit",
  },
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
  reloadToken = null,          // change de valeur => force un re-fetch des couches
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

  // ── Couches dynamiques + dessin (Active Learning) ──────────────────────────
  const [customLayers, setCustomLayers]     = useState([]);   // [{name,label,color,geomType}]
  const [selectedDrawLayer, setSelectedDrawLayer] = useState(null); // couche cible du tracé
  const [newLayerOpen, setNewLayerOpen]     = useState(false);
  const [pendingDraw, setPendingDraw]       = useState(null); // {action,layerName,geometry,leafletLayer,vbefore,vafter}
  const [submitting, setSubmitting]         = useState(false);
  const [submitErr, setSubmitErr]           = useState(null);
  const [calibLayers, setCalibLayers]       = useState({});   // statut AL pour les badges
  const [calibFlash, setCalibFlash]         = useState([]);   // calibration_updates récents
  const [drawSavedCount, setDrawSavedCount] = useState(0);    // déclenche refresh badges/panel

  // Correction queue
  const { queue, addDelete, addEdit, saveAll, saving, savedCount, error: saveError }
    = useCorrections(mapId, apiBaseUrl);

  // Style d'affichage d'une couche (base connue OU couche custom créée à la volée).
  const layerStyleOf = useCallback((name) => {
    const custom = customLayers.find((l) => l.name === name);
    if (custom) {
      const isLine = custom.geomType === "line";
      return {
        color: custom.color, weight: isLine ? 2.5 : 1.8,
        fillColor: isLine ? "none" : custom.color,
        fillOpacity: isLine ? 0 : 0.35,
      };
    }
    return LAYER_STYLES[name] || LAYER_STYLES.default;
  }, [customLayers]);

  const colorOf = useCallback(
    (name) => layerStyleOf(name).color,
    [layerStyleOf],
  );

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
  }, [mapId, apiBaseUrl, geojsonLayersProp, reloadToken]);

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

  // ── Statut de calibration Active Learning (pour les badges Calibrée/Défaut) ─
  // Léger : on lit /api/calibration/{series}/ et on rafraîchit après chaque
  // soumission de tracé. CalibrationPanel fait sa propre lecture détaillée.
  useEffect(() => {
    let cancelled = false;
    axios.get(`${apiBaseUrl}/calibration/${mapSeries}/`)
      .then((res) => { if (!cancelled) setCalibLayers(res.data?.layers || {}); })
      .catch(() => { if (!cancelled) setCalibLayers({}); });
    return () => { cancelled = true; };
  }, [apiBaseUrl, mapSeries, drawSavedCount]);

  // Couche cible de dessin par défaut : 'red_roads' si présente, sinon la 1ʳᵉ.
  useEffect(() => {
    if (selectedDrawLayer) return;
    const names = Object.keys(geojsonLayers);
    if (!names.length) return;
    setSelectedDrawLayer(names.includes("red_roads") ? "red_roads" : names[0]);
  }, [geojsonLayers, selectedDrawLayer]);

  // ── Création d'une couche personnalisée (client-side) ──────────────────────
  const handleCreateLayer = useCallback(({ name, label, color, geomType }) => {
    setCustomLayers((prev) => [...prev, { name, label, color, geomType }]);
    setGeojsonLayers((prev) =>
      prev[name] ? prev : { ...prev, [name]: { type: "FeatureCollection", features: [] } });
    setVisibility((v) => ({ ...v, [name]: true }));
    setSelectedDrawLayer(name);
    setNewLayerOpen(false);
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
    const featureId = getFeatureId(feature);
    if (featureId == null) return;

    // Optimistic: add to local deleted set
    setDeletedIds(ids => new Set([...ids, `${layerName}::${featureId}`]));

    // Queue for API
    addDelete(layerName, featureId);
    setSelectedFeature(null);
  }, [selectedFeature, addDelete]);

  // ── Tracé neuf terminé (Geoman pm:create) ─────────────────────────────────
  // On capture la géométrie (coords PIXEL — la carte est en CRS.Simple), on
  // l'affecte à la couche de dessin courante et on propose la recalibration.
  const handleShapeCreated = useCallback((leafletLayer, shape) => {
    const target = selectedDrawLayer || "red_roads";
    const geometry = leafletLayer.toGeoJSON().geometry;
    // Applique le style de la couche cible au tracé fraîchement posé.
    try { leafletLayer.setStyle?.(layerStyleOf(target)); } catch { /* markers n/a */ }
    setSubmitErr(null);
    setPendingDraw({
      action: "draw_new",
      layerName: target,
      geometry,
      leafletLayer,
      shape,
    });
  }, [selectedDrawLayer, layerStyleOf]);

  // ── Sommets modifiés (Geoman pm:edit) ──────────────────────────────────────
  // Si la forme éditée correspond à une feature existante sélectionnée, on garde
  // son id et sa couche (edit_existing). Sinon on retombe sur la couche cible.
  const handleShapeEdited = useCallback((leafletLayer) => {
    const geometry = leafletLayer.toGeoJSON().geometry;
    const isSelected = selectedFeature?.leafletLayer === leafletLayer;
    const layerName = isSelected ? selectedFeature.layerName : (selectedDrawLayer || "red_roads");
    const featureId = isSelected ? getFeatureId(selectedFeature.feature) : null;
    setSubmitErr(null);
    setPendingDraw({
      action: "edit_existing",
      layerName,
      featureId,
      geometry,
      leafletLayer,
    });
  }, [selectedFeature, selectedDrawLayer]);

  // ── Validation : envoie la correction et déclenche la recalibration HSV ────
  const confirmPendingDraw = useCallback(async () => {
    if (!pendingDraw) return;
    setSubmitting(true);
    setSubmitErr(null);
    try {
      const data = await sendMapCorrectionToServer(
        {
          map_upload_id: mapId,
          layer_name: pendingDraw.layerName,
          action_type: pendingDraw.action,
          feature_id: pendingDraw.featureId ?? undefined,
          geometry: pendingDraw.geometry,
        },
        { apiBaseUrl, tolerance: DEFAULT_SIMPLIFY_TOLERANCE },
      );

      // Tracé neuf : on l'injecte dans la couche locale pour qu'il s'affiche,
      // puis on retire la forme temporaire Geoman (sinon doublon visuel).
      if (pendingDraw.action === "draw_new") {
        const newFeature = {
          type: "Feature",
          properties: { layer: pendingDraw.layerName, label_id: data.feature_id, source: "hitl_draw" },
          geometry: pendingDraw.geometry,
        };
        setGeojsonLayers((prev) => {
          const fc = prev[pendingDraw.layerName] || { type: "FeatureCollection", features: [] };
          return {
            ...prev,
            [pendingDraw.layerName]: { ...fc, features: [...(fc.features || []), newFeature] },
          };
        });
        try { pendingDraw.leafletLayer.remove(); } catch { /* déjà retiré */ }
      }

      setCalibFlash(data.calibration_updates || []);
      setDrawSavedCount((n) => n + 1);
      setPendingDraw(null);
    } catch (err) {
      setSubmitErr(err.response?.data?.detail || err.message || "Échec de l'envoi.");
    } finally {
      setSubmitting(false);
    }
  }, [pendingDraw, mapId, apiBaseUrl]);

  const cancelPendingDraw = useCallback(() => {
    // Pour un tracé neuf annulé, on enlève la forme temporaire de la carte.
    if (pendingDraw?.action === "draw_new") {
      try { pendingDraw.leafletLayer.remove(); } catch { /* noop */ }
    }
    setSubmitErr(null);
    setPendingDraw(null);
  }, [pendingDraw]);

  // ── GeoJSON style function (per feature) ─────────────────────────────────
  const styleFeature = useCallback((layerName) => (feature) => {
    const featureId = getFeatureId(feature);
    const key = `${layerName}::${featureId}`;
    if (deletedIds.has(key)) return PENDING_DELETE_STYLE;
    return layerStyleOf(layerName);
  }, [deletedIds, layerStyleOf]);

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
              const fid = getFeatureId(f);
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

        {/* Geoman toolbar : dessin (Polygon/LineString) + édition + drag */}
        <GeomanControls
          onShapeCreated={handleShapeCreated}
          onShapeEdited={handleShapeEdited}
        />

      </MapContainer>

      {/* ── Badge mode (pixel vs SIG) ─────────────────────────────────── */}
      <div style={{
        position: "absolute", top: 12, right: 12, zIndex: 1100,
        padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        background: hasGeoreference ? "rgba(39,174,96,0.92)" : "rgba(52,73,94,0.92)",
        color: "#fff", border: "1px solid rgba(255,255,255,0.15)",
      }}
        title={hasGeoreference
          ? "Mode SIG : coordonnées géoréférencées (WGS84). Export QGIS disponible."
          : "Mode pixel : coordonnées image (CRS.Simple). Export QGIS désactivé."}
      >
        {hasGeoreference ? "🌍 SIG · WGS84" : "🖼️ PIXEL"}
      </div>

      {/* ── Indicateur de couche de dessin (haut-centre) ─────────────────── */}
      {selectedDrawLayer && (
        <div style={drawHintStyles.bar}>
          ✎ Dessin actif sur&nbsp;
          <span style={{ color: colorOf(selectedDrawLayer), fontWeight: 700 }}>
            {selectedDrawLayer}
          </span>
          <span style={drawHintStyles.hint}>
            &nbsp;· trace un polygone ou une ligne, puis valide la recalibration
          </span>
        </div>
      )}

      {/* ── Gestionnaire de couches (top-left) ────────────────────────── */}
      <LayerControls
        layers={layerCounts}
        visibility={visibility}
        onToggle={toggleLayer}
        calibLayers={calibLayers}
        selectedDrawLayer={selectedDrawLayer}
        onSelectDraw={setSelectedDrawLayer}
        onCreateLayer={() => setNewLayerOpen(true)}
        colorOf={colorOf}
      />

      {/* ── Dialogue de création de couche ──────────────────────────────── */}
      {newLayerOpen && (
        <NewLayerDialog
          existingNames={Object.keys(geojsonLayers)}
          onCreate={handleCreateLayer}
          onClose={() => setNewLayerOpen(false)}
        />
      )}

      {/* ── Prompt contextuel « Soumettre pour recalibration ? » ─────────── */}
      {pendingDraw && (
        <div style={promptStyles.card}>
          <div style={promptStyles.title}>Soumettre pour recalibration ?</div>
          <div style={promptStyles.detail}>
            {pendingDraw.action === "draw_new" ? "Nouveau tracé" : "Tracé modifié"}
            {" → couche "}
            <span style={{ color: colorOf(pendingDraw.layerName), fontWeight: 700 }}>
              {pendingDraw.layerName}
            </span>
            {" · "}
            {countVertices(pendingDraw.geometry)} sommets
          </div>
          {submitErr && <div style={promptStyles.err}>⚠ {submitErr}</div>}
          <div style={promptStyles.row}>
            <button
              type="button"
              style={promptStyles.cancelBtn}
              onClick={cancelPendingDraw}
              disabled={submitting}
            >
              Annuler
            </button>
            <button
              type="button"
              style={promptStyles.okBtn}
              onClick={confirmPendingDraw}
              disabled={submitting}
            >
              {submitting ? "Envoi…" : "✓ Recalibrer la couche"}
            </button>
          </div>
        </div>
      )}

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
              layerStyleOf(selectedFeature.layerName)
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
        correctionCount={queue.length + savedCount + drawSavedCount}
        liveUpdates={calibFlash}
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

const drawHintStyles = {
  bar: {
    position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
    zIndex: 1100, maxWidth: "70%",
    background: "rgba(15,20,30,0.92)", color: "#ecf0f1",
    padding: "5px 12px", borderRadius: 16, fontSize: 11,
    fontFamily: "'JetBrains Mono', monospace",
    border: "1px solid rgba(243,156,18,0.4)",
    boxShadow: "0 2px 12px rgba(0,0,0,0.4)",
    whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
  },
  hint: { color: "#7f8c8d" },
};

const promptStyles = {
  card: {
    position: "absolute", bottom: 20, left: "50%", transform: "translateX(-50%)",
    zIndex: 1600, width: 320,
    background: "rgba(15,20,30,0.97)", color: "#ecf0f1",
    borderRadius: 10, padding: "12px 16px",
    border: "1px solid rgba(243,156,18,0.5)",
    boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
    fontFamily: "system-ui, sans-serif",
  },
  title: { fontWeight: 700, fontSize: 14, color: "#f39c12", marginBottom: 6 },
  detail: { fontSize: 12, color: "#bdc3c7", marginBottom: 8 },
  err: { fontSize: 12, color: "#e74c3c", marginBottom: 8 },
  row: { display: "flex", justifyContent: "flex-end", gap: 8 },
  cancelBtn: {
    padding: "6px 14px", background: "transparent", color: "#bdc3c7",
    border: "1px solid rgba(255,255,255,0.2)", borderRadius: 6, cursor: "pointer", fontSize: 12,
  },
  okBtn: {
    padding: "6px 14px", background: "#27ae60", color: "#fff",
    border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 700, fontSize: 12,
  },
};
