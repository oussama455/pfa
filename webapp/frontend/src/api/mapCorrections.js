/**
 * webapp/frontend/src/api/mapCorrections.js
 *
 * Pont frontend ↔ Active Learning (Django) pour les corrections HITL dessinées
 * sur la carte (Leaflet-Geoman). Deux responsabilités :
 *
 *   1. simplifyGeometryPixel() — allège un tracé sinueux (Douglas-Peucker) AVANT
 *      l'envoi, pour ne pas noyer le serveur sous des milliers de points. Opère
 *      en espace pixel (L.CRS.Simple) : aucune projection géographique.
 *
 *   2. sendMapCorrectionToServer() — traduit le payload « idéal » de la directive
 *      vers le contrat réel de l'API existante et déclenche la recalibration.
 *
 * ── Contrat serveur réel (webapp/vectorizer/api_v2.py · MapCorrectionsV2View) ──
 *   PATCH /api/maps/{id}/corrections/
 *   body  : { corrections: [ { type, layer, feature_id, geometry, timestamp } ] }
 *           type ∈ { "delete", "edit" }   ← PAS de "draw_new" côté modèle/serializer
 *   resp  : {
 *             saved: N,
 *             map_id: id,
 *             active_learning: bool,
 *             calibration_updates: [
 *               { layer, series, corrections, active,
 *                 new_range: { H:[min,max], S:[min,max], V:[min,max] } }
 *             ]
 *           }
 *
 * ── Décision d'intégration (validée) ──────────────────────────────────────────
 *   La directive parle de action_type 'draw_new' | 'edit_existing'. Le backend
 *   n'expose que 'delete' | 'edit'. On garde action_type CÔTÉ CLIENT pour l'UX,
 *   et on l'aplatit vers type="edit" à l'envoi : dans les deux cas une géométrie
 *   pixel est jointe, et al_process() échantillonne le HSV sous ce polygone pour
 *   mettre à jour la charte via EMA (α=0.3). Zéro migration Django.
 */
import axios from "axios";

// Tolérance de simplification par défaut, en pixels. Réglable par appel.
// ~2 px : invisible à l'œil sur un scan 1:50000, mais divise fortement le nombre
// de sommets d'une longue route sinueuse.
export const DEFAULT_SIMPLIFY_TOLERANCE = 2.0;

// ── Douglas-Peucker (espace plan, coords [x, y] en pixels) ───────────────────

function perpendicularDistance(pt, lineStart, lineEnd) {
  const [x, y] = pt;
  const [x1, y1] = lineStart;
  const [x2, y2] = lineEnd;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const denom = dx * dx + dy * dy;
  if (denom === 0) {
    // Segment dégénéré : distance au point.
    return Math.hypot(x - x1, y - y1);
  }
  // Aire du parallélogramme / longueur de la base = hauteur (distance perpendiculaire).
  const num = Math.abs(dy * x - dx * y + x2 * y1 - y2 * x1);
  return num / Math.sqrt(denom);
}

/**
 * Simplifie une polyligne (tableau de [x, y]) par Douglas-Peucker récursif.
 * @param {Array<[number,number]>} points
 * @param {number} tolerance  tolérance en pixels
 * @returns {Array<[number,number]>}
 */
export function douglasPeucker(points, tolerance = DEFAULT_SIMPLIFY_TOLERANCE) {
  if (!Array.isArray(points) || points.length <= 2) return points;

  let maxDist = 0;
  let index = 0;
  const end = points.length - 1;

  for (let i = 1; i < end; i += 1) {
    const dist = perpendicularDistance(points[i], points[0], points[end]);
    if (dist > maxDist) {
      maxDist = dist;
      index = i;
    }
  }

  if (maxDist > tolerance) {
    const left = douglasPeucker(points.slice(0, index + 1), tolerance);
    const right = douglasPeucker(points.slice(index), tolerance);
    // Concatène en évitant de dupliquer le point de jonction.
    return left.slice(0, -1).concat(right);
  }
  // Aucun point au-delà de la tolérance : on garde juste les extrémités.
  return [points[0], points[end]];
}

/**
 * Simplifie une géométrie GeoJSON in-place-safe (renvoie une copie) selon son type.
 * Préserve la fermeture des anneaux de Polygon. N'altère pas les coordonnées :
 * elles restent en espace pixel, conformément à la formule pipeline
 *     X_final = (X_mask + x_offset) × inv_scale
 * (aucune conversion WGS84 ici).
 *
 * @param {object} geometry  GeoJSON geometry (Point|LineString|Polygon|MultiPolygon)
 * @param {number} tolerance pixels
 * @returns {object} nouvelle géométrie simplifiée
 */
export function simplifyGeometryPixel(geometry, tolerance = DEFAULT_SIMPLIFY_TOLERANCE) {
  if (!geometry || !geometry.type) return geometry;

  const closeRing = (ring) => {
    if (ring.length < 2) return ring;
    const [fx, fy] = ring[0];
    const [lx, ly] = ring[ring.length - 1];
    if (fx !== lx || fy !== ly) return [...ring, ring[0]];
    return ring;
  };

  switch (geometry.type) {
    case "Point":
      return geometry;

    case "LineString":
      return {
        ...geometry,
        coordinates: douglasPeucker(geometry.coordinates, tolerance),
      };

    case "MultiLineString":
      return {
        ...geometry,
        coordinates: geometry.coordinates.map((ls) => douglasPeucker(ls, tolerance)),
      };

    case "Polygon":
      return {
        ...geometry,
        coordinates: geometry.coordinates.map((ring) =>
          closeRing(douglasPeucker(ring, tolerance))),
      };

    case "MultiPolygon":
      return {
        ...geometry,
        coordinates: geometry.coordinates.map((poly) =>
          poly.map((ring) => closeRing(douglasPeucker(ring, tolerance)))),
      };

    default:
      return geometry;
  }
}

/** Compte les sommets d'une géométrie (pour le retour utilisateur / logs). */
export function countVertices(geometry) {
  if (!geometry || !geometry.coordinates) return 0;
  const walk = (c) => {
    if (typeof c[0] === "number") return 1;
    return c.reduce((sum, sub) => sum + walk(sub), 0);
  };
  return walk(geometry.coordinates);
}

// ── Envoi de la correction au serveur ────────────────────────────────────────

/**
 * Envoie une correction dessinée/éditée au backend Active Learning.
 *
 * @param {object} payloadCorrection
 *   @param {number}  .map_upload_id  PK de la carte (MapUpload).
 *   @param {string}  .layer_name     Nom de couche (ex: 'red_roads' ou un layer custom).
 *   @param {string}  .action_type    'draw_new' | 'edit_existing' (conceptuel, côté client).
 *   @param {object}  .geometry       GeoJSON geometry en coords pixel (CRS.Simple).
 *   @param {string} [.feature_id]    Identifiant de la feature (généré si absent).
 * @param {object} [opts]
 *   @param {string} [opts.apiBaseUrl="/api"]
 *   @param {number} [opts.tolerance=DEFAULT_SIMPLIFY_TOLERANCE]  simplification px.
 *   @param {boolean}[opts.simplify=true]  désactivable pour les petites formes.
 *
 * @returns {Promise<object>} data serveur : { saved, map_id, calibration_updates, active_learning }
 */
export async function sendMapCorrectionToServer(payloadCorrection, opts = {}) {
  const {
    apiBaseUrl = "/api",
    tolerance = DEFAULT_SIMPLIFY_TOLERANCE,
    simplify = true,
  } = opts;

  const {
    map_upload_id,
    layer_name,
    action_type = "edit_existing",
    geometry,
  } = payloadCorrection;

  if (map_upload_id == null) {
    throw new Error("sendMapCorrectionToServer: map_upload_id manquant.");
  }
  if (!layer_name) {
    throw new Error("sendMapCorrectionToServer: layer_name manquant.");
  }

  // Simplification facultative (allège les longs tracés avant envoi).
  const finalGeometry = simplify && geometry
    ? simplifyGeometryPixel(geometry, tolerance)
    : geometry;

  // feature_id : pour un tracé neuf on génère un id stable client ; pour une
  // édition on réutilise l'id fourni s'il existe.
  const featureId =
    payloadCorrection.feature_id != null
      ? String(payloadCorrection.feature_id)
      : `draw-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  // Traduction vers le contrat réel : action_type (client) → type=edit (serveur).
  // 'draw_new' et 'edit_existing' portent tous deux une géométrie => "edit".
  const body = {
    corrections: [
      {
        type: "edit",
        layer: layer_name,
        feature_id: featureId,
        geometry: finalGeometry,
        timestamp: new Date().toISOString(),
      },
    ],
  };

  const url = `${apiBaseUrl}/maps/${map_upload_id}/corrections/`;
  const { data } = await axios.patch(url, body, {
    headers: { "Content-Type": "application/json" },
  });

  // On renvoie aussi le feature_id généré + l'action conceptuelle, utiles côté UI
  // (injection locale de la feature, traçabilité).
  return { ...data, feature_id: featureId, action_type };
}

export default sendMapCorrectionToServer;
