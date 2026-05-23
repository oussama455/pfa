/**
 * WeightsSelector.jsx
 *
 * Dropdown pour choisir le fichier .pth a utiliser pour l'inference U-Net.
 * Recupere la liste depuis GET /api/weights/ et expose une valeur au parent.
 *
 * Props :
 *   value         : chemin actuellement selectionne (string ou null)
 *   onChange      : callback(newValue: string | null)
 *   apiBase       : prefixe API (defaut "/api")
 *   disabled      : booleen
 *   includeNone   : si true, ajoute une option "Aucun (pas d'U-Net)" en tete
 */
import React, { useEffect, useState } from "react";

export default function WeightsSelector({
  value,
  onChange,
  apiBase = "/api",
  disabled = false,
  includeNone = true,
}) {
  const [weights, setWeights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${apiBase}/weights/`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setWeights(data.weights || []);
    } catch (e) {
      setError(`Erreur chargement /api/weights/ : ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, [apiBase]);

  return (
    <div className="weights-selector">
      <label>
        Poids U-Net
        <button
          type="button"
          onClick={refresh}
          disabled={loading}
          title="Rafraichir la liste"
        >
          {loading ? "..." : "↻"}
        </button>
      </label>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        disabled={disabled || loading}
      >
        {includeNone && (
          <option value="">— Aucun (segmentation couleur seule) —</option>
        )}
        {weights.map((w) => (
          <option key={w.path} value={w.path}>
            {w.name} ({w.dataset}, {w.size_mb} Mo)
          </option>
        ))}
      </select>
      {error && (
        <p style={{ color: "#c00", fontSize: 12, marginTop: 4 }}>{error}</p>
      )}
      {!loading && weights.length === 0 && !error && (
        <p style={{ color: "#888", fontSize: 12, marginTop: 4 }}>
          Aucun fichier .pth dans external/weight/.
          Entraine un modele via le panneau "Training" ci-dessous.
        </p>
      )}
    </div>
  );
}
