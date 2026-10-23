"""Ejecutar en el entorno congelado para contrastar inferencia viva vs. CSV 07.01.

python scripts/smoke_inference.py
No reentrena modelos, no modifica los CSV ni depende del dataset raw.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from app_core import FEATURES, MODEL_COLUMNS, PricingEngine, load_simulation


def main() -> None:
    data = load_simulation(ROOT)
    reference = data["prospects"].iloc[:5]
    print("Cargando 3 familias de modelos congelados…")
    predictor = PricingEngine.from_root(ROOT)
    for idx, row in reference.iterrows():
        profile = {f: int(row[f]) for f in FEATURES}
        quote = predictor.predict(profile)
        quotes_by_name = quote.set_index("estrategia")
        for name in MODEL_COLUMNS:
            observed = float(quotes_by_name.loc[name, "prima_pura"])
            expected = float(row[f"pp__{name}"])
            np.testing.assert_allclose(
                observed, expected, rtol=1e-4, atol=0.15,
                err_msg=f"Modelo {name} no reproduce el prospecto {row['prospect_id']}",
            )
            np.testing.assert_allclose(
                float(quotes_by_name.loc[name, "prima_comercial"]), observed / 0.70,
                rtol=1e-10, atol=1e-8,
            )
        print(f"PASS: {row['prospect_id']}")
    print("PASS: 5 perfiles × 3 estrategias, reproducidas sin oracle en las entradas.")


if __name__ == "__main__":
    main()
