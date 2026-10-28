"""Motor ligero de validación y tarificación 

No entrena modelos ni simula nuevos targets. 
Las funciones de inferencia se reutilizan sin modificarlas (src/models/).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import numpy as np
import pandas as pd

FEATURES = ["edad", "zona", "anios_vehiculo", "cobertura", "historial_siniestros", "tipo_uso"]
MODEL_COLUMNS = {
    "Estadístico · GLM Tweedie": "pp_tweedie",
    "ML · XGBoost directo": "pp_xgb_direct",
    "DL · MLP Tweedie": "pp_mlp_tweedie",
}
STRATEGIES = list(MODEL_COLUMNS)
DEFAULT_EXPENSE_RATIO = .25
DEFAULT_TARGET_MARGIN = .05
ORIGINAL_DENOMINATOR = 1 - DEFAULT_EXPENSE_RATIO - DEFAULT_TARGET_MARGIN
DATASETS = {
    "prospects": "07_01_prospects.csv",
    "simulated": "07_01_strategy_metrics.csv",
    "expected": "07_01_expected_metrics.csv",
    "segments": "07_01_risk_segments.csv",
    "sensitivity": "07_01_demand_sensitivity.csv",
    "winners": "07_01_demand_winners.csv",
    "observable_segments": "07_01_segment_metrics.csv",
    "observable_findings": "07_01_segment_findings.csv",
}
# Categorías fijadas en la sección 12 de 07.01 (segmentaciones marginales).
SEGMENT_ORDER = {
    "Cobertura": ["Limitada", "RC", "Amplia"],
    "Zona": ["1", "2", "3", "4", "5"],
    "Historial": ["0", "1", "2", "3–4"],
    "Edad": ["18–29", "30–44", "45–59", "60–74"],
    "Antigüedad": ["0–4", "5–9", "10–14", "15–19"],
}

VALID_RANGES = {
    "edad": (18, 74),
    "zona": (1, 5),
    "anios_vehiculo": (0, 19),
    "cobertura": (0, 2),
    "historial_siniestros": (0, 4),
    "tipo_uso": (0, 2),
}


def _assert_close(observed: float, expected: float, name: str) -> None:
    if not np.isclose(float(observed), float(expected), rtol=2e-8, atol=1e-5):
        raise ValueError(f"Datos 07.01 inconsistentes en {name}: {observed} ≠ {expected}")


def load_simulation(root: Path) -> dict[str, pd.DataFrame]:
    folder = Path(root) / "results" / "pricing_simulation"
    missing = [file for file in DATASETS.values() if not (folder / file).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos de 07.01 en {folder}: {', '.join(missing)}")
    frames = {name: pd.read_csv(folder / filename) for name, filename in DATASETS.items()}
    validate_simulation(frames)
    return frames


def validate_simulation(data: dict[str, pd.DataFrame]) -> None:
    """Compara identidad comercial y KPI contra prospectos (sin re-simular nada)."""
    prospects = data["prospects"]
    n = len(prospects)
    if n == 0 or "prospect_id" not in prospects or prospects["prospect_id"].duplicated().any() or prospects["prospect_id"].isna().any():
        raise ValueError("Los prospectos deben tener prospect_id único y no nulo")
    expected_cols = [*FEATURES, "pure_premium_oracle", "willingness_to_pay", "u_accept"]
    for s in STRATEGIES:
        expected_cols.extend([f"pp__{s}", f"price__{s}", f"p_accept__{s}", f"accepted__{s}"])
    if missing := set(expected_cols) - set(prospects.columns):
        raise ValueError(f"Columnas de prospectos faltantes: {sorted(missing)}")
    if not set(data["simulated"]["estrategia"]) == set(STRATEGIES):
        raise ValueError("Las estrategias de la tabla simulada son incompatibles")
    if not set(data["expected"]["estrategia"]) == set(STRATEGIES):
        raise ValueError("Las estrategias de la tabla ponderada son incompatibles")
    p = prospects["pure_premium_oracle"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Los oracles de la simulación son inválidos")
    for s in STRATEGIES:
        quoted = prospects[f"price__{s}"].to_numpy(dtype=float)
        pp = prospects[f"pp__{s}"].to_numpy(dtype=float)
        q = prospects[f"p_accept__{s}"].to_numpy(dtype=float)
        if not (np.isfinite(quoted).all() and np.isfinite(pp).all() and np.isfinite(q).all()):
            raise ValueError(f"Valores no finitos: {s}")
        if (quoted <= 0).any() or (pp <= 0).any() or ((q < 0) | (q > 1)).any():
            raise ValueError(f"Tarifas o probabilidades inválidas: {s}")
        if not np.allclose(pp, ORIGINAL_DENOMINATOR * quoted, atol=1e-6, rtol=1e-9):
            raise ValueError(f"La política comercial 25% + 5% no coincide: {s}")
        if not np.array_equal(prospects[f"accepted__{s}"].to_numpy(dtype=bool), prospects["u_accept"].to_numpy(dtype=float) < q):
            raise ValueError(f"Compras simuladas inconsistentes: {s}")
        for mode, weights in (
            ("simulated", prospects[f"accepted__{s}"].to_numpy(dtype=float)),
            ("expected", q),
        ):
            metric_row = data[mode].set_index("estrategia").loc[s]
            count = float(weights.sum())
            premium = float(np.dot(weights, quoted))
            claims = float(np.dot(weights, p))
            expense = DEFAULT_EXPENSE_RATIO * premium
            checks = {
                "n_polizas": count,
                "conversion": count / n,
                "prima_emitida": premium,
                "siniestros_esperados": claims,
                "gastos_esperados": expense,
                "resultado_tecnico": premium - expense - claims,
            }
            for k, true_value in checks.items():
                _assert_close(metric_row[k], true_value, f"{mode}/{s}/{k}")
    sens = data["sensitivity"]
    if len(sens) != 27 or sens.duplicated(["L", "k", "estrategia"]).any():
        raise ValueError("La cuadrícula de sensibilidad debe tener 27 combinaciones únicas")
    base = sens.loc[np.isclose(sens["L"], 1.5) & np.isclose(sens["k"], 5.0)].set_index("estrategia")
    if set(base.index) != set(STRATEGIES):
        raise ValueError("Falta el escenario base L=1.5, k=5.0")
    expected_metrics = data["expected"].set_index("estrategia")
    for s in STRATEGIES:
        for k in ("conversion", "prima_emitida", "resultado_tecnico", "combined_ratio"):
            _assert_close(base.loc[s, k], expected_metrics.loc[s, k], f"sensibilidad/{s}/{k}")
    winners = data["winners"]
    if len(winners) != 9 or winners.duplicated(["L", "k"]).any():
        raise ValueError("Los nueve ganadores de sensibilidad no son únicos")
    for _, row in winners.iterrows():
        sample = sens.loc[np.isclose(sens["L"], row["L"]) & np.isclose(sens["k"], row["k"])]
        best = sample.loc[sample["resultado_tecnico"].idxmax()]
        if best["estrategia"] != row["estrategia"]:
            raise ValueError("Ganador de sensibilidad no coincide")
        _assert_close(row["resultado_tecnico"], best["resultado_tecnico"], "ganador sensibilidad")



    validate_observable_segments(data)


def validate_observable_segments(data: dict[str, pd.DataFrame]) -> None:
    """Valida CSV congelados de la sección 12 sin recalcular modelos ni demanda.

    Las cinco dimensiones son particiones separadas y superpuestas de la misma
    cartera. Solo se concilia cada dimensión individualmente con los KPI base.
    """
    metrics = data["observable_segments"]
    findings = data["observable_findings"]
    prospects = data["prospects"]
    required = {
        "dimension", "segmento", "estrategia", "n_prospectos",
        "prima_pura_oracle_media", "prima_pura_estimada_media",
        "sesgo_medio_todos", "sesgo_relativo_todos", "mae_oracle_todos",
        "sesgo_medio_aceptados_esperados", "sesgo_relativo_aceptados_esperados",
        "precio_medio_ofertado", "precio_medio_aceptados_esperados",
        "n_contratados_simulados", "conversion_simulada",
        "contrataciones_esperadas", "conversion_esperada",
        "prima_emitida_simulada", "resultado_tecnico_simulado",
        "prima_emitida_esperada", "siniestros_esperados", "gastos_esperados",
        "resultado_tecnico_esperado", "loss_ratio_esperado",
        "combined_ratio_esperado", "margen_tecnico_esperado",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Columnas de segmentación faltantes: {sorted(missing)}")
    if metrics[list(required)].isna().any().any():
        raise ValueError("La tabla segmentada contiene valores nulos")
    expected_keys = {(dim, label, strategy)
                     for dim, labels in SEGMENT_ORDER.items()
                     for label in labels for strategy in STRATEGIES}
    actual_keys = set(metrics[["dimension", "segmento", "estrategia"]].itertuples(index=False, name=None))
    if len(metrics) != 60 or len(actual_keys) != len(metrics) or actual_keys != expected_keys:
        raise ValueError("Los 60 cruces dimensión/segmento/estrategia no coinciden con 07.01")

    # Las etiquetas son exclusivamente verificaciones del CSV contra prospectos.
    # No se añaden columnas a prospectos ni se generan nuevas predicciones.
    labels = {
        "Cobertura": prospects["cobertura"].map({0: "Limitada", 1: "RC", 2: "Amplia"}),
        "Zona": prospects["zona"].astype(str),
        "Historial": prospects["historial_siniestros"].map({
            0: "0", 1: "1", 2: "2", 3: "3–4", 4: "3–4"
        }),
        "Edad": pd.cut(prospects["edad"], bins=[17, 29, 44, 59, 74],
                       labels=SEGMENT_ORDER["Edad"], include_lowest=True),
        "Antigüedad": pd.cut(prospects["anios_vehiculo"], bins=[-1, 4, 9, 14, 19],
                             labels=SEGMENT_ORDER["Antigüedad"], include_lowest=True),
    }
    n = len(prospects)
    expected_global = data["expected"].set_index("estrategia")
    simulated_global = data["simulated"].set_index("estrategia")
    for dim, ordered_labels in SEGMENT_ORDER.items():
        observed = labels[dim]
        if observed.isna().any():
            raise ValueError(f"{dim}: valores fuera de los rangos de segmentación")
        counts = observed.value_counts().reindex(ordered_labels, fill_value=0)
        part = metrics.loc[metrics["dimension"] == dim]
        for strategy in STRATEGIES:
            group = part.loc[part["estrategia"] == strategy].set_index("segmento").loc[ordered_labels]
            if not np.array_equal(group["n_prospectos"].to_numpy(), counts.to_numpy()):
                raise ValueError(f"{dim}/{strategy}: los tamaños no corresponden a los prospectos")
            if int(group["n_prospectos"].sum()) != n:
                raise ValueError(f"{dim}/{strategy}: partición incompleta")
            for segment_metric, global_metric in (
                ("contrataciones_esperadas", "n_polizas"),
                ("prima_emitida_esperada", "prima_emitida"),
                ("siniestros_esperados", "siniestros_esperados"),
                ("gastos_esperados", "gastos_esperados"),
                ("resultado_tecnico_esperado", "resultado_tecnico"),
            ):
                _assert_close(group[segment_metric].sum(), expected_global.loc[strategy, global_metric],
                              f"segmentos/{dim}/{strategy}/{segment_metric}")
            for segment_metric, global_metric in (
                ("n_contratados_simulados", "n_polizas"),
                ("prima_emitida_simulada", "prima_emitida"),
                ("resultado_tecnico_simulado", "resultado_tecnico"),
            ):
                _assert_close(group[segment_metric].sum(), simulated_global.loc[strategy, global_metric],
                              f"segmentos/{dim}/{strategy}/{segment_metric}")
            for segment_metric, global_metric in (
                ("sesgo_medio_todos", "sesgo_oracle_todos"),
                ("mae_oracle_todos", "mae_oracle_todos"),
            ):
                aggregate = np.dot(group["n_prospectos"], group[segment_metric]) / n
                _assert_close(aggregate, expected_global.loc[strategy, global_metric],
                              f"segmentos/{dim}/{strategy}/{segment_metric}")
            # Comprobaciones de coherencia por fila, sin recomputar inferencias.
            for label, row in group.iterrows():
                prefix = f"segmentos/{dim}/{label}/{strategy}"
                _assert_close(row["conversion_esperada"],
                              row["contrataciones_esperadas"] / row["n_prospectos"],
                              prefix + "/conversion")
                _assert_close(row["precio_medio_ofertado"],
                              row["prima_pura_estimada_media"] / ORIGINAL_DENOMINATOR,
                              prefix + "/precio")
                _assert_close(row["sesgo_relativo_todos"],
                              row["sesgo_medio_todos"] / row["prima_pura_oracle_media"],
                              prefix + "/sesgo_relativo")
                _assert_close(row["resultado_tecnico_esperado"],
                              row["prima_emitida_esperada"] - row["gastos_esperados"]
                              - row["siniestros_esperados"], prefix + "/resultado")
                _assert_close(row["combined_ratio_esperado"],
                              (row["siniestros_esperados"] + row["gastos_esperados"])
                              / row["prima_emitida_esperada"], prefix + "/combined_ratio")

    findings_cols = {"dimension", "estrategia", "segmento_mayor_subestimacion",
                     "n_prospectos", "sesgo_relativo_aceptados", "combined_ratio"}
    if findings_cols - set(findings.columns):
        raise ValueError("Faltan columnas de síntesis por segmentos")
    if findings[list(findings_cols)].isna().any().any() or len(findings) != 15 or findings.duplicated(
        ["dimension", "estrategia"]
    ).any():
        raise ValueError("La síntesis de segmentación debe contener 15 cruces únicos y completos")
    for _, finding in findings.iterrows():
        sample = metrics.loc[(metrics["dimension"] == finding["dimension"])
                             & (metrics["estrategia"] == finding["estrategia"])]
        if sample.empty:
            raise ValueError("Síntesis de segmento desconocido")
        worst = sample.loc[sample["sesgo_relativo_aceptados_esperados"].idxmin()]
        if worst["segmento"] != finding["segmento_mayor_subestimacion"]:
            raise ValueError("Síntesis de mayor subestimación incompatible")
        _assert_close(worst["n_prospectos"], finding["n_prospectos"], "síntesis/n")
        _assert_close(worst["sesgo_relativo_aceptados_esperados"],
                      finding["sesgo_relativo_aceptados"], "síntesis/sesgo")
        _assert_close(worst["combined_ratio_esperado"], finding["combined_ratio"],
                      "síntesis/combined_ratio")

def validate_profile(profile: dict[str, Any]) -> pd.DataFrame:
    missing = set(FEATURES) - set(profile)
    if missing:
        raise ValueError(f"Faltan características: {sorted(missing)}")
    validated = {}
    for feature in FEATURES:
        v = profile[feature]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{feature} debe ser un entero")
        lo, hi = VALID_RANGES[feature]
        if not lo <= int(v) <= hi:
            raise ValueError(f"{feature} fuera del rango sintético [{lo}, {hi}]")
        validated[feature] = int(v)
    return pd.DataFrame([{"policy_id": "COTIZACION-001", **validated}])


def quote_from_pp(pure_premium: float, expense_ratio: float = DEFAULT_EXPENSE_RATIO,
                  target_margin: float = DEFAULT_TARGET_MARGIN) -> float:
    vals = (pure_premium, expense_ratio, target_margin)
    if not all(math.isfinite(float(x)) for x in vals):
        raise ValueError("La prima y las cargas deben ser finitas")
    if pure_premium <= 0 or expense_ratio < 0 or target_margin < 0 or expense_ratio + target_margin >= 1:
        raise ValueError("Prima pura positiva y suma de cargas menor que 100% requeridas")
    return float(pure_premium / (1 - expense_ratio - target_margin))


@dataclass
class PricingEngine:
    """Carga y utiliza únicamente los estimadores ya persistidos."""
    statistical_bundle: Any
    ml_models: dict[str, Any]
    ml_manifest: dict[str, Any]
    dl_predictor: Any

    @classmethod
    def from_root(cls, root: Path) -> "PricingEngine":
        # Importaciones tardías: el dashboard funciona aunque falte TensorFlow.
        from src.models.statistical_predictor import load_statistical_bundle
        from src.models.ml_persistence import load_ml_models
        from src.models.dl_inference import FrozenDLPricing

        root = Path(root)
        stat = load_statistical_bundle(root / "models/statistical/statistical_models_v1.joblib")
        ml, manifest = load_ml_models(root / "models/machine_learning")
        dl = FrozenDLPricing(project_root=root, version="v1")
        return cls(stat, ml, manifest, dl)

    def predict(self, profile: dict[str, int], expense_ratio: float = DEFAULT_EXPENSE_RATIO,
                target_margin: float = DEFAULT_TARGET_MARGIN) -> pd.DataFrame:
        from src.models.statistical_predictor import predict_statistical_models
        from src.models.ml_persistence import predict_ml_pure_premium
        row = validate_profile(profile)
        statistical = predict_statistical_models(row, self.statistical_bundle, model="tweedie")
        ml = predict_ml_pure_premium(row, self.ml_models, self.ml_manifest["feature_order"])
        dl = self.dl_predictor.predict(row)
        by_name = {
            STRATEGIES[0]: float(statistical["pp_tweedie"].iloc[0]),
            STRATEGIES[1]: float(ml["pp_xgb_direct"].iloc[0]),
            STRATEGIES[2]: float(dl["pp_mlp_tweedie"].iloc[0]),
        }
        result = []
        for strategy, pp in by_name.items():
            premium = quote_from_pp(pp, expense_ratio, target_margin)
            result.append({"estrategia": strategy, "prima_pura": pp,
                           "prima_comercial": premium, "carga_total": premium - pp})
        return pd.DataFrame(result)
