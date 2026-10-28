"""Pruebas ligeras: no deserializan modelos ni requieren TensorFlow."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app_core import (load_simulation, quote_from_pp, validate_profile,
                      validate_simulation, SEGMENT_ORDER,
                      STRATEGIES, DEFAULT_EXPENSE_RATIO, DEFAULT_TARGET_MARGIN)


def test_csv_complete_and_consistent():
    tables = load_simulation(ROOT)
    assert len(tables["prospects"]) == 1500
    assert len(tables["sensitivity"]) == 27
    assert set(tables["simulated"]["estrategia"]) == set(STRATEGIES)


def test_quote_identity():
    assert quote_from_pp(700, .25, .05) == pytest.approx(1000)
    assert quote_from_pp(700, DEFAULT_EXPENSE_RATIO, DEFAULT_TARGET_MARGIN) == pytest.approx(1000)
    assert quote_from_pp(700, .0, .0) == pytest.approx(700)


@pytest.mark.parametrize("bad", [(0.25, 0.75), (-0.1, 0.05), (0.9, 0.25)])
def test_invalid_commercial_loadings(bad):
    with pytest.raises(ValueError):
        quote_from_pp(700, *bad)


def test_profile_schema():
    profile = {"edad": 40, "zona": 3, "anios_vehiculo": 5, "cobertura": 2,
               "historial_siniestros": 1, "tipo_uso": 0}
    frame = validate_profile(profile)
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "zona"] == 3
    with pytest.raises(ValueError):
        validate_profile({**profile, "edad": 99})
    with pytest.raises(ValueError):
        validate_profile({**profile, "cobertura": 4})
    with pytest.raises(ValueError):
        validate_profile({**profile, "tipo_uso": "Particular"})


def test_segment_exports_cover_five_dimensions_and_reconcile():
    frames = load_simulation(ROOT)
    segment_metrics = frames["observable_segments"]
    assert len(segment_metrics) == 60
    assert len(frames["observable_findings"]) == 15
    for dimension, segments in SEGMENT_ORDER.items():
        group = segment_metrics.loc[segment_metrics["dimension"] == dimension]
        assert set(group["segmento"]) == set(segments)
        for strategy in STRATEGIES:
            sample = group.loc[group["estrategia"] == strategy]
            assert sample["n_prospectos"].sum() == len(frames["prospects"])
            expected = frames["expected"].set_index("estrategia").loc[strategy]
            assert sample["resultado_tecnico_esperado"].sum() == pytest.approx(
                expected["resultado_tecnico"]
            )


@pytest.mark.parametrize("field,delta", [
    ("n_prospectos", 1),
    ("sesgo_relativo_todos", .05),
    ("resultado_tecnico_esperado", 5000),
])
def test_invalid_segment_metric_is_rejected(field, delta):
    frames = load_simulation(ROOT)
    frames["observable_segments"].loc[0, field] += delta
    with pytest.raises(ValueError):
        validate_simulation(frames)


def test_invalid_segment_findings_is_rejected():
    frames = load_simulation(ROOT)
    frames["observable_findings"].loc[0, "segmento_mayor_subestimacion"] = "inventado"
    with pytest.raises(ValueError):
        validate_simulation(frames)
