"""
tests/test_hypothesis_tests.py
=================================

Pruebas unitarias para `src/hypothesis_tests.py`, usando datos
sintéticos con relaciones conocidas de antemano para verificar que
cada test detecta correctamente presencia/ausencia de efecto.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.hypothesis_tests import (
    anova_test,
    chi_square_independence_test,
    compare_two_groups_auto,
    run_all_hypothesis_tests,
    simple_linear_regression_test,
)


def test_chi_square_detects_independence() -> None:
    """Variables generadas de forma independiente no deben rechazar H0."""
    rng = np.random.default_rng(42)
    n = 2000
    df = pd.DataFrame(
        {
            "CANAL": rng.choice(["POS", "WEB", "APP"], size=n),
            "LOCAL": rng.choice([1001, 1002, 1003], size=n),
        }
    )
    result = chi_square_independence_test(df, "CANAL", "LOCAL")
    assert result["reject_h0"] is False


def test_chi_square_detects_association() -> None:
    """Variables construidas con asociación fuerte deben rechazar H0."""
    n = 300
    canal = ["POS"] * 150 + ["WEB"] * 150
    local = [1001] * 140 + [1002] * 10 + [1001] * 10 + [1002] * 140
    df = pd.DataFrame({"CANAL": canal, "LOCAL": local})
    result = chi_square_independence_test(df, "CANAL", "LOCAL")
    assert result["reject_h0"] is True


def test_anova_detects_no_difference() -> None:
    """Tres grupos con la misma media no deben rechazar H0."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "MONTO_APLICADO": np.concatenate(
                [rng.normal(1000, 50, 100) for _ in range(3)]
            ),
            "CANAL": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        }
    )
    result = anova_test(df, "MONTO_APLICADO", "CANAL")
    assert result["reject_h0"] is False


def test_anova_detects_difference() -> None:
    """Grupos con medias claramente distintas deben rechazar H0."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "MONTO_APLICADO": np.concatenate(
                [rng.normal(1000, 50, 100), rng.normal(5000, 50, 100)]
            ),
            "CANAL": ["A"] * 100 + ["B"] * 100,
        }
    )
    result = anova_test(df, "MONTO_APLICADO", "CANAL")
    assert result["reject_h0"] is True


def test_compare_two_groups_selects_ttest_for_normal_data() -> None:
    """Con datos normales, debe seleccionarse el t-test, no Mann-Whitney."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "MONTO_APLICADO": np.concatenate(
                [rng.normal(1000, 50, 300), rng.normal(1200, 50, 300)]
            ),
            "CANAL": ["APP"] * 300 + ["WEB"] * 300,
        }
    )
    result = compare_two_groups_auto(
        df, "MONTO_APLICADO", "CANAL", "APP", "WEB", alternative="less"
    )
    assert "t-test" in result["method"]
    assert result["reject_h0"] is True  # APP < WEB por construcción


def test_simple_linear_regression_detects_positive_effect() -> None:
    """Una relación lineal fuerte y positiva debe rechazar H0 y reportar pendiente positiva."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 200)
    y = 5 + 10 * x + rng.normal(0, 0.1, 200)  # relación lineal fuerte
    df = pd.DataFrame({"PORCENTAJE_DESCUENTO": x, "UNIDADES": y})

    result = simple_linear_regression_test(df, "PORCENTAJE_DESCUENTO", "UNIDADES")
    assert result["reject_h0"] is True
    assert result["slope"] > 0
    assert result["r_squared"] > 0.9
    assert result["advertencia_varianza_cero"] is False


def test_simple_linear_regression_flags_zero_variance_dependent_variable() -> None:
    """
    Regresión: si la variable dependiente es constante (ej. UNIDADES=1 en
    el 100% de las filas, hallazgo real del archivo de producción), debe
    detectarse y marcarse explícitamente en vez de reportar p=1 sin
    contexto, que podría malinterpretarse como un hallazgo estadístico
    genuino en lugar de una tautología matemática.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "PORCENTAJE_DESCUENTO": rng.uniform(0, 0.3, 300),
            "UNIDADES": np.ones(300),  # constante
        }
    )
    result = simple_linear_regression_test(df, "PORCENTAJE_DESCUENTO", "UNIDADES")

    assert result["advertencia_varianza_cero"] is True
    assert result["p_value"] == 1.0
    assert result["slope"] == 0.0
    assert result["reject_h0"] is False


def test_run_all_hypothesis_tests_executes_without_error(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """El pipeline completo de las 5 hipótesis debe ejecutarse sin excepciones."""
    from config import COLUMN_RENAME_MAP
    from src.data_cleaning import clean_dataset
    from src.feature_engineering import engineer_features

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, _ = clean_dataset(dd.from_pandas(df, npartitions=2))
    ddf_final, _ = engineer_features(ddf_clean)
    df_final = ddf_final.compute()

    results = run_all_hypothesis_tests(df_final, seed=42)

    assert len(results) >= 5
    for result in results:
        assert "p_value" in result
        assert "reject_h0" in result
        assert 0.0 <= result["p_value"] <= 1.0
