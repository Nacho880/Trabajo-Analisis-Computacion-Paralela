"""
tests/test_modeling_regression.py
====================================

Pruebas unitarias para `src/modeling_regression.py`.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.modeling_regression import (
    breusch_pagan_test,
    compute_vif,
    fit_and_diagnose_regression,
    prepare_design_matrix,
)


def test_prepare_design_matrix_creates_dummies_and_drops_first() -> None:
    """Una variable categórica de 3 niveles debe generar 2 dummies (drop_first)."""
    df = pd.DataFrame(
        {
            "MONTO_APLICADO": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "UNIDADES": [1, 2, 3, 1, 2, 3],
            "PORCENTAJE_DESCUENTO": [0.1, 0.2, 0.0, 0.1, 0.2, 0.0],
            "CANAL": ["POS", "WEB", "APP", "POS", "WEB", "APP"],
        }
    )
    X, y = prepare_design_matrix(
        df, "MONTO_APLICADO", ["UNIDADES", "PORCENTAJE_DESCUENTO"], ["CANAL"]
    )
    canal_dummy_columns = [c for c in X.columns if c.startswith("CANAL_")]
    assert len(canal_dummy_columns) == 2  # 3 categorías - 1 (drop_first)
    assert len(X) == 6
    assert len(y) == 6


def test_prepare_design_matrix_drops_rows_with_nan() -> None:
    """Filas con NaN en cualquier columna relevante deben excluirse."""
    df = pd.DataFrame(
        {
            "MONTO_APLICADO": [100.0, np.nan, 300.0],
            "UNIDADES": [1, 2, 3],
            "CANAL": ["POS", "WEB", "APP"],
        }
    )
    X, y = prepare_design_matrix(df, "MONTO_APLICADO", ["UNIDADES"], ["CANAL"])
    assert len(X) == 2
    assert len(y) == 2


def test_compute_vif_low_for_independent_predictors() -> None:
    """Predictores generados de forma independiente deben tener VIF cercano a 1."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "A": rng.normal(0, 1, 500),
            "B": rng.normal(0, 1, 500),
            "C": rng.normal(0, 1, 500),
        }
    )
    vif_table = compute_vif(X)
    assert all(vif_table["VIF"] < 1.5)


def test_compute_vif_high_for_collinear_predictors() -> None:
    """Un predictor que es combinación lineal casi exacta de otro debe tener VIF muy alto."""
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 500)
    X = pd.DataFrame(
        {
            "A": a,
            "B": a * 2 + rng.normal(0, 0.001, 500),  # casi colineal con A
            "C": rng.normal(0, 1, 500),
        }
    )
    vif_table = compute_vif(X)
    max_vif = vif_table["VIF"].max()
    assert max_vif > 10  # multicolinealidad severa esperada


def test_breusch_pagan_detects_homoscedasticity() -> None:
    """Residuos con varianza constante no deben rechazar H0 (p >= 0.05 esperado)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"X1": rng.normal(0, 1, 500)})
    residuals = rng.normal(0, 1, 500)  # varianza constante por construcción
    p_value = breusch_pagan_test(X, residuals)
    assert p_value > 0.01


def test_breusch_pagan_detects_heteroscedasticity() -> None:
    """Residuos cuya varianza crece con X deben rechazar H0 (p < 0.05 esperado)."""
    rng = np.random.default_rng(42)
    x = np.linspace(1, 100, 1000)
    residuals = rng.normal(0, x)  # varianza proporcional a x: heterocedástico
    X = pd.DataFrame({"X1": x})
    p_value = breusch_pagan_test(X, residuals)
    assert p_value < 0.05


def test_fit_and_diagnose_regression_end_to_end(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """El pipeline completo de regresión debe ejecutarse sin excepciones."""
    from config import COLUMN_RENAME_MAP
    from src.data_cleaning import clean_dataset
    from src.feature_engineering import engineer_features

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, _ = clean_dataset(dd.from_pandas(df, npartitions=2))
    ddf_final, _ = engineer_features(ddf_clean)
    df_final = ddf_final.compute()

    model, diagnostics, metrics = fit_and_diagnose_regression(df_final, seed=42)

    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["r2_test"] <= 1.0  # R² no está acotado por abajo, solo por 1.0 arriba
    assert np.isfinite(metrics["r2_test"])
    assert not diagnostics.vif_table.empty
    assert "intercept" in diagnostics.coefficients


def test_fit_and_diagnose_regression_is_reproducible(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """La misma semilla debe producir exactamente los mismos coeficientes."""
    from config import COLUMN_RENAME_MAP
    from src.data_cleaning import clean_dataset
    from src.feature_engineering import engineer_features

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, _ = clean_dataset(dd.from_pandas(df, npartitions=2))
    ddf_final, _ = engineer_features(ddf_clean)
    df_final = ddf_final.compute()

    _, diag_1, metrics_1 = fit_and_diagnose_regression(df_final, seed=42)
    _, diag_2, metrics_2 = fit_and_diagnose_regression(df_final, seed=42)

    assert diag_1.coefficients == diag_2.coefficients
    assert metrics_1["rmse"] == metrics_2["rmse"]
