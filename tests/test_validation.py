"""
tests/test_validation.py
===========================

Pruebas unitarias para `src/validation.py`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_cleaning import CleaningReport
from src.modeling_clustering import ClusteringResult
from src.modeling_regression import RegressionDiagnostics
from src.validation import build_validation_summary, discuss_extrapolability


@pytest.fixture()
def good_regression_diagnostics() -> RegressionDiagnostics:
    """Diagnóstico de regresión "saludable": sin banderas de alerta."""
    return RegressionDiagnostics(
        shapiro_residuals_pvalue=0.5,
        breusch_pagan_pvalue=0.5,
        vif_table=pd.DataFrame({"variable": ["A", "B"], "VIF": [1.2, 1.1]}),
        r_squared=0.8,
        r_squared_adjusted=0.78,
        coefficients={"A": 1.0, "intercept": 0.0},
    )


@pytest.fixture()
def bad_regression_diagnostics() -> RegressionDiagnostics:
    """Diagnóstico de regresión con TODAS las banderas de alerta activadas."""
    return RegressionDiagnostics(
        shapiro_residuals_pvalue=0.001,
        breusch_pagan_pvalue=0.001,
        vif_table=pd.DataFrame({"variable": ["A", "B"], "VIF": [15.0, 20.0]}),
        r_squared=0.1,
        r_squared_adjusted=0.05,
        coefficients={"A": 1.0, "intercept": 0.0},
    )


@pytest.fixture()
def dummy_clustering_result() -> ClusteringResult:
    return ClusteringResult(
        elbow_table=pd.DataFrame({"k": [2, 3], "inertia": [100.0, 50.0]}),
        silhouette_table=pd.DataFrame({"k": [2, 3], "silhouette_score": [0.4, 0.6]}),
        optimal_k=3,
        labels=[0, 1, 2],
        cluster_profile=pd.DataFrame({"n_clientes": [1, 1, 1]}),
    )


def test_build_validation_summary_has_both_models(
    good_regression_diagnostics: RegressionDiagnostics,
    dummy_clustering_result: ClusteringResult,
) -> None:
    """La tabla consolidada debe incluir exactamente los 2 modelos exigidos."""
    regression_metrics = {
        "rmse": 100.0, "mae": 80.0, "r2_test": 0.75, "r2_adjusted": 0.73,
        "n_train": 140, "n_test": 60,
    }
    summary = build_validation_summary(
        regression_metrics, good_regression_diagnostics, dummy_clustering_result
    )
    assert len(summary) == 2
    assert "Regresión" in summary["modelo"].iloc[0]
    assert "K-means" in summary["modelo"].iloc[1]


def test_discuss_extrapolability_flags_no_warnings_for_healthy_model(
    good_regression_diagnostics: RegressionDiagnostics,
) -> None:
    """Un modelo sin problemas metodológicos no debe generar advertencias."""
    regression_metrics = {"r2_adjusted": 0.78}
    empty_cleaning_report = CleaningReport(outliers_detected={"MONTO_APLICADO": 0})

    result = discuss_extrapolability(
        regression_metrics, good_regression_diagnostics, empty_cleaning_report
    )
    assert result["es_extrapolable"] is True
    assert len(result["advertencias"]) == 0


def test_discuss_extrapolability_flags_all_warnings_for_bad_model(
    bad_regression_diagnostics: RegressionDiagnostics,
) -> None:
    """Un modelo con todos los problemas debe generar todas las advertencias esperadas."""
    regression_metrics = {"r2_adjusted": 0.05}
    cleaning_report = CleaningReport(outliers_detected={"MONTO_APLICADO": 10})

    result = discuss_extrapolability(
        regression_metrics, bad_regression_diagnostics, cleaning_report
    )
    assert result["es_extrapolable"] is False
    assert len(result["advertencias"]) == 5  # r2, vif, shapiro, bp, outliers
