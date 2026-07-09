"""
tests/test_eda.py
===================

Pruebas unitarias para `src/eda.py`.

No se testea la calidad estética de las figuras (fuera del alcance de
un test automatizado), pero sí que se generen sin excepciones y que
los cálculos numéricos subyacentes sean correctos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda import (
    compute_correlation_matrix_with_pvalues,
    compute_descriptive_stats,
    plot_boxplot_by_category,
    plot_histogram_with_density,
    check_normality,
)


@pytest.fixture()
def normal_dataframe() -> pd.DataFrame:
    """DataFrame con una columna normal y otra claramente no normal (uniforme)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "NORMAL_COL": rng.normal(loc=100, scale=15, size=500),
            "UNIFORM_COL": rng.uniform(0, 1, size=500),
            "CANAL": rng.choice(["POS", "WEB"], size=500),
        }
    )


def test_compute_descriptive_stats_known_values() -> None:
    """Verifica los estadísticos descriptivos contra un cálculo manual conocido."""
    df = pd.DataFrame({"X": [1, 2, 3, 4, 5]})
    result = compute_descriptive_stats(df, ["X"])

    row = result.iloc[0]
    assert row["n"] == 5
    assert row["mean"] == 3.0
    assert row["median"] == 3.0
    assert row["min"] == 1.0
    assert row["max"] == 5.0


def test_compute_descriptive_stats_skips_missing_column() -> None:
    """Una columna inexistente debe omitirse sin lanzar excepción."""
    df = pd.DataFrame({"X": [1, 2, 3]})
    result = compute_descriptive_stats(df, ["X", "NO_EXISTE"])
    assert len(result) == 1
    assert result.iloc[0]["variable"] == "X"


def test_normality_detects_normal_distribution(normal_dataframe: pd.DataFrame) -> None:
    """Una distribución generada como normal no debe rechazar H0 de normalidad."""
    result = check_normality(normal_dataframe, "NORMAL_COL", seed=42)
    assert result["p_value"] > 0.01  # margen amplio para evitar falsos negativos


def test_normality_detects_non_normal_distribution(normal_dataframe: pd.DataFrame) -> None:
    """Una distribución uniforme debe rechazar H0 de normalidad."""
    result = check_normality(normal_dataframe, "UNIFORM_COL", seed=42)
    assert result["p_value"] < 0.05


def test_correlation_matrix_diagonal_is_one() -> None:
    """La diagonal de la matriz de correlación debe ser siempre 1.0."""
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1], "C": [1, 3, 2, 5, 4]}
    )
    corr, pval = compute_correlation_matrix_with_pvalues(df, ["A", "B", "C"])
    assert all(np.isclose(corr.loc[c, c], 1.0) for c in ["A", "B", "C"])


def test_correlation_matrix_perfect_negative_correlation() -> None:
    """A y B son perfectamente anticorrelacionadas por construcción."""
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})
    corr, pval = compute_correlation_matrix_with_pvalues(df, ["A", "B"])
    assert np.isclose(corr.loc["A", "B"], -1.0)
    assert pval.loc["A", "B"] < 0.05


def test_correlation_matrix_invalid_method_raises() -> None:
    """Un método de correlación desconocido debe lanzar ValueError."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        compute_correlation_matrix_with_pvalues(df, ["A"], method="invalido")


def test_plot_histogram_returns_figure(normal_dataframe: pd.DataFrame) -> None:
    """La función debe retornar un objeto Figure sin lanzar excepciones."""
    import matplotlib.figure
    fig = plot_histogram_with_density(normal_dataframe, "NORMAL_COL")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_boxplot_returns_figure(normal_dataframe: pd.DataFrame) -> None:
    """La función debe retornar un objeto Figure sin lanzar excepciones."""
    import matplotlib.figure
    fig = plot_boxplot_by_category(normal_dataframe, "NORMAL_COL", "CANAL")
    assert isinstance(fig, matplotlib.figure.Figure)
