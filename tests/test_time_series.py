"""
tests/test_time_series.py
============================

Pruebas unitarias para `src/time_series.py`.

La lógica de construcción de la serie diaria (`build_daily_sales_series`)
no depende de statsmodels y se testea siempre. Las pruebas de
descomposición y ACF/PACF requieren `statsmodels` real y se omiten
automáticamente si no está instalado en el entorno de ejecución.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.time_series import build_daily_sales_series


def test_build_daily_sales_series_sums_by_day() -> None:
    """Dos transacciones el mismo día deben sumarse en un único valor diario."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(
                ["2026-01-01 08:00", "2026-01-01 15:00", "2026-01-02 10:00"]
            ),
            "MONTO_APLICADO": [1000.0, 500.0, 2000.0],
        }
    )
    series = build_daily_sales_series(df)

    assert series.loc["2026-01-01"] == 1500.0
    assert series.loc["2026-01-02"] == 2000.0


def test_build_daily_sales_series_fills_gaps_with_zero() -> None:
    """Un día sin transacciones dentro del rango debe aparecer con valor 0, no desaparecer."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-01-01", "2026-01-05"]),
            "MONTO_APLICADO": [1000.0, 2000.0],
        }
    )
    series = build_daily_sales_series(df)

    assert len(series) == 5  # 1,2,3,4,5 de enero
    assert series.loc["2026-01-03"] == 0.0


def test_decompose_time_series_raises_on_short_series() -> None:
    """Una serie con menos de 2 ciclos debe lanzar ValueError, no fallar silenciosamente."""
    pytest.importorskip("statsmodels")
    from src.time_series import decompose_time_series

    short_series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    with pytest.raises(ValueError):
        decompose_time_series(short_series, period=7)


def test_decompose_time_series_succeeds_on_long_series() -> None:
    """Una serie con suficientes ciclos debe descomponerse sin excepciones."""
    pytest.importorskip("statsmodels")
    from src.time_series import decompose_time_series

    rng = np.random.default_rng(42)
    n_days = 60
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    trend = np.linspace(1000, 2000, n_days)
    seasonal = 200 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
    noise = rng.normal(0, 20, n_days)
    series = pd.Series(trend + seasonal + noise, index=dates)

    result = decompose_time_series(series, period=7)
    assert "trend" in result and "seasonal" in result and "residual" in result


def test_compute_acf_pacf_returns_expected_length() -> None:
    """El resultado de ACF/PACF debe tener nlags+1 valores (incluye rezago 0)."""
    pytest.importorskip("statsmodels")
    from src.time_series import compute_acf_pacf

    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(100, 10, 100))

    result = compute_acf_pacf(series, nlags=10)
    assert len(result["acf"]) == 11
    assert len(result["pacf"]) == 11
    assert np.isclose(result["acf"][0], 1.0)  # autocorrelación en rezago 0 es siempre 1


def test_detect_and_trim_leading_gap_trims_leading_inactivity() -> None:
    """
    Regresión del hallazgo real: un lote de actividad aislado seguido de
    un hueco largo (>50% de ceros en la ventana de evaluación) debe
    recortarse; la serie completa NO se modifica, se retorna un recorte.
    """
    from src.time_series import detect_and_trim_leading_gap

    dates = pd.date_range("2023-11-09", periods=200, freq="D")
    values = np.zeros(200)
    values[0] = 96731.0  # lote aislado, día 1
    # Actividad sostenida a partir del día 150 (149 días de silencio total).
    values[150:] = 50000.0

    series = pd.Series(values, index=dates)
    trimmed, report = detect_and_trim_leading_gap(series, window_days=28, max_zero_fraction=0.5)

    assert report["trimmed_days"] > 100  # se detectó y recortó el hueco
    assert trimmed.index.min() >= dates[100]  # el recorte cae dentro del hueco de ceros
    assert len(series) == 200  # la serie original NO se modificó (inmutabilidad)


def test_detect_and_trim_leading_gap_is_noop_on_continuous_series() -> None:
    """Una serie sin huecos iniciales no debe recortarse (start_idx=0)."""
    from src.time_series import detect_and_trim_leading_gap

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    series = pd.Series(rng.uniform(1000, 5000, 100), index=dates)  # siempre > 0

    trimmed, report = detect_and_trim_leading_gap(series)

    assert report["trimmed_days"] == 0
    assert len(trimmed) == len(series)
