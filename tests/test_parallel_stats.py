"""
tests/test_parallel_stats.py
==============================

Pruebas unitarias, de reproducibilidad/correctitud y de rendimiento
para `src/parallel_stats.py`.

Nota sobre el entorno de ejecución:
    En máquinas con un único núcleo lógico disponible, el benchmark de
    rendimiento (`test_parallel_matches_sequential_speedup_reported`)
    verificará que el mecanismo de medición funciona correctamente
    (produce un speedup/eficiencia numéricos válidos), pero no exigirá
    speedup > 1, ya que en un solo núcleo la paralelización con
    procesos añade overhead sin beneficio real. La prueba de
    CORRECTITUD (que los resultados coincidan) sí se exige siempre,
    independientemente del número de núcleos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.parallel_stats import (
    benchmark_sequential_vs_parallel,
    compute_stats_parallel,
    compute_stats_sequential,
    plot_scalability_curve,
    run_scalability_study,
    stats_to_dataframe,
)


@pytest.fixture()
def multi_local_dataframe() -> pd.DataFrame:
    """DataFrame con 4 LOCAL distintos y tamaños de partición desiguales."""
    rng = np.random.default_rng(42)
    frames = []
    for local_id, n_rows in zip([1001, 1002, 1003, 1004], [50, 80, 30, 100]):
        frames.append(
            pd.DataFrame(
                {
                    "LOCAL": local_id,
                    "MONTO_APLICADO": rng.normal(
                        loc=local_id, scale=100, size=n_rows
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_sequential_and_parallel_produce_same_partitions(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """Ambas estrategias deben procesar exactamente las mismas particiones."""
    seq_results = compute_stats_sequential(
        multi_local_dataframe, "LOCAL", "MONTO_APLICADO"
    )
    par_results = compute_stats_parallel(
        multi_local_dataframe, "LOCAL", "MONTO_APLICADO", n_jobs=2
    )

    seq_keys = {r.partition_key for r in seq_results}
    par_keys = {r.partition_key for r in par_results}
    assert seq_keys == par_keys == {1001, 1002, 1003, 1004}


def test_sequential_and_parallel_produce_identical_statistics(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """Los estadísticos numéricos deben ser idénticos entre ambas versiones."""
    seq_results = {
        r.partition_key: r
        for r in compute_stats_sequential(
            multi_local_dataframe, "LOCAL", "MONTO_APLICADO"
        )
    }
    par_results = {
        r.partition_key: r
        for r in compute_stats_parallel(
            multi_local_dataframe, "LOCAL", "MONTO_APLICADO", n_jobs=2
        )
    }

    for key in seq_results:
        assert np.isclose(seq_results[key].mean, par_results[key].mean)
        assert np.isclose(seq_results[key].std, par_results[key].std)
        assert seq_results[key].n_rows == par_results[key].n_rows


def test_partition_sizes_match_expected_counts(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """El número de filas por partición debe coincidir con lo generado."""
    results = compute_stats_sequential(
        multi_local_dataframe, "LOCAL", "MONTO_APLICADO"
    )
    counts = {r.partition_key: r.n_rows for r in results}
    assert counts == {1001: 50, 1002: 80, 1003: 30, 1004: 100}


def test_benchmark_reports_valid_speedup_and_efficiency(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """
    El benchmark debe ejecutar ambas estrategias, verificar su
    correctitud internamente (sin lanzar AssertionError) y reportar
    speedup/eficiencia como números finitos y positivos.
    """
    results, benchmark = benchmark_sequential_vs_parallel(
        multi_local_dataframe, group_column="LOCAL",
        target_column="MONTO_APLICADO", n_jobs=2,
    )

    assert len(results) == 4
    assert benchmark.n_partitions == 4
    assert benchmark.n_jobs == 2
    assert benchmark.speedup > 0
    assert benchmark.efficiency > 0
    assert benchmark.sequential_time_seconds >= 0
    assert benchmark.parallel_time_seconds >= 0


def test_stats_to_dataframe_produces_sorted_table(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """La tabla resultante debe estar ordenada por LOCAL y tener 4 filas."""
    results = compute_stats_sequential(
        multi_local_dataframe, "LOCAL", "MONTO_APLICADO"
    )
    df = stats_to_dataframe(results)

    assert len(df) == 4
    assert list(df["LOCAL"]) == sorted(df["LOCAL"])


def test_empty_groups_are_skipped_without_error() -> None:
    """Una columna objetivo completamente nula en una partición no debe romper el cálculo."""
    df = pd.DataFrame(
        {
            "LOCAL": [1, 1, 2, 2],
            "MONTO_APLICADO": [10.0, 20.0, np.nan, np.nan],
        }
    )
    seq_results = compute_stats_sequential(df, "LOCAL", "MONTO_APLICADO")
    par_results = compute_stats_parallel(df, "LOCAL", "MONTO_APLICADO", n_jobs=2)

    # Solo LOCAL=1 tiene datos válidos; LOCAL=2 (todo NaN) se omite.
    assert {r.partition_key for r in seq_results} == {1}
    assert {r.partition_key for r in par_results} == {1}


def test_run_scalability_study_produces_valid_speedup_curve(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """
    El estudio de escalabilidad debe producir una fila por cada valor
    de n_jobs evaluado, con speedup/eficiencia numéricos válidos, y
    debe verificar correctitud internamente para cada configuración
    (sin lanzar AssertionError).
    """
    table = run_scalability_study(
        multi_local_dataframe, group_column="LOCAL",
        target_column="MONTO_APLICADO", n_jobs_values=(1, 2, 4),
    )

    assert list(table["n_jobs"]) == [1, 2, 4]
    assert (table["speedup"] > 0).all()
    assert (table["efficiency"] > 0).all()
    # El tiempo secuencial de referencia debe ser idéntico en todas las filas
    # (se mide una sola vez y se reutiliza para todas las comparaciones).
    assert table["sequential_time_seconds"].nunique() == 1


def test_run_scalability_study_uses_default_n_jobs_values(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """Sin especificar n_jobs_values, debe usar el conjunto por defecto (incluye 1 y N_JOBS)."""
    table = run_scalability_study(
        multi_local_dataframe, group_column="LOCAL", target_column="MONTO_APLICADO"
    )
    assert 1 in table["n_jobs"].values


def test_plot_scalability_curve_returns_figure(
    multi_local_dataframe: pd.DataFrame,
) -> None:
    """La función de graficado debe retornar un objeto Figure válido."""
    import matplotlib.figure

    table = run_scalability_study(
        multi_local_dataframe, group_column="LOCAL",
        target_column="MONTO_APLICADO", n_jobs_values=(1, 2),
    )
    fig = plot_scalability_curve(table)
    assert isinstance(fig, matplotlib.figure.Figure)

