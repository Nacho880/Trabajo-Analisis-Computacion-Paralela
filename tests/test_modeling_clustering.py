"""
tests/test_modeling_clustering.py
====================================

Pruebas unitarias para `src/modeling_clustering.py`.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from src.modeling_clustering import (
    build_customer_profile,
    compute_elbow_and_silhouette,
    fit_final_clustering,
    run_clustering_pipeline,
    select_optimal_k,
)


def test_build_customer_profile_aggregates_correctly() -> None:
    """El perfil debe tener una fila por cliente único con las agregaciones esperadas."""
    df = pd.DataFrame(
        {
            "CODIGO_CLIENTE": ["C1", "C1", "C2"],
            "MONTO_APLICADO": [100.0, 200.0, 500.0],
            "UNIDADES": [1, 2, 5],
            "PORCENTAJE_DESCUENTO": [0.1, 0.2, 0.0],
            "FRECUENCIA_COMPRA": [2, 2, 1],
        }
    )
    profile = build_customer_profile(df)

    assert len(profile) == 2  # C1, C2
    c1_row = profile[profile["CODIGO_CLIENTE"] == "C1"].iloc[0]
    assert c1_row["MONTO_APLICADO_sum"] == 300.0
    assert c1_row["MONTO_APLICADO_mean"] == 150.0


def test_compute_elbow_and_silhouette_produces_valid_scores() -> None:
    """El silhouette score debe estar en [-1, 1] para cada k evaluado."""
    rng = np.random.default_rng(42)
    cluster_1 = rng.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster_2 = rng.normal(loc=[10, 10], scale=0.5, size=(50, 2))
    cluster_3 = rng.normal(loc=[0, 10], scale=0.5, size=(50, 2))
    X = np.vstack([cluster_1, cluster_2, cluster_3])

    elbow_table, silhouette_table = compute_elbow_and_silhouette(X, range(2, 6), seed=42)

    assert len(elbow_table) == 4
    assert len(silhouette_table) == 4
    assert silhouette_table["silhouette_score"].between(-1.0, 1.0).all()
    assert elbow_table["inertia"].is_monotonic_decreasing


def test_compute_elbow_and_silhouette_raises_on_k_less_than_2() -> None:
    """k=1 no es válido para silhouette score; debe lanzar ValueError."""
    X = np.random.default_rng(42).normal(size=(20, 2))
    with pytest.raises(ValueError):
        compute_elbow_and_silhouette(X, range(1, 4), seed=42)


def test_compute_elbow_and_silhouette_samples_large_datasets() -> None:
    """
    Regresión: con datasets grandes, silhouette_score debe calcularse
    sobre una muestra acotada, no sobre la población completa.

    Bug de rendimiento real detectado con el archivo de producción
    (1.18 millones de clientes únicos): `silhouette_score` es O(n²) por
    defecto y la ejecución no terminaba en un tiempo razonable. Se
    verifica aquí que, al superar el umbral, el cómputo se complete
    rápidamente (evidencia indirecta de que se usó una muestra, no la
    población completa) y que el resultado siga siendo un score válido.
    """
    import time

    rng = np.random.default_rng(42)
    n = 50_000
    cluster_1 = rng.normal(loc=[0, 0], scale=1.0, size=(n // 2, 2))
    cluster_2 = rng.normal(loc=[15, 15], scale=1.0, size=(n // 2, 2))
    X = np.vstack([cluster_1, cluster_2])

    start = time.perf_counter()
    _, silhouette_table = compute_elbow_and_silhouette(
        X, range(2, 4), seed=42, silhouette_sample_size=2000
    )
    elapsed = time.perf_counter() - start

    # Con muestreo, 50k puntos deben procesarse en segundos, no minutos.
    assert elapsed < 30
    assert silhouette_table["silhouette_score"].between(-1.0, 1.0).all()


def test_select_optimal_k_picks_maximum_silhouette() -> None:
    """Debe seleccionar exactamente el k con mayor silhouette score."""
    table = pd.DataFrame(
        {"k": [2, 3, 4, 5], "silhouette_score": [0.3, 0.7, 0.5, 0.2]}
    )
    assert select_optimal_k(table) == 3


def test_fit_final_clustering_separates_well_defined_blobs() -> None:
    """Con 3 blobs muy separados, K-means con k=3 debe encontrar 3 clusters distintos."""
    rng = np.random.default_rng(42)
    cluster_1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    cluster_2 = rng.normal(loc=[20, 20], scale=0.3, size=(30, 2))
    cluster_3 = rng.normal(loc=[0, 20], scale=0.3, size=(30, 2))
    X = np.vstack([cluster_1, cluster_2, cluster_3])

    labels, cluster_profile = fit_final_clustering(X, ["f1", "f2"], optimal_k=3, seed=42)

    assert len(set(labels)) == 3
    assert len(cluster_profile) == 3
    assert cluster_profile["n_clientes"].sum() == 90


def test_run_clustering_pipeline_end_to_end(
    synthetic_raw_dataframe: pd.DataFrame,
) -> None:
    """El pipeline completo debe ejecutarse sin excepciones sobre datos sintéticos."""
    from config import COLUMN_RENAME_MAP
    from src.data_cleaning import clean_dataset
    from src.feature_engineering import engineer_features

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, _ = clean_dataset(dd.from_pandas(df, npartitions=2))
    ddf_final, _ = engineer_features(ddf_clean)
    df_final = ddf_final.compute()

    result = run_clustering_pipeline(
        df_final,
        feature_columns=["MONTO_APLICADO_sum", "UNIDADES_mean"],
        k_range=range(2, 5),
        seed=42,
    )

    assert result.optimal_k in range(2, 5)
    assert len(result.labels) > 0
    assert not result.cluster_profile.empty


def test_run_clustering_pipeline_handles_nullable_pandas_dtypes() -> None:
    """
    Regresión: columnas con tipos "nullable" de pandas (Int64, no int64)
    deben funcionar sin lanzar TypeError.

    Bug real detectado en el archivo de datos de producción: `UNIDADES`
    se declara como "Int64" (nullable) en `config.EXPECTED_DTYPES` para
    admitir valores faltantes. Esa nulabilidad se propaga a través de
    `groupby().agg()`, produciendo columnas "Float64" (nullable) en el
    perfil de cliente. Convertir eso a NumPy sin normalizar el dtype
    primero genera un array `dtype=object` en lugar de `float64` nativo,
    y `.std()` falla con "loop of ufunc does not support argument 0 of
    type float which has no callable sqrt method". El fix castea
    explícitamente a `float64` antes de `.to_numpy()`.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "CODIGO_CLIENTE": [f"C{i % 40}" for i in range(n)],
            "MONTO_APLICADO": pd.array(rng.uniform(100, 5000, n), dtype="float64"),
            "UNIDADES": pd.array(rng.integers(1, 5, n), dtype="Int64"),  # nullable
            "PORCENTAJE_DESCUENTO": pd.array(rng.uniform(0, 0.3, n), dtype="float64"),
            "BOLETA": np.arange(n),
            "FRECUENCIA_COMPRA": pd.array(rng.integers(1, 10, n), dtype="Int64"),
        }
    )

    result = run_clustering_pipeline(
        df,
        feature_columns=["MONTO_APLICADO_sum", "UNIDADES_mean"],
        k_range=range(2, 5),
        seed=42,
    )

    assert result.optimal_k in range(2, 5)
    assert not result.cluster_profile.empty
