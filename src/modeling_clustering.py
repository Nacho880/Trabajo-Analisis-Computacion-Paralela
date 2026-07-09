"""
src/modeling_clustering.py
=============================

Opción B del enunciado: segmentación de clientes usando K-means,
validada con el método del codo (elbow) y el silhouette score.

Responsabilidad única: a partir de un DataFrame con variables
numéricas ya estandarizadas (ver `feature_engineering.standardize_columns`),
construir el perfil de cliente (una fila por CODIGO_CLIENTE, agregando
sus transacciones), determinar el número óptimo de clusters, ajustar
K-means con la semilla del proyecto, y validar el resultado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClusteringResult:
    """
    Resultado completo del análisis de clustering.

    Attributes:
        elbow_table: DataFrame con columnas ["k", "inertia"] para el
            gráfico del método del codo.
        silhouette_table: DataFrame con columnas ["k", "silhouette_score"].
        optimal_k: número de clusters seleccionado (máximo silhouette).
        labels: asignación de cluster para cada fila del perfil de
            clientes (mismo orden que el DataFrame de entrada).
        cluster_profile: DataFrame con la media de cada variable por
            cluster, para interpretación de negocio.
    """

    elbow_table: pd.DataFrame
    silhouette_table: pd.DataFrame
    optimal_k: int
    labels: np.ndarray
    cluster_profile: pd.DataFrame

    def to_summary_dict(self) -> dict:
        """Serializa un resumen (sin las etiquetas por fila) apto para JSON."""
        return {
            "optimal_k": self.optimal_k,
            "best_silhouette_score": float(
                self.silhouette_table.loc[
                    self.silhouette_table["k"] == self.optimal_k, "silhouette_score"
                ].iloc[0]
            ),
        }


def build_customer_profile(
    df: pd.DataFrame,
    customer_column: str = "CODIGO_CLIENTE",
    aggregations: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Construye un perfil agregado por cliente a partir de las
    transacciones individuales, requisito previo para segmentar
    clientes (no transacciones).

    Args:
        df: DataFrame de transacciones limpio y transformado.
        customer_column: columna identificadora del cliente.
        aggregations: mapeo {columna: función de agregación} (por
            defecto: gasto total y promedio, unidades promedio,
            frecuencia de compra y descuento promedio recibido).

    Returns:
        pd.DataFrame: una fila por cliente único, indexado por
        `customer_column`.

    Complejidad:
        O(n) tiempo (agrupación por hash), O(c) espacio,
        c = número de clientes únicos.
    """
    aggregations = aggregations or {
        "MONTO_APLICADO": ["sum", "mean"],
        "UNIDADES": "mean",
        "PORCENTAJE_DESCUENTO": "mean",
        "FRECUENCIA_COMPRA": "first",
    }

    profile = df.groupby(customer_column).agg(aggregations)
    profile.columns = ["_".join(col).strip("_") for col in profile.columns]
    profile = profile.reset_index()

    logger.info(
        "Perfil de cliente construido: %d clientes únicos, %d variables.",
        len(profile), profile.shape[1] - 1,
    )
    return profile


def compute_elbow_and_silhouette(
    X: np.ndarray, k_range: range, seed: int = SEED, silhouette_sample_size: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ajusta K-means para cada valor de k en `k_range` y calcula la
    inercia (para el método del codo) y el silhouette score (para
    validar la cohesión/separación de los clusters).

    Muestreo para silhouette score en datasets grandes:
        `sklearn.metrics.silhouette_score` calcula, por defecto, la
        matriz de distancias entre TODOS los pares de puntos: O(n²)
        tiempo y espacio. Con datasets de cientos de miles o millones
        de filas (como el perfil de clientes de este proyecto, con
        más de un millón de clientes únicos), esto es computacionalmente
        inviable (se verificó empíricamente: la ejecución no terminaba
        en un tiempo razonable). La práctica estándar de la industria
        para este caso es calcular el silhouette score sobre una
        MUESTRA ALEATORIA REPRODUCIBLE de tamaño acotado
        (`silhouette_sample_size`, parámetro nativo `sample_size` de
        scikit-learn), en lugar de la población completa. Esto sigue
        siendo un estimador estadísticamente válido de la calidad del
        clustering (el silhouette score de una muestra aleatoria grande
        converge al valor poblacional), a un costo computacional
        acotado y constante independiente de n.

    Args:
        X: matriz de features numéricas, ya estandarizadas.
        k_range: rango de valores de k a evaluar (ej. range(2, 11)).
            k debe ser >= 2 (silhouette score no está definido para k=1).
        seed: semilla de reproducibilidad para K-means y para el
            muestreo del silhouette score.
        silhouette_sample_size: tamaño máximo de muestra para calcular
            el silhouette score. Si `len(X) <= silhouette_sample_size`,
            se usa la población completa sin submuestrear (el costo
            O(n²) es aceptable para datasets pequeños/medianos).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (elbow_table, silhouette_table).

    Raises:
        ValueError: si algún valor de k en `k_range` es menor a 2.

    Complejidad:
        O(|k_range| * n * k_max * i) tiempo para el ajuste de K-means
        (n = filas, i = iteraciones hasta convergencia); el cálculo del
        silhouette score queda acotado a
        O(|k_range| * min(n, silhouette_sample_size)²) en lugar de
        O(|k_range| * n²). O(n) espacio por ajuste de K-means.
    """
    if min(k_range) < 2:
        raise ValueError("El silhouette score requiere k >= 2 para todos los valores evaluados.")

    n_samples = X.shape[0]
    effective_sample_size = None if n_samples <= silhouette_sample_size else silhouette_sample_size
    if effective_sample_size is not None:
        logger.warning(
            "Dataset con %d filas supera el umbral de %d para silhouette_score "
            "completo (costo O(n²) inviable a esta escala). Se calculará sobre "
            "una muestra aleatoria reproducible de %d filas (seed=%d) en su "
            "lugar, un estimador estadísticamente válido y estándar en la "
            "industria para clustering a gran escala.",
            n_samples, silhouette_sample_size, effective_sample_size, seed,
        )

    elbow_rows = []
    silhouette_rows = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X)

        elbow_rows.append({"k": k, "inertia": float(kmeans.inertia_)})
        score = silhouette_score(
            X, labels, sample_size=effective_sample_size, random_state=seed
        )
        silhouette_rows.append({"k": k, "silhouette_score": float(score)})

        logger.info("K-means k=%d: inercia=%.2f, silhouette=%.4f.", k, kmeans.inertia_, score)

    return pd.DataFrame(elbow_rows), pd.DataFrame(silhouette_rows)


def select_optimal_k(silhouette_table: pd.DataFrame) -> int:
    """
    Selecciona el número óptimo de clusters como el que maximiza el
    silhouette score.

    Args:
        silhouette_table: salida de `compute_elbow_and_silhouette`.

    Returns:
        int: valor de k con mayor silhouette score.

    Complejidad:
        O(m) tiempo, m = número de filas de la tabla; O(1) espacio.
    """
    best_row = silhouette_table.loc[silhouette_table["silhouette_score"].idxmax()]
    optimal_k = int(best_row["k"])
    logger.info(
        "K óptimo seleccionado: k=%d (silhouette=%.4f).",
        optimal_k, best_row["silhouette_score"],
    )
    return optimal_k


def fit_final_clustering(
    X: np.ndarray, feature_names: list[str], optimal_k: int, seed: int = SEED
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Ajusta el modelo K-means final con `optimal_k` clusters y calcula
    el perfil promedio de cada cluster (para interpretación de negocio).

    Args:
        X: matriz de features numéricas estandarizadas.
        feature_names: nombres de las columnas de X, en el mismo orden.
        optimal_k: número de clusters a ajustar.
        seed: semilla de reproducibilidad.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: (etiquetas de cluster por fila,
        perfil promedio por cluster).

    Complejidad:
        O(n * k * i) tiempo, O(n) espacio, n = filas, k = optimal_k,
        i = iteraciones hasta convergencia.
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    profile = pd.DataFrame(X, columns=feature_names)
    profile["cluster"] = labels
    cluster_profile = profile.groupby("cluster").mean()
    cluster_profile["n_clientes"] = profile.groupby("cluster").size()

    logger.info(
        "Clustering final ajustado: k=%d, tamaños de cluster=%s.",
        optimal_k, cluster_profile["n_clientes"].to_dict(),
    )
    return labels, cluster_profile


def run_clustering_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str],
    customer_column: str = "CODIGO_CLIENTE",
    k_range: range = range(2, 8),
    seed: int = SEED,
) -> ClusteringResult:
    """
    Orquesta el pipeline completo de segmentación de clientes: perfil
    de cliente, estandarización de features, selección de k óptimo vía
    elbow+silhouette, y ajuste del modelo final.

    Args:
        df: DataFrame de transacciones limpio y transformado.
        feature_columns: columnas del perfil de cliente a usar como
            features de clustering (deben existir tras
            `build_customer_profile`, ej. "MONTO_APLICADO_sum").
        customer_column: columna identificadora del cliente.
        k_range: rango de valores de k a evaluar.
        seed: semilla de reproducibilidad.

    Returns:
        ClusteringResult: resultado completo, listo para exportar.

    Complejidad:
        O(|k_range| * n * k_max * i) tiempo, n = clientes únicos;
        O(n) espacio.
    """
    logger.info("=== Iniciando pipeline de clustering (Opción B) ===")

    profile = build_customer_profile(df, customer_column)
    profile_clean = profile.dropna(subset=feature_columns)

    # .astype("float64") ANTES de .to_numpy() es deliberado: las columnas de
    # origen pueden tener tipos "nullable" de pandas (ej. Int64, Float64 con
    # mayúscula inicial, usados en config.EXPECTED_DTYPES para admitir NA).
    # Esa nulabilidad se propaga a través de groupby().agg(), y convertir
    # directamente a NumPy sin normalizar el dtype primero produce un array
    # dtype=object (no float64 nativo), donde cada elemento es un `float`
    # de Python en lugar de un `numpy.float64`. Operaciones vectorizadas
    # como `.std()` fallan sobre ese array porque NumPy intenta invocar un
    # método `.sqrt()` inexistente en el objeto Python plano. Se verificó
    # empíricamente este error con el archivo real de este proyecto.
    X_raw = profile_clean[feature_columns].astype("float64").to_numpy()
    # Estandarización propia (z-score) para el espacio de clustering:
    # K-means es sensible a la escala, y las columnas del perfil (montos
    # en pesos vs frecuencias en unidades) tienen escalas muy distintas.
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0  # evita división por cero en columnas constantes
    X = (X_raw - means) / stds

    elbow_table, silhouette_table = compute_elbow_and_silhouette(X, k_range, seed)
    optimal_k = select_optimal_k(silhouette_table)
    labels, cluster_profile = fit_final_clustering(X, feature_columns, optimal_k, seed)

    logger.info("=== Pipeline de clustering finalizado ===")

    return ClusteringResult(
        elbow_table=elbow_table,
        silhouette_table=silhouette_table,
        optimal_k=optimal_k,
        labels=labels,
        cluster_profile=cluster_profile,
    )
