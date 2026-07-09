"""
src/eda.py
==========

Análisis Exploratorio Estadístico (EDA).

Responsabilidad única: calcular estadística descriptiva completa,
generar las visualizaciones obligatorias del enunciado (histogramas
con densidad y test de normalidad, boxplots por categoría, matriz de
correlación con p-values) y devolver los resultados en estructuras
tabulares listas para exportar.

Este módulo NO decide qué hacer con outliers ni nulos (eso ya se hizo
en `data_cleaning.py`); solo describe el dataset ya limpio.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend no interactivo: requerido en entornos sin display (servidores/CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils.io_utils import save_figure, save_table
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Umbral de tamaño muestral a partir del cual Shapiro-Wilk deja de ser
# recomendable (pierde potencia estadística y se vuelve computacionalmente
# costoso); por encima de este umbral se usa Kolmogorov-Smirnov sobre una
# muestra aleatoria reproducible.
_SHAPIRO_MAX_N = 5000


def compute_descriptive_stats(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """
    Calcula medidas de tendencia central, dispersión, asimetría y
    curtosis para cada columna numérica indicada.

    Args:
        df: DataFrame limpio.
        numeric_columns: lista de columnas numéricas a describir.

    Returns:
        pd.DataFrame: una fila por variable, con columnas
        [n, mean, std, min, q1, median, q3, max, skewness, kurtosis].
        Las columnas ausentes en `df` se omiten silenciosamente (se
        registra advertencia en el log).

    Complejidad:
        O(n * k log n) tiempo (dominado por el cálculo de cuantiles por
        columna), O(k) espacio para el resultado, n = filas, k = columnas.
    """
    rows = []
    for column in numeric_columns:
        if column not in df.columns:
            logger.warning("Columna '%s' no encontrada; se omite del resumen descriptivo.", column)
            continue

        series = df[column].dropna()
        if series.empty:
            logger.warning("Columna '%s' no tiene valores no nulos; se omite.", column)
            continue

        rows.append(
            {
                "variable": column,
                "n": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "q1": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q3": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": float(stats.skew(series)),
                "kurtosis": float(stats.kurtosis(series)),  # Fisher (exceso), normal=0
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Estadística descriptiva calculada para %d variables.", len(result))
    return result


def check_normality(df: pd.DataFrame, column: str, seed: int) -> dict:
    """
    Aplica un test de normalidad a `column`, eligiendo automáticamente
    el método según el tamaño muestral:
        - n <= _SHAPIRO_MAX_N: Shapiro-Wilk (más potente para muestras
          pequeñas/medianas).
        - n > _SHAPIRO_MAX_N: Kolmogorov-Smirnov contra una normal con
          media/std estimadas de los datos, sobre una muestra aleatoria
          de tamaño _SHAPIRO_MAX_N (reproducible vía `seed`) para
          mantener el costo computacional acotado.

    Args:
        df: DataFrame con la columna a testear.
        column: columna numérica.
        seed: semilla para el muestreo reproducible en el caso K-S.

    Returns:
        dict: {"method": str, "statistic": float, "p_value": float,
        "n_used": int, "is_normal_at_0.05": bool}.

    Complejidad:
        O(n log n) tiempo (Shapiro y K-S ambos requieren ordenar),
        O(n) espacio (o O(_SHAPIRO_MAX_N) si se submuestrea).
    """
    series = df[column].dropna()
    n = len(series)

    if n <= _SHAPIRO_MAX_N:
        statistic, p_value = stats.shapiro(series)
        method = "shapiro-wilk"
        n_used = n
    else:
        sample = series.sample(n=_SHAPIRO_MAX_N, random_state=seed)
        mean, std = sample.mean(), sample.std(ddof=1)
        statistic, p_value = stats.kstest(sample, "norm", args=(mean, std))
        method = "kolmogorov-smirnov"
        n_used = _SHAPIRO_MAX_N

    result = {
        "variable": column,
        "method": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_used": n_used,
        "is_normal_at_0.05": bool(p_value >= 0.05),
    }
    logger.info(
        "Test de normalidad (%s) para '%s': estadístico=%.4f, p-value=%.4f -> %s",
        method, column, statistic, p_value,
        "NORMAL" if result["is_normal_at_0.05"] else "NO NORMAL",
    )
    return result


def plot_histogram_with_density(df: pd.DataFrame, column: str) -> plt.Figure:
    """
    Genera un histograma con curva de densidad superpuesta (KDE) para
    `column`.

    Args:
        df: DataFrame con la columna a graficar.
        column: columna numérica.

    Returns:
        matplotlib.figure.Figure: figura lista para guardar con
        `src.utils.io_utils.save_figure`.

    Complejidad:
        O(n) tiempo (KDE es O(n) a O(n log n) según implementación),
        O(n) espacio.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#2E86AB")
    ax.set_title(f"Distribución de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def plot_boxplot_by_category(
    df: pd.DataFrame, value_column: str, category_column: str
) -> plt.Figure:
    """
    Genera un boxplot de `value_column` agrupado por `category_column`.

    Args:
        df: DataFrame con ambas columnas.
        value_column: columna numérica (ej. MONTO_APLICADO).
        category_column: columna categórica (ej. CANAL).

    Returns:
        matplotlib.figure.Figure: figura lista para guardar.

    Complejidad:
        O(n log n) tiempo (ordenamiento interno para cuartiles por
        categoría), O(n) espacio.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=category_column, y=value_column, ax=ax, hue=category_column,
                palette="Set2", legend=False)
    ax.set_title(f"{value_column} por {category_column}")
    fig.tight_layout()
    return fig


def compute_correlation_matrix_with_pvalues(
    df: pd.DataFrame, numeric_columns: list[str], method: str = "pearson"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula la matriz de correlación entre las columnas indicadas junto
    con la matriz de p-values de significancia correspondiente.

    Args:
        df: DataFrame con las columnas numéricas.
        numeric_columns: columnas a incluir en la matriz.
        method: "pearson" o "spearman".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (matriz_correlacion,
        matriz_p_values), ambas cuadradas de tamaño k x k,
        k = len(numeric_columns).

    Raises:
        ValueError: si `method` no es "pearson" ni "spearman".

    Complejidad:
        O(k^2 * n) tiempo (una prueba de correlación por cada par de
        columnas), O(k^2) espacio, k = número de columnas, n = filas.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Método de correlación desconocido: '{method}'.")

    corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr

    valid_columns = [c for c in numeric_columns if c in df.columns]
    n_cols = len(valid_columns)
    corr_matrix = pd.DataFrame(np.eye(n_cols), index=valid_columns, columns=valid_columns)
    pval_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=valid_columns, columns=valid_columns)

    for i, col_i in enumerate(valid_columns):
        for j, col_j in enumerate(valid_columns):
            if j <= i:
                continue
            paired = df[[col_i, col_j]].dropna()
            if len(paired) < 2:
                corr_value, p_value = np.nan, np.nan
            else:
                corr_value, p_value = corr_func(paired[col_i], paired[col_j])
            corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i] = corr_value
            pval_matrix.iloc[i, j] = pval_matrix.iloc[j, i] = p_value

    logger.info(
        "Matriz de correlación (%s) calculada para %d variables.", method, n_cols
    )
    return corr_matrix, pval_matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, method: str) -> plt.Figure:
    """
    Genera un heatmap de la matriz de correlación.

    Args:
        corr_matrix: matriz de correlación (salida de
            `compute_correlation_matrix_with_pvalues`).
        method: nombre del método usado (solo para el título).

    Returns:
        matplotlib.figure.Figure: figura lista para guardar.

    Complejidad:
        O(k^2) tiempo y espacio, k = número de variables.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr_matrix.astype(float), annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, ax=ax, square=True,
    )
    ax.set_title(f"Matriz de correlación ({method})")
    fig.tight_layout()
    return fig


def run_descriptive_analysis(
    df: pd.DataFrame,
    numeric_columns: list[str],
    category_column: str,
    boxplot_value_column: str,
    correlation_columns: list[str],
    seed: int,
) -> dict:
    """
    Orquesta el análisis exploratorio completo exigido por el
    enunciado: estadística descriptiva, tests de normalidad,
    histogramas+densidad, boxplots por categoría y matriz de
    correlación con p-values. Exporta automáticamente todos los
    artefactos generados.

    Args:
        df: DataFrame limpio y transformado.
        numeric_columns: columnas numéricas a describir y testear normalidad.
        category_column: columna categórica para los boxplots (ej. CANAL).
        boxplot_value_column: columna numérica graficada en el boxplot.
        correlation_columns: columnas incluidas en la matriz de correlación.
        seed: semilla de reproducibilidad (para submuestreo en K-S).

    Returns:
        dict: resumen con las claves "descriptive_stats" (DataFrame),
        "normality_tests" (list[dict]), "correlation_pearson"
        (DataFrame), "correlation_spearman" (DataFrame), y las rutas de
        las figuras generadas.

    Complejidad:
        O(n * k log n) tiempo dominado por la estadística descriptiva y
        las correlaciones (ver funciones individuales); O(n + k^2)
        espacio.
    """
    logger.info("=== Iniciando Análisis Exploratorio Estadístico ===")

    descriptive_stats = compute_descriptive_stats(df, numeric_columns)
    save_table(descriptive_stats, "eda_estadisticos_descriptivos")

    normality_results = [check_normality(df, col, seed) for col in numeric_columns]
    save_table(pd.DataFrame(normality_results), "eda_tests_normalidad")

    figure_paths = []
    for column in numeric_columns:
        fig = plot_histogram_with_density(df, column)
        figure_paths.append(save_figure(fig, f"histograma_{column.lower()}"))

    boxplot_fig = plot_boxplot_by_category(df, boxplot_value_column, category_column)
    figure_paths.append(
        save_figure(boxplot_fig, f"boxplot_{boxplot_value_column.lower()}_por_{category_column.lower()}")
    )

    corr_pearson, pval_pearson = compute_correlation_matrix_with_pvalues(
        df, correlation_columns, method="pearson"
    )
    corr_spearman, pval_spearman = compute_correlation_matrix_with_pvalues(
        df, correlation_columns, method="spearman"
    )
    save_table(corr_pearson, "eda_correlacion_pearson", index=True)
    save_table(pval_pearson, "eda_correlacion_pearson_pvalues", index=True)
    save_table(corr_spearman, "eda_correlacion_spearman", index=True)
    save_table(pval_spearman, "eda_correlacion_spearman_pvalues", index=True)

    figure_paths.append(save_figure(plot_correlation_heatmap(corr_pearson, "pearson"), "heatmap_correlacion_pearson"))
    figure_paths.append(save_figure(plot_correlation_heatmap(corr_spearman, "spearman"), "heatmap_correlacion_spearman"))

    logger.info("=== Análisis Exploratorio Estadístico finalizado ===")

    return {
        "descriptive_stats": descriptive_stats,
        "normality_tests": normality_results,
        "correlation_pearson": corr_pearson,
        "correlation_pearson_pvalues": pval_pearson,
        "correlation_spearman": corr_spearman,
        "correlation_spearman_pvalues": pval_spearman,
        "figure_paths": figure_paths,
    }
