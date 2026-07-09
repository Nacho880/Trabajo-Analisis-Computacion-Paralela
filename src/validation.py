"""
src/validation.py
===================

Consolidación de la validación de modelos y discusión de
extrapolabilidad/limitaciones, tal como exige el enunciado.

Responsabilidad única: este módulo NO reajusta modelos ni recalcula
métricas desde cero (eso ya ocurre en `modeling_regression.py` y
`modeling_clustering.py`, donde el split train/test 70/30 y las
métricas RMSE/MAE/R² y silhouette se calculan de forma reproducible).
Aquí se consolidan esos resultados en una tabla única exportable y se
genera la discusión estructurada de limitaciones que el informe
técnico debe presentar en lenguaje no técnico.
"""

from __future__ import annotations

import pandas as pd

from src.data_cleaning import CleaningReport
from src.modeling_clustering import ClusteringResult
from src.modeling_regression import RegressionDiagnostics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_validation_summary(
    regression_metrics: dict,
    regression_diagnostics: RegressionDiagnostics,
    clustering_result: ClusteringResult,
) -> pd.DataFrame:
    """
    Consolida las métricas de validación de ambos modelos (Opción A:
    Regresión, Opción B: Clustering) en una única tabla comparable.

    Args:
        regression_metrics: dict retornado por
            `modeling_regression.fit_and_diagnose_regression` (rmse,
            mae, r2_test, r2_adjusted, n_train, n_test).
        regression_diagnostics: diagnóstico de supuestos de la regresión.
        clustering_result: resultado del pipeline de clustering.

    Returns:
        pd.DataFrame: tabla con una fila por modelo y sus métricas
        clave, lista para incluir en el informe técnico.

    Complejidad:
        O(1) tiempo y espacio (número fijo de modelos y métricas).
    """
    rows = [
        {
            "modelo": "Regresión lineal múltiple (Opción A)",
            "metrica_principal": "RMSE",
            "valor_principal": regression_metrics["rmse"],
            "metrica_secundaria": "R² ajustado",
            "valor_secundario": regression_metrics["r2_adjusted"],
            "n_train": regression_metrics["n_train"],
            "n_test": regression_metrics["n_test"],
        },
        {
            "modelo": "K-means clustering (Opción B)",
            "metrica_principal": "Silhouette score",
            "valor_principal": clustering_result.to_summary_dict()["best_silhouette_score"],
            "metrica_secundaria": "k óptimo",
            "valor_secundario": clustering_result.optimal_k,
            "n_train": None,
            "n_test": None,
        },
    ]
    summary = pd.DataFrame(rows)
    logger.info("Resumen de validación consolidado para %d modelos.", len(summary))
    return summary


def discuss_extrapolability(
    regression_metrics: dict,
    regression_diagnostics: RegressionDiagnostics,
    cleaning_report: CleaningReport,
) -> dict:
    """
    Genera una discusión estructurada (no técnica) sobre si el modelo
    de regresión es extrapolable a datos futuros o de otras sucursales,
    y qué limitaciones deben considerarse.

    La evaluación se basa en criterios objetivos derivados de los
    diagnósticos ya calculados:
        - R² ajustado bajo (< 0.3) sugiere que el modelo captura poca
          variabilidad y no debería usarse para predicciones puntuales
          de negocio, solo para entender tendencias generales.
        - VIF máximo alto (> 10) sugiere que los coeficientes
          individuales son inestables ante pequeños cambios en los
          datos de entrenamiento, limitando su interpretabilidad causal.
        - Residuos no normales o heterocedásticos invalidan los
          intervalos de confianza clásicos de la regresión, aunque no
          necesariamente el uso del modelo para predicción puntual.
        - Un alto porcentaje de outliers marcados en la variable
          objetivo sugiere que el modelo fue entrenado con casos
          extremos que pueden no representar el comportamiento típico.

    Args:
        regression_metrics: métricas de la regresión (rmse, r2_adjusted, etc.).
        regression_diagnostics: diagnóstico de supuestos de la regresión.
        cleaning_report: reporte de limpieza (para contextualizar
            outliers y valores faltantes tratados).

    Returns:
        dict: {"es_extrapolable": bool, "advertencias": list[str],
        "recomendacion": str}.

    Complejidad:
        O(1) tiempo y espacio.
    """
    warnings: list[str] = []

    if regression_metrics["r2_adjusted"] < 0.3:
        warnings.append(
            "El R² ajustado es bajo (< 0.3): el modelo explica poca "
            "variabilidad de MONTO_APLICADO. Es útil para entender "
            "tendencias generales, pero no debe usarse para predicciones "
            "puntuales de alta precisión."
        )

    max_vif = regression_diagnostics.vif_table["VIF"].max()
    if max_vif > 10:
        warnings.append(
            f"Se detectó multicolinealidad severa (VIF máximo={max_vif:.1f} > 10). "
            "Los coeficientes individuales pueden ser inestables; interprételos "
            "con cautela y evite atribuir causalidad a variables específicas."
        )

    if regression_diagnostics.shapiro_residuals_pvalue < 0.05:
        warnings.append(
            "Los residuos del modelo no siguen una distribución normal "
            "(p < 0.05 en Shapiro-Wilk). Los intervalos de confianza "
            "clásicos de la regresión pueden no ser precisos."
        )

    if regression_diagnostics.breusch_pagan_pvalue < 0.05:
        warnings.append(
            "Se detectó heterocedasticidad (p < 0.05 en Breusch-Pagan): "
            "la varianza del error no es constante, por lo que el modelo "
            "puede ser menos confiable para montos de compra muy altos o "
            "muy bajos que para el rango medio de los datos."
        )

    total_outliers = sum(cleaning_report.outliers_detected.values())
    if total_outliers > 0:
        warnings.append(
            f"Se marcaron {total_outliers} transacciones como outliers durante "
            "la limpieza (no se eliminaron). Si representan una proporción alta "
            "del dataset, pueden estar influyendo desproporcionadamente en los "
            "coeficientes del modelo."
        )

    es_extrapolable = len(warnings) == 0
    if es_extrapolable:
        recomendacion = (
            "El modelo cumple los supuestos evaluados y explica una proporción "
            "razonable de la variabilidad de los datos. Puede usarse con "
            "confianza moderada para estimar montos de venta bajo condiciones "
            "similares a las observadas en el período analizado."
        )
    else:
        recomendacion = (
            "El modelo presenta limitaciones metodológicas documentadas arriba. "
            "Se recomienda usarlo como herramienta exploratoria/descriptiva y "
            "no como predictor definitivo para decisiones de negocio de alto "
            "impacto, especialmente al extrapolar a sucursales, períodos o "
            "rangos de monto no representados en los datos de entrenamiento."
        )

    logger.info(
        "Discusión de extrapolabilidad generada: %d advertencias, "
        "es_extrapolable=%s.", len(warnings), es_extrapolable,
    )

    return {
        "es_extrapolable": es_extrapolable,
        "advertencias": warnings,
        "recomendacion": recomendacion,
    }
