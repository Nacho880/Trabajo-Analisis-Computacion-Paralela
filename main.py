#!/usr/bin/env python3
"""
main.py
========

Punto de entrada del proyecto "Análisis Estadístico de Datos de
Ventas: Inferencia y Modelado" (Cruz Morada).

Orquesta el pipeline completo:
    1. Carga y validación del CSV (Dask, lazy, chunking).
    1b. Persistencia en memoria (Dask, eager: `.persist()`).
    2. Limpieza (Dask, lazy: valores faltantes, outliers).
    3. Ingeniería de variables (Dask, lazy).
    3b. Materialización única a Pandas (`.compute()`).
    4. Estadísticos por partición con paralelismo real (benchmark
       secuencial vs paralelo, Pandas + ThreadPoolExecutor propio).
    5. Análisis Exploratorio Estadístico (EDA).
    6. Pruebas de hipótesis (5 hipótesis).
    7. Análisis de patrones temporales.
    8. Modelado: regresión (Opción A) y clustering (Opción B).
    9. Validación consolidada y discusión de extrapolabilidad.
    10. Exportación de un reporte de rendimiento global.

Uso:
    python main.py --input data/ventas_completas.csv

Variables de entorno relevantes:
    CPYD_SEED: semilla de reproducibilidad (default 42).
    CPYD_N_JOBS: número de procesos para el paralelismo explícito
        (default: todos los núcleos disponibles).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import psutil
import pandas as pd

from config import SEED, N_JOBS, setup_logging
from src.data_cleaning import clean_dataset
from src.data_loader import load_and_validate
from src.eda import run_descriptive_analysis
from src.feature_engineering import engineer_features
from src.hypothesis_tests import anova_test, run_all_hypothesis_tests
from src.modeling_clustering import run_clustering_pipeline
from src.modeling_regression import fit_and_diagnose_regression
from src.parallel_stats import (
    benchmark_sequential_vs_parallel,
    plot_scalability_curve,
    run_scalability_study,
    stats_to_dataframe,
)
from src.time_series import run_time_series_analysis
from src.utils.io_utils import save_figure, save_json, save_model, save_table
from src.utils.logger import get_logger
from src.utils.validators import DataValidationError
from src.validation import build_validation_summary, discuss_extrapolability

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos del programa.

    Returns:
        argparse.Namespace: con el atributo `input` (ruta al CSV).

    Complejidad:
        O(1) tiempo y espacio.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline de análisis estadístico y modelado de ventas para "
            "Cruz Morada (Computación Paralela y Distribuida)."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/ventas_completas.csv",
        help="Ruta al archivo CSV de ventas (default: data/ventas_completas.csv).",
    )
    return parser.parse_args()


def _log_stage(stage_name: str):
    """
    Decorador simple que registra inicio, fin y duración de una etapa
    del pipeline, sin necesidad de repetir código de timing en cada
    función de `main`.

    Args:
        stage_name: nombre descriptivo de la etapa (para el log).

    Returns:
        Callable: decorador que envuelve la función de la etapa.

    Complejidad:
        O(1) overhead adicional por llamada (además del costo propio
        de la función decorada).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(">>> INICIO: %s", stage_name)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(">>> FIN: %s (%.2f segundos)", stage_name, elapsed)
            return result, elapsed
        return wrapper
    return decorator


def run_pipeline(input_path: str) -> int:
    """
    Ejecuta el pipeline completo de extremo a extremo.

    Args:
        input_path: ruta al archivo CSV de entrada.

    Returns:
        int: código de salida del proceso (0 = éxito, 1 = error
        controlado de validación de datos, 2 = error inesperado).

    Complejidad:
        Dominada por el módulo más costoso del pipeline
        (`parallel_stats`, O(n log n)); overhead de orquestación O(1).
    """
    stage_times: dict[str, float] = {}
    process = psutil.Process()
    total_start = time.perf_counter()

    logger.info(
        "Iniciando pipeline. SEED=%d, N_JOBS=%d, núcleos lógicos disponibles=%d.",
        SEED, N_JOBS, psutil.cpu_count(logical=True),
    )

    try:
        # --- 1. Carga (Dask, lazy) ---
        (ddf, elapsed) = _log_stage("Carga de datos (Dask, lazy)")(load_and_validate)(input_path)
        stage_times["carga"] = elapsed

        # --- 1b. Persistencia en memoria (Dask, eager) ---
        # `.persist()` dispara la lectura/parseo del CSV completo UNA sola
        # vez y guarda las particiones ya materializadas en memoria, pero
        # SIN abandonar la API de Dask (el resultado sigue siendo un
        # dask.dataframe.DataFrame). Esto es lo que hace viable encadenar
        # `clean_dataset` y `engineer_features` -que internamente hacen
        # varios `.compute()` de valores pequeños para el log/reporte- sin
        # que cada uno de esos `.compute()` dispare una relectura completa
        # del archivo desde cero (que fue la causa de que una versión
        # anterior, sin este `.persist()`, tomara ~15 minutos en vez de
        # los ~3 minutos de este pipeline).
        (ddf, elapsed) = _log_stage("Persistencia en memoria (Dask, eager)")(
            lambda: ddf.persist()
        )()
        stage_times["persistencia"] = elapsed

        # --- 2. Limpieza (Dask, lazy sobre datos ya persistidos) ---
        (cleaning_output, elapsed) = _log_stage("Limpieza de datos (Dask, lazy)")(
            clean_dataset
        )(ddf)
        stage_times["limpieza"] = elapsed
        ddf_clean, cleaning_report = cleaning_output
        save_json(cleaning_report.to_dict(), "reporte_limpieza")

        # --- 3. Ingeniería de variables (Dask, lazy) ---
        (feature_output, elapsed) = _log_stage("Ingeniería de variables (Dask, lazy)")(
            engineer_features
        )(ddf_clean)
        stage_times["feature_engineering"] = elapsed
        ddf_final, scaling_params = feature_output
        save_json(scaling_params.to_dict(), "parametros_estandarizacion")

        # --- 3b. Materialización única (.compute()): única frontera
        # Dask -> Pandas de todo el pipeline. A partir de aquí, el
        # algoritmo paralelo propio (ThreadPoolExecutor sobre particiones
        # lógicas), el EDA, las pruebas de hipótesis, el análisis de
        # series de tiempo y el modelado (regresión/clustering) operan
        # sobre Pandas puro, ya que dependen de scipy/statsmodels/
        # scikit-learn/matplotlib, que no entienden objetos Dask.
        (df_final, elapsed) = _log_stage("Materialización en memoria (.compute())")(
            lambda: ddf_final.compute()
        )()
        stage_times["materializacion"] = elapsed
        logger.info(
            "Dataset materializado: %d filas, %d columnas, %.2f MB en memoria.",
            df_final.shape[0], df_final.shape[1],
            df_final.memory_usage(deep=True).sum() / (1024 ** 2),
        )

        # --- 4. Estadísticos paralelos + benchmark ---
        (parallel_output, elapsed) = _log_stage(
            "Estadísticos por partición (paralelo, benchmark vs secuencial)"
        )(benchmark_sequential_vs_parallel)(df_final)
        stage_times["parallel_stats"] = elapsed
        partition_stats, benchmark = parallel_output
        save_table(stats_to_dataframe(partition_stats), "estadisticos_por_local")
        save_json(benchmark.to_dict(), "benchmark_rendimiento_paralelo")

        # --- 4b. Estudio de escalabilidad (speedup vs número de hilos) ---
        (scalability_table, elapsed) = _log_stage(
            "Estudio de escalabilidad (speedup vs n_jobs)"
        )(run_scalability_study)(df_final)
        stage_times["scalability_study"] = elapsed
        save_table(scalability_table, "escalabilidad_speedup_vs_n_jobs")
        save_figure(
            plot_scalability_curve(scalability_table), "curva_escalabilidad_speedup"
        )

        # --- 5. EDA ---
        (eda_results, elapsed) = _log_stage("Análisis Exploratorio Estadístico")(
            run_descriptive_analysis
        )(
            df_final,
            numeric_columns=["MONTO_APLICADO", "UNIDADES", "PORCENTAJE_DESCUENTO", "EDAD"],
            category_column="CANAL",
            boxplot_value_column="MONTO_APLICADO",
            correlation_columns=["UNIDADES", "MONTO_APLICADO", "PORCENTAJE_DESCUENTO"],
            seed=SEED,
        )
        stage_times["eda"] = elapsed

        # --- 5b. Análisis de asociación: ANOVA (exigido por el enunciado en
        # la sección de EDA: "¿Hay diferencias significativas en el MONTO
        # APLICADO entre CANAL o LOCAL?"). Se ejecuta como paso separado de
        # las 5 hipótesis del punto 6, ya que el enunciado lo ubica como
        # parte del Análisis Exploratorio, no como una hipótesis de negocio.
        (anova_results, elapsed) = _log_stage(
            "Análisis de asociación (ANOVA: MONTO_APLICADO por CANAL/LOCAL)"
        )(
            lambda: [
                anova_test(df_final, "MONTO_APLICADO", "CANAL"),
                anova_test(df_final, "MONTO_APLICADO", "LOCAL"),
            ]
        )()
        stage_times["anova_asociacion"] = elapsed
        save_table(pd.DataFrame(anova_results), "eda_anova_monto_por_canal_local")

        # --- 6. Pruebas de hipótesis ---
        (hypothesis_results, elapsed) = _log_stage("Pruebas de hipótesis")(
            run_all_hypothesis_tests
        )(df_final, seed=SEED)
        stage_times["hipotesis"] = elapsed
        save_table(pd.DataFrame(hypothesis_results), "resultados_hipotesis")

        # --- 7. Patrones temporales ---
        (_, elapsed) = _log_stage("Análisis de patrones temporales")(
            run_time_series_analysis
        )(df_final)
        stage_times["series_tiempo"] = elapsed

        # --- 8a. Modelado: Regresión (Opción A) ---
        (regression_output, elapsed) = _log_stage("Modelado: Regresión (Opción A)")(
            fit_and_diagnose_regression
        )(df_final, seed=SEED)
        stage_times["regresion"] = elapsed
        regression_model, regression_diagnostics, regression_metrics = regression_output
        save_model(regression_model, "modelo_regresion")
        save_json(regression_diagnostics.to_dict(), "diagnostico_regresion")
        save_json(regression_metrics, "metricas_regresion")
        save_table(regression_diagnostics.vif_table, "vif_regresion")

        # --- 8b. Modelado: Clustering (Opción B) ---
        (clustering_result, elapsed) = _log_stage("Modelado: Clustering (Opción B)")(
            run_clustering_pipeline
        )(
            df_final,
            feature_columns=["MONTO_APLICADO_sum", "UNIDADES_mean"],
        )
        stage_times["clustering"] = elapsed
        save_table(clustering_result.elbow_table, "clustering_elbow")
        save_table(clustering_result.silhouette_table, "clustering_silhouette")
        save_table(
            clustering_result.cluster_profile.reset_index(), "clustering_perfil_clusters"
        )
        save_json(clustering_result.to_summary_dict(), "clustering_resumen")

        # --- 9. Validación consolidada ---
        validation_summary = build_validation_summary(
            regression_metrics, regression_diagnostics, clustering_result
        )
        save_table(validation_summary, "resumen_validacion_modelos")

        extrapolability = discuss_extrapolability(
            regression_metrics, regression_diagnostics, cleaning_report
        )
        save_json(extrapolability, "discusion_extrapolabilidad")

        # --- 10. Reporte de rendimiento global ---
        total_elapsed = time.perf_counter() - total_start
        performance_report = {
            "tiempos_por_etapa_segundos": stage_times,
            "tiempo_total_segundos": total_elapsed,
            "memoria_rss_mb": process.memory_info().rss / (1024 ** 2),
            "cpu_percent": process.cpu_percent(interval=0.5),
            "nucleos_logicos": psutil.cpu_count(logical=True),
            "n_jobs_usados": N_JOBS,
            "seed": SEED,
            "speedup_paralelismo": benchmark.speedup,
            "eficiencia_paralelismo": benchmark.efficiency,
        }
        save_json(performance_report, "performance_report")

        logger.info(
            "=== PIPELINE FINALIZADO EXITOSAMENTE en %.2f segundos ===", total_elapsed
        )
        return 0

    except DataValidationError as exc:
        logger.error("Error de validación de datos: %s", exc)
        return 1
    except Exception:  # noqa: BLE001 - se captura para nunca terminar abruptamente
        logger.error("Error inesperado en el pipeline:\n%s", traceback.format_exc())
        return 2


def main() -> None:
    """
    Función de entrada del script. Configura logging, parsea
    argumentos, ejecuta el pipeline y establece el código de salida
    del proceso según el resultado.
    """
    setup_logging()
    args = parse_arguments()

    input_path = Path(args.input)
    exit_code = run_pipeline(str(input_path))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
