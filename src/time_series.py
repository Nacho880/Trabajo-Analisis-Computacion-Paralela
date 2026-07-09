"""
src/time_series.py
====================

Análisis de patrones temporales en las ventas.

Responsabilidad única: construir una serie de tiempo agregada
(ventas diarias), descomponerla en tendencia/estacionalidad/residuo, y
calcular la autocorrelación (ACF) y autocorrelación parcial (PACF)
para identificar dependencia temporal.

Uso de statsmodels vía import diferido:
    Igual que en `data_loader.py` con dask, `statsmodels` se importa
    solo dentro de las funciones que lo requieren estrictamente
    (`decompose_time_series`, `compute_acf_pacf`). Esto permite que la
    construcción de la serie agregada (`build_daily_sales_series`),
    que no depende de statsmodels, sea testeable de forma aislada en
    cualquier entorno.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io_utils import save_figure, save_json, save_table
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_and_trim_leading_gap(
    daily_series: pd.Series,
    window_days: int = 28,
    max_zero_fraction: float = 0.5,
) -> tuple[pd.Series, dict]:
    """
    Detecta y recorta un eventual período inicial de inactividad casi
    total al comienzo de la serie, ÚNICAMENTE de cara a la
    descomposición temporal y el ACF/PACF (ver alcance más abajo).

    Postura metodológica adoptada para este proyecto (justificada,
    no una limpieza de datos encubierta):
        Consultado el profesor sobre cómo tratar el hallazgo descrito
        abajo, la indicación fue que cada grupo tome una postura y la
        justifique explícitamente, siendo válida cualquier decisión
        bien fundamentada. La postura de este proyecto es: SÍ excluir
        este período de la descomposición estacional y el ACF/PACF
        (y solo de esos dos análisis), por los motivos estadísticos y
        de negocio documentados a continuación.

    Motivación (hallazgo empírico real de este proyecto):
        En el archivo de producción, la serie diaria construida por
        `build_daily_sales_series` comienza con un lote de 8
        transacciones (distintos clientes, boletas y productos, todas
        en un mismo local, LOCAL 371) concentradas en una sola fecha
        (2023-11-09), seguido de 149 días consecutivos en cero
        (ninguna venta en NINGÚN local de la cadena, incluyendo ese
        mismo local) antes de que empiece la actividad sostenida y
        multi-local (2024-04-07 en adelante). No es una fila aislada
        con error de tipeo -son transacciones con estructura interna
        consistente (boletas, descuentos y productos distintos)-, por
        lo que no se puede simplemente atribuir a un error de captura
        puntual. Dos hipótesis de negocio son plausibles: (a) un lote
        de pruebas/UAT del sistema cargado con la fecha real en que se
        ejecutó la prueba, meses antes del lanzamiento comercial, o
        (b) una apertura piloto real de un único local que luego
        quedó inactivo hasta el lanzamiento general de la cadena. El
        dataset no contiene información adicional (ej. un campo de
        tipo de transacción) para distinguir entre ambas con certeza;
        se documenta la ambigüedad explícitamente en el informe en
        lugar de asumir una causa.

        Independientemente de cuál hipótesis sea la correcta, incluir
        ese hueco de 149 días en la descomposición temporal y el
        ACF/PACF distorsiona la tendencia estimada (un salto artificial
        de ~0 a niveles reales) y contamina la estacionalidad semanal
        estimada, sin aportar señal útil sobre el comportamiento
        recurrente del negocio que esos dos análisis buscan capturar.

    Alcance de la exclusión (transparencia; nada se elimina en
    silencio, y nada se elimina del dataset en sí):
        Este recorte NO reinterpreta ni descarta esas 8 transacciones
        del resto del pipeline (siguen presentes en la limpieza, el
        EDA, las pruebas de hipótesis, la regresión y el clustering,
        que no dependen de continuidad temporal, y `ts_serie_diaria.csv`
        conserva la serie completa sin recortar). Solo se excluyen de
        `decompose_time_series` y `compute_acf_pacf`, que sí requieren
        una serie sin huecos artificiales para producir resultados
        estadísticamente válidos.

    Criterio de detección (genérico, no hardcodeado a una fecha
    específica, para que la postura sea reproducible ante cualquier
    corrida futura del pipeline con datos distintos):
        Se recorre la serie día a día. Para cada posición `i`, se
        evalúa la fracción de días en cero dentro de la ventana
        [i, i + window_days). El primer `i` cuya ventana tiene una
        fracción de ceros <= `max_zero_fraction` se considera el
        inicio de la actividad "sostenida", y la serie se recorta para
        comenzar ahí. Si nunca se encuentra tal ventana (o el hueco
        inicial no califica), no se recorta nada: la función es un
        no-op seguro sobre series sin este patrón.

    Args:
        daily_series: serie diaria completa, sin huecos (ver
            `build_daily_sales_series`).
        window_days: tamaño de la ventana de evaluación, en días
            (default 28 = 4 semanas, suficiente para promediar sobre
            el patrón semanal sin ser tan corta que un solo día activo
            aislado la haga pasar el umbral).
        max_zero_fraction: fracción máxima de días en cero tolerada
            dentro de la ventana para considerarla "actividad
            sostenida" (default 0.5 = menos de la mitad de los días de
            la ventana están en cero).

    Returns:
        tuple[pd.Series, dict]: la serie recortada (o la original si no
        se detectó ningún hueco inicial que califique) y un reporte
        con las claves "trimmed_days", "original_start", "new_start".

    Complejidad:
        O(n * window_days) tiempo en el peor caso (ventana deslizante
        no vectorizada, pero `n` es del orden de cientos de días, no
        de filas del dataset), O(n) espacio.
    """
    zero_mask = (daily_series <= 0).to_numpy()
    n = len(daily_series)
    start_idx = 0

    for i in range(n):
        window = zero_mask[i : i + window_days]
        if len(window) < window_days:
            # No queda una ventana completa por evaluar: no se encontró
            # un punto de inicio "sostenido" claro. No se recorta nada.
            start_idx = 0
            break
        if window.mean() <= max_zero_fraction:
            start_idx = i
            break

    trimmed = daily_series.iloc[start_idx:]
    report = {
        "trimmed_days": int(start_idx),
        "original_start": str(daily_series.index.min().date()),
        "new_start": str(trimmed.index.min().date()) if len(trimmed) else None,
        "window_days": window_days,
        "max_zero_fraction": max_zero_fraction,
    }

    if start_idx > 0:
        logger.warning(
            "Se detectó un período inicial de %d días con actividad "
            "prácticamente nula: del %s al %s, previo a un lote aislado de "
            "transacciones en un único local (ver docstring de "
            "detect_and_trim_leading_gap para las hipótesis de negocio "
            "consideradas y la postura metodológica adoptada). Se excluye "
            "de la descomposición temporal y el ACF/PACF (pero se conserva "
            "íntegro en ts_serie_diaria.csv y en el resto del pipeline). "
            "Análisis temporal recortado a partir de %s.",
            report["trimmed_days"], report["original_start"],
            (trimmed.index[0] - pd.Timedelta(days=1)).date(), report["new_start"],
        )

    return trimmed, report


def build_daily_sales_series(
    df: pd.DataFrame, date_column: str = "FECHA", value_column: str = "MONTO_APLICADO"
) -> pd.Series:
    """
    Agrega las transacciones a nivel diario, sumando `value_column` por
    día, y reindexa la serie a un rango de fechas completo y continuo
    (rellenando días sin ventas con 0), requisito indispensable para
    que la descomposición estacional y el ACF/PACF sean válidos (no
    toleran huecos temporales).

    Args:
        df: DataFrame con las transacciones limpias.
        date_column: columna de fecha/hora de la transacción.
        value_column: columna numérica a agregar (ej. MONTO_APLICADO).

    Returns:
        pd.Series: serie indexada por fecha diaria (freq="D"), sin huecos.

    Complejidad:
        O(n log n) tiempo (agrupación y ordenamiento por fecha), O(d)
        espacio, d = número de días en el rango cubierto por los datos.
    """
    daily = (
        df.set_index(date_column)[value_column]
        .resample("D")
        .sum()
    )
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.index.name = "FECHA"

    logger.info(
        "Serie diaria construida: %d días (%s a %s), total=%.2f.",
        len(daily), daily.index.min().date(), daily.index.max().date(), daily.sum(),
    )
    return daily


def decompose_time_series(series: pd.Series, period: int = 7) -> dict:
    """
    Descompone la serie temporal en tendencia, estacionalidad y
    residuo, usando un modelo aditivo (apropiado cuando la
    variabilidad estacional no crece proporcionalmente con el nivel de
    la serie, caso típico de ventas diarias agregadas de una farmacia).

    Args:
        series: serie de tiempo diaria sin huecos (ver
            `build_daily_sales_series`).
        period: periodicidad estacional en días (7 = patrón semanal,
            el más natural para ventas de retail).

    Returns:
        dict: {"trend": pd.Series, "seasonal": pd.Series,
        "residual": pd.Series, "observed": pd.Series}.

    Raises:
        ValueError: si la serie tiene menos de 2 ciclos completos
            (2 * period observaciones), insuficiente para descomponer.

    Complejidad:
        O(n) tiempo (medias móviles), O(n) espacio,
        n = longitud de la serie.
    """
    if len(series) < 2 * period:
        raise ValueError(
            f"La serie tiene {len(series)} observaciones; se requieren al "
            f"menos {2 * period} (2 ciclos de periodo={period}) para "
            "descomponer de forma confiable."
        )

    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series, model="additive", period=period)

    logger.info(
        "Descomposición temporal completada (modelo=additive, period=%d).", period
    )
    return {
        "observed": result.observed,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    }


def plot_decomposition(decomposition: dict) -> plt.Figure:
    """
    Genera una figura con 4 paneles (observado, tendencia,
    estacionalidad, residuo), tal como exige el enunciado.

    Args:
        decomposition: salida de `decompose_time_series`.

    Returns:
        matplotlib.figure.Figure: figura lista para guardar.

    Complejidad:
        O(n) tiempo y espacio, n = longitud de la serie.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    components = ["observed", "trend", "seasonal", "residual"]
    titles = ["Serie observada", "Tendencia", "Estacionalidad", "Residuo"]

    for ax, component, title in zip(axes, components, titles):
        ax.plot(decomposition[component].index, decomposition[component].values,
                color="#2E86AB")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def compute_acf_pacf(series: pd.Series, nlags: int = 30) -> dict:
    """
    Calcula la función de autocorrelación (ACF) y autocorrelación
    parcial (PACF) de la serie, para identificar dependencia temporal
    de corto y largo plazo.

    Args:
        series: serie de tiempo diaria.
        nlags: número máximo de rezagos a calcular.

    Returns:
        dict: {"acf": np.ndarray, "pacf": np.ndarray, "nlags": int}.

    Complejidad:
        O(n * nlags) tiempo para ACF, O(n * nlags^2) para PACF
        (método de Yule-Walker por defecto de statsmodels),
        O(nlags) espacio.
    """
    from statsmodels.tsa.stattools import acf, pacf

    max_lags = min(nlags, len(series) // 2 - 1)
    acf_values = acf(series, nlags=max_lags, fft=True)
    pacf_values = pacf(series, nlags=max_lags)

    logger.info("ACF/PACF calculados con %d rezagos.", max_lags)
    return {"acf": acf_values, "pacf": pacf_values, "nlags": max_lags}


def plot_acf_pacf(acf_pacf_result: dict) -> plt.Figure:
    """
    Genera un gráfico de barras para ACF y PACF, con bandas de
    significancia aproximadas al 95% (±1.96/sqrt(n)).

    Args:
        acf_pacf_result: salida de `compute_acf_pacf`.

    Returns:
        matplotlib.figure.Figure: figura lista para guardar.

    Complejidad:
        O(nlags) tiempo y espacio.
    """
    acf_values = acf_pacf_result["acf"]
    pacf_values = acf_pacf_result["pacf"]
    n = len(acf_values)
    confidence_band = 1.96 / np.sqrt(n)

    fig, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(10, 6))

    for ax, values, title in (
        (ax_acf, acf_values, "Autocorrelación (ACF)"),
        (ax_pacf, pacf_values, "Autocorrelación Parcial (PACF)"),
    ):
        ax.bar(range(len(values)), values, color="#2E86AB", width=0.4)
        ax.axhline(confidence_band, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(-confidence_band, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Rezago (días)")

    fig.tight_layout()
    return fig


def run_time_series_analysis(
    df: pd.DataFrame,
    date_column: str = "FECHA",
    value_column: str = "MONTO_APLICADO",
    period: int = 7,
) -> dict:
    """
    Orquesta el análisis temporal completo: construcción de la serie
    diaria, descomposición, ACF/PACF, y exportación automática de
    tablas y figuras.

    Args:
        df: DataFrame limpio y transformado.
        date_column: columna de fecha de la transacción.
        value_column: columna numérica a analizar.
        period: periodicidad estacional (días).

    Returns:
        dict: {"daily_series": pd.Series, "decomposition": dict,
        "acf_pacf": dict} o, si la serie es demasiado corta para
        descomponer, un dict con "decomposition": None y una advertencia
        registrada en el log.

    Complejidad:
        O(n log n + d) tiempo, n = filas del DataFrame, d = días de la
        serie agregada; O(n + d) espacio.
    """
    logger.info("=== Iniciando análisis de patrones temporales ===")

    daily_series = build_daily_sales_series(df, date_column, value_column)
    save_table(daily_series.reset_index(), "ts_serie_diaria")

    # La serie completa (con el eventual hueco inicial) se conserva en el
    # CSV exportado arriba. Para la descomposición y el ACF/PACF -que sí
    # requieren continuidad temporal para producir resultados válidos- se
    # usa la versión recortada, según la postura metodológica documentada
    # en `detect_and_trim_leading_gap`.
    trimmed_series, gap_report = detect_and_trim_leading_gap(daily_series)
    save_json(gap_report, "ts_periodo_inicial_excluido")

    result = {
        "daily_series": daily_series,
        "decomposition": None,
        "acf_pacf": None,
        "leading_gap_report": gap_report,
    }

    try:
        decomposition = decompose_time_series(trimmed_series, period=period)
        result["decomposition"] = decomposition
        save_figure(plot_decomposition(decomposition), "ts_descomposicion")
    except ValueError as exc:
        logger.warning(
            "No se pudo descomponer la serie temporal: %s. Se omite esta "
            "sección del análisis (dataset con rango de fechas insuficiente).",
            exc,
        )

    acf_pacf = compute_acf_pacf(trimmed_series)
    result["acf_pacf"] = acf_pacf
    save_figure(plot_acf_pacf(acf_pacf), "ts_acf_pacf")

    logger.info("=== Análisis de patrones temporales finalizado ===")
    return result
