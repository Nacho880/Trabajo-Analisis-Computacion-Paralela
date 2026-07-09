"""
src/utils/io_utils.py
======================

Funciones genéricas de entrada/salida para exportar resultados.

Centralizar el guardado aquí evita duplicar `plt.savefig(...)` o
`df.to_csv(...)` con manejo de errores repetido en cada módulo de
análisis (DRY). Todo módulo que produzca una figura, tabla o modelo
debe usar estas funciones para exportarlo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.figure
import pandas as pd

from config import FIGURES_DIR, MODELS_DIR, TABLES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_figure(fig: matplotlib.figure.Figure, filename: str) -> Path:
    """
    Guarda una figura de Matplotlib en `output/figures/`.

    Args:
        fig: figura de Matplotlib a guardar.
        filename: nombre de archivo (ej. "histograma_montos.png").
            Si no incluye extensión, se asume `.png`.

    Returns:
        Path: ruta absoluta del archivo guardado.

    Raises:
        OSError: si no se puede escribir en el directorio de salida
            (ej. problemas de permisos), tras registrarse en el log.

    Complejidad:
        O(1) en términos del volumen de datos ya graficado (el costo real
        de renderizado ya fue pagado al construir `fig`).
    """
    if not filename.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        filename = f"{filename}.png"

    output_path = FIGURES_DIR / filename
    try:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Figura guardada en: %s", output_path)
    except OSError as exc:
        logger.error("No se pudo guardar la figura '%s': %s", output_path, exc)
        raise
    finally:
        # Libera memoria de matplotlib incluso si el guardado falla, para no
        # acumular figuras abiertas en pipelines con muchas visualizaciones.
        import matplotlib.pyplot as plt
        plt.close(fig)

    return output_path


def save_table(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    """
    Guarda un DataFrame de Pandas como CSV en `output/tables/`.

    Args:
        df: DataFrame a exportar.
        filename: nombre de archivo (ej. "estadisticos_descriptivos.csv").
            Si no incluye extensión, se asume `.csv`.
        index: si se debe escribir el índice del DataFrame como columna.
            Por defecto False, salvo que el índice porte información
            relevante (ej. nombre de variable en una tabla resumen).

    Returns:
        Path: ruta absoluta del archivo guardado.

    Raises:
        OSError: si no se puede escribir el archivo, tras registrarse en el log.

    Complejidad:
        O(n * m) tiempo, donde n = filas, m = columnas del DataFrame
        (costo inherente de serializar a CSV); O(1) espacio adicional.
    """
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"

    output_path = TABLES_DIR / filename
    try:
        df.to_csv(output_path, index=index, encoding="utf-8")
        logger.info(
            "Tabla guardada en: %s (%d filas, %d columnas)",
            output_path, df.shape[0], df.shape[1],
        )
    except OSError as exc:
        logger.error("No se pudo guardar la tabla '%s': %s", output_path, exc)
        raise

    return output_path


def save_json(data: dict[str, Any], filename: str) -> Path:
    """
    Guarda un diccionario como JSON en `output/tables/`.

    Útil para resultados de tests de hipótesis, métricas de validación
    y reportes de rendimiento que no calzan naturalmente en una tabla
    tabular plana.

    Args:
        data: diccionario serializable a JSON (se aplica `default=str`
            para tolerar tipos no nativos como `numpy.float64`).
        filename: nombre de archivo (ej. "resultados_hipotesis.json").
            Si no incluye extensión, se asume `.json`.

    Returns:
        Path: ruta absoluta del archivo guardado.

    Raises:
        OSError: si no se puede escribir el archivo.

    Complejidad:
        O(n) tiempo y espacio, donde n = tamaño total de la estructura `data`.
    """
    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    output_path = TABLES_DIR / filename
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("JSON guardado en: %s", output_path)
    except OSError as exc:
        logger.error("No se pudo guardar el JSON '%s': %s", output_path, exc)
        raise

    return output_path


def save_model(model: Any, filename: str) -> Path:
    """
    Serializa un modelo (scikit-learn u otro objeto Python) con joblib
    en `output/models/`.

    Args:
        model: objeto del modelo entrenado a persistir.
        filename: nombre de archivo (ej. "regresion_monto.joblib").
            Si no incluye extensión, se asume `.joblib`.

    Returns:
        Path: ruta absoluta del archivo guardado.

    Raises:
        OSError: si no se puede escribir el archivo.

    Complejidad:
        O(s) tiempo y espacio, donde s = tamaño en memoria del objeto modelo.
    """
    if not filename.endswith(".joblib"):
        filename = f"{filename}.joblib"

    output_path = MODELS_DIR / filename
    try:
        joblib.dump(model, output_path)
        logger.info("Modelo guardado en: %s", output_path)
    except OSError as exc:
        logger.error("No se pudo guardar el modelo '%s': %s", output_path, exc)
        raise

    return output_path
