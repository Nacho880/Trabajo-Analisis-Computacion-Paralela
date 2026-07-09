"""
config.py
=========

Módulo de configuración global del proyecto.

Centraliza:
    - Lectura de la semilla de reproducibilidad (CPYD_SEED).
    - Definición de rutas de entrada/salida.
    - Parámetros globales usados por múltiples módulos (nivel de significancia,
      proporción train/test, número de núcleos a utilizar).

Diseño:
    Este módulo NO debe importar nada de `src/` para evitar dependencias
    circulares. Es la base de la que todos los demás módulos dependen.

Variables de entorno soportadas:
    CPYD_SEED (int, opcional): semilla global para todo proceso aleatorio
        (imputación estocástica, muestreo train/test, inicialización de
        K-means, etc.). Si no está definida, se usa 42 por defecto, tal
        como exige el enunciado del proyecto.
    CPYD_N_JOBS (int, opcional): número de procesos a usar en el paralelismo
        explícito de `src/parallel_stats.py`. Si no está definida, se usa
        `os.cpu_count()`.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path


# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# BASE_DIR apunta a la raíz del proyecto (directorio donde vive este archivo).
BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data"
OUTPUT_DIR: Path = BASE_DIR / "output"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
MODELS_DIR: Path = OUTPUT_DIR / "models"
LOGS_DIR: Path = OUTPUT_DIR / "logs"

# Se asegura la existencia de los directorios de salida en tiempo de import.
# Esto evita errores de "No such file or directory" al primer guardado.
for _directory in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, LOGS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def get_seed() -> int:
    """
    Obtiene la semilla global de reproducibilidad.

    Lee la variable de entorno ``CPYD_SEED``. Si no existe o no es un
    entero válido, retorna 42 (valor por defecto exigido por el enunciado).

    Returns:
        int: semilla a utilizar en todos los procesos aleatorios del
        proyecto (numpy, scikit-learn, muestreo, etc.).

    Complejidad:
        O(1) tiempo y espacio.
    """
    raw_value = os.environ.get("CPYD_SEED")
    if raw_value is None:
        return 42
    try:
        return int(raw_value)
    except ValueError:
        logging.getLogger(__name__).warning(
            "CPYD_SEED='%s' no es un entero válido. Se usará el valor por "
            "defecto (42).",
            raw_value,
        )
        return 42


SEED: int = get_seed()


# ---------------------------------------------------------------------------
# Paralelismo
# ---------------------------------------------------------------------------
def get_n_jobs() -> int:
    """
    Obtiene el número de procesos a utilizar en el paralelismo explícito.

    Lee la variable de entorno ``CPYD_N_JOBS``. Si no existe o es inválida,
    utiliza la totalidad de núcleos lógicos disponibles (``os.cpu_count()``).

    Returns:
        int: número de procesos worker a utilizar (mínimo 1).

    Complejidad:
        O(1) tiempo y espacio.
    """
    raw_value = os.environ.get("CPYD_N_JOBS")
    cpu_count = os.cpu_count() or 1
    if raw_value is None:
        return cpu_count
    try:
        n_jobs = int(raw_value)
        return max(1, n_jobs)
    except ValueError:
        logging.getLogger(__name__).warning(
            "CPYD_N_JOBS='%s' no es un entero válido. Se usarán %d núcleos.",
            raw_value,
            cpu_count,
        )
        return cpu_count


N_JOBS: int = get_n_jobs()


# ---------------------------------------------------------------------------
# Parámetros estadísticos y de modelado globales
# ---------------------------------------------------------------------------
ALPHA: float = 0.05                 # Nivel de significancia estándar para todos los tests.
TEST_SIZE: float = 0.30             # Proporción de test en el split train/test (70/30).

# Delimitador real del CSV de origen. El enunciado lo describe con comas en
# su tabla de referencia, pero el archivo real distribuido usa punto y coma
# (verificado empíricamente sobre el archivo real), un formato común en
# exportaciones de Excel en configuración regional latinoamericana/europea.
# Se centraliza aquí para que `data_loader.py` sea la única fuente de
# verdad y no queden "magic strings" de delimitador repetidos.
CSV_SEPARATOR: str = ";"
CSV_QUOTECHAR: str = '"'

RAW_EXPECTED_COLUMNS: tuple[str, ...] = (
    "FECHA",
    "CANAL",
    "SKU",
    "PRODUCTO",
    "UNIDADES",
    "PORCENTAJE DESCUENTO",
    "MONTO APLICADO",
    "BOLETA",
    "LOCAL",
    "CODIGO CLIENTE",
    "RUN CLIENTE",
    "NOMBRES",
    "APELLIDOS",
    "FECHA NACIMIENTO",
    "GENERO",
)

COLUMN_RENAME_MAP: dict[str, str] = {
    "FECHA": "FECHA",
    "CANAL": "CANAL",
    "SKU": "SKU",
    "PRODUCTO": "PRODUCTO",
    "UNIDADES": "UNIDADES",
    "PORCENTAJE DESCUENTO": "PORCENTAJE_DESCUENTO",
    "MONTO APLICADO": "MONTO_APLICADO",
    "BOLETA": "BOLETA",
    "LOCAL": "LOCAL",
    "CODIGO CLIENTE": "CODIGO_CLIENTE",
    "RUN CLIENTE": "RUN_CLIENTE",
    "NOMBRES": "NOMBRES",
    "APELLIDOS": "APELLIDOS",
    "FECHA NACIMIENTO": "FECHA_NACIMIENTO",
    "GENERO": "GENERO",
}

STANDARD_COLUMNS: tuple[str, ...] = tuple(COLUMN_RENAME_MAP.values())

EXPECTED_DTYPES: dict[str, str] = {
    # CANAL se lee como "string" (no "category") a propósito: ver
    # comentario en `data_loader.load_and_validate` sobre por qué
    # convertirla a categórica con `dtype="category"` directamente en
    # `dask.dataframe.read_csv` produce un orden de categorías NO
    # reproducible entre corridas.
    "CANAL": "string",
    "SKU": "Int64",
    "PRODUCTO": "string",
    "UNIDADES": "Int64",
    "PORCENTAJE_DESCUENTO": "float64",
    "MONTO_APLICADO": "float64",
    "BOLETA": "Int64",
    "LOCAL": "Int64",
    "CODIGO_CLIENTE": "string",
    "RUN_CLIENTE": "string",
    "NOMBRES": "string",
    "APELLIDOS": "string",
    "GENERO": "Int64",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class _ConsoleProgressFilter(logging.Filter):
    """
    Filtra qué registros llegan a la consola para mantenerla legible en
    ejecuciones largas sobre datasets grandes.

    Criterio: la consola solo debe mostrar información "de alto nivel"
    (arranque del pipeline, inicio/fin de cada etapa con su tiempo), el
    benchmark de paralelismo (resultado central del proyecto) y
    cualquier WARNING/ERROR real que requiera atención. El detalle fino
    de cada submódulo (conteos por columna, métricas intermedias,
    resultados de cada test estadístico, etc.) sigue registrándose
    íntegro en el archivo de log (FileHandler) para el informe técnico,
    pero no satura la terminal.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        # "src.parallel_stats" se permite además de "__main__"/"root":
        # los tiempos secuencial/paralelo y el speedup son un resultado
        # central del proyecto (paralelismo), no detalle interno de
        # implementación, y conviene verlos en vivo en consola.
        return record.name in ("__main__", "root", "src.parallel_stats")


def setup_logging(log_filename: str = "pipeline.log") -> logging.Logger:
    """
    Configura el logging global del proyecto (consola + archivo).

    Crea un logger raíz con dos handlers:
        1. StreamHandler -> salida por consola (nivel INFO).
        2. FileHandler   -> archivo en ``output/logs/<log_filename>``
           (nivel DEBUG, para trazabilidad completa).

    Idempotente: si se llama más de una vez, no duplica handlers.

    Args:
        log_filename: nombre del archivo de log dentro de ``output/logs/``.

    Returns:
        logging.Logger: logger raíz ya configurado.

    Complejidad:
        O(1) tiempo y espacio (configuración fija, independiente del volumen
        de datos que luego se registre).
    """
    root_logger = logging.getLogger()

    # Evita agregar handlers duplicados si setup_logging() se llama más de una vez
    # (por ejemplo, en tests que importan main varias veces).
    if getattr(root_logger, "_cpyd_configured", False):
        return root_logger

    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ConsoleProgressFilter())

    log_path = LOGS_DIR / log_filename
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger._cpyd_configured = True  # marca idempotencia

    root_logger.info(
        "Logging inicializado. Seed=%d | N_JOBS=%d | log_file=%s",
        SEED,
        N_JOBS,
        log_path,
    )
    return root_logger
