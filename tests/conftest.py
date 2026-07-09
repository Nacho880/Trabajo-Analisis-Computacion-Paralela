"""
tests/conftest.py
==================

Fixtures compartidas por toda la suite de tests.

Se genera un dataset sintético pequeño (no el CSV real de producción)
para que los tests sean rápidos, deterministas y no dependan de la
descarga del archivo real desde Google Drive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Aseguramos que la semilla sea determinista en TODOS los tests,
# independientemente de si el entorno definió CPYD_SEED.
SEED = 42


@pytest.fixture(scope="session")
def synthetic_raw_dataframe() -> pd.DataFrame:
    """
    Construye un DataFrame sintético de 200 filas con la MISMA
    estructura (nombres de columna crudos, con espacios/tildes) que el
    CSV real descrito en el enunciado del proyecto.

    Incluye deliberadamente:
        - Valores nulos en PORCENTAJE DESCUENTO y FECHA NACIMIENTO.
        - Un outlier extremo en MONTO APLICADO.
        - Dos canales (POS, WEB, APP) y tres locales para permitir
          pruebas de Chi-cuadrado, ANOVA y t-test.

    Returns:
        pd.DataFrame: dataset sintético con columnas crudas (sin renombrar).
    """
    rng = np.random.default_rng(SEED)
    n = 200

    canales = rng.choice(["POS", "WEB", "APP"], size=n, p=[0.5, 0.25, 0.25])
    locales = rng.choice([1001, 1002, 1003], size=n)
    unidades = rng.integers(1, 5, size=n)
    descuento = rng.uniform(0.0, 0.3, size=n)
    # Introducimos nulos deliberados en el descuento (~10%).
    null_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    descuento[null_idx] = np.nan

    precio_base = rng.uniform(1000, 20000, size=n)
    monto = precio_base * unidades * (1 - np.nan_to_num(descuento))
    # Outlier extremo intencional.
    monto[0] = monto.max() * 50

    fechas = pd.date_range("2026-01-01", periods=n, freq="h")

    fecha_nac = pd.to_datetime(
        rng.integers(
            pd.Timestamp("1950-01-01").value,
            pd.Timestamp("2005-01-01").value,
            size=n,
        )
    )
    fecha_nac = fecha_nac.to_series().reset_index(drop=True)
    fecha_nac_null_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    fecha_nac = fecha_nac.astype("datetime64[ns]")
    fecha_nac.iloc[fecha_nac_null_idx] = pd.NaT

    codigo_cliente = [f"CLI-{i % 50:04d}" for i in range(n)]  # 50 clientes distintos

    df = pd.DataFrame(
        {
            "FECHA": fechas,
            "CANAL": canales,
            "SKU": rng.integers(1000, 2000, size=n),
            "PRODUCTO": [f"PRODUCTO_{i % 20}" for i in range(n)],
            "UNIDADES": unidades,
            "PORCENTAJE DESCUENTO": descuento,
            "MONTO APLICADO": monto,
            "BOLETA": np.arange(100000, 100000 + n),
            "LOCAL": locales,
            "CODIGO CLIENTE": codigo_cliente,
            "RUN CLIENTE": [f"{10_000_000 + i}-{i % 10}" for i in range(n)],
            "NOMBRES": [f"NOMBRE_{i % 50}" for i in range(n)],
            "APELLIDOS": [f"APELLIDO_{i % 50}" for i in range(n)],
            "FECHA NACIMIENTO": fecha_nac,
            "GENERO": rng.integers(1, 3, size=n),
        }
    )
    return df


@pytest.fixture()
def synthetic_csv_path(tmp_path: Path, synthetic_raw_dataframe: pd.DataFrame) -> Path:
    """
    Escribe `synthetic_raw_dataframe` a un archivo CSV temporal
    delimitado por punto y coma (`;`), replicando el formato real del
    archivo `ventas_completas.csv` (verificado empíricamente sobre el
    archivo real de la cátedra), y retorna su ruta.

    Args:
        tmp_path: directorio temporal provisto por pytest (aislado por test).
        synthetic_raw_dataframe: fixture con los datos sintéticos.

    Returns:
        Path: ruta al CSV temporal generado.
    """
    from config import CSV_SEPARATOR

    csv_path = tmp_path / "ventas_completas_test.csv"
    synthetic_raw_dataframe.to_csv(csv_path, index=False, sep=CSV_SEPARATOR)
    return csv_path


@pytest.fixture()
def empty_csv_path(tmp_path: Path) -> Path:
    """Crea un archivo CSV vacío (0 bytes) para probar el manejo de errores."""
    path = tmp_path / "vacio.csv"
    path.touch()
    return path


@pytest.fixture()
def corrupt_csv_path(tmp_path: Path) -> Path:
    """Crea un archivo con contenido binario no parseable como CSV."""
    path = tmp_path / "corrupto.csv"
    path.write_bytes(bytes([0xFF, 0xFE, 0x00, 0x01, 0x02] * 20))
    return path
