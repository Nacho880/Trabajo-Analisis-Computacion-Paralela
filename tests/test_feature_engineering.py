"""
tests/test_feature_engineering.py
====================================

Pruebas unitarias para `src/feature_engineering.py`.

Como el módulo ahora opera sobre `dask.dataframe.DataFrame` de punta a
punta (ver docstring de `src/feature_engineering.py`), cada test
construye un `pd.DataFrame` pequeño y lo envuelve con
`dd.from_pandas(..., npartitions=2)` antes de pasarlo a las funciones
bajo prueba, materializando (`.compute()`) el resultado perezoso antes
de cada aserción.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.feature_engineering import (
    add_edad,
    add_frecuencia_compra,
    add_monto_por_unidad,
    engineer_features,
    standardize_columns,
)


def _to_dask(df: pd.DataFrame, npartitions: int = 2) -> dd.DataFrame:
    """Envuelve un DataFrame pequeño de Pandas como Dask, para los tests."""
    return dd.from_pandas(df, npartitions=npartitions)


def test_add_monto_por_unidad_basic_division() -> None:
    """MONTO_POR_UNIDAD debe ser exactamente MONTO_APLICADO / UNIDADES."""
    df = pd.DataFrame({"MONTO_APLICADO": [1000.0, 2000.0], "UNIDADES": [2, 4]})
    result = add_monto_por_unidad(_to_dask(df)).compute()
    assert list(result["MONTO_POR_UNIDAD"]) == [500.0, 500.0]


def test_add_monto_por_unidad_handles_zero_units() -> None:
    """UNIDADES=0 debe producir NaN, nunca inf ni una excepción."""
    df = pd.DataFrame({"MONTO_APLICADO": [1000.0, 500.0], "UNIDADES": [0, 5]})
    result = add_monto_por_unidad(_to_dask(df)).compute()
    assert np.isnan(result["MONTO_POR_UNIDAD"].iloc[0])
    assert not np.isinf(result["MONTO_POR_UNIDAD"].iloc[0])
    assert result["MONTO_POR_UNIDAD"].iloc[1] == 100.0


def test_add_edad_computes_years_between_dates() -> None:
    """EDAD debe calcularse respecto a FECHA de transacción, no a hoy."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-01-01"]),
            "FECHA_NACIMIENTO": pd.to_datetime(["1996-01-01"]),
        }
    )
    result = add_edad(_to_dask(df, npartitions=1)).compute()
    assert abs(result["EDAD"].iloc[0] - 30.0) < 0.1


def test_add_edad_flags_invalid_ages_as_nan() -> None:
    """Edades fuera de [0, 110] deben convertirse a NaN, no eliminarse la fila."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "FECHA_NACIMIENTO": pd.to_datetime(["2027-01-01", "1900-01-01"]),
        }
    )
    result = add_edad(_to_dask(df)).compute()
    assert len(result) == 2  # ninguna fila eliminada
    assert result["EDAD"].isna().sum() == 2  # ambas inválidas (futura y >110 años)


def test_add_edad_accounts_for_birthday_not_yet_occurred_this_year() -> None:
    """Si el cumpleaños aún no ocurre este año, la edad debe ser un año menor."""
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-06-15", "2026-06-15"]),
            # Cumpleaños en diciembre (aún no ocurre) vs. en enero (ya ocurrió).
            "FECHA_NACIMIENTO": pd.to_datetime(["1996-12-25", "1996-01-01"]),
        }
    )
    result = add_edad(_to_dask(df)).compute()
    assert result["EDAD"].iloc[0] == 29.0  # aún no cumple años este 2026
    assert result["EDAD"].iloc[1] == 30.0  # ya cumplió años este 2026


def test_add_edad_does_not_overflow_with_extreme_birth_dates() -> None:
    """
    Regresión: una FECHA_NACIMIENTO extremadamente antigua no debe lanzar
    OverflowError (bug real detectado en el archivo de datos de producción:
    restar datetimes directamente construye un Timedelta limitado a ~292
    años en su representación interna de nanosegundos int64; el cálculo
    por componentes año/mes/día es inmune a esto).
    """
    df = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2026-06-15"]),
            "FECHA_NACIMIENTO": pd.to_datetime(["1700-01-01"]),
        }
    )
    result = add_edad(_to_dask(df, npartitions=1)).compute()  # no debe lanzar excepción
    assert len(result) == 1
    assert result["EDAD"].isna().iloc[0]  # 326 años > 110 -> filtrada como inválida


def test_add_frecuencia_compra_counts_unique_boletas() -> None:
    """FRECUENCIA_COMPRA debe contar boletas ÚNICAS, no filas."""
    df = pd.DataFrame(
        {
            "CODIGO_CLIENTE": ["C1", "C1", "C1", "C2"],
            "BOLETA": [100, 100, 101, 200],  # C1 tiene 2 boletas únicas (100 repetida)
        }
    )
    result = add_frecuencia_compra(_to_dask(df)).compute()
    c1_freq = result.loc[result["CODIGO_CLIENTE"] == "C1", "FRECUENCIA_COMPRA"].iloc[0]
    c2_freq = result.loc[result["CODIGO_CLIENTE"] == "C2", "FRECUENCIA_COMPRA"].iloc[0]
    assert c1_freq == 2
    assert c2_freq == 1


def test_standardize_columns_produces_zero_mean_unit_std() -> None:
    """Una columna estandarizada debe tener media ~0 y std ~1."""
    df = pd.DataFrame({"X": [10.0, 20.0, 30.0, 40.0, 50.0]})
    ddf_result, params = standardize_columns(_to_dask(df), ["X"])
    result = ddf_result.compute()

    assert abs(result["X_STD"].mean()) < 1e-9
    assert abs(result["X_STD"].std(ddof=1) - 1.0) < 1e-9
    assert params.means["X"] == 30.0


def test_standardize_columns_handles_constant_column() -> None:
    """
    Una columna constante (std=0) no debe generar división por cero, y
    NO debe crear una columna `_STD` engañosa (que daría la falsa
    impresión de haber sido estandarizada). Los parámetros (media, std=0)
    igualmente quedan documentados en `ScalingParameters`.
    """
    df = pd.DataFrame({"X": [5.0, 5.0, 5.0]})
    ddf_result, params = standardize_columns(_to_dask(df), ["X"])
    result = ddf_result.compute()

    assert "X_STD" not in result.columns
    assert params.stds["X"] == 0.0
    assert params.means["X"] == 5.0


def test_engineer_features_end_to_end(synthetic_raw_dataframe: pd.DataFrame) -> None:
    """El pipeline completo debe ejecutarse sin excepciones y agregar todas las columnas."""
    from config import COLUMN_RENAME_MAP
    from src.data_cleaning import clean_dataset

    df = synthetic_raw_dataframe.rename(columns=COLUMN_RENAME_MAP)
    ddf_clean, _ = clean_dataset(_to_dask(df))
    ddf_final, scaling_params = engineer_features(ddf_clean)
    df_final = ddf_final.compute()

    for expected_col in ("MONTO_POR_UNIDAD", "EDAD", "FRECUENCIA_COMPRA",
                          "MONTO_APLICADO_STD", "UNIDADES_STD",
                          "PORCENTAJE_DESCUENTO_STD"):
        assert expected_col in df_final.columns

    assert len(df_final) == len(df)  # sin eliminación de filas
    assert "MONTO_APLICADO" in scaling_params.means
