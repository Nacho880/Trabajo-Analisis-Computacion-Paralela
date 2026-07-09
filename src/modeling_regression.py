"""
src/modeling_regression.py
=============================

Opción A del enunciado: modelado de MONTO_APLICADO en función de
CANAL, LOCAL, UNIDADES y PORCENTAJE_DESCUENTO, con diagnóstico
completo de supuestos.

Decisión de diseño (VIF y homocedasticidad sin `statsmodels`):
    El cálculo de VIF (Variance Inflation Factor) y el test de
    Breusch-Pagan se implementan aquí manualmente con `scikit-learn` y
    `scipy`, en lugar de depender de `statsmodels`, por dos razones:
        1. Ambos se derivan de fórmulas estadísticas simples y bien
           conocidas (ver docstrings de `compute_vif` y
           `breusch_pagan_test`), por lo que reimplementarlos no
           introduce riesgo de error metodológico.
        2. Reduce el acoplamiento del módulo de modelado a una
           dependencia adicional pesada, manteniendo el principio KISS:
           `statsmodels` ya se usa en `time_series.py` porque ahí SÍ es
           indispensable (descomposición STL, ACF/PACF con bandas de
           confianza), pero aquí no aporta nada que scikit-learn/scipy
           no puedan calcular directamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import SEED, TEST_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegressionDiagnostics:
    """
    Resultado del diagnóstico de supuestos de la regresión lineal.

    Attributes:
        shapiro_residuals_pvalue: p-value del test de normalidad de
            Shapiro-Wilk sobre los residuos (H0: residuos normales).
        breusch_pagan_pvalue: p-value del test de Breusch-Pagan para
            homocedasticidad (H0: varianza constante de los residuos).
        vif_table: DataFrame con el VIF de cada variable predictora
            (VIF > 10 sugiere multicolinealidad problemática, > 5 es
            zona de atención según convención estándar).
        r_squared: coeficiente de determinación sobre el set de test.
        r_squared_adjusted: R² ajustado por número de predictores.
        coefficients: mapeo {nombre_variable: coeficiente} del modelo
            ajustado, incluyendo el intercepto.
    """

    shapiro_residuals_pvalue: float
    breusch_pagan_pvalue: float
    vif_table: pd.DataFrame
    r_squared: float
    r_squared_adjusted: float
    coefficients: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializa el diagnóstico a un diccionario plano apto para JSON."""
        return {
            "shapiro_residuals_pvalue": self.shapiro_residuals_pvalue,
            "residuals_normal_at_0.05": self.shapiro_residuals_pvalue >= 0.05,
            "breusch_pagan_pvalue": self.breusch_pagan_pvalue,
            "homoscedastic_at_0.05": self.breusch_pagan_pvalue >= 0.05,
            "r_squared": self.r_squared,
            "r_squared_adjusted": self.r_squared_adjusted,
            "coefficients": self.coefficients,
            "max_vif": float(self.vif_table["VIF"].max()) if not self.vif_table.empty else None,
        }


def prepare_design_matrix(
    df: pd.DataFrame,
    target_column: str,
    numeric_predictors: list[str],
    categorical_predictors: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye la matriz de diseño (X) y el vector objetivo (y) para la
    regresión, aplicando codificación one-hot (dummy) a las variables
    categóricas (con `drop_first=True` para evitar la trampa de la
    variable dummy / multicolinealidad perfecta por construcción).

    Exclusión automática de predictores con varianza cero:
        Un predictor constante (ej. UNIDADES=1 en el 100% de las
        transacciones, hallazgo real verificado en el archivo de
        producción de este proyecto) no aporta ninguna información al
        modelo: no puede explicar variabilidad en `target_column`
        porque él mismo no varía. Peor aún, incluirlo en la regresión
        produce multicolinealidad perfecta con el intercepto (el
        predictor es, en la práctica, un múltiplo constante de la
        columna de unos del intercepto), lo que hace que su VIF sea
        matemáticamente infinito y puede desestabilizar numéricamente
        la estimación del resto de los coeficientes. La práctica
        estándar (y la que se aplica aquí) es excluir estos predictores
        ANTES de ajustar el modelo, registrando la decisión en el log
        para que quede documentada en el informe técnico.

    Args:
        df: DataFrame limpio y transformado.
        target_column: variable respuesta (ej. MONTO_APLICADO).
        numeric_predictors: columnas numéricas predictoras candidatas
            (las que tengan varianza cero se excluirán automáticamente).
        categorical_predictors: columnas categóricas predictoras.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y), con filas que tenían
        NaN en cualquiera de las columnas involucradas ya excluidas, y
        sin predictores numéricos de varianza cero.

    Complejidad:
        O(n * k) tiempo y espacio, n = filas, k = número de columnas
        involucradas (incluyendo las dummies generadas).
    """
    relevant_columns = [target_column] + numeric_predictors + categorical_predictors
    data = df[relevant_columns].dropna()

    # Detección y exclusión de predictores numéricos con varianza cero.
    # Se hace ANTES de construir X para no malgastar cómputo de VIF/ajuste
    # sobre una columna que se sabe de antemano que es matemáticamente
    # problemática.
    zero_variance_predictors = [
        col for col in numeric_predictors if data[col].std(ddof=1) == 0
    ]
    effective_numeric_predictors = [
        col for col in numeric_predictors if col not in zero_variance_predictors
    ]
    if zero_variance_predictors:
        logger.warning(
            "Predictor(es) numérico(s) %s excluido(s) automáticamente: "
            "varianza cero en los datos (columna constante). Un predictor "
            "constante no puede explicar variabilidad de '%s' y produce "
            "multicolinealidad perfecta (VIF infinito) con el intercepto.",
            zero_variance_predictors, target_column,
        )

    X_numeric = data[effective_numeric_predictors]
    X_categorical = pd.get_dummies(
        data[categorical_predictors], drop_first=True, dtype=float
    )
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = data[target_column]

    logger.info(
        "Matriz de diseño construida: %d filas, %d predictores (%d numéricos + "
        "%d dummies de %d categóricas). Filas descartadas por NaN: %d. "
        "Predictores excluidos por varianza cero: %d.",
        len(X), X.shape[1], len(effective_numeric_predictors), X_categorical.shape[1],
        len(categorical_predictors), len(df) - len(data), len(zero_variance_predictors),
    )
    return X, y


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Factor de Inflación de Varianza (VIF) para cada
    predictor en `X`.

    Fórmula: VIF_i = 1 / (1 - R_i^2), donde R_i^2 es el coeficiente de
    determinación de regresionar la variable i sobre TODAS las demás
    variables predictoras. Un VIF alto indica que la variable i puede
    explicarse en gran medida por las otras (multicolinealidad).

    Args:
        X: matriz de diseño (solo predictores, sin la variable objetivo).

    Returns:
        pd.DataFrame: columnas ["variable", "VIF"], una fila por
        predictor.

    Complejidad:
        O(k * n * k) = O(n * k^2) tiempo (una regresión auxiliar por
        cada una de las k columnas, cada una sobre las k-1 restantes),
        O(k) espacio para el resultado.
    """
    vif_rows = []
    columns = list(X.columns)

    for i, column in enumerate(columns):
        other_columns = [c for c in columns if c != column]
        if not other_columns:
            vif_rows.append({"variable": column, "VIF": 1.0})
            continue

        aux_model = LinearRegression()
        aux_model.fit(X[other_columns], X[column])
        r_squared_i = aux_model.score(X[other_columns], X[column])

        # Si R²=1 exactamente (colinealidad perfecta), VIF es infinito;
        # se reporta como tal en lugar de fallar por división por cero.
        vif = float("inf") if np.isclose(r_squared_i, 1.0) else 1.0 / (1.0 - r_squared_i)
        vif_rows.append({"variable": column, "VIF": vif})

    result = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    logger.info(
        "VIF calculado para %d predictores. VIF máximo=%.2f (%s).",
        len(result), result["VIF"].iloc[0], result["variable"].iloc[0],
    )
    return result


def breusch_pagan_test(X: pd.DataFrame, residuals: np.ndarray) -> float:
    """
    Test de Breusch-Pagan para homocedasticidad de los residuos.

    H0: la varianza de los residuos es constante (homocedasticidad).
    H1: la varianza depende de los predictores (heterocedasticidad).

    Procedimiento:
        1. Se regresionan los residuos al cuadrado sobre los
           predictores originales (regresión auxiliar).
        2. El estadístico de prueba es n * R² de esa regresión
           auxiliar, que sigue asintóticamente una distribución
           Chi-cuadrado con k grados de libertad (k = número de
           predictores).

    Args:
        X: matriz de diseño usada en el modelo original.
        residuals: residuos del modelo original (y_true - y_pred).

    Returns:
        float: p-value del test.

    Complejidad:
        O(n * k^2) tiempo (una regresión lineal múltiple), O(n * k) espacio.
    """
    squared_residuals = residuals ** 2
    aux_model = LinearRegression()
    aux_model.fit(X, squared_residuals)
    r_squared_aux = aux_model.score(X, squared_residuals)

    n = len(residuals)
    k = X.shape[1]
    statistic = n * r_squared_aux
    p_value = float(1 - stats.chi2.cdf(statistic, df=k))

    logger.info(
        "Test de Breusch-Pagan: estadístico=%.4f (n=%d, k=%d), p-value=%.4f -> %s",
        statistic, n, k, p_value,
        "homocedástico" if p_value >= 0.05 else "heterocedástico",
    )
    return p_value


def fit_and_diagnose_regression(
    df: pd.DataFrame,
    target_column: str = "MONTO_APLICADO",
    numeric_predictors: list[str] | None = None,
    categorical_predictors: list[str] | None = None,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
) -> tuple[LinearRegression, RegressionDiagnostics, dict]:
    """
    Ajusta una regresión lineal múltiple de `target_column` sobre los
    predictores indicados, divide los datos en train/test de forma
    reproducible, y ejecuta el diagnóstico completo de supuestos
    exigido por el enunciado: normalidad de residuos, homocedasticidad,
    multicolinealidad (VIF), y R²/R² ajustado.

    Args:
        df: DataFrame limpio y transformado.
        target_column: variable respuesta.
        numeric_predictors: predictores numéricos (por defecto:
            ["UNIDADES", "PORCENTAJE_DESCUENTO"]).
        categorical_predictors: predictores categóricos (por defecto:
            ["CANAL", "LOCAL"]).
        seed: semilla de reproducibilidad para el split train/test.
        test_size: proporción del set de test.

    Returns:
        tuple[LinearRegression, RegressionDiagnostics, dict]: el modelo
        entrenado, el diagnóstico de supuestos, y un dict con métricas
        de validación en test ("rmse", "mae", "r2_test").

    Complejidad:
        O(n * k^2) tiempo (dominado por VIF, que requiere k regresiones
        auxiliares), O(n * k) espacio.
    """
    numeric_predictors = numeric_predictors or ["UNIDADES", "PORCENTAJE_DESCUENTO"]
    categorical_predictors = categorical_predictors or ["CANAL", "LOCAL"]

    X, y = prepare_design_matrix(df, target_column, numeric_predictors, categorical_predictors)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    logger.info(
        "Split train/test reproducible (seed=%d): %d train, %d test (%.0f%% test).",
        seed, len(X_train), len(X_test), test_size * 100,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    residuals_train = (y_train - y_pred_train).to_numpy()

    # El diagnóstico de supuestos se evalúa sobre los residuos de
    # ENTRENAMIENTO, ya que los supuestos del modelo lineal (normalidad,
    # homocedasticidad) son propiedades del proceso generador de datos
    # que el modelo intenta capturar, no del desempeño predictivo
    # (eso se evalúa por separado en el set de test).
    shapiro_stat, shapiro_p = stats.shapiro(
        residuals_train if len(residuals_train) <= 5000
        else pd.Series(residuals_train).sample(5000, random_state=seed)
    )
    bp_p_value = breusch_pagan_test(X_train, residuals_train)
    vif_table = compute_vif(X_train)

    r2_test = r2_score(y_test, y_pred_test)
    n_test, k_predictors = X_test.shape
    r2_adjusted = 1 - (1 - r2_test) * (n_test - 1) / (n_test - k_predictors - 1)

    coefficients = dict(zip(X.columns, model.coef_))
    coefficients["intercept"] = float(model.intercept_)

    diagnostics = RegressionDiagnostics(
        shapiro_residuals_pvalue=float(shapiro_p),
        breusch_pagan_pvalue=bp_p_value,
        vif_table=vif_table,
        r_squared=float(r2_test),
        r_squared_adjusted=float(r2_adjusted),
        coefficients={k: float(v) for k, v in coefficients.items()},
    )

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "r2_test": float(r2_test),
        "r2_adjusted": float(r2_adjusted),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    logger.info(
        "Regresión ajustada. RMSE=%.2f, MAE=%.2f, R²_test=%.4f, R²_ajustado=%.4f.",
        metrics["rmse"], metrics["mae"], metrics["r2_test"], metrics["r2_adjusted"],
    )

    return model, diagnostics, metrics
