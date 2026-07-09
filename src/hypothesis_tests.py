"""
src/hypothesis_tests.py
=========================

Pruebas de hipótesis estadísticas.

Responsabilidad única: implementar funciones de test reutilizables
(Chi-cuadrado, ANOVA, t-test/Mann-Whitney con selección automática
según normalidad) y orquestar las 5 hipótesis exigidas por el
enunciado:

    Obligatorias:
        H1: El ticket promedio (MONTO_APLICADO) en APP es mayor que en WEB.
        H2: El % de descuento afecta significativamente las unidades vendidas.

    Propias (3 adicionales, con justificación de negocio):
        H3: El monto promedio de compra difiere entre géneros.
        H4: Existe asociación entre el CANAL de compra y el LOCAL
            (los clientes no eligen canal de forma independiente de
            la sucursal, ej. por diferencias de digitalización entre
            sucursales).
        H5: Los clientes de mayor EDAD tienden a gastar un MONTO_APLICADO
            distinto por transacción (posible mayor/menor poder
            adquisitivo o patrón de consumo asociado a la edad). Se usa
            MONTO_APLICADO y no UNIDADES como variable respuesta porque,
            en el archivo real de este proyecto, UNIDADES resultó ser
            constante (std=0) — ver nota de "varianza cero" en
            `simple_linear_regression_test`.

Cada test retorna un diccionario estandarizado con:
    {statistic, p_value, alpha, reject_h0, method, interpretation}
para poder tabular todos los resultados de forma homogénea en el
informe técnico.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from config import ALPHA
from src.eda import check_normality
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_result(
    test_name: str,
    statistic: float,
    p_value: float,
    alpha: float,
    method: str,
    interpretation_h0: str,
    interpretation_h1: str,
) -> dict:
    """
    Construye el diccionario de resultado estandarizado para cualquier
    test de hipótesis del módulo.

    Args:
        test_name: nombre identificador de la hipótesis (ej. "H1").
        statistic: estadístico de prueba calculado.
        p_value: p-value asociado.
        alpha: nivel de significancia usado.
        method: nombre del test estadístico aplicado.
        interpretation_h0: texto no técnico si NO se rechaza H0.
        interpretation_h1: texto no técnico si SÍ se rechaza H0.

    Returns:
        dict: resultado estandarizado, listo para tabular o serializar.

    Complejidad:
        O(1) tiempo y espacio.
    """
    reject_h0 = p_value < alpha
    interpretation = interpretation_h1 if reject_h0 else interpretation_h0

    logger.info(
        "[%s] método=%s | estadístico=%.4f | p-value=%.6f | alpha=%.2f | "
        "%s H0 -> %s",
        test_name, method, statistic, p_value, alpha,
        "SE RECHAZA" if reject_h0 else "NO se rechaza", interpretation,
    )

    return {
        "test": test_name,
        "method": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject_h0": bool(reject_h0),
        "interpretation": interpretation,
    }


def chi_square_independence_test(
    df: pd.DataFrame, column_a: str, column_b: str, alpha: float = ALPHA
) -> dict:
    """
    Prueba de independencia Chi-cuadrado entre dos variables categóricas.

    H0: las variables son independientes.
    H1: existe asociación entre las variables.

    Args:
        df: DataFrame con ambas columnas categóricas.
        column_a: primera columna categórica.
        column_b: segunda columna categórica.
        alpha: nivel de significancia.

    Returns:
        dict: resultado estandarizado del test.

    Complejidad:
        O(n + r*c) tiempo, donde n = filas, r = categorías de column_a,
        c = categorías de column_b (construcción de la tabla de
        contingencia domina sobre el cálculo del estadístico).
    """
    contingency_table = pd.crosstab(df[column_a], df[column_b])
    statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return _build_result(
        test_name=f"chi2_{column_a}_vs_{column_b}",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        method="chi-cuadrado de independencia",
        interpretation_h0=f"No hay evidencia de asociación entre {column_a} y {column_b}.",
        interpretation_h1=f"Existe asociación significativa entre {column_a} y {column_b}.",
    )


def anova_test(
    df: pd.DataFrame, value_column: str, group_column: str, alpha: float = ALPHA
) -> dict:
    """
    Análisis de varianza (ANOVA) de un factor para comparar la media de
    `value_column` entre los grupos de `group_column`.

    H0: todas las medias grupales son iguales.
    H1: al menos una media grupal difiere de las demás.

    Args:
        df: DataFrame con ambas columnas.
        value_column: columna numérica (ej. MONTO_APLICADO).
        group_column: columna categórica que define los grupos (ej. CANAL).
        alpha: nivel de significancia.

    Returns:
        dict: resultado estandarizado del test.

    Complejidad:
        O(n) tiempo, O(g) espacio adicional, g = número de grupos.
    """
    # .astype("float64") previene el mismo bug de dtypes "nullable" de
    # pandas (Int64/Float64) documentado en modeling_clustering.py: sin
    # esta conversión explícita, un value_column declarado como "Int64"
    # en config.EXPECTED_DTYPES (ej. UNIDADES) produciría un array
    # dtype=object, sobre el cual scipy.stats.f_oneway falla de forma
    # confusa.
    groups = [
        group[value_column].dropna().astype("float64").to_numpy()
        for _, group in df.groupby(group_column, observed=True)
        if group[value_column].dropna().shape[0] > 0
    ]
    statistic, p_value = stats.f_oneway(*groups)

    return _build_result(
        test_name=f"anova_{value_column}_por_{group_column}",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        method="ANOVA de un factor",
        interpretation_h0=f"No hay diferencias significativas de {value_column} entre los grupos de {group_column}.",
        interpretation_h1=f"Existen diferencias significativas de {value_column} entre al menos dos grupos de {group_column}.",
    )


def compare_two_groups_auto(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a: str,
    group_b: str,
    alpha: float = ALPHA,
    seed: int = 42,
    alternative: str = "two-sided",
) -> dict:
    """
    Compara `value_column` entre dos grupos (`group_a` vs `group_b`),
    eligiendo automáticamente el test según normalidad:
        - Si ambos grupos son razonablemente normales (Shapiro/K-S,
          p >= 0.05): t-test de Student para muestras independientes.
        - En caso contrario: Mann-Whitney U (no paramétrico).

    Args:
        df: DataFrame con ambas columnas.
        value_column: columna numérica a comparar.
        group_column: columna categórica que define los grupos.
        group_a: etiqueta del primer grupo (ej. "APP").
        group_b: etiqueta del segundo grupo (ej. "WEB").
        alpha: nivel de significancia.
        seed: semilla para el test de normalidad (submuestreo K-S si aplica).
        alternative: "two-sided", "less" o "greater" (dirección de H1).

    Returns:
        dict: resultado estandarizado del test, incluyendo qué método
        se seleccionó automáticamente y por qué.

    Complejidad:
        O(n log n) tiempo (dominado por los tests de normalidad y el
        ordenamiento interno de Mann-Whitney si se usa), O(n) espacio.
    """
    values_a = df.loc[df[group_column] == group_a, value_column].dropna()
    values_b = df.loc[df[group_column] == group_b, value_column].dropna()

    df_a = pd.DataFrame({value_column: values_a})
    df_b = pd.DataFrame({value_column: values_b})
    normality_a = check_normality(df_a, value_column, seed)
    normality_b = check_normality(df_b, value_column, seed)
    both_normal = normality_a["is_normal_at_0.05"] and normality_b["is_normal_at_0.05"]

    if both_normal:
        statistic, p_value = stats.ttest_ind(
            values_a, values_b, alternative=alternative, equal_var=False
        )
        method = "t-test de Student (Welch, varianzas desiguales)"
    else:
        statistic, p_value = stats.mannwhitneyu(
            values_a, values_b, alternative=alternative
        )
        method = "Mann-Whitney U (no paramétrico)"

    logger.info(
        "Selección automática de test para %s (%s vs %s): normal_A=%s, "
        "normal_B=%s -> %s",
        value_column, group_a, group_b,
        normality_a["is_normal_at_0.05"], normality_b["is_normal_at_0.05"], method,
    )

    return _build_result(
        test_name=f"comparacion_{value_column}_{group_a}_vs_{group_b}",
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        method=method,
        interpretation_h0=f"No hay diferencia significativa de {value_column} entre {group_a} y {group_b}.",
        interpretation_h1=f"{value_column} difiere significativamente entre {group_a} y {group_b}.",
    )


def simple_linear_regression_test(
    df: pd.DataFrame, x_column: str, y_column: str, alpha: float = ALPHA
) -> dict:
    """
    Regresión lineal simple usada como prueba de hipótesis sobre la
    pendiente: evalúa si `x_column` tiene un efecto lineal
    significativo sobre `y_column`.

    H0: la pendiente de la regresión es 0 (sin efecto lineal).
    H1: la pendiente es distinta de 0 (existe efecto lineal).

    Caso especial: variable dependiente con varianza cero.
        Si `y_column` es constante (ej. UNIDADES=1 en el 100% de las
        transacciones, hallazgo real verificado en el archivo de
        producción de este proyecto), el resultado (pendiente=0, p=1)
        está matemáticamente GARANTIZADO de antemano: no hay ninguna
        variabilidad en `y_column` que `x_column` pueda explicar o
        dejar de explicar. Esto es cualitativamente distinto de un
        verdadero resultado nulo (donde SÍ hay variabilidad, pero
        `x_column` no la explica): aquí no se está aportando evidencia
        empírica de ausencia de efecto, sino verificando una tautología
        aritmética. Se detecta este caso y se documenta explícitamente
        en el resultado (clave `"advertencia_varianza_cero"`) para que
        el informe técnico no interprete erróneamente "p=1" como un
        hallazgo estadístico genuino.

    Args:
        df: DataFrame con ambas columnas.
        x_column: variable predictora (ej. PORCENTAJE_DESCUENTO).
        y_column: variable respuesta (ej. UNIDADES).
        alpha: nivel de significancia.

    Returns:
        dict: resultado estandarizado, incluyendo además la pendiente
        (`slope`) y el R² para contexto adicional en el informe. Incluye
        `"advertencia_varianza_cero": True` si `y_column` resultó ser
        constante en los datos.

    Complejidad:
        O(n) tiempo, O(n) espacio (mínimos cuadrados sobre pares
        completos de datos).
    """
    paired = df[[x_column, y_column]].dropna()

    if paired[y_column].std(ddof=1) == 0:
        logger.warning(
            "La variable dependiente '%s' es CONSTANTE en los datos (std=0). "
            "El resultado de esta prueba (pendiente=0, p=1) está "
            "matemáticamente garantizado de antemano y NO constituye "
            "evidencia empírica de ausencia de efecto: no existe "
            "variabilidad en '%s' que '%s' pueda explicar.",
            y_column, y_column, x_column,
        )
        result = _build_result(
            test_name=f"regresion_{x_column}_sobre_{y_column}",
            statistic=0.0,
            p_value=1.0,
            alpha=alpha,
            method="regresión lineal simple (test de pendiente)",
            interpretation_h0=(
                f"No aplica: '{y_column}' es constante en los datos, por lo "
                f"que no puede evaluarse si '{x_column}' la afecta."
            ),
            interpretation_h1="No aplica (ver nota de varianza cero).",
        )
        result["slope"] = 0.0
        result["r_squared"] = float("nan")
        result["advertencia_varianza_cero"] = True
        return result

    regression = stats.linregress(paired[x_column], paired[y_column])

    result = _build_result(
        test_name=f"regresion_{x_column}_sobre_{y_column}",
        statistic=regression.slope,
        p_value=regression.pvalue,
        alpha=alpha,
        method="regresión lineal simple (test de pendiente)",
        interpretation_h0=f"{x_column} no tiene efecto lineal significativo sobre {y_column}.",
        interpretation_h1=f"{x_column} SÍ afecta significativamente a {y_column} (pendiente={regression.slope:.4f}).",
    )
    result["slope"] = float(regression.slope)
    result["r_squared"] = float(regression.rvalue ** 2)
    result["advertencia_varianza_cero"] = False
    return result


def run_all_hypothesis_tests(df: pd.DataFrame, seed: int = 42) -> list[dict]:
    """
    Ejecuta las 5 hipótesis del proyecto (2 obligatorias + 3 propias) y
    retorna todos los resultados en una lista homogénea.

    Args:
        df: DataFrame limpio y transformado, con columnas CANAL, LOCAL,
            MONTO_APLICADO, UNIDADES, PORCENTAJE_DESCUENTO, GENERO, EDAD.
        seed: semilla de reproducibilidad.

    Returns:
        list[dict]: un resultado estandarizado por cada hipótesis.

    Complejidad:
        O(n log n) tiempo total (dominado por los tests individuales
        más costosos), O(n) espacio.
    """
    logger.info("=== Iniciando validación de las 5 hipótesis del proyecto ===")
    results = []

    # H1 (obligatoria): ticket promedio en APP > WEB.
    results.append(
        compare_two_groups_auto(
            df, "MONTO_APLICADO", "CANAL", "APP", "WEB", seed=seed,
            alternative="greater",
        )
    )

    # H2 (obligatoria): % descuento afecta unidades vendidas.
    results.append(
        simple_linear_regression_test(df, "PORCENTAJE_DESCUENTO", "UNIDADES")
    )

    # H3 (propia): el monto promedio difiere entre géneros (1=Masculino, 2=Femenino).
    if "GENERO" in df.columns:
        df_genero = df.dropna(subset=["GENERO"])
        genero_labels = sorted(df_genero["GENERO"].unique())
        if len(genero_labels) >= 2:
            results.append(
                compare_two_groups_auto(
                    df_genero, "MONTO_APLICADO", "GENERO",
                    genero_labels[0], genero_labels[1], seed=seed,
                )
            )

    # H4 (propia): asociación entre CANAL y LOCAL.
    results.append(chi_square_independence_test(df, "CANAL", "LOCAL"))

    # H5 (propia): EDAD se asocia linealmente con el MONTO_APLICADO por
    # transacción. Se eligió MONTO_APLICADO (y no UNIDADES) como variable
    # respuesta porque, en el archivo real de este proyecto, UNIDADES
    # resultó ser CONSTANTE (std=0) — probar "¿EDAD afecta a UNIDADES?"
    # sobre una variable sin varianza es una tautología matemática (ver
    # `simple_linear_regression_test`), no una pregunta de negocio con
    # contenido empírico real. MONTO_APLICADO sí varía sustancialmente y
    # es una pregunta de negocio genuinamente interesante: ¿los clientes
    # de mayor edad gastan más o menos por transacción?
    df_edad = df.dropna(subset=["EDAD"])
    results.append(simple_linear_regression_test(df_edad, "EDAD", "MONTO_APLICADO"))

    logger.info(
        "=== Validación de hipótesis finalizada: %d tests ejecutados, %d "
        "rechazan H0 ===",
        len(results), sum(r["reject_h0"] for r in results),
    )
    return results
