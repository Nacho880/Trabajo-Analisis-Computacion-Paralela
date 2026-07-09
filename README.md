# Trabajo Analisis — Cruz Morada

Curso: Computación Paralela y Distribuida
Institución: Universidad Tecnológica Metropolitana (UTEM)
Entrega: 10 de julio de 2026
Integrantes:

[Matias Fernandez 20.969.062-4]
[Camilo Moya 21.230.348-8]
[Ignacio Ortega 21.481.176-6]

Pipeline completo de carga, limpieza, análisis
exploratorio, inferencia estadística y modelado predictivo/descriptivo sobre
los datos de ventas de la cadena de farmacias Cruz Morada, con procesamiento
paralelo real y reproducibilidad total.

## Descripción

El proyecto procesa `data/ventas_completas.csv` (transacciones + datos de
clientes) para:

1. Limpiar y transformar los datos (valores faltantes, outliers, variables
   derivadas).
2. Calcular estadísticos por sucursal (`LOCAL`) usando un **algoritmo
   paralelo real** (múltiples hilos de CPU), con benchmark cuantitativo
   contra la versión secuencial equivalente.
3. Ejecutar un Análisis Exploratorio Estadístico completo (descriptivos,
   normalidad, correlaciones con significancia).
4. Validar 5 hipótesis de negocio (2 exigidas por la cátedra + 3 propias).
5. Analizar patrones temporales de ventas (tendencia, estacionalidad,
   autocorrelación).
6. Ajustar y diagnosticar un modelo de **regresión lineal** (Opción A) y un
   modelo de **clustering K-means** (Opción B), con validación train/test y
   discusión de extrapolabilidad.

Todos los resultados (tablas, figuras, modelos, logs, reporte de
rendimiento) se exportan automáticamente a `output/`.

## Arquitectura

```
main.py (orquestador CLI)
  → src/data_loader.py         Carga lazy con Dask (chunking real)
  → .persist()                 Materialización en memoria (aún como colección Dask)
  → src/data_cleaning.py       Nulos (test tipo MCAR) + outliers (IQR)        [Dask, lazy]
  → src/feature_engineering.py Variables derivadas + estandarización         [Dask, lazy]
  → .compute()                 Única frontera Dask → Pandas de todo el pipeline
  → src/parallel_stats.py      ★ Paralelismo real (ThreadPoolExecutor)       [Pandas]
  → src/eda.py                 Estadística descriptiva + visualizaciones     [Pandas]
  → src/hypothesis_tests.py    5 pruebas de hipótesis                       [Pandas]
  → src/time_series.py         Descomposición + ACF/PACF                    [Pandas]
  → src/modeling_regression.py Opción A: regresión + diagnóstico + VIF      [Pandas]
  → src/modeling_clustering.py Opción B: K-means + elbow + silhouette       [Pandas]
  → src/validation.py          Consolidación + discusión de extrapolabilidad
  → src/utils/                 logger, io_utils (exportación), validators
```

### Dónde vive el paralelismo y por qué

`src/parallel_stats.py` implementa el algoritmo paralelo **explícito** del
proyecto: particiona el dataset por `LOCAL` (particiones lógicas
independientes, sin overlap), agrupa esas particiones en `n_jobs` chunks
balanceados, y calcula estadísticos descriptivos de cada chunk en un hilo
worker separado (`concurrent.futures.ThreadPoolExecutor`).

- **Por qué se paraleliza**: es un problema *embarazosamente paralelo* (el
  resultado de una sucursal no depende de otra).
- **Qué se comparte entre hilos**: nada de forma mutable con riesgo real —
  aunque los hilos comparten memoria del proceso, cada uno opera solo sobre
  su propio chunk (subconjunto disjunto de particiones) y escribe únicamente
  en su propia lista de resultados local.
- **Cómo se evitan condiciones de carrera**: por partición de datos disjunta
  entre hilos, no por locks — ningún hilo toca el chunk de otro.
- **Cómo se evitan deadlocks**: no hay locks ni dependencias entre tareas;
  cada chunk se procesa de forma completamente independiente.
- **Verificación de correctitud**: `benchmark_sequential_vs_parallel` compara
  automáticamente los resultados secuencial y paralelo y lanza
  `AssertionError` si difieren, antes de reportar cualquier métrica de
  rendimiento.

#### Por qué hilos (`ThreadPoolExecutor`) y no procesos

Se evaluó `ProcessPoolExecutor` como alternativa, con el objetivo teórico de
evitar el GIL de Python. En la práctica midió peor: en Windows, `spawn`
obliga a cada proceso worker a levantar un intérprete nuevo y reimportar
todo el stack científico (pandas, NumPy, SciPy, scikit-learn) antes de poder
calcular nada, un costo de arranque que domina por completo sobre el cómputo
real (microsegundos por partición). Se optó por `ThreadPoolExecutor`: es
válido porque `numpy.mean`/`numpy.std`/`numpy.percentile` liberan el GIL en
su implementación en C, permitiendo paralelismo real de CPU con hilos sin
pagar el costo de arrancar procesos nuevos (la misma técnica que usa el
scheduler "threaded" de Dask para trabajo NumPy-bound).

Aun con hilos, el speedup medido es modesto (cercano a 1x): las tareas por
partición son de grano muy fino (~4096 filas, del orden de microsegundos),
por debajo del intervalo de cambio de contexto del GIL (5ms por defecto), por
lo que el overhead de coordinación entre hilos compite en magnitud con el
beneficio real del paralelismo. Esto se documenta explícitamente, en vez de
perseguir un número más "bonito": `run_scalability_study()` mide speedup y
eficiencia para varios valores de `n_jobs` (1, 2, 4, 8, `N_JOBS`) sobre la
misma base secuencial, generando la curva de escalabilidad clásica de
Computación Paralela (`output/figures/curva_escalabilidad_speedup.png`,
`output/tables/escalabilidad_speedup_vs_n_jobs.csv`). El detalle completo del
proceso de evaluación (ambas alternativas, con las métricas exactas medidas
en cada una) está en la sección 6 del informe técnico.

### Dónde vive Dask y por qué se acota a carga + limpieza + features

Dask maneja las tres etapas que operan sobre el dataset completo sin reducir
(`src/data_loader.py`, `src/data_cleaning.py`, `src/feature_engineering.py`),
de forma perezosa (*lazy*) y particionada, evitando cargar los ~830 MB del
CSV de una sola vez. Inmediatamente después de la carga se invoca
`.persist()`, que materializa esa lectura una única vez en memoria (sin
abandonar la API de Dask), permitiendo encadenar limpieza y feature
engineering sin que cada estadístico intermedio (usado para logging/reportes)
dispare una relectura completa del archivo desde cero.

La materialización a Pandas (`.compute()`) ocurre una única vez, después de
feature engineering. A partir de ahí, el algoritmo paralelo propio
(`ThreadPoolExecutor`), el EDA, las pruebas de hipótesis, el análisis de
series de tiempo y el modelado (regresión/clustering) operan sobre Pandas
puro, ya que dependen de scipy/statsmodels/scikit-learn/matplotlib, que no
entienden objetos Dask. Es una frontera deliberada: Dask resuelve el
problema de memoria/I-O sobre el volumen completo de datos; Pandas resuelve
el análisis estadístico una vez que el dataset ya está en un tamaño
manejable en memoria (~830 MB).

## Instalación

Requiere Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dependencias principales

| Librería | Uso |
|---|---|
| `dask` | Carga, limpieza y feature engineering perezosos/particionados sobre el CSV completo (`.persist()` tras la carga; frontera única a Pandas vía `.compute()` después de feature engineering) |
| `pandas` / `numpy` | Manipulación de datos y cómputo numérico |
| `scipy` | Tests estadísticos (normalidad, Chi², ANOVA, Mann-Whitney, etc.) |
| `scikit-learn` | Regresión lineal, K-means, métricas de validación |
| `statsmodels` | Descomposición de series de tiempo, ACF/PACF |
| `matplotlib` / `seaborn` | Visualizaciones |
| `joblib` | Serialización del modelo de regresión |
| `psutil` | Métricas de CPU/memoria para el reporte de rendimiento |
| `pytest` | Suite de tests unitarios/funcionales/rendimiento/reproducibilidad |

> **Nota de diseño**: el VIF (multicolinealidad) y el test de Breusch-Pagan
> (homocedasticidad) se implementan manualmente en
> `src/modeling_regression.py` con `scikit-learn`/`scipy` en lugar de
> depender de `statsmodels` para esa parte, ya que ambos se derivan de
> fórmulas estadísticas simples (ver docstrings del módulo) y esto reduce
> el acoplamiento a una dependencia pesada que ya se usa, de forma
> justificada, únicamente en `time_series.py`.

## Variables de entorno

Ver `.env.example`. Las relevantes son:

- `CPYD_SEED` (default `42`): semilla de reproducibilidad global.
- `CPYD_N_JOBS` (default: todos los núcleos disponibles): hilos worker
  para el paralelismo explícito.

## Ejecución

> **Nota sobre el formato del CSV**: el enunciado describe las columnas
> separadas por coma, pero el archivo real distribuido usa **punto y coma
> (`;`)** como delimitador y la columna de género se llama `GENERO` (sin
> tilde). Esto ya está resuelto en `config.CSV_SEPARATOR` y
> `config.RAW_EXPECTED_COLUMNS`; no requiere ninguna acción de su parte.

1. Descargue `ventas_completas.csv` desde el enlace del enunciado y colóquelo
   en `data/ventas_completas.csv`.
2. Ejecute:

```bash
python main.py --input data/ventas_completas.csv
```

3. Revise los resultados en `output/`:
   - `output/figures/` — histogramas, boxplots, heatmaps, descomposición temporal.
   - `output/tables/` — estadísticos, resultados de tests, métricas, reportes JSON.
   - `output/models/` — modelo de regresión serializado (`.joblib`).
   - `output/logs/` — log completo de la ejecución (`pipeline.log`).

## Ejecutar los tests

```bash
pytest tests/ -v
```

> Algunos tests de `test_data_loader.py` y `test_time_series.py` requieren
> `dask`/`statsmodels` reales y se omiten automáticamente
> (`pytest.importorskip`) si no están instalados, sin afectar el resto de la
> suite.

## Descripción de módulos

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Semilla, rutas, logging, esquema de columnas |
| `src/data_loader.py` | Carga y validación de esquema (Dask, lazy) |
| `src/data_cleaning.py` | Nulos (test tipo MCAR + mediana), outliers (IQR/Z-score, marcado no eliminación) — Dask, lazy |
| `src/feature_engineering.py` | MONTO_POR_UNIDAD, EDAD, FRECUENCIA_COMPRA, estandarización — Dask, lazy |
| `src/parallel_stats.py` | Algoritmo paralelo real + benchmark secuencial vs paralelo |
| `src/eda.py` | Descriptivos, normalidad, histogramas, boxplots, correlaciones con p-value |
| `src/hypothesis_tests.py` | Chi², ANOVA, t-test/Mann-Whitney automático, regresión simple, 5 hipótesis |
| `src/time_series.py` | Serie diaria, descomposición aditiva, ACF/PACF |
| `src/modeling_regression.py` | Regresión múltiple, VIF propio, Breusch-Pagan propio, R² ajustado |
| `src/modeling_clustering.py` | Perfil de cliente, elbow + silhouette, K-means final |
| `src/validation.py` | Tabla comparativa de modelos + discusión de extrapolabilidad |
| `src/utils/logger.py` | Logger consistente por módulo |
| `src/utils/io_utils.py` | Exportación uniforme (figuras, tablas, JSON, modelos) |
| `src/utils/validators.py` | Excepciones de dominio y validaciones reutilizables |

## Resultados

Cada ejecución genera automáticamente (nombres de archivo exactos en
`output/`):

- `eda_estadisticos_descriptivos.csv`, `eda_tests_normalidad.csv`
- `eda_correlacion_{pearson,spearman}[_pvalues].csv` + heatmaps
- `eda_anova_monto_por_canal_local.csv` (ANOVA de MONTO_APLICADO por CANAL y por LOCAL)
- `resultados_hipotesis.csv` (5 hipótesis, con interpretación no técnica)
- `estadisticos_por_local.csv` + `benchmark_rendimiento_paralelo.json`
  (speedup y eficiencia medidos del algoritmo paralelo)
- `escalabilidad_speedup_vs_n_jobs.csv` + `curva_escalabilidad_speedup.png`
  (speedup y eficiencia medidos para distintos números de hilos: 1, 2, 4, 8
  y `N_JOBS`, comparados contra el speedup ideal lineal)
- `ts_serie_diaria.csv`, `ts_descomposicion.png`, `ts_acf_pacf.png`
- `modelo_regresion.joblib`, `diagnostico_regresion.json`, `vif_regresion.csv`
- `clustering_elbow.csv`, `clustering_silhouette.csv`,
  `clustering_perfil_clusters.csv`
- `resumen_validacion_modelos.csv`, `discusion_extrapolabilidad.json`
- `performance_report.json` (tiempos por etapa, memoria, CPU, speedup)

## Limitaciones

- Cinco módulos (`modeling_regression.py`, `hypothesis_tests.py`, `eda.py`,
  `data_cleaning.py`, `parallel_stats.py`) superan levemente la guía de
  ~300 líneas por archivo. En los cinco casos la razón es la misma: cada
  función pública lleva un docstring extenso (Args, Returns, Raises y
  Complejidad temporal/espacial, exigidos explícitamente por el estándar de
  documentación del proyecto) y, en `parallel_stats.py` en particular, un
  bloque de documentación del diseño de paralelismo (justificación de
  seguridad de concurrencia). El código ejecutable propiamente tal se
  mantiene cohesionado y de responsabilidad única por módulo; dividir estos
  archivos habría fragmentado funciones estrechamente relacionadas (ej.
  separar `compute_vif` de `breusch_pagan_test`) sin beneficio real de
  mantenibilidad.
- El VIF y Breusch-Pagan implementados manualmente asumen regresiones
  auxiliares OLS estándar; para datasets con miles de predictores dummy
  (alta cardinalidad categórica) su costo computacional crece
  cuadráticamente con el número de columnas (ver complejidad documentada en
  `compute_vif`).
- La imputación de `PORCENTAJE_DESCUENTO` usa mediana global; no se
  implementó imputación condicional por grupo (ej. por `CANAL`), lo que
  podría mejorar la precisión si el patrón de descuento difiere fuertemente
  entre canales.
- El test de tipo MCAR implementado es una aproximación práctica
  (Mann-Whitney comparando grupos con/sin nulo) y no el test formal de
  Little's MCAR multivariado.
- La descomposición temporal requiere al menos 2 ciclos estacionales
  completos (14 días para periodicidad semanal); con rangos de fecha más
  cortos esa sección se omite automáticamente con una advertencia en el log.

## Trabajo futuro

- Agregar la Opción C (reglas de asociación con Apriori) como análisis
  complementario de canasta de compra.
- Migrar `parallel_stats.py` a `dask.dataframe.groupby` nativo como
  alternativa adicional al `ThreadPoolExecutor` manual, para comparar
  ambos enfoques de paralelismo en el mismo informe.
- Implementar validación cruzada (k-fold) además del split simple 70/30
  para una estimación más robusta del error de generalización.
- Añadir un test de Little's MCAR multivariado completo (actualmente
  aproximado con Mann-Whitney bivariado).
