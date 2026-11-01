# TC5061-Proyecto-Equipo28

**Proyecto MiBici: propuesta y base reproducible**

Tecnológico de Monterrey | TC5061 — MLOps  
Equipo 28

## 1. Descripción del proyecto

Este proyecto tiene como objetivo desarrollar una solución reproducible de Machine Learning para analizar los patrones de utilización del sistema de bicicletas públicas MiBici del Área Metropolitana de Guadalajara.

La solución contempla modelos de regresión, clasificación y clustering, utilizando los registros históricos de viajes y la información de las estaciones.

El proyecto implementará buenas prácticas de MLOps, incluyendo versionado de código y datos, modularización, seguimiento de experimentos con MLflow y colaboración mediante GitHub.

## 2. Preguntas del proyecto

| ID | Tipo de problema | Objetivo |
|---|---|---|
| P1 | Regresión | Predecir la cantidad de viajes que se iniciarán en una estación o zona durante un periodo futuro. |
| P2 | Clasificación | Estimar el riesgo de que una estación se quede sin bicicletas o sin anclajes disponibles. |
| P3 | Clustering | Identificar grupos de estaciones que presentan patrones de uso similares. |

Para P2 se desarrollará una aproximación del riesgo utilizando los registros de viajes y la capacidad de las estaciones, debido a que los datos abiertos no incluyen inventarios históricos de bicicletas.


## 3. Fuentes de datos

Los datos provienen del portal oficial de [Datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).

### 3.1. Registros históricos de viajes

Se utilizan los doce archivos CSV correspondientes a enero–diciembre de 2025.

| Característica | Valor |
|---|---|
| Periodo | Enero–diciembre de 2025 |
| Archivos originales | 12 CSV |
| Viajes registrados | 4,532,032 |
| Estaciones observadas | 371 |
| Tamaño total aproximado | 393 MB |
| Codificaciones | UTF-8 y Windows-1252 |

Los archivos de referencia son:

- `references/mibici_sources_2025.csv`: manifiesto con las URLs oficiales.
- `references/mibici_inventory_2025.csv`: inventario de archivos, tamaños y huellas SHA-256.

### 3.2. Catálogo de estaciones

Se utiliza el archivo `nomenclatura_2026_09.csv`, correspondiente a septiembre de 2026, con 484 estaciones.

Este catálogo es posterior al periodo analizado. Por tanto, no se asume que todas sus estaciones o capacidades estuvieran vigentes durante 2025.

### 3.3. Privacidad y almacenamiento

Durante la estandarización se excluyen los campos `Usuario_Id`, `Genero` y `Año_de_nacimiento`.

Los archivos originales, intermedios y procesados se almacenan localmente y no se incluyen en Git.

### 3.4. Limitaciones de los datos

Los registros de viajes no contienen inventarios históricos de bicicletas ni disponibilidad real de anclajes.

Por ello, el problema P2 utilizará indicadores proxy basados en los flujos de salidas y llegadas, sin afirmar que representan estados reales de estaciones vacías o llenas.

Asimismo, las horas sin viajes no garantizan que una estación estuviera operativa.

## 4. Estructura del repositorio

```text
TC5061-Proyecto-Equipo28/
├── data/                 # Datos originales, intermedios y procesados
├── docs/                 # Reporte de la segunda entrega (resultados, selección y evidencias)
├── mibici/               # Código fuente del proyecto
│   ├── clustering.py     # Agrupamiento de estaciones (P3)
│   ├── config.py         # Configuración centralizada
│   ├── dataset.py        # Ingesta y preparación de datos
│   ├── features.py       # Ingeniería de características
│   ├── tracking.py       # Registro del pipeline de datos en MLflow
│   ├── validation.py     # Integridad y drift de P1/P2 con Deepchecks
│   ├── workflow.py       # Encadena los entry points bajo un run padre
│   └── modeling/         # Modelos P1 y P2, runs de MLflow, búsqueda y Model Registry
├── models/               # Modelos generados localmente
├── notebooks/            # Exploración de datos
├── references/           # Manifiestos de datos, handoff P1/P2 y auditoría del catálogo
├── reports/
│   └── figures/          # Figuras y visualizaciones
├── tests/                # Pruebas del código
├── .gitignore            # Exclusiones de Git
├── Makefile              # Atajos para los entry points de MLflow
├── MLproject             # Entry points de MLflow Projects
├── pyproject.toml        # Configuración del paquete
├── python_env.yaml       # Entorno de MLflow Projects
├── requirements.txt      # Dependencias directas
└── requirements-lock.txt # Versiones del entorno
```

## 5. Entorno de desarrollo

El proyecto se ha inicializado utilizando:

- Python 3.14.4.
- WSL Ubuntu.
- Visual Studio Code.
- Cookiecutter Data Science.
- Git y GitHub.
- pandas, NumPy, scikit-learn y PyArrow.
- MLflow para seguimiento de experimentos.
- pytest y Ruff para pruebas y calidad del código.

Las dependencias se mantienen con versiones fijadas para facilitar la reproducción del entorno.

## 6. Instalación

Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd TC5061-Proyecto-Equipo28
```

Crear y activar el entorno virtual:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Verificar la instalación:

```bash
python -m pip check
```

La integración de validación usa el extra `validation` de `mibici` (Deepchecks 0.19.1 y setuptools 80.10.2, requerido por `pkg_resources` en esta versión de Deepchecks). Ya está incluido en `requirements.txt`; para instalarlo por separado en un entorno existente:

```bash
python -m pip install -e ".[validation]"
```

## 7. Ejecución con MLflow Projects

El archivo `MLproject` define los entry points que MLflow puede ejecutar y `python_env.yaml` fija las versiones de Python y librerías.

### 7.1. Preparar la terminal

En cada terminal nueva, activa el ambiente de la sección 6 y elige el experimento:

```bash
source .venv/bin/activate
export MLFLOW_EXPERIMENT_NAME=mibici-equipo28   # si no, los runs caen en "Default"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 # servidor de la sección 7.2
```

En Windows (PowerShell) usa `$env:NOMBRE="valor"` y define también `$env:PYTHONUTF8=1`: sin él, `mlflow run` falla con `UnicodeEncodeError` si rediriges la salida a un archivo o la canalizas.

### 7.2. Levantar el servidor de tracking

El Model Registry necesita un backend con base de datos. En una terminal aparte, con el ambiente activado, levanta el servidor de MLflow con un backend sqlite:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

También puedes usar `make mlflow-server`. Deja esa terminal abierta mientras trabajas: los runs, los modelos registrados y sus artefactos se guardan en `mlflow.db` y `mlartifacts/`, y la UI queda disponible en http://127.0.0.1:5000.

Con `MLFLOW_TRACKING_URI` definido (sección 7.1), todos los comandos `mlflow run` y `make mlflow-*` registran en ese servidor. Si no lo defines, MLflow usa `mlflow.db` local directamente.

### 7.3. Correr el proyecto

```bash
mlflow run . -e ingest --env-manager local
```

MLflow lee el archivo `MLproject`, toma el entry point `ingest` y ejecuta
`python -m mibici.dataset run-pipeline`, es decir, el pipeline completo de la sección 8.4.
La opción `--env-manager local` usa el ambiente activado en lugar de crear uno nuevo.
Al final debe decir `=== Run (ID '...') succeeded ===`.

Requiere conexión a Internet para descargar los archivos que aún no estén disponibles localmente.

`ingest` acepta los parámetros `year` y `manifest_path`, cuyos valores por defecto (`2025` y `references/mibici_sources_2025.csv`) vienen de `INGEST_YEAR` y `SOURCES_MANIFEST` en `mibici/config.py`:

```bash
mlflow run . -e ingest -P year=2025 -P manifest_path=references/mibici_sources_2025.csv --env-manager local
```

### 7.4. Entry points disponibles

| Entry point | Comando | Con `make` | Qué hace |
|---|---|---|---|
| `ingest` | `mlflow run . -e ingest --env-manager local` | `make mlflow-ingest` | Ejecuta el pipeline de datos (descarga, preparación y datasets horarios) |
| `p1_train` | `mlflow run . -e p1_train --env-manager local` | `make mlflow-p1` | Entrena y valida los modelos de regresión P1 |
| `p2_catalog` | `mlflow run . -e p2_catalog --env-manager local` | `make mlflow-p2` | Descarga y audita el catálogo de capacidades de estaciones |
| `p2_targets` | `mlflow run . -e p2_targets --env-manager local` | `make mlflow-p2` | Construye las etiquetas proxy de P2 para train y validation |
| `p2_train` | `mlflow run . -e p2_train --env-manager local` | `make mlflow-p2` | Verifica las etiquetas y entrena y valida los clasificadores P2 |
| `p3_train` | `mlflow run . -e p3_train --env-manager local` | `make mlflow-p3` | Agrupa las estaciones por perfil de uso con K-Means y un baseline por cuantiles |
| `train` | `mlflow run . -e train -P question=p1 --env-manager local` | `make mlflow-train QUESTION=p1` | Entrena una pregunta con el modelo e hiperparámetros elegidos (sección 7.5) |
| `search` | `mlflow run . -e search -P question=p1 --env-manager local` | `make mlflow-search QUESTION=p1` | Busca hiperparámetros de P1 o P2: un run padre con los baselines y un hijo por combinación (sección 7.5) |
| `register` | `mlflow run . -e register --env-manager local` | `make mlflow-register` | Registra los mejores modelos en el Model Registry y actualiza el alias `champion` (sección 7.7) |
| `evaluate` | `mlflow run . -e evaluate --env-manager local` | — | Evalúa en test (noviembre-diciembre) los `champion` de P1 y P2 y sus baselines, sin reentrenar; resultados en [reports/test_evaluation.md](reports/test_evaluation.md) |
| `workflow` | `mlflow run . -e workflow --env-manager local` | `make mlflow-workflow` | Encadena `ingest`, los pasos de P1, P2 y P3 y `register` como runs hijos de un run padre |

Cada paso usa los archivos que genera el anterior: P1, P2 y P3 leen el dataset horario de `ingest`, y `p2_train` necesita el catálogo y las etiquetas de `p2_catalog` y `p2_targets`. `p3_train` usa el catálogo para dibujar el mapa de grupos; si no está descargado, omite el mapa. `make mlflow-p2` ejecuta los tres pasos de P2 en orden. `p2_catalog` requiere conexión a Internet.

Los pasos de modelado solo leen datos de enero a octubre de 2025 (train y validation); noviembre y diciembre quedan reservados como test y ningún paso calcula métricas sobre ellos.

Los comandos de `make` ya definen `MLFLOW_EXPERIMENT_NAME=mibici-equipo28`, por lo que no hace falta ese `export` de la sección 7.1; `MLFLOW_TRACKING_URI` sí debe exportarse para usar el servidor.

### 7.5. Entrenar con parámetros

El entry point `train` entrena una pregunta (`question`: `p1`, `p2` o `p3`) y registra sus runs igual que `p1_train`, `p2_train` y `p3_train`. Sus valores por defecto vienen de `HGB_PARAMS` y `P3_MAX_K` en `mibici/config.py`:

| Parámetro | Por defecto | Aplica a | Qué hace |
|---|---|---|---|
| `question` | (obligatorio) | Todas | Pregunta a entrenar: `p1`, `p2` o `p3` |
| `model` | `all` | Todas | Modelo a registrar: `all`, o uno de `dummy_mean`, `seasonal_weekly`, `hist_gradient_boosting_poisson` (P1); `dummy_prior`, `hist_gradient_boosting_balanced` (P2); `baseline_activity_quantiles`, `kmeans` (P3) |
| `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` | `0.1`, `100`, `31`, `50`, `1.0` | P1 y P2 | Hiperparámetros de HistGradientBoosting |
| `k` | `0` | P3 | Cantidad de grupos seleccionada; `0` la elige por silhouette |
| `max_k` | `8` | P3 | Mayor k evaluado |

```bash
mlflow run . -e train -P question=p1 -P learning_rate=0.05 -P max_iter=200 --env-manager local
mlflow run . -e train -P question=p3 -P k=4 --env-manager local
```

P2 requiere haber corrido antes `p2_catalog` y `p2_targets`.

**Búsqueda de hiperparámetros.** El entry point `search` reutiliza el entrenamiento de P1 o P2 y evalúa cada combinación de `learning_rates` y `max_leaf_nodes` de HistGradientBoosting con la misma partición temporal (entrenamiento de enero a agosto, validation de septiembre y octubre). Sus valores por defecto vienen de `SEARCH_LEARNING_RATES` y `SEARCH_MAX_LEAF_NODES` en `mibici/config.py`:

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `question` | (obligatorio) | `p1` o `p2` |
| `learning_rates` | `0.05,0.1,0.2` | Valores de `learning_rate`, separados por comas |
| `max_leaf_nodes` | `15,31,63` | Valores de `max_leaf_nodes`, separados por comas |

```bash
mlflow run . -e search -P question=p1 --env-manager local
mlflow run . -e search -P question=p2 -P learning_rates=0.05,0.1 --env-manager local
```

El run padre se llama `P1-search` o `P2-search`. Sus hijos son los baselines y un candidato por combinación (`P1-regression-search-lr0.05-leaf15`; en P2, `P2-bike-pressure-search-lr0.05-leaf15` y su equivalente `dock-pressure`). El padre guarda `reports/selection_summary.json` con el mejor candidato frente al mejor baseline y los tags `selected_run_*`. Como `register` toma el entrenamiento más reciente de cada pregunta, la búsqueda corrida al final define el champion.

### 7.6. Ver los resultados

Con el servidor de la sección 7.2 corriendo, abre http://127.0.0.1:5000. Si no usas el servidor, ejecuta `mlflow ui` y abre la misma dirección. Selecciona el experimento `mibici-equipo28` para revisar los runs, sus parámetros y artefactos.

No hace falta tener la UI abierta mientras corres los runs: puedes abrirla después.

El paso `ingest` registra en su run:

| Tipo | Qué se registra |
|---|---|
| Parámetros | `year` y el nombre del manifiesto de fuentes |
| Tag `data_version` | Huella SHA-256 de los doce CSV originales; cambia solo si cambian los datos de entrada |
| Métricas | Viajes totales y por mes (`trips_month`), indicadores de calidad (`*_count`, `*_pct`), filas de los datasets horarios, estaciones observadas, porcentaje de horas sin actividad y duración del pipeline |
| Artefactos | `reports/raw_inventory.csv`, `reports/quality_by_month.csv` y los dos datasets horarios en `processed/` |

Los pasos `p1_train`, `p2_train` y `p3_train` registran un run padre por problema (`P1-regression`, `P2-classification` y `P3-clustering`) y, dentro de él, un run hijo por modelo:

| Problema | Runs hijos |
|---|---|
| P1 | `P1-regression-dummy-mean`, `P1-regression-seasonal-168h`, `P1-regression-hist-gradient-boosting-poisson` |
| P2 | `P2-bike-pressure-dummy-prior`, `P2-bike-pressure-hist-gradient-boosting-balanced`, `P2-dock-pressure-dummy-prior`, `P2-dock-pressure-hist-gradient-boosting-balanced` |
| P3 | `P3-clustering-baseline-activity-quantiles` y `P3-clustering-kmeans-k2` a `P3-clustering-kmeans-k8`; el tag `selected` marca el k elegido por silhouette |

Cada run hijo registra:

| Tipo | Qué se registra |
|---|---|
| Parámetros | Clase e hiperparámetros del modelo, `random_state`, columnas predictoras, periodos y tamaños de train/validation; en P2, también los umbrales proxy y el umbral de clasificación; en P3, el periodo de datos, el número de estaciones, los indicadores y su transformación |
| Tags | Pregunta, objetivo, equipo, commit de Git, versiones de Python y scikit-learn y `train_validation_data_version` (huella del dataset horario de enero a octubre); en P2, también la versión y el hash del catálogo y el hash de las etiquetas; en P1 y P2, `model_family` y `stage_of_work` (`baseline`, `training` o `search`) |
| Métricas | Prefijo `val_`: `val_mae` y `val_rmse` en P1; `val_average_precision`, `val_precision`, `val_recall`, `val_f1`, prevalencia y celdas de la matriz de confusión en P2. En P3 (sin partición de validación): `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, tamaño de los grupos e `inertia` |
| Artefactos | Modelo registrado (baselines y candidatos) con su firma y ejemplo de entrada en P1 y P2; en `evaluation/`, el perfil horario real contra predicho en P1; la matriz de confusión y la curva Precision-Recall en P2; y la asignación de estaciones, los perfiles por grupo, la proyección PCA y el mapa de grupos en P3. En P1 y P2 también `manifests/` (procedencia de los datos y manifiestos de fuentes e inventario) y `config/run_config.json` |

El run padre guarda la tabla comparativa de los modelos y un reporte JSON con los parámetros y la procedencia de los datos en `reports/`.

`mlflow.db`, `mlruns/` y `mlartifacts/` se crean solos al correr y están excluidos de Git.

### 7.7. Model Registry y alias `champion`

El entry point `register` publica en el Model Registry un modelo por pregunta (dos en P2, uno por objetivo):

| Modelo registrado | Runs que compara | Métrica de selección |
|---|---|---|
| `mibici-p1-departures` | Hijos del último run `P1-regression` | `val_mae` (menor es mejor) |
| `mibici-p2-bike-pressure` | Hijos `bike_pressure` del último run `P2-classification` | `val_average_precision` (mayor es mejor) |
| `mibici-p2-dock-pressure` | Hijos `dock_pressure` del último run `P2-classification` | `val_average_precision` (mayor es mejor) |
| `mibici-p3-station-clusters` | Hijo con `selected=True` del último run `P3-clustering` | `silhouette_score` (mayor es mejor) |

Para cada modelo, `register` toma el mejor run hijo del entrenamiento más reciente de su pregunta y lo registra como una versión nueva, con tags de la métrica, la versión de datos y el commit. El alias `champion` (el modelo en producción) solo se mueve a esa versión si mejora la métrica del champion actual; si no, la versión queda registrada sin alias. Volver a correr `register` sin entrenar de nuevo no crea versiones duplicadas.

El flujo es: entrenar (`train` o `workflow`), correr `register` y revisar en la pestaña **Models** de la UI qué versión tiene el alias. `workflow` ya ejecuta `register` como último paso.

Para cargar el modelo en producción desde Python:

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/mibici-p1-departures@champion")
```

### 7.8. Validación de datos con Deepchecks

Ejecuta la validación desde la raíz del repositorio:

```bash
python -m mibici.validation --output-dir reports/deepchecks
```

El comando genera `summary.json` y reportes HTML para:

- P1: integridad de train, validation y test, además de drift de train contra validation y contra el test reservado.
- P2: integridad de train y validation, más comparaciones independientes train/validation para `target_bike_pressure` y `target_dock_pressure`.

El test de P1 solo se usa para informar integridad y drift: no se entrena ni selecciona ningún modelo con él. P2 excluye test y cada comparación incluye únicamente sus predictores históricos permitidos y la etiqueta que se está revisando, nunca la otra etiqueta. Los HTML usan un muestreo determinista de hasta 100,000 filas por partición; `summary.json` informa los conteos de filas y nulos calculados sobre los datos completos, incluidos los nulos de las características de rezago. Cada entrada de `reports` identifica las particiones, el tamaño de muestra, los chequeos ejecutados y excluidos, los chequeos que no pasaron con sus detalles y los errores de ejecución. Nulos en claves o etiquetas, infinitos y claves `station_id-hour` duplicadas detienen la ejecución antes de escribir reportes.

La selección de chequeos evita errores conocidos de Deepchecks 0.19.1 con el entorno fijado (pandas 3.0.6, NumPy 2.5.4 y scikit-learn 1.9.1): se excluyen `Outlier Sample Detection` (`quantile(interpolation)`), `Feature Feature Correlation` (`np.NaN`), `Train Test Samples Mix` (`DataFrame.applymap`) y `Multivariate Drift` (incompatibilidad de `__sklearn_tags__`). También se omite `Index Train Test Leakage`, porque `station_id` se repite intencionalmente y no es un índice único de observación. En P1, `New Label Train Test` no aplica a una etiqueta de regresión; en la integridad conjunta de P2 no se designa una de las dos etiquetas como objetivo. Las particiones comparadas sí incluyen `hour` como metadato temporal para ejecutar los chequeos de solapamiento de fechas.

Se pueden sustituir las rutas de entrada y salida:

```bash
python -m mibici.validation --p1-input data/processed/station_hourly_2025_complete.parquet --p2-input data/processed/p2_targets_2025_train_validation.parquet --output-dir reports/deepchecks
```

Si faltan los archivos de entrada, genera el dataset horario de P1 con `mlflow run . -e ingest --env-manager local`. Para P2, ejecuta `mlflow run . -e p2_catalog --env-manager local` y luego `mlflow run . -e p2_targets --env-manager local`; la descarga del catálogo de estaciones requiere conexión a Internet.

#### Evidencia de ejecución local

Validación completa ejecutada el 1 de noviembre de 2026 con los Parquets locales. Se generaron nueve HTML y un resumen JSON; los reportes usaron hasta 100,000 filas por partición. No hubo errores de ejecución en los chequeos seleccionados.

![Resumen de DeepChecks: nueve reportes generados y cero errores de ejecución](docs/evidence/deepchecks-results.png)

| Reportes | Resultado observado |
|---|---|
| Integridad P1 (train, validation, test) | 9 chequeos por partición, 0 condiciones no aprobadas y 0 errores |
| Integridad P2 (train, validation) | 7 chequeos por partición, 0 condiciones no aprobadas y 0 errores |
| Drift P1 (train/validation y train/test) | 8 chequeos por comparación; `Feature Drift` señaló `month` en ambas. `Label Drift` pasó con KS 0.02 y 0.01, respectivamente |
| Drift P2 (una comparación por etiqueta) | 9 chequeos por comparación; `Feature Drift` señaló `month` en ambas. `Label Drift` pasó con Cramer's V 0.00622 y 0.00634 |

El resumen de datos completos registró 2,137,920 filas train, 537,640 validation y 536,981 test para P1. Para cada etiqueta P2 registró 2,137,182 filas train y 536,904 validation. No se encontraron nulos en claves o etiquetas, infinitos ni claves estación-hora duplicadas. En cada grupo de features `departures`, `arrivals` y `net_flow`, las columnas de rezago de 1h/24h/168h tienen 369/8,856/61,992 nulos en train y 1/24/168 en validation; P1 test también tiene 1/24/168. Las medias móviles P1 de 24h y 168h tienen 8,856/61,992 nulos en train y 24/168 en validation y test. El hallazgo de drift en `month` se reporta para revisión, no se oculta como error de ejecución.

## 8. Pipeline reproducible de ingeniería de datos

El módulo `mibici/dataset.py` implementa la descarga, validación, preparación y agregación de los datos de MiBici 2025.

### 8.1. Descargar los archivos originales

```bash
python -m mibici.dataset download-year
```

Descarga los doce CSV definidos en `references/mibici_sources_2025.csv`. Los archivos existentes se reutilizan sin volver a descargarlos.

### 8.2. Preparar los datos mensuales

```bash
python -m mibici.dataset prepare-year
```

Lee y valida los CSV, reconoce sus codificaciones y genera doce archivos Parquet estandarizados en `data/interim/`.

Se calculan indicadores de calidad para fechas inválidas, duraciones no positivas y viajes superiores a una hora.

### 8.3. Construir los datasets horarios

```bash
python -m mibici.dataset build-hourly
```

Integra los doce meses, valida los identificadores de viaje y agrega las salidas y llegadas por estación y hora.

Genera:

- `data/processed/station_hourly_2025.parquet`: 1,887,819 registros estación-hora con actividad observada.
- `data/processed/station_hourly_2025_complete.parquet`: 3,212,541 registros estación-hora dentro de 2025, incluyendo horas completadas con cero viajes.

Las horas completadas no garantizan que la estación estuviera operativa.

El dataset observado también incluye llegadas de viajes que finalizaron en enero de 2026, mientras que el dataset completado se restringe al calendario 2025.

### 8.4. Ejecutar el pipeline completo

```bash
python -m mibici.dataset run-pipeline
```

Ejecuta las tres etapas anteriores en orden. Requiere conexión a Internet para descargar los archivos que aún no estén disponibles localmente.

### 8.5. Ejecutar pruebas automatizadas

El proyecto utiliza PyTest para verificar el procesamiento de datos, la agregación horaria y el comportamiento de los modelos. Las pruebas se encuentran en la carpeta `tests/`.

Para ejecutar todas las pruebas desde la raíz del repositorio:

```bash
python -m pytest tests/ -v
```

Para ejecutar únicamente las pruebas de procesamiento de datos:

```bash
python -m pytest -q
```

Pytest descubre los archivos `tests/test_*.py` desde la raíz. La suite cubre casos normales y límite para datos, features, modelos, evaluación, etiquetas P2 y los reportes de Deepchecks. Incluye pruebas de reproducibilidad: P1 ajusta el candidato dos veces y compara métricas con tiempos excluidos; P2 verifica la semilla fija y compara exactamente las probabilidades entre ejecuciones.

Para repetir las pruebas enfocadas de Deepchecks y reproducibilidad:

```bash
python -m pytest tests/test_validation.py tests/test_modeling.py::test_candidate_evaluation_is_deterministic_and_ignores_test_by_default tests/test_p2_train.py::test_train_objective_uses_balanced_weights_and_explicit_seed tests/test_p2_train.py::test_run_validation_uses_same_rows_and_is_reproducible -v
```

Evidencia local: PyTest ejecutó 126 pruebas y todas pasaron; la salida registró 112 advertencias y 612.39 segundos. La captura muestra el resultado de `python -m pytest -q`.

![PyTest: 126 pruebas aprobadas](docs/evidence/pytest-126-passed.png)

Después de la corrección de compatibilidad de DeepChecks, el comando enfocado anterior ejecutó 10 pruebas y obtuvo 10 aprobadas, 1 advertencia y 19.26 segundos; `python -m pytest --collect-only -q` encontró 126 pruebas en 4.00 segundos. No se repitió la suite completa para evitar volver a ejecutar sus pruebas de MLflow.

### 8.6. Experimentación

Los modelos de regresión, clasificación y clustering se ejecutan mediante MLflow Projects y se registran en el experimento `mibici-equipo28` (sección 7). La búsqueda de hiperparámetros de P1 y P2 y la selección del mejor modelo están en [docs/entrega2_mlflow.md](docs/entrega2_mlflow.md).

## 9. Flujo de colaboración

El equipo utilizará GitHub Flow.

- `main` será la rama principal protegida.
- Cada integrante desarrollará sus cambios en una rama independiente.
- Los cambios se integrarán mediante Pull Requests revisados.
- Cada integrante deberá abrir al menos un Pull Request y revisar al menos uno de otro integrante.
- Los commits seguirán la convención Conventional Commits.

Convención de ramas:

```text
feature/descripcion
fix/descripcion
docs/descripcion
chore/descripcion
```

Ejemplo de commit:

```text
chore: initialize MiBici project structure
```


## 10. Integrantes y responsabilidades

| Integrante | Responsabilidad |
|---|---|
| Iván Fernando García Nuñez | MLOps, seguimiento de experimentos con MLflow e integración del proyecto. |
| Anthony Gerardo Gutarra Sánchez | Modelos de regresión (P1) y clasificación (P2). |
| Miguel Ángel Lara López | Análisis exploratorio y clustering de estaciones (P3) y pruebas unitarias con PyTest|
| Josafat Misael Rodríguez Nieto | Ingeniería de datos, ingesta, validación, procesamiento y arquitectura de datos. |

## 11. Documentación

Los documentos de formulación del problema y diseño de datos están en `references/`. El reporte de la segunda entrega (resultados, selección del mejor modelo, reproducción y evidencias) está en [docs/entrega2_mlflow.md](docs/entrega2_mlflow.md).


## 12. Estado del proyecto

**Ingeniería de datos:** Implementada y validada mediante 24 pruebas automatizadas.

**Datos disponibles:** Doce archivos mensuales de 2025, con 4,532,032 viajes y 371 estaciones observadas.

**Pipeline reproducible:** Implementado mediante cuatro comandos CLI para descarga, preparación, agregación y ejecución integral.

**Modelado:** P1 y P2 tienen baselines y una búsqueda de hiperparámetros de HistGradientBoosting (9 combinaciones) evaluada en validation; P3 tiene K-Means y un baseline por cuantiles. El test (noviembre y diciembre de 2025) se evaluó solo para los `champion` de P1 y P2: [reports/test_evaluation.md](reports/test_evaluation.md).

**MLOps:** Los entry points de MLflow Projects (`ingest`, `train`, `search`, `register`, entre otros) se ejecutan contra un servidor con SQLite. Los modelos mejores de P1, P2 (uno por objetivo) y P3 están en el Model Registry con el alias `champion`.

**Documentación:** Fuentes de datos, pipeline, resultados de la búsqueda, selección del mejor modelo y evidencias en [docs/entrega2_mlflow.md](docs/entrega2_mlflow.md). Pendiente integrar la arquitectura.