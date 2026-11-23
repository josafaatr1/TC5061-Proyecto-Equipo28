# TC5061-Proyecto-Equipo28

[![CI - MiBici](https://github.com/josafaatr1/TC5061-Proyecto-Equipo28/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/josafaatr1/TC5061-Proyecto-Equipo28/actions/workflows/ci.yml)

**Proyecto MiBici: propuesta y base reproducible**

Tecnológico de Monterrey | TC5061 — MLOps  
Equipo 28

> **Inicio rápido (Entregable 7).** Con los modelos P2 exportados (sección 14.2), `docker compose up -d --build --wait` levanta la API y MLflow con un solo comando. Pipeline de CI/CD, pruebas de carga y análisis FinOps: [sección 16](#16-operación-del-sistema-entregable-7).

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
├── .github/workflows/
│   └── ci.yml            # Pipeline de CI/CD: pruebas, Ruff y construcción de la imagen
├── data/                 # Datos originales, intermedios y procesados
├── docs/                 # Reportes de entregas, guía de Docker Compose y evidencias
├── mibici/               # Código fuente del proyecto
│   ├── clustering.py     # Agrupamiento de estaciones (P3)
│   ├── config.py         # Configuración centralizada
│   ├── dataset.py        # Ingesta y preparación de datos
│   ├── features.py       # Ingeniería de características
│   ├── finops.py         # Costo por inferencia con capacidad medida y precios de AWS
│   ├── tracking.py       # Registro del pipeline de datos en MLflow
│   ├── validation.py     # Integridad y drift de P1/P2 con Deepchecks
│   ├── workflow.py       # Encadena los entry points bajo un run padre
│   ├── api/              # Servicio REST P2 y exportación de modelos
│   ├── monitoring/       # Reportes de drift y calidad de datos con Evidently
│   └── modeling/         # Modelos P1 y P2, runs de MLflow, búsqueda y Model Registry
├── models/               # Modelos generados localmente
├── notebooks/            # Exploración de datos
├── references/           # Manifiestos de datos, handoff P1/P2, auditoría del catálogo y precios de AWS
├── reports/
│   ├── figures/          # Figuras y visualizaciones
│   └── finops/           # Análisis FinOps generado (reporte, resumen JSON y gráficas)
├── tests/                # Pruebas (misma estructura que mibici/)
│   ├── conftest.py       # Fixtures compartidas (tracking store temporal de MLflow)
│   ├── factories.py      # Datos sintéticos reutilizados por varias pruebas
│   ├── test_*.py         # Pruebas de mibici/*.py (dataset, features, clustering, finops, ...)
│   ├── api/              # Pruebas de la API REST
│   ├── load/             # Pruebas de carga y capacidad con Locust, monitoreo y resultados
│   ├── modeling/         # Pruebas de mibici/modeling/*.py
│   └── monitoring/       # Pruebas de mibici/monitoring/*.py
├── .gitignore            # Exclusiones de Git
├── Containerfile         # Imagen de la API (compatible con Docker y Podman)
├── docker-compose.yml    # Orquestación de la API y MLflow con límites de recursos
├── Makefile              # Atajos para MLflow, imagen, Compose, prueba de capacidad y FinOps
├── MLproject             # Entry points de MLflow Projects
├── pyproject.toml        # Configuración del paquete
├── python_env.yaml       # Entorno de MLflow Projects
├── requirements.txt      # Dependencias directas
├── requirements-api.txt  # Dependencias de la imagen de la API
├── requirements-load.txt # Dependencias de las pruebas de carga (Locust)
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

El proyecto utiliza PyTest para verificar el procesamiento de datos, la agregación horaria y el comportamiento de los modelos. Todas las pruebas están en la carpeta `tests/`, organizada igual que el paquete `mibici/`: cada módulo tiene su archivo `test_<módulo>.py`.

```text
tests/
├── conftest.py           # Fixtures compartidas
├── factories.py          # Datos sintéticos compartidos
├── test_clustering.py    # mibici/clustering.py
├── test_dataset.py       # mibici/dataset.py
├── test_features.py      # mibici/features.py
├── test_finops.py        # mibici/finops.py
├── test_monitor_resources.py  # tests/load/monitor_resources.py (Docker y Podman)
├── test_tracking.py      # mibici/tracking.py
├── test_validation.py    # mibici/validation.py
├── api/
│   └── test_main.py          # mibici/api/main.py
├── load/                     # Escenarios de Locust (no son pruebas de pytest; sección 16.4)
├── monitoring/
│   └── test_drift_report.py  # mibici/monitoring/drift_report.py
└── modeling/
    ├── test_evaluate.py      # mibici/modeling/evaluate.py
    ├── test_mlflow_runs.py   # mibici/modeling/mlflow_runs.py
    ├── test_p2_catalog.py    # mibici/modeling/p2_catalog.py
    ├── test_p2_targets.py    # mibici/modeling/p2_targets.py
    ├── test_p2_train.py      # mibici/modeling/p2_train.py
    ├── test_registry.py      # mibici/modeling/registry.py
    └── test_train.py         # mibici/modeling/train.py
```

Para ejecutar todas las pruebas, desde la raíz del repositorio, basta un solo comando (la configuración está en `pyproject.toml`):

```bash
pytest          # o: make test
```

Pytest descubre los archivos `test_*.py` dentro de `tests/` y sus subcarpetas. La suite cubre casos normales y límite para datos, features, modelos, evaluación, etiquetas P2 y los reportes de Deepchecks. Incluye pruebas de reproducibilidad: P1 ajusta el candidato dos veces y compara métricas con tiempos excluidos; P2 verifica la semilla fija y compara exactamente las probabilidades entre ejecuciones.

Para ejecutar solo una parte de la suite:

```bash
pytest tests/test_dataset.py -v      # procesamiento de datos
pytest tests/test_validation.py -v   # reportes de Deepchecks
pytest tests/modeling -v             # modelado (P1, P2, MLflow y Model Registry)
```

Para repetir las pruebas enfocadas de Deepchecks y reproducibilidad:

```bash
pytest tests/test_validation.py tests/modeling/test_train.py::test_candidate_evaluation_is_deterministic_and_ignores_test_by_default tests/modeling/test_p2_train.py::test_train_objective_uses_balanced_weights_and_explicit_seed tests/modeling/test_p2_train.py::test_run_validation_uses_same_rows_and_is_reproducible -v
```

Evidencia local: con la estructura actual de `tests/`, PyTest ejecutó 131 pruebas y todas pasaron; la salida registró 112 advertencias (casi todas de MLflow, scikit-learn y Deepchecks) y 239.59 segundos. La captura muestra el resultado de `python -m pytest -q`; se omite la parte intermedia del resumen de advertencias y la ruta del entorno virtual aparece abreviada como `$VENV`.

![PyTest: 131 pruebas aprobadas](docs/evidence/pytest-131-passed.png)

El comando enfocado de Deepchecks y reproducibilidad ejecutó 10 pruebas y obtuvo 10 aprobadas, con 1 advertencia, en 15.77 segundos.

### 8.6. Chequeos de calidad y seguridad

Además de las pruebas, el repositorio tiene tres chequeos que se ejecutan desde la raíz del repositorio con el entorno activado:

| Chequeo | Comando | Qué revisa | Detalle y evidencia |
|---|---|---|---|
| Pruebas | `pytest` (o `make test`) | Toda la suite de `tests/` | Sección 8.5 |
| Estilo y lint | `make lint` | Formato (`ruff format --check`) y reglas de Ruff sobre `mibici/` | Esta sección |
| Calidad de datos | `python -m mibici.validation --output-dir reports/deepchecks` | Integridad y drift de P1 y P2 con Deepchecks | Sección 7.8 |
| Seguridad | `python -m bandit -r mibici/ -f txt` | Análisis estático del código Python con Bandit | Sección 13 |

Ruff ya viene en `requirements.txt`. `make lint` solo revisa; si falla por formato, `make format` aplica las correcciones automáticas. Evidencia de una ejecución correcta:

![make lint: 18 archivos con formato correcto y todas las reglas de Ruff aprobadas](docs/evidence/ruff-lint-passed.png)

### 8.7. Experimentación

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

Las aportaciones de cada integrante al Entregable 7 (Compose, CI/CD, pruebas de carga y FinOps) están en la [sección 16](#16-operación-del-sistema-entregable-7).

## 11. Documentación

Los documentos de formulación del problema y diseño de datos están en `references/`. El reporte de la segunda entrega (resultados, selección del mejor modelo, reproducción y evidencias) está en [docs/entrega2_mlflow.md](docs/entrega2_mlflow.md).

El despliegue REST se describe en la sección 14; la auditoría y los resultados de integración están en [docs/api_validation.md](docs/api_validation.md). La integración histórica de inferencia P1 se conserva en [docs/mlflow_model_serving.md](docs/mlflow_model_serving.md).

La operación del sistema (Entregable 7) se describe en la sección 16, con tres documentos de detalle: la [guía de Docker Compose](docs/compose_operations.md), las [pruebas de carga y capacidad](tests/load/README.md) y el [análisis FinOps](reports/finops/finops_report.md).


## 12. Estado del proyecto

**Ingeniería de datos:** Implementada y validada mediante 24 pruebas automatizadas.

**Datos disponibles:** Doce archivos mensuales de 2025, con 4,532,032 viajes y 371 estaciones observadas.

**Pipeline reproducible:** Implementado mediante cuatro comandos CLI para descarga, preparación, agregación y ejecución integral.

**Modelado:** P1 y P2 tienen baselines y una búsqueda de hiperparámetros de HistGradientBoosting (9 combinaciones) evaluada en validation; P3 tiene K-Means y un baseline por cuantiles. El test (noviembre y diciembre de 2025) se evaluó solo para los `champion` de P1 y P2: [reports/test_evaluation.md](reports/test_evaluation.md).

**MLOps:** Los entry points de MLflow Projects (`ingest`, `train`, `search`, `register`, entre otros) se ejecutan contra un servidor con SQLite. Los modelos mejores de P1, P2 (uno por objetivo) y P3 están en el Model Registry con el alias `champion`.

**Despliegue y monitoreo:** API REST P2 en contenedor (sección 14) y reportes de drift y calidad con Evidently (sección 15).

**Operación:** Docker Compose levanta la API y MLflow con un solo comando y límites de recursos; GitHub Actions ejecuta pruebas, Ruff y la construcción de la imagen en cada push y pull request; Locust mide desempeño y capacidad, y el análisis FinOps estima el costo por inferencia con precios de AWS (sección 16).

**Documentación:** Fuentes de datos, pipeline, resultados de la búsqueda, selección del mejor modelo y evidencias en [docs/entrega2_mlflow.md](docs/entrega2_mlflow.md); arquitectura de operación en la sección 16.

## 13. Auditoría de seguridad

El proyecto incluye una revisión básica de credenciales y archivos sensibles, así como un análisis estático del código Python mediante Bandit.

Para ejecutar el análisis desde la raíz del repositorio:

```bash
python -m pip install bandit==1.9.4
python -m bandit -r mibici/ -f txt
```

Bandit termina con código de salida 1 cuando reporta hallazgos, aunque el análisis se haya completado correctamente.

El análisis más reciente (que ya incluye `mibici/validation.py`) examinó 4,550 líneas de código y reportó tres hallazgos de severidad baja, sin hallazgos medios ni altos. Los tres corresponden a la llamada a `git rev-parse HEAD` en `mibici/modeling/mlflow_runs.py` (B404, B603 y B607). Los resultados y su evaluación se documentan en [reports/security/README.md](reports/security/README.md).

#### Evidencia de ejecución local

![Bandit: 4,550 líneas analizadas, 3 hallazgos de severidad baja y ninguno medio o alto](docs/evidence/bandit-results.png)

La salida completa se encuentra en [reports/security/bandit_report.txt](reports/security/bandit_report.txt).

## 14. API REST P2 en contenedor

### 14.1. Propósito y arquitectura

FastAPI recibe los 15 predictores históricos de una estación-hora y devuelve dos clasificaciones de presión proxy: falta de bicicletas (`bike`) y falta de anclajes (`dock`). No mide inventarios reales ni calcula los predictores a partir de una estación y una fecha.

```text
MLflow Registry -> export_models -> models/ + metadata.json -> imagen OCI
Cliente HTTP -> FastAPI -> modelos P2 cargados al iniciar -> JSON
```

Los modelos se exportan una vez y se empaquetan en la imagen. El contenedor no necesita contactar MLflow durante la inferencia. Un cambio del alias requiere volver a exportar, construir y desplegar; no actualiza contenedores existentes.

Con los modelos ya exportados, `docker compose up -d --build --wait` inicia API y MLflow juntos (sección 16.2). Arquitectura, operaciones PowerShell, persistencia y pruebas: [guía de Docker Compose](docs/compose_operations.md).

La inferencia por solicitud permite consultar una estación-hora sin esperar un proceso por lotes. La construcción de variables históricas sigue siendo responsabilidad del consumidor; para consultas masivas puede resultar más apropiado un proceso batch. La latencia y la capacidad se midieron localmente con Locust (sección 16.4); no son mediciones de un entorno de producción.

| Objetivo | URI de origen | Carpeta exportada | Versión local verificada |
|---|---|---|---|
| Bicicletas | `models:/mibici-p2-bike-pressure@champion` | `models/p2-bike-pressure` | 1 |
| Anclajes | `models:/mibici-p2-dock-pressure@champion` | `models/p2-dock-pressure` | 1 |

Las versiones no están fijadas en código: [mibici/api/export_models.py](mibici/api/export_models.py) resuelve el alias y descarga esa versión explícita. `models/metadata.json` conserva nombre, alias, versión y run de origen. Los modelos y la base de datos están excluidos de Git; clonar el repositorio no los proporciona.

### 14.2. Prerrequisitos y exportación

Se requiere Python 3.14, el entorno de desarrollo de la sección 6, acceso al Registry y sus artefactos, y Podman o Docker con motor Linux activo. GNU Make es opcional: debajo se incluyen comandos equivalentes. Las dependencias de servicio están fijadas en [requirements-api.txt](requirements-api.txt); la imagen las instala automáticamente, sin incluir Jupyter, datos ni herramientas de entrenamiento.

Para usar la API localmente, instalar también sus dependencias en un entorno virtual, nunca globalmente:

```bash
python -m pip install -r requirements-api.txt
```

Para las pruebas se necesitan `pytest` y `httpx` en el entorno de desarrollo (`python -m pip install pytest==9.1.1 httpx==0.28.1` si faltan). El servidor de Registry necesita MLflow completo, incluido en [requirements.txt](requirements.txt), no únicamente `mlflow-skinny`.

Con el servidor de la sección 7.2 activo, configurar en la terminal de exportación:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

En PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:PYTHONUTF8 = "1"
```

Desde la raíz, con el entorno activado:

```bash
make export-models
# Equivalente cuando no hay make:
python -m mibici.api.export_models
```

Comprobar que existen `models/metadata.json`, `models/p2-bike-pressure/MLmodel`, `models/p2-dock-pressure/MLmodel` y sus respectivos `model.skops`. No sustituirlos por archivos con nombres inventados ni ejecutar el registro para cambiar versiones como parte del despliegue. Cargar únicamente modelos de un Registry de confianza.

### 14.3. Construcción y ejecución

Docker es el motor principal del proyecto: lo usan el pipeline de CI, Docker Compose y las pruebas de carga.

```bash
make image
make container-run
```

Sin Make, con el puerto limitado a localhost:

```bash
docker build -t mibici-api -f Containerfile .
docker run --rm -p 127.0.0.1:8000:8000 --name mibici-api mibici-api
```

El target `container-run` publica `8000:8000` en todas las interfaces; los comandos explícitos limitan la exposición local.

Alternativa con Podman:

```bash
make image CONTAINER_ENGINE=podman
make container-run CONTAINER_ENGINE=podman
# Sin Make:
podman build --format docker -t mibici-api -f Containerfile .
podman run --rm -p 127.0.0.1:8000:8000 --name mibici-api mibici-api
```

`--format docker` conserva el `HEALTHCHECK` en Podman; el Makefile lo agrega automáticamente con `CONTAINER_ENGINE=podman`.

La ejecución queda en primer plano; realizar las consultas desde otra terminal. Para segundo plano añadir `-d`. Detener con `podman stop mibici-api` o `docker stop mibici-api`, según el motor utilizado.

El [Containerfile](Containerfile) conserva la base fijada por digest, el usuario `app` (UID 10001), el puerto 8000 y un healthcheck que utiliza `PORT`. [.containerignore](.containerignore) y [.dockerignore](.dockerignore) excluyen datos, credenciales, cachés y modelos ajenos al servicio. No hacen falta montajes de la base de datos ni credenciales MLflow dentro del contenedor.

### 14.4. Consultas y contrato

Documentación interactiva: http://127.0.0.1:8000/docs. OpenAPI: http://127.0.0.1:8000/openapi.json.

Los comandos siguientes son de una sola línea. En PowerShell utilizar `curl.exe` en lugar de `curl` para evitar el alias de Windows PowerShell.

```bash
curl -sS -i http://127.0.0.1:8000/health
```

Respuesta verificada, HTTP 200:

```json
{"status":"healthy","models_loaded":["bike","dock"]}
```

Entrada completa disponible en [docs/examples/p2-request.json](docs/examples/p2-request.json). Es una entrada sintética de prueba, no una observación histórica:

```json
{
    "hour_of_day": 8,
    "day_of_week": 2,
    "month": 11,
    "is_weekend": false,
    "departures_lag_1h": 1.0,
    "departures_lag_24h": 2.0,
    "departures_lag_168h": 1.0,
    "arrivals_lag_1h": 1.0,
    "arrivals_lag_24h": 2.0,
    "arrivals_lag_168h": 1.0,
    "net_flow_lag_1h": -1.0,
    "net_flow_lag_24h": 0.0,
    "net_flow_lag_168h": 1.0,
    "departures_rolling_mean_24h": 1.5,
    "departures_rolling_mean_168h": 1.2
}
```

```bash
curl -sS -i -H "Content-Type: application/json" --data-binary "@docs/examples/p2-request.json" http://127.0.0.1:8000/predict
```

Respuesta real obtenida del contenedor con las versiones 1 exportadas:

```json
{
    "predictions": {
        "bike": {"prediction": 0, "probability": 0.3171913481120226},
        "dock": {"prediction": 1, "probability": 0.5846188145068164}
    },
    "decision_threshold": 0.5
}
```

`probability` es `P(clase=1)` de cada clasificador, no la confianza de la clase predicha ni una probabilidad de inventario real. No se promete calibración. `prediction` conserva `model.predict()`, con umbral nominal 0.5: para estos HGB binarios, el empate exacto devuelve 0. Las métricas históricas P2 utilizan `>= 0.5`; esta diferencia en empates se documenta sin cambiar la API ni reescribir métricas anteriores.

El esquema acepta enteros para hora (0-23), día (0-6, lunes=0) y mes (1-12), y un booleano JSON para `is_weekend`. Los once predictores restantes son números finitos representables como `float32`; solo los tres flujos netos pueden ser negativos. Todas las variables son obligatorias. Se rechazan `null`, campos extra, conversiones desde cadenas, booleanos en campos numéricos, NaN e Infinity con HTTP 422.

Los modelos P2 registrados tienen firma tensorial `float32`, forma `[-1, 15]`. La API ordena con `PREDICTOR_COLUMNS`; no usa el orden de las claves JSON. Ambos modelos se cargan una vez durante el arranque. Si falla la carga, el servicio no inicia; si falta un modelo durante la operación, `/health` y `/predict` responden 503. Una salida inválida o un fallo de inferencia devuelve 500 sin detalles internos.

Para comprobar un 422 desde PowerShell usando la misma entrada completa:

```powershell
$invalid = Get-Content docs/examples/p2-request.json -Raw | ConvertFrom-Json
$invalid.hour_of_day = 24
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body ($invalid | ConvertTo-Json)
```

El error HTTP 422 es el resultado esperado. En Swagger, ejecutar la entrada válida y luego cambiar `hour_of_day` a 24 permite ver ambos estados.

### 14.5. Pruebas y evidencias

```bash
python -m pytest -q
python -m pytest tests/api/test_main.py -q
make lint
python -m pip check
```

Sin Make, `python -m ruff format --check` y `python -m ruff check` son los dos chequeos de `make lint`. Para incluir explícitamente la nueva suite: `python -m ruff check tests/api --config 'include=["*.py"]'`.

Las pruebas unitarias usan `TestClient`, fixtures y mocks del cargador, sin Registry real. Las dos integraciones externas se omiten por defecto. Con Registry activo, modelos exportados y contenedor funcionando, habilitarlas en PowerShell:

```powershell
$env:MIBICI_API_INTEGRATION = "1"
$env:MIBICI_API_CONTAINER = "1"
python -m pytest tests/api/test_main.py -q -k "real_registry_export or live_container"
Remove-Item Env:MIBICI_API_INTEGRATION, Env:MIBICI_API_CONTAINER
```

En Bash:

```bash
MIBICI_API_INTEGRATION=1 MIBICI_API_CONTAINER=1 python -m pytest tests/api/test_main.py -q -k 'real_registry_export or live_container'
```

La prueba Registry compara alias, versión, run, firma, orden, umbral y probabilidades de los modelos cargados y exportados. La prueba HTTP verifica el contenedor en `MIBICI_API_URL` (por defecto `http://127.0.0.1:8000`). No se presentan mocks como prueba de extremo a extremo.

Resultado global: **279 aprobadas y 2 integraciones omitidas**, más **2 integraciones reales aprobadas por separado**. Los detalles, advertencias, limitaciones y matriz de requisitos están en [docs/api_validation.md](docs/api_validation.md).

Captura auténtica de Swagger ejecutando `POST /predict` contra el contenedor Docker, con HTTP 200 y ambos modelos. No es una respuesta simulada:

![Predicción HTTP 200 de la API P2 ejecutada en Docker](docs/evidence/api-container-prediction.png)

### 14.6. Problemas frecuentes

| Problema | Acción |
|---|---|
| Registry inaccesible o alias ausente | Revisar `MLFLOW_TRACKING_URI`, servidor, nombre y alias; obtener acceso al Registry del equipo. No inventar versiones ni registrar otro modelo para ocultar el problema. |
| Faltan `MLmodel` o `model.skops` | Repetir la exportación con artefactos accesibles y reconstruir. Los modelos no vienen del clon de Git. |
| Docker no conecta con el motor | Iniciar Docker Desktop en modo Linux (`docker desktop start`, si está disponible) y verificar `docker info`. |
| Podman en Windows no conecta | Verificar que la máquina de Podman esté iniciada con `podman machine list` y `podman machine start`. |
| `make` no existe | Usar los comandos Python y Podman/Docker explícitos anteriores. |
| Puerto o nombre ocupado | Detener solo el contenedor propio o usar otro nombre y `-p 127.0.0.1:8001:8000`; ajustar la URL y `MIBICI_API_URL`. |
| Incompatibilidad de scikit-learn/skops | Usar las versiones fijadas para la imagen y revisar las registradas en `MLmodel`; no actualizar el modelo entrenado durante el despliegue. |
| HTTP 422 | Revisar tipos JSON, campos obligatorios, valores finitos y rangos en `/docs`; no enviar IDs ni datos personales. |
| Swagger sin interfaz | `/docs` utiliza recursos CDN; revisar acceso de red. `/openapi.json` y las consultas HTTP no necesitan esos recursos externos. |

Este despliegue es una demostración local sin autenticación ni TLS. No exponerlo públicamente sin controles adicionales; no se afirma que sea un servicio de producción.

## 15. Monitoreo de Data Drift y Data Quality

El módulo `mibici/monitoring/drift_report.py` usa Evidently 0.7.23 para comparar los datos con los que se entrenó el modelo P2 contra un periodo posterior que representa producción. La interpretación completa de los resultados está en [docs/drift_interpretation.md](docs/drift_interpretation.md).

### 15.1. Conjuntos de datos

| Conjunto | Periodo | Filas | Origen |
|---|---|---:|---|
| Reference | Enero-agosto de 2025 | 2,137,182 | Partición train de `data/processed/p2_targets_2025_train_validation.parquet`, con la que se entrenó el champion de P2 |
| Current | Noviembre-diciembre de 2025 | 536,245 | `build_p2_test_dataset()` sobre `station_hourly_2025_complete.parquet`, con etiquetas calculadas con los umbrales de presión de train |

Septiembre-octubre no se usa como current porque es la partición validation con la que se eligió el champion. Noviembre-diciembre es el primer periodo que el modelo no vio en entrenamiento ni en selección.

Se comparan las 17 columnas que importan al modelo: sus 15 variables de entrada (`PREDICTOR_COLUMNS`: calendario, rezagos y medias móviles) y las etiquetas `target_bike_pressure` y `target_dock_pressure`. Las variables de calendario y las etiquetas se tratan como categóricas; los rezagos y las medias móviles, como numéricas. Evidently elige la prueba de cada variable (Wasserstein o Jensen-Shannon, umbral 0.1), salvo en las etiquetas, que usan la prueba Z de proporciones (p < 0.05) porque Jensen-Shannon casi no reacciona a cambios de prevalencia en clases tan desbalanceadas.

### 15.2. Ejecución

Requiere los Parquets de `ingest` y `p2_targets` y el catálogo histórico de septiembre de 2026 con su archivo de procedencia. El catálogo predeterminado del proyecto ahora es el de octubre y no reproduce estos resultados. Antes de generar los reportes se verifica que las etiquetas de reference se reproduzcan exactamente con esos archivos (`verify_label_reproducibility`), para que ambos periodos usen los mismos umbrales de presión; si no coinciden, el comando falla. Evidently ya está en `requirements.txt`. Desde la raíz del repositorio:

```bash
python -m mibici.monitoring.drift_report --catalog-path data/external/nomenclatura_2026_09.csv
```

Tarda alrededor de un minuto y escribe en `reports/evidently/`:

| Archivo | Contenido |
|---|---|
| `data_drift_report.html` | Drift por variable: distribución reference contra current, prueba estadística y puntaje; la pestaña *Tests* muestra la decisión de drift de cada variable |
| `data_quality_report.html` | Resumen de calidad de ambos periodos: faltantes, columnas constantes o duplicadas y distribución de cada variable |
| `drift_summary.json` | Periodos, filas, umbrales de presión, prevalencia de las etiquetas y resultado de drift por variable (la misma decisión que muestra el HTML) |

Los HTML se abren directamente en el navegador. Las pruebas del módulo están en `tests/monitoring/test_drift_report.py`.

### 15.3. Resultados

![Resumen de Data Drift de Evidently: 3 de 17 variables con drift](docs/evidence/evidently-data-drift.png)

- **Drift significativo en 3 de 17 variables.** El encabezado "Dataset Drift is NOT detected" solo indica que no se alcanza el criterio global por defecto de Evidently (50 % de las variables).
- **`month`** (Jensen-Shannon 0.83): train contiene los meses 1-8 y current los meses 11-12, valores que el modelo nunca vio.
- Los rezagos y las medias móviles, la señal principal del modelo, no superan el umbral (Wasserstein ≤ 0.06).
- **Las dos etiquetas** (prueba Z, p < 0.001): la prevalencia de presión de bicicletas baja de 4.34 % a 3.09 % (−29 %) y la de anclajes de 4.10 % a 3.43 % (−16 %). La menor prevalencia puede contribuir a la caída de precision y AP, pero no demuestra su causa; las ventanas solapadas también limitan la interpretación del p-value.
- Calidad: sin columnas vacías, constantes ni duplicadas; los únicos faltantes son las primeras horas sin historia de los rezagos.

La conclusión y las recomendaciones (reentrenar incluyendo meses recientes y revisar el umbral de decisión) están en [docs/drift_interpretation.md](docs/drift_interpretation.md).

La comparación retrospectiva entre el champion y otra variante HGB, sus limitaciones y los comandos reproducibles están en [docs/ab_testing_summary.md](docs/ab_testing_summary.md). Los insumos y su estado frente a la rúbrica están en [docs/deliverable8_audit_summary.md](docs/deliverable8_audit_summary.md).

## 16. Operación del sistema (Entregable 7)

La API contenerizada (sección 14) y los modelos registrados en MLflow (sección 7) se integran en un sistema que se levanta con un solo comando, se prueba y construye automáticamente en cada cambio, y se analiza bajo carga y desde la perspectiva de costo.

| Componente | Archivos | Integrante | Pull Request |
|---|---|---|---|
| Orquestación con Docker Compose | `docker-compose.yml`, [docs/compose_operations.md](docs/compose_operations.md) | Anthony Gerardo Gutarra Sánchez | #31 |
| Pipeline de CI/CD | `.github/workflows/ci.yml` | Miguel Ángel Lara López | #29 |
| Pruebas de carga y monitoreo de recursos | `tests/load/`, `requirements-load.txt` | Josafat Misael Rodríguez Nieto | #30 |
| Prueba de capacidad, límites de recursos, análisis FinOps e integración del README | `mibici/finops.py`, `tests/load/locustfile_capacity.py`, `reports/finops/`, `docker-compose.yml` (límites) | Iván Fernando García Nuñez | `feature/finops-integracion` |

### 16.1. Arquitectura

```text
docker compose up -d --build --wait
│
├── api     FastAPI + modelos P2 empaquetados en la imagen   127.0.0.1:8000   1 vCPU, 512 MiB
│           (healthcheck /health; arranca cuando mlflow está saludable)
│
└── mlflow  servidor de tracking y artefactos (SQLite)       127.0.0.1:5000   1 vCPU, 2.5 GiB
            └── volumen mibici-compose_mlflow-data (base de datos y artefactos persistentes)

red bridge "mibici": la API resuelve http://mlflow:5000 por nombre
```

La API recibe `MLFLOW_TRACKING_URI`, pero **la inferencia no consulta MLflow**: los modelos se exportan del Registry y se empaquetan en la imagen (sección 14.2). Si MLflow se detiene, `/predict` sigue respondiendo. Ambos servicios eliminan todas las capabilities, usan `no-new-privileges`, publican sus puertos solo en `127.0.0.1` y limitan sus logs.

Los límites de recursos permiten que ambos servicios quepan en una instancia de 2 vCPU y 4 GiB, la referencia del análisis FinOps. MLflow usa entre 1.2 y 1.7 GiB en reposo, según la ejecución, y llegó a 1.8 GiB durante las pruebas; por eso su límite es de 2.5 GiB.

### 16.2. Levantar el sistema con un solo comando

Requisitos: Docker Engine o Docker Desktop con Docker Compose v2 y los modelos P2 exportados en `models/` (sección 14.2; el clon de Git no los incluye).

```bash
docker compose up -d --build --wait     # o: make compose-up
```

`--wait` regresa cuando ambos servicios están saludables. Verificación:

```bash
docker compose ps
curl -sS http://127.0.0.1:8000/health
curl -sS -H "Content-Type: application/json" --data-binary "@docs/examples/p2-request.json" http://127.0.0.1:8000/predict
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/
docker compose stats --no-stream
```

Respuesta esperada de `/health`: `{"status":"healthy","models_loaded":["bike","dock"]}`. La interfaz de MLflow queda en http://127.0.0.1:5000 y la documentación de la API en http://127.0.0.1:8000/docs. Los comandos equivalentes en PowerShell están en la [guía de Docker Compose](docs/compose_operations.md).

Para detener el sistema:

```bash
docker compose down                     # o: make compose-down
```

`down` conserva el volumen de MLflow. No usar `docker compose down -v`, porque borra la base de datos y los artefactos.

Los límites y los procesos de la API se ajustan con variables de entorno, sin editar el archivo:

| Variable | Valor por defecto | Uso |
|---|---|---|
| `API_CPUS` | `1.0` | vCPU de la API |
| `API_MEMORY` | `512M` | Memoria de la API |
| `API_WORKERS` | `1` | Procesos de uvicorn (`WEB_CONCURRENCY`) |
| `MLFLOW_CPUS` | `1.0` | vCPU de MLflow |
| `MLFLOW_MEMORY` | `2560M` | Memoria de MLflow |

Por ejemplo, la configuración con más capacidad medida (sección 16.4):

```bash
API_CPUS=2 API_MEMORY=1G API_WORKERS=2 docker compose up -d --wait api
```

`OMP_NUM_THREADS=1` está fijo en la API: con más de un proceso evita que cada uno abra un hilo de OpenMP por núcleo del host.

Se validó con Docker Compose v5.1.3 sobre Docker 29.4.1. El archivo sigue la especificación de Compose, así que `podman compose` es la alternativa con Podman, aunque no se validó en esta entrega.

### 16.3. Pipeline de CI/CD

El workflow [.github/workflows/ci.yml](.github/workflows/ci.yml) se ejecuta en cada push a `main` o a ramas `feature/**` y en cada pull request hacia `main`. Un solo job en `ubuntu-latest`:

| Paso | Qué hace |
|---|---|
| Checkout y Python 3.14 | Descarga el código e instala Python con caché de pip ligada a `requirements.txt` |
| Dependencias | `pip install -r requirements.txt` |
| Pruebas | `python -m pytest -q` (toda la suite de `tests/`) |
| Formato | `ruff format --check` |
| Lint | `ruff check` |
| Imagen | `docker build -t mibici-api -f Containerfile .` |

El estado se ve en el badge del inicio de este README, en la pestaña **Actions** del repositorio y en los *checks* de cada pull request. Con la CLI de GitHub:

```bash
gh run list --workflow ci.yml --limit 5
gh run watch                            # sigue la ejecución en curso
```

Para reproducir el pipeline localmente antes de hacer push, con el entorno de la sección 6:

```bash
python -m pytest -q
ruff format --check && ruff check       # o: make lint
docker build -t mibici-api -f Containerfile .
```

Alcance: Ruff revisa `mibici/` (configuración en `pyproject.toml`). La imagen se construye sin modelos, porque `models/` solo contiene `.gitkeep` en Git: el paso valida el Containerfile y las dependencias de la API, no la inferencia. La inferencia real se valida localmente con Compose (sección 16.2).

### 16.4. Pruebas de carga y monitoreo de recursos

Instalar las dependencias de carga en el entorno de desarrollo:

```bash
python -m pip install -r requirements-load.txt
```

Hay dos tipos de prueba, ambas sobre `POST /predict` con validación del contenido de la respuesta:

| Prueba | Archivo | Usuarios | Propósito |
|---|---|---|---|
| Escenarios de uso | `tests/load/locustfile.py` | 1, 5, 10 y 20, con 1-2 s de espera entre peticiones | Desempeño con un patrón de uso parecido al de una app |
| Capacidad | `tests/load/locustfile_capacity.py` | Escalones de 1 a 32, sin espera | Capacidad máxima sostenible, insumo del análisis FinOps |

Ejemplo de un escenario de uso, con la API levantada y el monitoreo en otra terminal:

```bash
python tests/load/monitor_resources.py --container "$(docker compose ps -q api)" \
  --output tests/load/results/<prefijo>_resources.csv
locust -f tests/load/locustfile.py --headless --host http://127.0.0.1:8000 \
  -u 20 -r 20 -t 60s --csv tests/load/results/<prefijo> --only-summary
python tests/load/plot_results.py       # gráficas de latencia, throughput, CPU, memoria y errores
```

Prueba de capacidad, que también monitorea la API y MLflow:

```bash
make capacity-test CAPACITY_NAME=<prefijo>
```

Con Podman, el monitor se ejecuta con `--engine podman`. Todos los comandos, la metodología y la interpretación completa están en [tests/load/README.md](tests/load/README.md).

**Escenarios de uso (sin límites de recursos, PR #30):**

| Usuarios | Peticiones | Errores | Throughput | Latencia promedio | p95 | CPU promedio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39 | 0 | 0.67 req/s | 12.7 ms | 18 ms | 5.8 % |
| 5 | 199 | 0 | 3.39 req/s | 24.5 ms | 76 ms | 25.5 % |
| 10 | 382 | 0 | 6.48 req/s | 48.0 ms | 91 ms | 52.7 % |
| 20 | 747 | 0 | 12.67 req/s | 89.7 ms | 130 ms | 114.7 % |

El throughput crece en proporción a los usuarios porque lo limita su tiempo de espera, no la API. La latencia sí aumenta con la concurrencia y la memoria se mantiene estable (~215 MiB).

| Latencia | Throughput | Tasa de error |
|---|---|---|
| ![Latencia](tests/load/results/graphs/01_response_latency.png) | ![Throughput](tests/load/results/graphs/02_throughput.png) | ![Tasa de error](tests/load/results/graphs/05_error_rate.png) |

**Prueba de capacidad (con los límites de Compose):**

| Configuración de la API | Capacidad sostenible (p95 ≤ 300 ms) | Con 16 usuarios |
|---|---|---|
| 1 vCPU, 1 proceso (por defecto) | 71.4 req/s con 1 usuario | 26.5 req/s, p95 920 ms |
| 2 vCPU, 1 proceso | 66.6 req/s con 1 usuario | 31.2 req/s, p95 555 ms |
| 2 vCPU, 2 procesos, `OMP_NUM_THREADS=1` | 95.1 req/s con 16 usuarios | 95.1 req/s, p95 270 ms |

Con un solo proceso, el throughput **baja** al aumentar la concurrencia, porque los hilos compiten por el GIL de Python, y duplicar la CPU no lo corrige. Dos procesos con un hilo de OpenMP cada uno multiplican por 3.6 la capacidad con concurrencia. Ninguna prueba registró errores. MLflow ocupó ~1.8 GiB de memoria con menos del 3 % de CPU.

### 16.5. Análisis FinOps

[mibici/finops.py](mibici/finops.py) combina la capacidad medida, los precios on-demand oficiales de AWS en la región México (Central) ([references/aws_pricing_mx-central-1.json](references/aws_pricing_mx-central-1.json), con fuentes y fechas) y la demanda real de MiBici en 2025. Requiere `data/processed/station_hourly_2025.parquet` (sección 8):

```bash
make finops                             # o: python -m mibici.finops
```

Escribe en `reports/finops/` el [reporte completo](reports/finops/finops_report.md), `finops_summary.json` y dos gráficas. Sus funciones tienen pruebas en `tests/test_finops.py`.

**Costo por inferencia de cada opción de despliegue:**

| Opción | USD/mes | Capacidad | USD por millón de inferencias (al 70 % de capacidad) |
|---|---:|---:|---:|
| Compose actual: API + MLflow en t3.medium | $31.90 | 71.4 req/s | $0.243 |
| API sola optimizada en t3.small | $15.91 | 95.1 req/s | $0.091 |
| API sola optimizada en t4g.small (Graviton) | $12.85 | 95.1 req/s | $0.073 |
| t4g.small apagada de 1:00 a 5:00 | $10.71 | 95.1 req/s | $0.073 |
| Fargate 1 vCPU / 2 GB siempre encendido | $37.84 | 71.4 req/s | $0.288 |

![Costo por opción de despliegue](reports/finops/cost_by_deployment_option.png)

**Con la demanda real el costo lo domina la capacidad ociosa.** Una app con 3 consultas por viaje generaría ~1.13 millones de inferencias al mes, con un pico de 1.5 req/s. Una sola instancia basta y opera al 0.5 % de su capacidad, así que cada millón de inferencias cuesta ~$9.48 en la opción más barata, unas 184 veces su costo a plena capacidad. Por eso las optimizaciones con más impacto reducen horas encendidas y tamaño de instancia:

| Optimización | Ahorro | Justificación con datos |
|---|---:|---|
| Separar MLflow de la API | 50.1 % | MLflow ocupa ~1.8 GiB y no participa en la inferencia; la API sola cabe en 2 GiB (t3.medium → t3.small) |
| Graviton (ARM) | 19.3 % | t4g.small cuesta menos que t3.small con los mismos 2 vCPU y 2 GiB; requiere construir y validar la imagen arm64 |
| Apagar en horas sin servicio | 16.7 % | MiBici no registra viajes de 1:00 a 5:00 |
| Las tres combinadas | 66.4 % | De $31.90 a $10.71 al mes |
| Lote horario en Fargate efímero | 98.6 % frente a Fargate siempre encendido | Las 371 predicciones de cada hora tardan ~35 s; se cobran 60 s por corrida: $0.53 al mes |
| 2 procesos con `OMP_NUM_THREADS=1` | 3.6 veces más capacidad con concurrencia | Pospone la necesidad de réplicas cuando la demanda crezca |

Supuestos y limitaciones: las capacidades se midieron en una laptop (Docker Desktop, AMD Ryzen 7 3700U), no en AWS. t3 y t4g son instancias burstable: sostienen el 20 % de cada vCPU sin consumir créditos, suficiente para la utilización real (< 10 %) pero no para operar a plena capacidad. La región México (Central) cuesta ~5 % más que US East (N. Virginia), a cambio de menor latencia para usuarios de Guadalajara. No se incluyen transferencia de datos, almacenamiento ni balanceador de carga.

### 16.6. Reporte

El reporte en PDF del Entregable 7, `TC5061_Operacion_FinOps_Equipo28.pdf`, con las gráficas de Locust y el análisis FinOps, se entrega en Canvas.
