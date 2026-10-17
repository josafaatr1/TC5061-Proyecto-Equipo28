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
├── docs/                 # Documentación adicional
├── mibici/               # Código fuente del proyecto
│   ├── config.py         # Configuración centralizada
│   ├── dataset.py        # Ingesta y preparación de datos
│   ├── features.py       # Ingeniería de características
│   ├── modeling/         # Entrenamiento y predicción
│   └── plots.py          # Visualizaciones
├── models/               # Modelos generados localmente
├── notebooks/            # Exploración de datos
├── references/           # ML Canvas y arquitectura
├── reports/
│   └── figures/          # Figuras y visualizaciones
├── tests/                # Pruebas del código
├── .gitignore            # Exclusiones de Git
├── pyproject.toml        # Configuración del paquete
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


## 7. Ejecución con MLflow Projects

El archivo `MLproject` define los entry points que MLflow puede ejecutar y `python_env.yaml` fija las versiones de Python y librerías.

### 7.1. Preparar la terminal

En cada terminal nueva, activa el ambiente de la sección 6 y elige el experimento:

```bash
source .venv/bin/activate
export MLFLOW_EXPERIMENT_NAME=mibici-equipo28   # si no, los runs caen en "Default"
```

### 7.2. Correr el proyecto

```bash
mlflow run . -e load --env-manager local
```

MLflow lee el archivo `MLproject`, toma el entry point `load` y ejecuta
`python -m mibici.dataset run-pipeline`, es decir, el pipeline completo de la sección 8.4.
La opción `--env-manager local` usa el ambiente activado en lugar de crear uno nuevo.
Al final debe decir `=== Run (ID '...') succeeded ===`.

Requiere conexión a Internet para descargar los archivos que aún no estén disponibles localmente.

### 7.3. Entry points disponibles

| Entry point | Comando | Con `make` | Qué hace |
|---|---|---|---|
| `load` | `mlflow run . -e load --env-manager local` | `make mlflow-load` | Ejecuta el pipeline de datos (descarga, preparación y datasets horarios) |
| `workflow` | `mlflow run . -e workflow --env-manager local` | `make mlflow-workflow` | Encadena los pasos del proyecto como runs hijos de un run padre |

Por ahora `workflow` solo contiene el paso `load`; los pasos de modelado se agregarán en `mibici/workflow.py` conforme se implementen.

Los comandos de `make` ya definen `MLFLOW_EXPERIMENT_NAME=mibici-equipo28`, por lo que no hace falta el `export` de la sección 7.1.

### 7.4. Ver los resultados

```bash
mlflow ui
```

Abre http://127.0.0.1:5000 y selecciona el experimento `mibici-equipo28` para revisar los runs, sus parámetros y artefactos.

No hace falta tener la UI abierta mientras corres los runs: puedes abrirla después.

El paso `load` registra en su run:

| Tipo | Qué se registra |
|---|---|
| Parámetros | `year` y el nombre del manifiesto de fuentes |
| Tag `data_version` | Huella SHA-256 de los doce CSV originales; cambia solo si cambian los datos de entrada |
| Métricas | Viajes totales y por mes (`trips_month`), indicadores de calidad (`*_count`, `*_pct`), filas de los datasets horarios, estaciones observadas, porcentaje de horas sin actividad y duración del pipeline |
| Artefactos | `reports/raw_inventory.csv`, `reports/quality_by_month.csv` y los dos datasets horarios en `processed/` |

Los pasos de modelado podrán descargar los datasets horarios con el `run_id` del paso `load`.

`mlflow.db`, `mlruns/` y `mlartifacts/` se crean solos al correr y están excluidos de Git.


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

```bash
python -m pytest tests/test_data.py -v
```

La implementación cuenta con 24 pruebas aprobadas, que verifican la lectura, validación, transformación, agregación y orquestación del pipeline.

### 8.6. Experimentación

Los modelos de regresión, clasificación y clustering se desarrollan de forma independiente por los integrantes del equipo.

Se utilizará MLflow para registrar los experimentos bajo el nombre previsto `mibici-equipo28`.

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
| Miguel Ángel Lara López | Análisis exploratorio y clustering de estaciones (P3). |
| Josafat Misael Rodríguez Nieto | Ingeniería de datos, ingesta, validación, procesamiento y arquitectura de datos. |

## 11. Documentación

Los documentos de formulación del problema y diseño de arquitectura se almacenarán en `references/`.

La documentación se ampliará conforme avance el desarrollo del proyecto integrador.


## 12. Estado del proyecto

**Ingeniería de datos:** Implementada y validada mediante 24 pruebas automatizadas.

**Datos disponibles:** Doce archivos mensuales de 2025, con 4,532,032 viajes y 371 estaciones observadas.

**Pipeline reproducible:** Implementado mediante cuatro comandos CLI para descarga, preparación, agregación y ejecución integral.

**Modelado:** Los modelos de regresión, clasificación y clustering se encuentran en desarrollo por los integrantes responsables.

**MLOps:** Integración de experimentos y seguimiento mediante MLflow en desarrollo.

**Documentación:** Fuentes de datos y pipeline documentados. Pendiente integrar la arquitectura, los resultados de modelado y la evaluación final.