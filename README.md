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

Se utilizarán los datos abiertos publicados por MiBici:

- Registros mensuales de viajes en formato CSV.
- Nomenclatura de estaciones, incluyendo identificadores, ubicación geográfica y capacidad.

Los archivos se descargarán mediante un proceso automatizado y se almacenarán localmente. Los datos originales y procesados no se incluirán en el repositorio Git.

El rango de fechas definitivo y las condiciones de uso se documentarán una vez que el equipo complete la selección y revisión de los datos.

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

## 7. Ingesta y experimentación

La ingesta reproducible se implementará en `mibici/dataset.py` y las transformaciones reutilizables en `mibici/features.py`.

Los experimentos base de P1, P2 y P3 se registrarán en MLflow bajo el nombre:

`mibici-equipo28`

Los comandos definitivos de ingesta y ejecución se incorporarán cuando los scripts estén implementados y validados.

## 8. Flujo de colaboración

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

## 9. Integrantes y responsabilidades

| Integrante | Responsabilidad |
|---|---|
| Iván Fernando García Nuñez | Pendiente de asignación |
| Anthony Gerardo Gutarra Sánchez | Pendiente de asignación |
| Miguel Ángel Lara López | Pendiente de asignación |
| Josafat Misael Rodríguez Nieto | Pendiente de asignación |

## 10. Documentación

Los documentos de formulación del problema y diseño de arquitectura se almacenarán en `references/`.

La documentación se ampliará conforme avance el desarrollo del proyecto integrador.

## 11. Estado del proyecto

**Fase actual:** Inicialización de la estructura reproducible.

**Pendiente:** Selección definitiva de datos, ML Canvas, arquitectura, ingesta automatizada, feature engineering y experimentos base de MLflow.