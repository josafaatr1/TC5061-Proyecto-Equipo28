import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configuración general del proyecto
PROJECT_NAME = "MiBici"
TEAM_NUMBER = 28
RANDOM_SEED = 42

# Configuración del conjunto de datos
# Fechas preliminares, pendientes de confirmación por el equipo.
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# Directorios adicionales del proyecto
REFERENCES_DIR = PROJ_ROOT / "references"
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"

# Valores por defecto del entry point `ingest` (deben coincidir con MLproject).
INGEST_YEAR = int(START_DATE[:4])
SOURCES_MANIFEST = REFERENCES_DIR / "mibici_sources_2025.csv"

# Hiperparámetros por defecto del entry point `train` (deben coincidir con MLproject).
# P1 y P2 comparten los de HistGradientBoosting; P3 elige k entre 2 y P3_MAX_K.
HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}
P3_MAX_K = 8
P3_N_INIT = 10

# MLflow: experimento del equipo y alias del modelo en producción.
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mibici-equipo28")
CHAMPION_ALIAS = "champion"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
