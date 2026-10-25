"""Registro en MLflow del pipeline de datos de MiBici.

Las funciones `summarize_*` solo leen los archivos generados por el pipeline
y no dependen de MLflow, por lo que pueden probarse de forma aislada.
"""

import hashlib
from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import pandas as pd

from mibici.config import INTERIM_DATA_DIR, RAW_DATA_DIR

# Indicadores de calidad calculados por `prepare_trips`.
QUALITY_FLAGS = {
    "has_invalid_timestamp": "invalid_timestamp",
    "has_nonpositive_duration": "nonpositive_duration",
    "is_long_trip": "long_trips",
}


def summarize_raw_files(
    year: int = 2025,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Construir el inventario de los CSV originales con su huella SHA-256."""
    rows = []

    for month in range(1, 13):
        file_path = raw_dir / f"datos_abiertos_{year}_{month:02d}.csv"

        with file_path.open("rb") as file:
            file_hash = hashlib.file_digest(file, "sha256").hexdigest()

        rows.append(
            {
                "month": month,
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "sha256": file_hash,
            }
        )

    return pd.DataFrame(rows)


def calculate_data_version(inventory: pd.DataFrame) -> str:
    """Calcular una huella única a partir de las huellas de los CSV."""
    combined = "".join(inventory.sort_values("month")["sha256"])

    return hashlib.sha256(combined.encode()).hexdigest()


def summarize_quality(
    year: int = 2025,
    interim_dir: Path = INTERIM_DATA_DIR,
) -> pd.DataFrame:
    """Contar viajes e indicadores de calidad de cada mes preparado."""
    rows = []

    for month in range(1, 13):
        file_path = interim_dir / f"trips_{year}_{month:02d}.parquet"

        # Leer únicamente las columnas de indicadores.
        flags = pd.read_parquet(file_path, columns=list(QUALITY_FLAGS))

        row = {"month": month, "trips": len(flags)}

        for column, name in QUALITY_FLAGS.items():
            row[name] = int(flags[column].sum())

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_hourly(
    observed_path: Path,
    completed_path: Path,
) -> dict[str, float]:
    """Calcular el tamaño y la cobertura de los datasets horarios."""
    observed = pd.read_parquet(observed_path, columns=["station_id"])
    completed = pd.read_parquet(completed_path, columns=["total_activity"])

    return {
        "observed_rows": len(observed),
        "completed_rows": len(completed),
        "stations_observed": observed["station_id"].nunique(),
        "zero_activity_pct": (100 * (completed["total_activity"] == 0).mean()),
    }


def log_pipeline_run(
    year: int,
    manifest_path: Path,
    observed_path: Path,
    completed_path: Path,
    duration_seconds: float,
) -> None:
    """Registrar parámetros, métricas y artefactos del pipeline en MLflow."""
    inventory = summarize_raw_files(year)
    quality = summarize_quality(year)
    hourly = summarize_hourly(observed_path, completed_path)

    total_trips = int(quality["trips"].sum())

    # Identificar la versión de los datos de entrada.
    mlflow.set_tag("data_version", calculate_data_version(inventory))

    mlflow.log_params(
        {
            "year": year,
            "manifest": manifest_path.name,
        }
    )

    # Volumen y calidad de los datos.
    mlflow.log_metrics(
        {
            "raw_files": len(inventory),
            "raw_size_mb": inventory["size_bytes"].sum() / 1024**2,
            "trips_total": total_trips,
            "pipeline_seconds": duration_seconds,
            **hourly,
        }
    )

    for name in QUALITY_FLAGS.values():
        count = int(quality[name].sum())
        mlflow.log_metric(f"{name}_count", count)
        mlflow.log_metric(f"{name}_pct", 100 * count / total_trips)

    # Serie mensual: la UI la muestra como gráfica por mes.
    for row in quality.itertuples(index=False):
        mlflow.log_metric("trips_month", row.trips, step=row.month)

    # Guardar los reportes y los datasets procesados.
    with tempfile.TemporaryDirectory() as tmp_dir:
        inventory_path = Path(tmp_dir) / "raw_inventory.csv"
        quality_path = Path(tmp_dir) / "quality_by_month.csv"

        inventory.to_csv(inventory_path, index=False)
        quality.to_csv(quality_path, index=False)

        mlflow.log_artifact(str(inventory_path), artifact_path="reports")
        mlflow.log_artifact(str(quality_path), artifact_path="reports")

    # Conservar el manifiesto de fuentes para reproducir la ingesta.
    mlflow.log_artifact(str(manifest_path), artifact_path="manifests")
    mlflow.log_artifact(str(observed_path), artifact_path="processed")
    mlflow.log_artifact(str(completed_path), artifact_path="processed")

    logger.success("Pipeline registrado en MLflow.")
