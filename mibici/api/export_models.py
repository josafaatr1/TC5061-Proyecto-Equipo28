"""Exporta del Model Registry los modelos que sirve la API para empaquetarlos en la imagen.

Uso (con el servidor de MLflow corriendo y MLFLOW_TRACKING_URI definido):

    python -m mibici.api.export_models

Descarga cada modelo registrado con su alias en `models/<carpeta>/` y escribe
`models/metadata.json` con el nombre, la versión y el run de origen de cada uno.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
import typer

from mibici.config import MODELS_DIR

# Modelo registrado -> carpeta dentro de MODELS_DIR. Debe coincidir con BIKE_MODEL_URI y
# DOCK_MODEL_URI del Containerfile.
EXPORTED_MODELS = {
    "mibici-p2-bike-pressure": "p2-bike-pressure",
    "mibici-p2-dock-pressure": "p2-dock-pressure",
}
DEFAULT_ALIAS = "champion"

app = typer.Typer()


def export_model(
    client: mlflow.MlflowClient, name: str, folder: str, alias: str, output_dir: Path
) -> dict:
    """Descarga una versión registrada (por alias) y devuelve su metadata."""
    version = client.get_model_version_by_alias(name, alias)
    destination = output_dir / folder
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{version.version}", dst_path=str(destination)
    )
    logger.info(f"{name}@{alias} (versión {version.version}) -> {destination}")
    return {
        "registered_model": name,
        "alias": alias,
        "version": version.version,
        "run_id": version.run_id,
        "path": folder,
    }


@app.command()
def main(alias: str = DEFAULT_ALIAS, output_dir: Path = MODELS_DIR):
    """Exporta los modelos de la API desde el Model Registry a `output_dir`."""
    client = mlflow.MlflowClient()
    exported = [
        export_model(client, name, folder, alias, output_dir)
        for name, folder in EXPORTED_MODELS.items()
    ]
    metadata = {
        "tracking_uri": mlflow.get_tracking_uri(),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": exported,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.success(f"{len(exported)} modelos exportados a {output_dir}")


if __name__ == "__main__":
    app()
