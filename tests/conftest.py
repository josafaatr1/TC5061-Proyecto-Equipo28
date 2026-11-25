from pathlib import Path

import mlflow
import pytest


@pytest.fixture
def tracking_store(tmp_path: Path, monkeypatch):
    """Usar un tracking store temporal con artefactos dentro de tmp_path."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    experiment_id = mlflow.create_experiment(
        "test",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )

    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment_id)

    # Explicitly reset MLflow's active experiment for each test.
    mlflow.set_experiment("test")

    yield mlflow.MlflowClient(), experiment_id

    mlflow.end_run()
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)
