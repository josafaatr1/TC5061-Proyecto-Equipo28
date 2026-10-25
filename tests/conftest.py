from pathlib import Path

import mlflow
import pytest


@pytest.fixture
def tracking_store(tmp_path: Path, monkeypatch):
    """Usar un tracking store temporal con artefactos dentro de tmp_path."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment_id)
    yield mlflow.MlflowClient(), experiment_id
    mlflow.set_tracking_uri(None)
