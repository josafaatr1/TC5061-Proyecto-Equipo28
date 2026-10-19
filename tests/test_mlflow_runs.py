from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from tests.test_modeling import make_hourly_data

from mibici.modeling.mlflow_runs import (
    P1_RUN_NAMES,
    ActivityQuantileModel,
    calculate_data_version,
    p2_run_name,
    precision_recall_table,
    run_p1,
    run_p3,
)
from mibici.modeling.train import (
    evaluate_candidate,
    prepare_supervised_data,
    split_chronologically,
    train_candidate_model,
)


def make_train_validation_hourly() -> pd.DataFrame:
    """Crear dos estaciones sintéticas de enero a octubre de 2025."""
    hours = pd.date_range("2025-01-01", "2025-11-01", inclusive="left", freq="h")
    return pd.concat(
        [make_hourly_data(hours, station_id=1), make_hourly_data(hours, station_id=2)],
        ignore_index=True,
    )


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


def test_data_version_ignores_row_order_and_detects_changes():
    """La huella depende del contenido, no del orden de las filas."""
    data = make_train_validation_hourly()

    original = calculate_data_version(data)
    shuffled = calculate_data_version(data.sample(frac=1, random_state=0))

    changed = data.copy()
    changed.loc[10, "departures"] += 1
    changed.loc[10, "total_activity"] += 1

    assert original == shuffled
    assert original != calculate_data_version(changed)


def test_p2_run_names_follow_handoff_contract():
    """Los nombres de run P2 coinciden con la sección 9 del handoff."""
    assert p2_run_name("bike_pressure", "dummy_prior") == "P2-bike-pressure-dummy-prior"
    assert (
        p2_run_name("dock_pressure", "hist_gradient_boosting_balanced")
        == "P2-dock-pressure-hist-gradient-boosting-balanced"
    )


def test_precision_recall_table_is_reduced_to_max_points():
    """La curva PR se reduce sin perder sus extremos."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=5000)
    probabilities = rng.random(5000)

    table = precision_recall_table(y_true, probabilities, max_points=100)

    assert len(table) <= 100
    assert list(table.columns) == ["threshold", "precision", "recall"]
    assert table["threshold"].is_monotonic_increasing
    assert table["recall"].iloc[0] == 1


def test_evaluate_candidate_reuses_trained_model():
    """Evaluar un modelo ya ajustado produce las mismas métricas."""
    splits = split_chronologically(
        prepare_supervised_data(make_train_validation_hourly()),
        evaluate_test=False,
    )
    model = train_candidate_model(splits["train"])

    internal = evaluate_candidate(splits)
    reused = evaluate_candidate(splits, model=model, fit_seconds=1.5)

    columns = ["evaluation_set", "mae", "rmse", "n_evaluated"]
    pd.testing.assert_frame_equal(internal[columns], reused[columns])
    assert reused["fit_seconds"].eq(1.5).all()


def test_run_p1_logs_parent_and_one_child_per_model(tmp_path: Path, tracking_store):
    """P1 registra un run padre y un hijo por modelo con métricas val_."""
    client, experiment_id = tracking_store
    hourly_path = tmp_path / "hourly.parquet"
    make_train_validation_hourly().to_parquet(hourly_path)

    run_p1(hourly_path)

    runs = client.search_runs([experiment_id])
    parent = next(run for run in runs if run.info.run_name == "P1-regression")
    children = [
        run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    ]

    assert {run.info.run_name for run in children} == set(P1_RUN_NAMES.values())
    for run in children:
        assert {"val_mae", "val_rmse", "val_n_evaluated"} <= set(run.data.metrics)
        assert run.data.tags["question"] == "P1"
        assert len(run.data.tags["train_validation_data_version"]) == 64
        assert run.data.params["validation_period"] == "2025-09-01/2025-11-01"

    # Cada modelo, incluidos los baselines, tiene artefactos y un modelo cargable.
    splits = split_chronologically(
        prepare_supervised_data(make_train_validation_hourly()),
        evaluate_test=False,
    )
    validation = splits["validation"].dropna(subset=["departures_lag_168h"])
    for run in children:
        artifacts = {item.path for item in client.list_artifacts(run.info.run_id, "evaluation")}
        assert {"evaluation/hourly_profile.csv", "evaluation/hourly_profile.png"} <= artifacts

        logged_models = client.search_logged_models(
            [experiment_id], filter_string=f"source_run_id = '{run.info.run_id}'"
        )
        assert len(logged_models) == 1

        model = mlflow.pyfunc.load_model(logged_models[0].model_uri)
        predictions = model.predict(validation)
        assert np.isfinite(np.asarray(predictions)).all()


def make_station_profiles(n_stations: int = 12) -> pd.DataFrame:
    """Crear estaciones sintéticas de enero con niveles de actividad distintos."""
    hours = pd.date_range("2025-01-01", "2025-02-01", inclusive="left", freq="h")
    stations = []
    for station_id in range(1, n_stations + 1):
        departures = [(index % 7) * (1 + station_id % 4) for index in range(len(hours))]
        stations.append(make_hourly_data(hours, departures=departures, station_id=station_id))

    return pd.concat(stations, ignore_index=True)


def test_activity_quantile_baseline_creates_groups_of_equal_size():
    """El baseline de P3 reparte las estaciones en grupos por cuantiles."""
    features = pd.DataFrame({"total_activity": np.arange(1, 13, dtype=float)})

    model = ActivityQuantileModel.fit(features["total_activity"], n_groups=3)
    labels = model.predict(None, features)

    assert np.bincount(labels).tolist() == [4, 4, 4]
    assert (np.diff(labels) >= 0).all()


def test_run_p3_logs_baseline_and_one_child_per_k(tmp_path: Path, tracking_store):
    """P3 registra un run padre, el baseline y un hijo por k con mapa y modelo."""
    client, experiment_id = tracking_store
    hourly_path = tmp_path / "hourly.parquet"
    make_station_profiles().to_parquet(hourly_path)
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "id": range(1, 13),
            "name": [f"Estación {i}" for i in range(1, 13)],
            "latitude": np.linspace(20.6, 20.7, 12),
            "longitude": np.linspace(-103.4, -103.3, 12),
            "dpcapacity": 15,
        }
    ).to_csv(catalog_path, index=False)

    summary = run_p3(hourly_path, catalog_path)

    runs = client.search_runs([experiment_id])
    parent = next(run for run in runs if run.info.run_name == "P3-clustering")
    children = [
        run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    ]

    expected = {"P3-clustering-baseline-activity-quantiles"} | {
        f"P3-clustering-kmeans-k{k}" for k in range(2, 9)
    }
    assert {run.info.run_name for run in children} == expected
    assert summary["selected"].sum() == 1
    assert parent.data.params["data_period"] == "2025-01-01/2025-11-01"

    for run in children:
        assert run.data.tags["question"] == "P3"
        assert "silhouette_score" in run.data.metrics
        artifacts = {item.path for item in client.list_artifacts(run.info.run_id, "evaluation")}
        assert "evaluation/station_map.png" in artifacts

        logged_models = client.search_logged_models(
            [experiment_id], filter_string=f"source_run_id = '{run.info.run_id}'"
        )
        assert len(logged_models) == 1
