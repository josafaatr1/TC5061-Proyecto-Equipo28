from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from tests.test_modeling import make_hourly_data
from tests.test_p2_train import make_labeled_data
import yaml

from mibici import clustering, config
from mibici.config import PROJ_ROOT
from mibici.modeling import p2_train
from mibici.modeling.mlflow_runs import (
    MODEL_FAMILIES,
    P1_RUN_NAMES,
    ActivityQuantileModel,
    calculate_data_version,
    p2_run_name,
    parse_search_grid,
    precision_recall_table,
    run_p1,
    run_p2,
    run_p3,
    search_suffix,
)
from mibici.modeling.train import (
    evaluate_candidate,
    prepare_supervised_data,
    split_chronologically,
    train_candidate_model,
)

MLPROJECT_PATH = PROJ_ROOT / "MLproject"


def make_train_validation_hourly() -> pd.DataFrame:
    """Crear dos estaciones sintéticas de enero a octubre de 2025."""
    hours = pd.date_range("2025-01-01", "2025-11-01", inclusive="left", freq="h")
    return pd.concat(
        [make_hourly_data(hours, station_id=1), make_hourly_data(hours, station_id=2)],
        ignore_index=True,
    )


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


def test_run_p1_applies_hyperparameters_and_model_filter(tmp_path: Path, tracking_store):
    """`train` puede registrar un solo modelo con hiperparámetros propios."""
    client, experiment_id = tracking_store
    hourly_path = tmp_path / "hourly.parquet"
    make_train_validation_hourly().to_parquet(hourly_path)

    run_p1(
        hourly_path,
        hgb_params={"learning_rate": 0.05, "max_iter": 20},
        model="hist_gradient_boosting_poisson",
    )

    runs = client.search_runs([experiment_id])
    children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]
    assert [run.info.run_name for run in children] == [
        P1_RUN_NAMES["hist_gradient_boosting_poisson"]
    ]
    assert children[0].data.params["learning_rate"] == "0.05"
    assert children[0].data.params["max_iter"] == "20"
    assert children[0].data.params["loss"] == "poisson"


def test_fit_clusters_accepts_fixed_k():
    """P3 usa el k fijo recibido en lugar del elegido por silhouette."""
    features = clustering.build_station_features(make_station_profiles())

    result = clustering.fit_clusters(features, max_k=5, k=4)

    assert result.best_k == 4
    assert len(np.unique(result.labels)) == 4
    with pytest.raises(ValueError, match="k=9"):
        clustering.fit_clusters(features, max_k=5, k=9)


def test_mlproject_defaults_match_config():
    """Los valores por defecto de MLproject son los de config.py."""
    entry_points = yaml.safe_load(MLPROJECT_PATH.read_text())["entry_points"]

    ingest = entry_points["ingest"]["parameters"]
    assert ingest["year"]["default"] == config.INGEST_YEAR
    assert PROJ_ROOT / ingest["manifest_path"]["default"] == config.SOURCES_MANIFEST

    train = entry_points["train"]["parameters"]
    assert "default" not in train["question"]
    for name, value in config.HGB_PARAMS.items():
        assert train[name]["default"] == value
    assert train["max_k"]["default"] == config.P3_MAX_K

    search = entry_points["search"]["parameters"]
    assert "default" not in search["question"]
    assert search["learning_rates"]["default"] == ",".join(map(str, config.SEARCH_LEARNING_RATES))
    assert search["max_leaf_nodes"]["default"] == ",".join(map(str, config.SEARCH_MAX_LEAF_NODES))


def test_parse_search_grid_builds_every_combination():
    """La rejilla es el producto de learning_rate y max_leaf_nodes."""
    grid = parse_search_grid("0.05,0.1", "15,31")

    assert grid == [
        {"learning_rate": 0.05, "max_leaf_nodes": 15},
        {"learning_rate": 0.05, "max_leaf_nodes": 31},
        {"learning_rate": 0.1, "max_leaf_nodes": 15},
        {"learning_rate": 0.1, "max_leaf_nodes": 31},
    ]
    assert search_suffix(grid[0]) == "lr0.05-leaf15"
    with pytest.raises(ValueError):
        parse_search_grid("", "15")


def children_of(client, experiment_id: str, parent_name: str):
    """Run padre por nombre y sus hijos indexados por nombre de run."""
    runs = client.search_runs([experiment_id])
    parent = next(run for run in runs if run.info.run_name == parent_name)
    children = {
        run.info.run_name: run
        for run in runs
        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    }
    return parent, children


def assert_candidate_artifacts(client, experiment_id: str, run, plots: set[str]):
    """Cada run hijo guarda sus carpetas de artefactos y un modelo con firma."""
    run_id = run.info.run_id
    for folder, expected in {
        "manifests": {
            "manifests/data_manifest.json",
            "manifests/mibici_sources_2025.csv",
            "manifests/mibici_inventory_2025.csv",
        },
        "config": {"config/run_config.json"},
        "evaluation": plots,
    }.items():
        assert expected <= {item.path for item in client.list_artifacts(run_id, folder)}

    logged_models = client.search_logged_models(
        [experiment_id], filter_string=f"source_run_id = '{run_id}'"
    )
    assert len(logged_models) == 1
    loaded = mlflow.pyfunc.load_model(logged_models[0].model_uri)
    assert loaded.metadata.signature is not None
    assert loaded.metadata.saved_input_example_info is not None

    tags = run.data.tags
    assert tags["model_family"] == MODEL_FAMILIES[tags["model_name"]]
    assert tags["stage_of_work"] in {"baseline", "search"}
    assert len(tags["train_validation_data_version"]) == 64


def test_run_p1_search_logs_one_child_per_combination(tmp_path: Path, tracking_store):
    """La búsqueda de P1 registra baselines y un candidato hijo por combinación."""
    client, experiment_id = tracking_store
    hourly_path = tmp_path / "hourly.parquet"
    make_train_validation_hourly().to_parquet(hourly_path)
    grid = [{**combo, "max_iter": 10} for combo in parse_search_grid("0.05,0.2", "15,31")]

    run_p1(hourly_path, search_grid=grid)

    parent, children = children_of(client, experiment_id, "P1-search")
    candidates = {
        f"P1-regression-search-lr{rate}-leaf{leaf}" for rate in (0.05, 0.2) for leaf in (15, 31)
    }
    assert set(children) == candidates | {
        P1_RUN_NAMES["dummy_mean"],
        P1_RUN_NAMES["seasonal_weekly"],
    }
    assert parent.data.tags["stage_of_work"] == "search"
    assert parent.data.tags["question"] == "P1"
    assert parent.data.tags["selected_run_departures"] in candidates
    assert {"reports/selection_summary.json", "reports/p1_model_comparison.csv"} <= {
        item.path for item in client.list_artifacts(parent.info.run_id, "reports")
    }

    for name, run in children.items():
        assert run.data.tags["stage_of_work"] == ("search" if name in candidates else "baseline")
        assert {"val_mae", "val_rmse", "fit_seconds"} <= set(run.data.metrics)
        assert_candidate_artifacts(
            client,
            experiment_id,
            run,
            {"evaluation/hourly_profile.csv", "evaluation/hourly_profile.png"},
        )
    assert children["P1-regression-search-lr0.05-leaf15"].data.params["max_iter"] == "10"


def test_run_p2_search_logs_candidates_for_each_objective(
    tmp_path: Path, tracking_store, monkeypatch
):
    """La búsqueda de P2 registra baselines y candidatos por combinación y objetivo."""
    client, experiment_id = tracking_store
    hourly_path = tmp_path / "hourly.parquet"
    make_train_validation_hourly().to_parquet(hourly_path)
    label_path = tmp_path / "labels.parquet"
    make_labeled_data(include_test=False).to_parquet(label_path)
    provenance = {
        "catalog_version": "2026-09",
        "catalog_sha256": "a" * 64,
        "label_artifact_sha256": "b" * 64,
        "labels_reproduced_exactly": True,
        "label_pressure_quantile": 0.9,
        "label_pressure_thresholds_train_only": {
            "bike_pressure_ratio": 0.3,
            "dock_pressure_ratio": 0.3,
        },
    }
    monkeypatch.setattr(p2_train, "verify_label_reproducibility", lambda **_: provenance)
    grid = [{**combo, "max_iter": 10} for combo in parse_search_grid("0.05,0.2", "15")]

    run_p2(label_path, hourly_path, tmp_path / "catalog.csv", search_grid=grid)

    parent, children = children_of(client, experiment_id, "P2-search")
    objectives = ("bike-pressure", "dock-pressure")
    candidates = {
        f"P2-{objective}-search-lr{rate}-leaf15" for objective in objectives for rate in (0.05, 0.2)
    }
    baselines = {f"P2-{objective}-dummy-prior" for objective in objectives}
    assert set(children) == candidates | baselines
    assert parent.data.tags["stage_of_work"] == "search"
    assert parent.data.tags["selected_run_bike_pressure"] in candidates
    assert parent.data.tags["selected_run_dock_pressure"] in candidates

    for name, run in children.items():
        assert run.data.tags["stage_of_work"] == ("search" if name in candidates else "baseline")
        assert run.data.tags["objective"] in {"bike_pressure", "dock_pressure"}
        assert {"val_average_precision", "val_recall", "val_f1"} <= set(run.data.metrics)
        assert_candidate_artifacts(
            client,
            experiment_id,
            run,
            {
                "evaluation/confusion_matrix.png",
                "evaluation/precision_recall.png",
                "evaluation/precision_recall.csv",
            },
        )
