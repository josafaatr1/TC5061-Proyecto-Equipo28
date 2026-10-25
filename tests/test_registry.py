import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from mibici.config import CHAMPION_ALIAS
from mibici.modeling.registry import (
    REGISTERED_MODELS,
    best_run_id,
    improves,
    latest_training_children,
    register_models,
)

P1_MODEL = "mibici-p1-departures"


def log_training(question: str, children: dict[str, dict]) -> dict[str, str]:
    """Simular un entrenamiento: un run padre y un hijo con modelo por entrada."""
    model = DummyRegressor().fit(np.zeros((2, 1)), [0.0, 1.0])
    run_ids = {}
    with mlflow.start_run(run_name=f"{question}-parent"):
        mlflow.set_tag("question", question)
        for model_name, child in children.items():
            with mlflow.start_run(run_name=model_name, nested=True) as run:
                mlflow.set_tags({"question": question, "model_name": model_name, **child["tags"]})
                mlflow.log_metrics(child["metrics"])
                mlflow.sklearn.log_model(model, name="model")
                run_ids[model_name] = run.info.run_id

    return run_ids


def p1_training(dummy_mae: float, candidate_mae: float) -> dict[str, str]:
    return log_training(
        "P1",
        {
            "dummy_mean": {"tags": {}, "metrics": {"val_mae": dummy_mae}},
            "hist_gradient_boosting_poisson": {"tags": {}, "metrics": {"val_mae": candidate_mae}},
        },
    )


def test_improves_respects_metric_direction():
    assert improves(0.9, 1.0, higher_is_better=False)
    assert not improves(1.0, 1.0, higher_is_better=False)
    assert improves(0.6, 0.5, higher_is_better=True)
    assert not improves(0.4, 0.5, higher_is_better=True)


def test_best_run_uses_latest_training_and_objective(tracking_store):
    """El mejor run sale del último entrenamiento y del objetivo de cada modelo."""
    _, experiment_id = tracking_store
    p1_training(dummy_mae=0.1, candidate_mae=0.2)
    latest = p1_training(dummy_mae=2.0, candidate_mae=1.0)
    p2 = log_training(
        "P2",
        {
            "bike": {"tags": {"objective": "bike_pressure"}, "metrics": {"val_average_precision": 0.3}},
            "dock": {"tags": {"objective": "dock_pressure"}, "metrics": {"val_average_precision": 0.9}},
        },
    )
    specs = {spec.name: spec for spec in REGISTERED_MODELS}

    p1_children = latest_training_children(experiment_id, "P1")
    p2_children = latest_training_children(experiment_id, "P2")

    assert set(p1_children["run_id"]) == set(latest.values())
    assert best_run_id(p1_children, specs[P1_MODEL]) == latest["hist_gradient_boosting_poisson"]
    assert best_run_id(p2_children, specs["mibici-p2-bike-pressure"]) == p2["bike"]
    assert best_run_id(p2_children, specs["mibici-p2-dock-pressure"]) == p2["dock"]


def test_champion_moves_only_when_metric_improves(tracking_store):
    """Un modelo peor queda registrado sin alias; uno mejor se vuelve champion."""
    client, _ = tracking_store

    first = p1_training(dummy_mae=2.0, candidate_mae=1.0)
    summary = register_models("test").set_index("model")
    assert summary.loc[P1_MODEL, "status"] == "nuevo champion"
    champion = client.get_model_version_by_alias(P1_MODEL, CHAMPION_ALIAS)
    assert champion.run_id == first["hist_gradient_boosting_poisson"]
    assert champion.tags["selection_metric"] == "val_mae"

    # Volver a registrar sin entrenar no crea versiones nuevas.
    register_models("test")
    assert len(client.search_model_versions(f"name = '{P1_MODEL}'")) == 1

    p1_training(dummy_mae=2.0, candidate_mae=1.5)
    summary = register_models("test").set_index("model")
    assert summary.loc[P1_MODEL, "status"] == "registrado sin alias"
    assert client.get_model_version_by_alias(P1_MODEL, CHAMPION_ALIAS).version == champion.version

    better = p1_training(dummy_mae=2.0, candidate_mae=0.5)
    summary = register_models("test").set_index("model")
    assert summary.loc[P1_MODEL, "status"] == "nuevo champion"
    new_champion = client.get_model_version_by_alias(P1_MODEL, CHAMPION_ALIAS)
    assert new_champion.run_id == better["hist_gradient_boosting_poisson"]
    assert len(client.search_model_versions(f"name = '{P1_MODEL}'")) == 3

    # El champion se carga por alias; las preguntas sin runs se omiten.
    model = mlflow.pyfunc.load_model(f"models:/{P1_MODEL}@{CHAMPION_ALIAS}")
    assert len(model.predict(pd.DataFrame({"x": [0.0]}))) == 1
    assert summary.loc["mibici-p3-station-clusters", "status"] == "sin runs"
