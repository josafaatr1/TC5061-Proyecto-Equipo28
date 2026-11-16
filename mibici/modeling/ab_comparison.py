"""Comparación retrospectiva de dos variantes P2 guardadas en MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.models import Model
import pandas as pd
import typer

from mibici.config import EXTERNAL_DATA_DIR
from mibici.features import REQUIRED_COLUMNS
from mibici.modeling import p2_train
from mibici.modeling.mlflow_runs import calculate_data_version
from mibici.modeling.p2_catalog import load_catalog
from mibici.modeling.p2_targets import VALIDATION_END, PressureThresholds, build_p2_test_dataset

app = typer.Typer()

# Parejas fijadas antes de evaluar: champion seleccionado por validation y variante
# de configuración inicial (lr=0.1, 31 hojas) del mismo P2-search.
CANDIDATES = {
    "bike_pressure": (
        ("champion_v1", "d306fe5f7f34496a94f8cc3d0db2f62b", "m-c65129c4cf4d4744b31f026ab2a4c1c1"),
        ("alternativa", "c6bcec6501404d57b0a728bfa3e0f3fd", "m-e41812f5920347b59086800d817a538f"),
    ),
    "dock_pressure": (
        ("champion_v1", "81f93c6f160e4f44afe3db913eac5030", "m-d8d16e84adc54d57bb7a6c0ba7f7d68c"),
        ("alternativa", "6c2da2372de840b08e06258acbe07a2b", "m-1b54aa649f974629a7ff43ac3eb7f4ca"),
    ),
}


def validate_configs(configs: list[dict], objective: str, provenance: dict) -> PressureThresholds:
    """Comprobar que ambos runs describen el mismo problema y datos de origen."""
    first = configs[0]
    tag_keys = (
        "question",
        "objective",
        "model_family",
        "train_validation_data_version",
        "catalog_sha256",
        "label_artifact_sha256",
    )
    param_keys = (
        "train_period",
        "validation_period",
        "predictor_columns",
        "decision_threshold",
        "bike_pressure_ratio_threshold",
        "dock_pressure_ratio_threshold",
    )
    if first["tags"]["question"] != "P2" or first["tags"]["objective"] != objective:
        raise ValueError(f"Objetivo incompatible: {objective}")
    if first["tags"]["model_family"] != "hist_gradient_boosting":
        raise ValueError("La familia del modelo no es HGB.")
    if first["params"]["predictor_columns"].split(",") != list(p2_train.PREDICTOR_COLUMNS):
        raise ValueError("Las columnas no coinciden con los predictores actuales.")
    if float(first["params"]["decision_threshold"]) != p2_train.DECISION_THRESHOLD:
        raise ValueError("El umbral de decisión no coincide con el de evaluación.")
    for config in configs:
        for key in tag_keys:
            if config["tags"][key] != first["tags"][key]:
                raise ValueError(f"Los runs difieren en {key}.")
        for key in param_keys:
            if config["params"][key] != first["params"][key]:
                raise ValueError(f"Los runs difieren en {key}.")
    for key in ("catalog_sha256", "label_artifact_sha256", "train_validation_data_version"):
        if first["tags"][key] != provenance[key]:
            raise ValueError(f"La procedencia local no coincide: {key}.")
    thresholds = PressureThresholds(
        float(first["params"]["bike_pressure_ratio_threshold"]),
        float(first["params"]["dock_pressure_ratio_threshold"]),
    )
    if vars(thresholds) != provenance["label_pressure_thresholds_train_only"]:
        raise ValueError("Los umbrales de etiquetas no coinciden con train.")
    return thresholds


@app.command()
def main(
    artifacts_dir: Path = Path("mlartifacts") / "1",
    output: Path = Path("reports") / "ab_testing" / "p2_comparison.csv",
    label_path: Path = p2_train.LABEL_DATA_PATH,
    hourly_path: Path = p2_train.HOURLY_DATA_PATH,
    catalog_path: Path = EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv",
) -> None:
    """Evaluar variantes guardadas localmente, sin modificar MLflow ni entrenar."""
    provenance = p2_train.verify_label_reproducibility(label_path, hourly_path, catalog_path)
    hourly = pd.read_parquet(hourly_path, columns=list(REQUIRED_COLUMNS))
    provenance["train_validation_data_version"] = calculate_data_version(
        hourly.loc[hourly["hour"] < VALIDATION_END]
    )
    full_year_data_version = calculate_data_version(hourly)
    targets = {name: column for column, name in p2_train.TARGET_NAMES.items()}
    records = []
    current = None
    expected_thresholds = None
    for objective, candidates in CANDIDATES.items():
        configs = [
            json.loads(
                (artifacts_dir / run_id / "artifacts" / "config" / "run_config.json").read_text(
                    encoding="utf-8"
                )
            )
            for _, run_id, _ in candidates
        ]
        thresholds = validate_configs(configs, objective, provenance)
        if current is None:
            current = build_p2_test_dataset(
                hourly, load_catalog(catalog_path, strict=False), thresholds
            )
            expected_thresholds = thresholds
        elif thresholds != expected_thresholds:
            raise ValueError("Los objetivos no comparten los umbrales de etiquetas.")
        features = p2_train._feature_array(current)
        target = current[targets[objective]].to_numpy()
        for (role, run_id, model_id), config in zip(candidates, configs, strict=True):
            path = artifacts_dir / "models" / model_id / "artifacts"
            metadata = Model.load(str(path / "MLmodel"))
            if metadata.run_id != run_id or metadata.model_id != model_id:
                raise ValueError(f"El artefacto {model_id} no corresponde al run {run_id}.")
            model = mlflow.sklearn.load_model(str(path))
            probabilities = p2_train._positive_probability(model, features)
            metrics = p2_train.evaluate_probabilities(target, probabilities)
            records.append(
                {
                    "objective": objective,
                    "role": role,
                    "run_id": run_id,
                    "model_id": model_id,
                    "model_sha256": p2_train._file_sha256(path / "model.skops"),
                    "full_year_data_version": full_year_data_version,
                    "learning_rate": config["params"]["learning_rate"],
                    "max_leaf_nodes": config["params"]["max_leaf_nodes"],
                    "evaluation_period": "2025-11-01/2026-01-01",
                    "decision_threshold": p2_train.DECISION_THRESHOLD,
                    **metrics,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    typer.echo(f"Comparación guardada en {output}")


if __name__ == "__main__":
    app()
