"""Reentrenar y comparar una versión P2 de bicicletas sin mover alias."""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.models import infer_signature
import pandas as pd
import typer

from mibici.config import EXTERNAL_DATA_DIR, MLFLOW_EXPERIMENT_NAME
from mibici.features import REQUIRED_COLUMNS
from mibici.modeling import p2_train
from mibici.modeling.mlflow_runs import HGB_TRUSTED_TYPES, calculate_data_version
from mibici.modeling.p2_catalog import load_catalog
from mibici.modeling.p2_targets import VALIDATION_END, PressureThresholds, build_p2_test_dataset

app = typer.Typer()
MODEL_NAME = "mibici-p2-bike-pressure"
CHAMPION_RUN = "d306fe5f7f34496a94f8cc3d0db2f62b"
RECIPE = "p2-bike-refit-jan-oct-2025-v1"
CATALOG = EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"


def training_rows(labeled: pd.DataFrame) -> pd.DataFrame:
    """Usar exclusivamente las ventanas train y validation previas a noviembre."""
    splits = p2_train.prepare_p2_splits(labeled)
    training = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    if training["hour"].ge(VALIDATION_END).any():
        raise ValueError("El entrenamiento contiene fechas de test.")
    return training


def verified_inputs(
    label_path: Path, hourly_path: Path, catalog_path: Path, include_current: bool = False
) -> tuple[dict, pd.DataFrame]:
    provenance = p2_train.verify_label_reproducibility(label_path, hourly_path, catalog_path)
    filters = None if include_current else [("hour", "<", VALIDATION_END)]
    hourly = pd.read_parquet(hourly_path, columns=list(REQUIRED_COLUMNS), filters=filters)
    provenance["train_validation_data_version"] = calculate_data_version(
        hourly.loc[hourly["hour"] < VALIDATION_END]
    )
    return provenance, hourly


def check_champion(client: mlflow.MlflowClient, provenance: dict):
    version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if str(version.version) != "1" or version.run_id != CHAMPION_RUN:
        raise ValueError("El champion no coincide con la versión 1 esperada.")
    run = client.get_run(CHAMPION_RUN)
    for key in ("catalog_sha256", "label_artifact_sha256", "train_validation_data_version"):
        if run.data.tags[key] != provenance[key]:
            raise ValueError(f"La procedencia no coincide con champion: {key}.")
    saved = run.data.params
    expected = provenance["label_pressure_thresholds_train_only"]
    for target in ("bike", "dock"):
        if (
            float(saved[f"{target}_pressure_ratio_threshold"])
            != expected[f"{target}_pressure_ratio"]
        ):
            raise ValueError(f"El umbral {target} no coincide con champion.")
    return run


@app.command()
def train(
    label_path: Path = p2_train.LABEL_DATA_PATH,
    hourly_path: Path = p2_train.HOURLY_DATA_PATH,
    catalog_path: Path = CATALOG,
) -> None:
    """Entrenar enero-octubre y registrar B sin modificar el alias champion."""
    client = mlflow.MlflowClient()
    existing = client.search_model_versions(f"name = '{MODEL_NAME}'")
    if any(version.tags.get("candidate_recipe") == RECIPE for version in existing):
        raise ValueError("Esta receta ya tiene una versión registrada; no se creará otra.")
    provenance, _ = verified_inputs(label_path, hourly_path, catalog_path)
    champion = check_champion(client, provenance)
    labeled = pd.read_parquet(
        label_path, filters=[[("partition", "==", "train")], [("partition", "==", "validation")]]
    )
    training = training_rows(labeled)
    parameters = {
        **p2_train.CANDIDATE_PARAMS,
        "learning_rate": float(champion.data.params["learning_rate"]),
        "max_leaf_nodes": int(champion.data.params["max_leaf_nodes"]),
    }
    fitted = p2_train.train_objective(
        training, p2_train.TARGET_BIKE_COLUMN, candidate_params=parameters
    )
    features = p2_train._feature_array(training)
    example = features[:5]
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P2-bike-refit-jan-oct-2025") as run:
        mlflow.set_tags(
            {
                "question": "P2",
                "objective": "bike_pressure",
                "candidate_recipe": RECIPE,
                "champion_source_run": CHAMPION_RUN,
                "catalog_sha256": provenance["catalog_sha256"],
                "label_artifact_sha256": provenance["label_artifact_sha256"],
                "train_validation_data_version": provenance["train_validation_data_version"],
            }
        )
        mlflow.log_params(
            {
                **parameters,
                "train_period": "2025-01-01/2025-11-01",
                "training_rows": len(training),
                "predictor_columns": ",".join(p2_train.PREDICTOR_COLUMNS),
                "decision_threshold": p2_train.DECISION_THRESHOLD,
                **{
                    f"{key}_threshold": value
                    for key, value in provenance["label_pressure_thresholds_train_only"].items()
                },
                "hyperparameters_selected_using": "champion validation 2025-09/2025-10",
            }
        )
        mlflow.log_metrics(
            {
                "train_positive_prevalence": fitted.training_prevalence,
                "fit_seconds": fitted.candidate_fit_seconds,
            }
        )
        mlflow.log_dict(provenance, "provenance/inputs.json")
        model = mlflow.sklearn.log_model(
            fitted.candidate,
            name="model",
            signature=infer_signature(example, fitted.candidate.predict(example)),
            input_example=example,
            skops_trusted_types=HGB_TRUSTED_TYPES,
        )
        registered = mlflow.register_model(
            model.model_uri, MODEL_NAME, tags={"candidate_recipe": RECIPE, "role": "challenger"}
        )
        mlflow.set_tag("registered_version", str(registered.version))
        mlflow.set_tag("logged_model_uri", model.model_uri)
        if client.get_model_version_by_alias(MODEL_NAME, "champion").run_id != CHAMPION_RUN:
            raise RuntimeError("El alias champion cambió durante el registro; revisar MLflow.")
        typer.echo(
            f"Registrado {MODEL_NAME} v{registered.version}, run {run.info.run_id}, {model.model_uri}"
        )


def comparison_rows(version: str, run_id: str, current: pd.DataFrame) -> list[dict]:
    """Calcular métricas y diferencias de dos modelos sobre el mismo dataframe."""
    features = p2_train._feature_array(current)
    target = current[p2_train.TARGET_BIKE_COLUMN].to_numpy()
    rows = []
    for role, model_version, source_run in (("A", "1", CHAMPION_RUN), ("B", version, run_id)):
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{model_version}")
        metrics = p2_train.evaluate_probabilities(
            target, p2_train._positive_probability(model, features)
        )
        rows.append(
            {
                "role": role,
                "registered_model": MODEL_NAME,
                "version": model_version,
                "source_run_id": source_run,
                "evaluation_period": "2025-11-01/2026-01-01",
                "decision_threshold": p2_train.DECISION_THRESHOLD,
                **metrics,
            }
        )
    for metric in ("average_precision", "precision", "recall", "f1", "tn", "fp", "fn", "tp"):
        a, b = rows[0][metric], rows[1][metric]
        rows[1][f"delta_{metric}"] = b - a
        rows[1][f"relative_{metric}"] = (b - a) / a if a != 0 else None
    return rows


@app.command()
def compare(
    version: str = typer.Option(..., help="Versión registrada de B."),
    output: Path = Path("reports") / "ab_testing" / "p2_bike_registered_comparison.csv",
    label_path: Path = p2_train.LABEL_DATA_PATH,
    hourly_path: Path = p2_train.HOURLY_DATA_PATH,
    catalog_path: Path = CATALOG,
) -> None:
    """Evaluar A y B en el mismo current y registrar solo la evaluación."""
    client = mlflow.MlflowClient()
    provenance, hourly = verified_inputs(
        label_path, hourly_path, catalog_path, include_current=True
    )
    check_champion(client, provenance)
    candidate = client.get_model_version(MODEL_NAME, version)
    if str(version) == "1" or candidate.tags.get("candidate_recipe") != RECIPE:
        raise ValueError("La versión indicada no es el challenger reentrenado de esta receta.")
    candidate_run = client.get_run(candidate.run_id)
    for key in ("catalog_sha256", "label_artifact_sha256", "train_validation_data_version"):
        if candidate_run.data.tags[key] != provenance[key]:
            raise ValueError(f"Procedencia del challenger incompatible: {key}.")
    thresholds = PressureThresholds(**provenance["label_pressure_thresholds_train_only"])
    current = build_p2_test_dataset(hourly, load_catalog(catalog_path, strict=False), thresholds)
    rows = comparison_rows(version, candidate.run_id, current)
    full_year_version = calculate_data_version(hourly)
    for row in rows:
        row["full_year_data_version"] = full_year_version
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P2-bike-retrospective-comparison") as run:
        mlflow.set_tags(
            {
                "evaluation_type": "retrospective_offline",
                "candidate_run_id": candidate.run_id,
                "candidate_version": str(version),
                "champion_run_id": CHAMPION_RUN,
                "full_year_data_version": full_year_version,
                **{key: provenance[key] for key in ("catalog_sha256", "label_artifact_sha256")},
            }
        )
        mlflow.log_params(
            {"evaluation_period": "2025-11-01/2026-01-01", "decision_threshold": 0.5}
        )
        mlflow.log_metrics(
            {
                f"{row['role']}_{key}": float(row[key])
                for row in rows
                for key in (
                    "average_precision",
                    "precision",
                    "recall",
                    "f1",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                    "positive_prevalence",
                    "n_evaluated",
                )
            }
        )
        mlflow.log_text(table.to_csv(index=False), "reports/p2_bike_registered_comparison.csv")
        typer.echo(f"Evaluación registrada: {run.info.run_id}")
    table.to_csv(output, index=False)
    typer.echo(f"Comparación guardada en {output}")


if __name__ == "__main__":
    app()
