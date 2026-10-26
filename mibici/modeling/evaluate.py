"""Evaluación final en test (noviembre-diciembre de 2025) de los champion de P1 y P2.

Carga cada champion del Model Registry sin reentrenarlo, reproduce sus variables y,
en P2, etiqueta test con los umbrales de train guardados en su run de origen. Los
baselines se evalúan con los modelos ya registrados en el mismo run padre. Test solo
se reporta: no se usa para elegir hiperparámetros ni modelos.

    mlflow run . -e evaluate --env-manager local
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import mlflow
from mlflow.entities import Run
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import typer

from mibici.config import CHAMPION_ALIAS, EXTERNAL_DATA_DIR, RANDOM_SEED
from mibici.features import REQUIRED_COLUMNS
from mibici.modeling import p2_train
from mibici.modeling.mlflow_runs import (
    HOURLY_PATH,
    calculate_data_version,
    child_tags,
    common_tags,
)
from mibici.modeling.p2_catalog import CATALOG_FILENAME, load_catalog
from mibici.modeling.p2_targets import (
    VALIDATION_END,
    YEAR_END,
    PressureThresholds,
    build_p2_test_dataset,
)
from mibici.modeling.registry import REGISTERED_MODELS, RegistrySpec
from mibici.modeling.train import (
    SEASONAL_COLUMN,
    TARGET_COLUMN,
    prepare_supervised_data,
    split_chronologically,
)

app = typer.Typer()

SPLIT_PREFIXES = {"validation": "val", "test": "test"}
TARGET_BY_OBJECTIVE = {name: column for column, name in p2_train.TARGET_NAMES.items()}


@dataclass(frozen=True)
class Candidate:
    """Un modelo registrado a evaluar y el run del que proviene."""

    role: str
    model_uri: str
    source: Run
    version: str | None = None


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """MAE y RMSE con las mismas fórmulas de ``train.evaluate_candidate``."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "n_evaluated": len(y_true),
    }


def p1_cohorts(hourly_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Filas de validation y test con rezago semanal, como en la selección de modelos."""
    splits = split_chronologically(prepare_supervised_data(hourly_data), evaluate_test=True)
    return {
        name: splits[name].loc[splits[name][SEASONAL_COLUMN].notna()]
        for name in ("validation", "test")
    }


def evaluate_p1(model_uri: str, cohorts: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Predecir con el modelo registrado usando las columnas de su firma."""
    model = mlflow.pyfunc.load_model(model_uri)
    columns = model.metadata.get_input_schema().input_names()
    return {
        split: regression_metrics(
            cohort[TARGET_COLUMN].to_numpy(dtype=np.float64), model.predict(cohort.loc[:, columns])
        )
        for split, cohort in cohorts.items()
    }


def evaluate_p2(model_uri: str, frames: dict[str, pd.DataFrame], target: str) -> dict[str, dict]:
    """Métricas de P2 con el umbral de decisión original de 0.5."""
    model = mlflow.sklearn.load_model(model_uri)
    return {
        split: p2_train.evaluate_probabilities(
            frame[target].to_numpy(),
            p2_train._positive_probability(model, p2_train._feature_array(frame)),
        )
        for split, frame in frames.items()
    }


def find_candidates(client: mlflow.MlflowClient, spec: RegistrySpec) -> list[Candidate]:
    """El champion y los baselines registrados en el mismo run padre que él."""
    version = client.get_model_version_by_alias(spec.name, CHAMPION_ALIAS)
    champion = client.get_run(version.run_id)
    found = [
        Candidate("champion", f"models:/{spec.name}@{CHAMPION_ALIAS}", champion, version.version)
    ]

    parent_id = champion.data.tags["mlflow.parentRunId"]
    filter_string = f"tags.mlflow.parentRunId = '{parent_id}' and tags.stage_of_work = 'baseline'"
    if spec.objective:
        filter_string += f" and tags.objective = '{spec.objective}'"
    runs = client.search_runs([champion.info.experiment_id], filter_string=filter_string)
    for run in sorted(runs, key=lambda item: item.info.run_name):
        logged = client.search_logged_models(
            [run.info.experiment_id], filter_string=f"source_run_id = '{run.info.run_id}'"
        )
        found.append(Candidate("baseline", logged[0].model_uri, run))

    return found


def check_validation_reproduced(spec: RegistrySpec, item: Candidate, metrics: dict) -> None:
    """La validation recalculada debe coincidir con la guardada al seleccionar el modelo."""
    actual = metrics["validation"][spec.metric.removeprefix("val_")]
    expected = item.source.data.metrics[spec.metric]
    if not np.isclose(actual, expected, rtol=1e-6):
        raise ValueError(
            f"{item.source.info.run_name}: {spec.metric} recalculada {actual} "
            f"no coincide con la registrada {expected}."
        )


def log_evaluation(
    spec: RegistrySpec, item: Candidate, metrics: dict, tags: dict[str, str], params: dict
) -> list[dict]:
    """Registrar un run hijo con las métricas de validation y test de un modelo."""
    source = item.source
    prefix = "P1" if spec.question == "P1" else f"P2-{spec.objective}".replace("_", "-")
    label = "champion" if item.role == "champion" else source.data.tags["model_name"]
    extra = {
        "evaluated_role": item.role,
        "evaluated_model_uri": item.model_uri,
        "source_run_id": source.info.run_id,
        "source_run_name": source.info.run_name,
    }
    if spec.objective:
        extra["objective"] = spec.objective
    if item.version:
        extra |= {"evaluated_model_name": spec.name, "evaluated_model_version": item.version}

    with mlflow.start_run(run_name=f"{prefix}-test-{label.replace('_', '-')}", nested=True):
        mlflow.set_tags(child_tags(tags, source.data.tags["model_name"], "test", **extra))
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                f"{SPLIT_PREFIXES[split]}_{name}": value
                for split, values in metrics.items()
                for name, value in values.items()
            }
        )

    return [
        {
            "question": spec.question,
            "objective": spec.objective or "departures",
            "role": item.role,
            "model_name": source.data.tags["model_name"],
            "run_name": source.info.run_name,
            "evaluation_set": split,
            **values,
        }
        for split, values in metrics.items()
    ]


def run_evaluation(
    hourly_path: Path = HOURLY_PATH,
    label_path: Path = p2_train.LABEL_DATA_PATH,
    catalog_path: Path = EXTERNAL_DATA_DIR / CATALOG_FILENAME,
) -> pd.DataFrame:
    """Evaluar en test los champion de P1 y P2 y sus baselines, y registrar los runs."""
    client = mlflow.MlflowClient()
    plan = {
        spec: find_candidates(client, spec)
        for spec in REGISTERED_MODELS
        if spec.question in ("P1", "P2")
    }
    p2_source = next(items[0].source for spec, items in plan.items() if spec.question == "P2")

    hourly = pd.read_parquet(hourly_path, columns=list(REQUIRED_COLUMNS))
    version = calculate_data_version(hourly.loc[hourly["hour"] < VALIDATION_END])
    for items in plan.values():
        if items[0].source.data.tags["train_validation_data_version"] != version:
            raise ValueError(
                "El dataset horario no coincide con el usado para seleccionar modelos."
            )

    catalog_hash = p2_train._file_sha256(catalog_path)
    label_hash = p2_train._file_sha256(label_path)
    if catalog_hash != p2_source.data.tags["catalog_sha256"]:
        raise ValueError("El catálogo local no coincide con el usado para etiquetar P2.")
    if label_hash != p2_source.data.tags["label_artifact_sha256"]:
        raise ValueError("Las etiquetas de train/validation no coinciden con las usadas en P2.")

    # Umbrales de train guardados en el run de origen; test nunca los recalcula.
    saved = p2_source.data.params
    thresholds = PressureThresholds(
        float(saved["bike_pressure_ratio_threshold"]),
        float(saved["dock_pressure_ratio_threshold"]),
    )
    p2_frames = {
        "validation": pd.read_parquet(
            label_path,
            columns=[*p2_train.PREDICTOR_COLUMNS, *p2_train.TARGET_COLUMNS],
            filters=[("partition", "==", "validation")],
        ),
        "test": build_p2_test_dataset(
            hourly, load_catalog(catalog_path, strict=False), thresholds
        ),
    }
    p1_frames = p1_cohorts(hourly)
    test_period = f"{VALIDATION_END.date()}/{YEAR_END.date()}"

    rows = []
    with mlflow.start_run():
        parent_tags = {
            "stage_of_work": "test",
            "full_year_data_version": calculate_data_version(hourly),
            "label_artifact_sha256": label_hash,
            "catalog_sha256": catalog_hash,
        }
        mlflow.set_tags({**parent_tags, "mlflow.runName": "test-evaluation"})
        mlflow.log_params({"test_period": test_period, "random_state": RANDOM_SEED})

        for spec, items in plan.items():
            tags = {**common_tags(spec.question, version), **parent_tags}
            params = {"test_period": test_period}
            if spec.question == "P2":
                params |= {
                    "decision_threshold": p2_train.DECISION_THRESHOLD,
                    **{f"{name}_threshold": value for name, value in vars(thresholds).items()},
                }
            for item in items:
                if spec.question == "P1":
                    metrics = evaluate_p1(item.model_uri, p1_frames)
                else:
                    target = TARGET_BY_OBJECTIVE[spec.objective]
                    metrics = evaluate_p2(item.model_uri, p2_frames, target)
                check_validation_reproduced(spec, item, metrics)
                rows += log_evaluation(spec, item, metrics, tags, params)

        table = pd.DataFrame(rows)
        mlflow.log_text(table.to_csv(index=False), "reports/test_evaluation.csv")

    logger.success("Evaluación final en test registrada en MLflow.")
    return table


@app.command()
def main():
    """Evaluar en test los champion de P1 y P2 y sus baselines, sin reentrenar."""
    table = run_evaluation()
    logger.info(f"Evaluación final:\n{table.to_string(index=False)}")


if __name__ == "__main__":
    app()
