"""Registro de modelos en el Model Registry de MLflow con el alias ``champion``.

Para cada modelo registrado toma el último entrenamiento de su pregunta,
elige el mejor run hijo según su métrica de validación y lo registra como una
versión nueva. El alias ``champion`` solo se mueve a esa versión si mejora la
métrica del champion actual; si no, la versión queda registrada sin alias.

    mlflow run . -e register --env-manager local
"""

from dataclasses import dataclass

from loguru import logger
import mlflow
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
import pandas as pd
import typer

from mibici.config import CHAMPION_ALIAS, MLFLOW_EXPERIMENT_NAME

app = typer.Typer()


@dataclass(frozen=True)
class RegistrySpec:
    """Qué runs alimentan un modelo registrado y cómo se comparan."""

    name: str
    question: str
    metric: str
    higher_is_better: bool
    description: str
    objective: str | None = None
    # P3 no tiene validación: se registra el k elegido en el entrenamiento.
    selected_only: bool = False


REGISTERED_MODELS = (
    RegistrySpec(
        name="mibici-p1-departures",
        question="P1",
        metric="val_mae",
        higher_is_better=False,
        description="P1: salidas por estación-hora. Se compara por val_mae (menor es mejor).",
    ),
    RegistrySpec(
        name="mibici-p2-bike-pressure",
        question="P2",
        objective="bike_pressure",
        metric="val_average_precision",
        higher_is_better=True,
        description=(
            "P2: riesgo de presión de bicicletas. "
            "Se compara por val_average_precision (mayor es mejor)."
        ),
    ),
    RegistrySpec(
        name="mibici-p2-dock-pressure",
        question="P2",
        objective="dock_pressure",
        metric="val_average_precision",
        higher_is_better=True,
        description=(
            "P2: riesgo de presión de anclajes. "
            "Se compara por val_average_precision (mayor es mejor)."
        ),
    ),
    RegistrySpec(
        name="mibici-p3-station-clusters",
        question="P3",
        metric="silhouette_score",
        higher_is_better=True,
        selected_only=True,
        description=(
            "P3: grupos de estaciones por perfil de uso. "
            "Se compara por silhouette_score (mayor es mejor)."
        ),
    ),
)


def latest_training_children(experiment_id: str, question: str) -> pd.DataFrame:
    """Runs hijos terminados del entrenamiento más reciente de una pregunta."""
    runs = mlflow.search_runs(
        [experiment_id],
        filter_string=f"tags.question = '{question}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty or "tags.model_name" not in runs:
        return runs.iloc[0:0]

    # Los runs padre son los únicos sin la etiqueta model_name.
    parents = runs.loc[runs["tags.model_name"].isna()]
    if parents.empty:
        return runs.iloc[0:0]

    latest_parent = parents.iloc[0]["run_id"]
    return runs.loc[runs["tags.mlflow.parentRunId"] == latest_parent]


def best_run_id(children: pd.DataFrame, spec: RegistrySpec) -> str | None:
    """Elegir el run hijo con la mejor métrica de validación del modelo."""
    candidates = children
    if spec.objective is not None:
        if "tags.objective" not in candidates:
            return None
        candidates = candidates.loc[candidates["tags.objective"] == spec.objective]
    if spec.selected_only:
        if "tags.selected" not in candidates:
            return None
        candidates = candidates.loc[candidates["tags.selected"] == "True"]

    column = f"metrics.{spec.metric}"
    if column not in candidates:
        return None
    candidates = candidates.dropna(subset=[column])
    if candidates.empty:
        return None

    best = candidates[column].idxmax() if spec.higher_is_better else candidates[column].idxmin()
    return candidates.loc[best, "run_id"]


def improves(new_value: float, current_value: float, higher_is_better: bool) -> bool:
    """Indicar si la métrica nueva es estrictamente mejor que la actual."""
    return new_value > current_value if higher_is_better else new_value < current_value


def ensure_version(client: mlflow.MlflowClient, spec: RegistrySpec, run: Run):
    """Registrar el modelo del run (o reutilizar su versión si ya existe)."""
    try:
        client.get_registered_model(spec.name)
    except MlflowException:
        client.create_registered_model(spec.name, description=spec.description)

    existing = client.search_model_versions(
        f"name = '{spec.name}' and run_id = '{run.info.run_id}'"
    )
    if existing:
        return existing[0]

    logged_models = client.search_logged_models(
        [run.info.experiment_id], filter_string=f"source_run_id = '{run.info.run_id}'"
    )
    if not logged_models:
        raise ValueError(f"El run {run.info.run_id} no tiene un modelo registrado.")

    tags = run.data.tags
    version = mlflow.register_model(
        logged_models[0].model_uri,
        spec.name,
        tags={
            "question": spec.question,
            "model_name": tags["model_name"],
            "selection_metric": spec.metric,
            "selection_metric_value": str(run.data.metrics[spec.metric]),
            "train_validation_data_version": tags.get("train_validation_data_version", ""),
            "git_commit": tags.get("mlflow.source.git.commit", ""),
        },
    )
    client.update_model_version(
        spec.name,
        version.version,
        description=(
            f"{tags['model_name']} del run {run.info.run_name}: "
            f"{spec.metric} = {run.data.metrics[spec.metric]:.6f}."
        ),
    )
    return version


def register_spec(client: mlflow.MlflowClient, experiment_id: str, spec: RegistrySpec) -> dict:
    """Registrar el mejor run del último entrenamiento y decidir el champion."""
    run_id = best_run_id(latest_training_children(experiment_id, spec.question), spec)
    if run_id is None:
        logger.warning(f"{spec.name}: no hay runs de {spec.question} para registrar.")
        return {"model": spec.name, "status": "sin runs"}

    run = client.get_run(run_id)
    version = ensure_version(client, spec, run)
    new_value = run.data.metrics[spec.metric]

    try:
        champion = client.get_model_version_by_alias(spec.name, CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    summary = {
        "model": spec.name,
        "version": int(version.version),
        "run_id": run_id,
        "model_name": run.data.tags["model_name"],
        spec.metric: new_value,
    }
    if champion is not None and champion.version == version.version:
        logger.info(f"{spec.name}: la versión {version.version} ya es {CHAMPION_ALIAS}.")
        return {**summary, "status": "ya es champion"}

    if champion is not None:
        current_value = float(champion.tags["selection_metric_value"])
        summary[f"champion_{spec.metric}"] = current_value
        if not improves(new_value, current_value, spec.higher_is_better):
            logger.info(
                f"{spec.name}: v{version.version} ({spec.metric}={new_value:.6f}) no mejora "
                f"a v{champion.version} ({current_value:.6f}); el champion no cambia."
            )
            return {**summary, "status": "registrado sin alias"}

    client.set_registered_model_alias(spec.name, CHAMPION_ALIAS, version.version)
    logger.success(
        f"{spec.name}: v{version.version} ({spec.metric}={new_value:.6f}) es el nuevo "
        f"{CHAMPION_ALIAS}."
    )
    return {**summary, "status": "nuevo champion"}


def register_models(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> pd.DataFrame:
    """Registrar los modelos de P1, P2 y P3 y devolver un resumen."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No existe el experimento {experiment_name}. Entrena primero.")

    client = mlflow.MlflowClient()
    summary = pd.DataFrame(
        [register_spec(client, experiment.experiment_id, spec) for spec in REGISTERED_MODELS]
    )
    if mlflow.active_run() is not None:
        mlflow.log_text(summary.to_csv(index=False), "reports/registry_summary.csv")

    return summary


@app.command()
def main(experiment_name: str = MLFLOW_EXPERIMENT_NAME):
    """Registrar en el Model Registry el mejor modelo de cada pregunta."""
    # Sin argumentos reutiliza el run abierto por `mlflow run` (o crea uno).
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "register-models")
        summary = register_models(experiment_name)
    logger.info(f"Model Registry:\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    app()
