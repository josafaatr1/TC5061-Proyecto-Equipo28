"""Registro en MLflow de los experimentos P1, P2 y P3.

Reutiliza el entrenamiento y la evaluación de ``train.py``, ``p2_train.py`` y
``clustering.py`` sin duplicar su lógica y aplica el contrato de
``references/p1_p2_mlflow_handoff.md``: un run padre por problema y un run
hijo por modelo, con métricas de validación prefijadas con ``val_``. Nunca
lee registros desde noviembre de 2025 (test reservado).

    mlflow run . -e p1_train --env-manager local
    mlflow run . -e p2_train --env-manager local
    mlflow run . -e p3_train --env-manager local
    mlflow run . -e train -P question=p1 -P learning_rate=0.05 --env-manager local
    mlflow run . -e search -P question=p1 --env-manager local
"""

from dataclasses import dataclass
import hashlib
from itertools import product
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from loguru import logger
from matplotlib.figure import Figure
import mlflow
from mlflow.data.dataset import Dataset
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    precision_recall_curve,
    silhouette_score,
)
import typer

from mibici import clustering
from mibici.config import (
    DATA_INVENTORY,
    EXTERNAL_DATA_DIR,
    HGB_PARAMS,
    P3_MAX_K,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RANDOM_SEED,
    SEARCH_LEARNING_RATES,
    SEARCH_MAX_LEAF_NODES,
    SOURCES_MANIFEST,
)
from mibici.features import REQUIRED_COLUMNS
from mibici.modeling import p2_train
from mibici.modeling.p2_catalog import AUDIT_FILENAME, CATALOG_FILENAME, load_catalog
from mibici.modeling.p2_targets import TRAIN_END, VALIDATION_END, YEAR_START
from mibici.modeling.train import (
    CANDIDATE_PARAMS,
    SEASONAL_COLUMN,
    TARGET_COLUMN,
    evaluate_baselines,
    evaluate_candidate,
    prepare_supervised_data,
    split_chronologically,
    train_candidate_model,
    train_dummy_regressor,
)
from mibici.tracking import calculate_data_version as calculate_raw_data_version

app = typer.Typer()

TEAM = "Equipo 28"
HOURLY_PATH = PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet"
DATASET_NAME = "station_hourly_2025_train_validation"
PR_CURVE_MAX_POINTS = 1000

# skops (serializador de MLflow) exige declarar los tipos internos de los
# árboles de HistGradientBoosting; los modelos se entrenan en este proyecto.
HGB_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
# La transformación logarítmica del pipeline de P3 es una función del proyecto.
P3_TRUSTED_TYPES = ["mibici.clustering.log_transform"]
P3_BASELINE_NAME = "baseline_activity_quantiles"
P3_KMEANS_NAME = "kmeans"
P1_CANDIDATE_NAME = "hist_gradient_boosting_poisson"

# Nombres de run del contrato (sección 9 del handoff).
P1_RUN_NAMES = {
    "dummy_mean": "P1-regression-dummy-mean",
    "seasonal_weekly": "P1-regression-seasonal-168h",
    "hist_gradient_boosting_poisson": "P1-regression-hist-gradient-boosting-poisson",
}
P1_MODEL_PARAMS = {
    "dummy_mean": {
        "model_class": "sklearn.dummy.DummyRegressor",
        "strategy": "mean",
    },
    "seasonal_weekly": {
        "model_class": "mibici.modeling.mlflow_runs.SeasonalNaiveModel",
        "lag_column": SEASONAL_COLUMN,
    },
    "hist_gradient_boosting_poisson": {
        "model_class": "sklearn.ensemble.HistGradientBoostingRegressor",
        **CANDIDATE_PARAMS,
    },
}
P2_MODEL_PARAMS = {
    p2_train.BASELINE_NAME: {
        "model_class": "sklearn.dummy.DummyClassifier",
        "strategy": "prior",
    },
    p2_train.CANDIDATE_NAME: {
        "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
        **p2_train.CANDIDATE_PARAMS,
    },
}
# Valores aceptados por el parámetro `model` del entry point `train`.
TRAIN_MODELS = {
    "p1": ("dummy_mean", "seasonal_weekly", P1_CANDIDATE_NAME),
    "p2": (p2_train.BASELINE_NAME, p2_train.CANDIDATE_NAME),
    "p3": (P3_BASELINE_NAME, P3_KMEANS_NAME),
}
# Familia del modelo: etiqueta `model_family` de cada run hijo de P1 y P2.
MODEL_FAMILIES = {
    "dummy_mean": "dummy",
    "seasonal_weekly": "seasonal_naive",
    P1_CANDIDATE_NAME: "hist_gradient_boosting",
    p2_train.BASELINE_NAME: "dummy",
    p2_train.CANDIDATE_NAME: "hist_gradient_boosting",
}
P2_VALIDATION_METRICS = (
    "average_precision",
    "precision",
    "recall",
    "f1",
    "positive_prevalence",
    "n_evaluated",
    "n_excluded",
    "tn",
    "fp",
    "fn",
    "tp",
)


class SeasonalNaiveModel(mlflow.pyfunc.PythonModel):
    """Baseline estacional de P1: predice las salidas de hace 168 horas."""

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        """Devolver el rezago semanal como predicción."""
        return model_input[SEASONAL_COLUMN].to_numpy(dtype=np.float64)


class ActivityQuantileModel(mlflow.pyfunc.PythonModel):
    """Baseline de P3: agrupa estaciones por cuantiles de actividad total."""

    def __init__(self, edges: list[float]):
        self.edges = edges

    @classmethod
    def fit(cls, total_activity: pd.Series, n_groups: int) -> ActivityQuantileModel:
        """Calcular los cortes de ``n_groups`` grupos de igual tamaño."""
        quantiles = np.linspace(0, 1, n_groups + 1)[1:-1]
        return cls(total_activity.quantile(quantiles).tolist())

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        """Asignar a cada estación el grupo de su actividad total."""
        return np.searchsorted(self.edges, model_input["total_activity"], side="right")


def calculate_data_version(hourly_data: pd.DataFrame) -> str:
    """Calcular una huella del contenido estación-hora usado por los modelos.

    La huella no depende del orden de las filas ni del índice, y cambia si
    cambia cualquier valor de las columnas del esquema horario.
    """
    data = (
        hourly_data.loc[:, list(REQUIRED_COLUMNS)]
        .sort_values(["station_id", "hour"], kind="stable")
        .reset_index(drop=True)
    )
    row_hashes = pd.util.hash_pandas_object(data, index=False)

    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def read_train_validation_hourly(
    hourly_path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Leer el Parquet horario solo de enero a octubre de 2025."""
    return pd.read_parquet(
        hourly_path,
        columns=columns,
        filters=[("hour", ">=", YEAR_START), ("hour", "<", VALIDATION_END)],
    )


def git_commit() -> str | None:
    """Obtener el commit de Git ejecutado, si el proyecto está en un repositorio."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None

    return result.stdout.strip()


def common_tags(question: str, data_version: str) -> dict[str, str]:
    """Tags de trazabilidad compartidos por el run padre y sus hijos."""
    tags = {
        "question": question,
        "team": TEAM,
        "train_validation_data_version": data_version,
        "python_version": platform.python_version(),
        "scikit_learn_version": SKLEARN_VERSION,
    }
    commit = git_commit()
    if commit:
        tags["mlflow.source.git.commit"] = commit

    return tags


def period_params(train_rows: int, validation_rows: int) -> dict[str, object]:
    """Periodos (fin exclusivo) y tamaños de train/validation."""
    return {
        "random_state": RANDOM_SEED,
        "train_period": f"{YEAR_START.date()}/{TRAIN_END.date()}",
        "validation_period": f"{TRAIN_END.date()}/{VALIDATION_END.date()}",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
    }


def p2_run_name(target_name: str, model_name: str) -> str:
    """Nombre de run P2 del contrato, p. ej. ``P2-bike-pressure-dummy-prior``."""
    return f"P2-{target_name}-{model_name}".replace("_", "-")


def precision_recall_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    max_points: int = PR_CURVE_MAX_POINTS,
) -> pd.DataFrame:
    """Curva Precision-Recall reducida a ``max_points`` umbrales equiespaciados."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    if len(table) > max_points:
        positions = np.unique(np.linspace(0, len(table) - 1, max_points).round().astype(int))
        table = table.iloc[positions].reset_index(drop=True)

    return table


def hourly_profile_table(
    validation: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Promedio real y predicho de salidas por hora del día en validation."""
    return (
        pd.DataFrame(
            {
                "hour_of_day": validation["hour_of_day"].to_numpy(),
                "actual": validation[TARGET_COLUMN].to_numpy(dtype=np.float64),
                "predicted": predictions,
            }
        )
        .groupby("hour_of_day", as_index=False)
        .mean()
    )


def hourly_profile_figure(table: pd.DataFrame, title: str) -> Figure:
    """Graficar el perfil horario real contra el predicho."""
    figure = Figure(figsize=(6.5, 4), layout="constrained")
    axes = figure.subplots()
    axes.plot(table["hour_of_day"], table["actual"], marker="o", label="Real")
    axes.plot(table["hour_of_day"], table["predicted"], marker="o", label="Predicho")
    axes.set_xticks(range(0, 24, 2))
    axes.set_xlabel("Hora del día")
    axes.set_ylabel("Salidas promedio por estación")
    axes.set_title(title)
    axes.legend()

    return figure


def confusion_matrix_figure(tn: int, fp: int, fn: int, tp: int, title: str) -> Figure:
    """Graficar la matriz de confusión ``[[TN, FP], [FN, TP]]``."""
    matrix = np.array([[tn, fp], [fn, tp]])
    figure = Figure(figsize=(5, 4.5), layout="constrained")
    axes = figure.subplots()
    axes.imshow(matrix, cmap="Blues")

    for (row, column), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        axes.text(column, row, f"{value:,}", ha="center", va="center", color=color)

    axes.set_xticks([0, 1], labels=["Negativo", "Positivo"])
    axes.set_yticks([0, 1], labels=["Negativo", "Positivo"])
    axes.set_xlabel("Predicho")
    axes.set_ylabel("Real")
    axes.set_title(title)

    return figure


def precision_recall_figure(
    table: pd.DataFrame,
    average_precision: float,
    prevalence: float,
    title: str,
) -> Figure:
    """Graficar la curva Precision-Recall junto a la prevalencia positiva."""
    figure = Figure(figsize=(5.5, 4.5), layout="constrained")
    axes = figure.subplots()
    axes.plot(table["recall"], table["precision"], label=f"AP = {average_precision:.3f}")
    axes.axhline(prevalence, linestyle="--", color="gray", label="Prevalencia")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(title)
    axes.legend()

    return figure


@dataclass(frozen=True)
class FittedRun:
    """Modelo ajustado de P1 y lo necesario para registrar su run hijo."""

    run_name: str
    model_name: str
    estimator: object
    columns: list[str]
    params: dict[str, object]
    stage: str


def parse_search_grid(learning_rates: str, max_leaf_nodes: str) -> list[dict[str, float | int]]:
    """Combinaciones de hiperparámetros a partir de listas separadas por comas."""
    return [
        {"learning_rate": float(rate), "max_leaf_nodes": int(leaves)}
        for rate, leaves in product(learning_rates.split(","), max_leaf_nodes.split(","))
    ]


def search_suffix(params: dict[str, float | int]) -> str:
    """Sufijo del nombre de un run de búsqueda, p. ej. ``lr0.05-leaf15``."""
    return f"lr{params['learning_rate']}-leaf{params['max_leaf_nodes']}"


def data_manifest(data_version: str, dataset_path: Path, **provenance: object) -> dict:
    """Procedencia de los datos de un run: huellas, archivo de origen y periodos."""
    inventory = pd.read_csv(DATA_INVENTORY, dtype=str)
    return {
        "train_validation_data_version": data_version,
        "raw_data_version": calculate_raw_data_version(inventory),
        "dataset_file": dataset_path.name,
        "sources_manifest": SOURCES_MANIFEST.name,
        "raw_inventory": DATA_INVENTORY.name,
        **provenance,
    }


def child_tags(tags: dict[str, str], model_name: str, stage: str, **extra: str) -> dict[str, str]:
    """Tags de un run hijo: los comunes más modelo, familia y etapa de trabajo."""
    return {
        **tags,
        **extra,
        "model_name": model_name,
        "model_family": MODEL_FAMILIES[model_name],
        "stage_of_work": stage,
    }


def log_child_metadata(
    tags: dict[str, str],
    params: dict[str, object],
    metrics: dict[str, float],
    dataset: Dataset,
    manifest: dict,
) -> None:
    """Registrar lo común de un run hijo: tags, parámetros, datos, métricas y procedencia."""
    mlflow.set_tags(tags)
    mlflow.log_params(params)
    mlflow.log_input(dataset, context="training")
    mlflow.log_metrics(metrics)
    mlflow.log_dict({"tags": tags, "params": params}, "config/run_config.json")
    mlflow.log_dict(manifest, "manifests/data_manifest.json")
    for path in (SOURCES_MANIFEST, DATA_INVENTORY):
        mlflow.log_artifact(str(path), artifact_path="manifests")


def predict_with(estimator: object, model_input: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predecir con un modelo de scikit-learn o con uno pyfunc del proyecto."""
    if isinstance(estimator, mlflow.pyfunc.PythonModel):
        return estimator.predict(None, model_input)
    return estimator.predict(model_input)


def log_signed_model(estimator: object, example: pd.DataFrame | np.ndarray, model_name: str):
    """Registrar el modelo con un ejemplo de entrada y la firma inferida de sus predicciones."""
    signature = infer_signature(example, predict_with(estimator, example))
    if isinstance(estimator, mlflow.pyfunc.PythonModel):
        mlflow.pyfunc.log_model(
            name="model", python_model=estimator, input_example=example, signature=signature
        )
        return

    is_hgb = MODEL_FAMILIES[model_name] == "hist_gradient_boosting"
    mlflow.sklearn.log_model(
        estimator,
        name="model",
        input_example=example,
        signature=signature,
        skops_trusted_types=HGB_TRUSTED_TYPES if is_hgb else None,
    )


def selection_summary(table: pd.DataFrame, metric: str, higher_is_better: bool) -> dict:
    """Comparar el mejor candidato con el mejor baseline según una métrica de validación.

    ``table`` tiene una fila por run, con las columnas ``run_name``, ``stage`` y ``metric``.
    """

    def best(rows: pd.DataFrame) -> pd.Series:
        values = rows[metric]
        return rows.loc[values.idxmax() if higher_is_better else values.idxmin()]

    is_baseline = table["stage"] == "baseline"
    baseline = best(table.loc[is_baseline])
    candidate = best(table.loc[~is_baseline])
    gain = candidate[metric] - baseline[metric]
    if not higher_is_better:
        gain = -gain

    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "best_baseline": {"run_name": baseline["run_name"], "value": float(baseline[metric])},
        "best_candidate": {"run_name": candidate["run_name"], "value": float(candidate[metric])},
        "n_candidates": int((~is_baseline).sum()),
        "improvement": float(gain),
        "improvement_pct": float(100 * gain / abs(baseline[metric])),
    }


def log_selection(summaries: dict[str, dict]) -> None:
    """Guardar en el run padre qué candidato gana frente al mejor baseline."""
    mlflow.log_dict(summaries, "reports/selection_summary.json")
    for label, summary in summaries.items():
        mlflow.set_tag(f"selected_run_{label}", summary["best_candidate"]["run_name"])
        mlflow.log_metric(f"{label}_improvement_pct", summary["improvement_pct"])


def run_p1(
    hourly_path: Path = HOURLY_PATH,
    hgb_params: dict | None = None,
    model: str = "all",
    search_grid: list[dict] | None = None,
) -> pd.DataFrame:
    """Entrenar y evaluar P1 en train/validation y registrar un run por modelo.

    ``hgb_params`` reemplaza hiperparámetros del candidato y ``model`` limita
    los runs hijos a un solo modelo (``"all"`` registra todos). ``search_grid``
    (un diccionario de hiperparámetros por combinación) registra una búsqueda:
    el run padre es ``P1-search`` y cada combinación es un hijo candidato.
    """
    searching = search_grid is not None
    configs = search_grid if searching else [hgb_params or {}]
    stage = "search" if searching else "training"
    hourly_data = read_train_validation_hourly(hourly_path)
    data_version = calculate_data_version(hourly_data)

    supervised_data = prepare_supervised_data(hourly_data)
    splits = split_chronologically(supervised_data, evaluate_test=False)
    baseline_metrics = evaluate_baselines(splits)
    baseline_metrics = baseline_metrics.assign(
        run_name=baseline_metrics["model"].map(P1_RUN_NAMES), stage="baseline"
    )

    # Cohorte de evaluación de train.py: filas de validation con rezago semanal.
    cohort = splits["validation"]
    cohort = cohort.loc[cohort[SEASONAL_COLUMN].notna()]
    fitted = [
        FittedRun(
            P1_RUN_NAMES["dummy_mean"],
            "dummy_mean",
            train_dummy_regressor(splits["train"]),
            ["hour_of_day"],
            {**P1_MODEL_PARAMS["dummy_mean"], "predictor_columns": "ninguno (media de train)"},
            "baseline",
        ),
        FittedRun(
            P1_RUN_NAMES["seasonal_weekly"],
            "seasonal_weekly",
            SeasonalNaiveModel(),
            [SEASONAL_COLUMN],
            {**P1_MODEL_PARAMS["seasonal_weekly"], "predictor_columns": SEASONAL_COLUMN},
            "baseline",
        ),
    ]

    all_metrics = [baseline_metrics]
    for config in configs:
        candidate_params = {**CANDIDATE_PARAMS, **config}
        run_name = (
            f"P1-regression-search-{search_suffix(config)}"
            if searching
            else P1_RUN_NAMES[P1_CANDIDATE_NAME]
        )

        # Entrenar una sola vez para poder registrar el modelo evaluado.
        fit_start = perf_counter()
        candidate = train_candidate_model(splits["train"], candidate_params)
        fit_seconds = perf_counter() - fit_start
        candidate_metrics = evaluate_candidate(splits, model=candidate, fit_seconds=fit_seconds)
        all_metrics.append(candidate_metrics.assign(run_name=run_name, stage=stage, **config))

        feature_columns = list(candidate.feature_names_in_)
        params = {
            **P1_MODEL_PARAMS[P1_CANDIDATE_NAME],
            **candidate_params,
            "predictor_columns": ",".join(feature_columns),
        }
        fitted.append(
            FittedRun(run_name, P1_CANDIDATE_NAME, candidate, feature_columns, params, stage)
        )
    metrics = pd.concat(all_metrics, ignore_index=True)

    tags = {**common_tags("P1", data_version), "objective": "departures por estación-hora"}
    shared_params = period_params(len(splits["train"]), len(splits["validation"]))
    dataset = mlflow.data.from_pandas(hourly_data, source=str(hourly_path), name=DATASET_NAME)
    manifest = data_manifest(data_version, hourly_path, **shared_params)
    validation_table = metrics.loc[metrics["evaluation_set"] == "validation"].rename(
        columns={"mae": "val_mae"}
    )

    with mlflow.start_run():
        mlflow.set_tags(
            {
                **tags,
                "mlflow.runName": "P1-search" if searching else "P1-regression",
                "stage_of_work": stage,
            }
        )
        mlflow.log_params(shared_params)
        mlflow.log_text(metrics.to_csv(index=False), "reports/p1_model_comparison.csv")
        mlflow.log_dict(
            {
                "tags": tags,
                "params": shared_params,
                "predictor_columns": feature_columns,
                "excluded_predictor_columns": ["station_id", "hour", "target_departures"],
                "candidate_parameters": [{**CANDIDATE_PARAMS, **config} for config in configs],
            },
            "reports/p1_run_report.json",
        )
        log_selection({"departures": selection_summary(validation_table, "val_mae", False)})

        for item in fitted:
            if model not in ("all", item.model_name):
                continue
            by_split = metrics.loc[metrics["run_name"] == item.run_name].set_index(
                "evaluation_set"
            )
            validation = by_split.loc["validation"]
            train = by_split.loc["train"]

            with mlflow.start_run(run_name=item.run_name, nested=True):
                log_child_metadata(
                    child_tags(tags, item.model_name, item.stage),
                    {**shared_params, **item.params},
                    {
                        "val_mae": validation["mae"],
                        "val_rmse": validation["rmse"],
                        "val_n_total": validation["n_total"],
                        "val_n_evaluated": validation["n_evaluated"],
                        "val_n_excluded": validation["n_excluded"],
                        "train_mae": train["mae"],
                        "train_rmse": train["rmse"],
                        "fit_seconds": validation["fit_seconds"],
                        "evaluation_seconds": validation["evaluation_seconds"],
                    },
                    dataset,
                    manifest,
                )

                model_input = cohort.loc[:, item.columns]
                profile = hourly_profile_table(cohort, predict_with(item.estimator, model_input))
                mlflow.log_text(profile.to_csv(index=False), "evaluation/hourly_profile.csv")
                mlflow.log_figure(
                    hourly_profile_figure(profile, f"Validation · {item.run_name}"),
                    "evaluation/hourly_profile.png",
                )
                log_signed_model(item.estimator, model_input.head(5), item.model_name)

    logger.success("Runs P1 registrados en MLflow.")
    return metrics


def run_p2(
    label_path: Path = p2_train.LABEL_DATA_PATH,
    hourly_path: Path = HOURLY_PATH,
    catalog_path: Path = EXTERNAL_DATA_DIR / CATALOG_FILENAME,
    hgb_params: dict | None = None,
    model: str = "all",
    search_grid: list[dict] | None = None,
) -> pd.DataFrame:
    """Reproducir etiquetas, entrenar P2 y registrar un run por objetivo y modelo.

    ``hgb_params`` reemplaza hiperparámetros del candidato y ``model`` limita
    los runs hijos a un solo modelo (``"all"`` registra todos). ``search_grid``
    (un diccionario de hiperparámetros por combinación) registra una búsqueda:
    el run padre es ``P2-search`` y cada combinación y objetivo es un hijo.
    """
    searching = search_grid is not None
    configs = search_grid if searching else [hgb_params or {}]
    stage = "search" if searching else "training"
    provenance = p2_train.verify_label_reproducibility(
        label_path=label_path,
        hourly_path=hourly_path,
        catalog_path=catalog_path,
    )
    data_version = calculate_data_version(
        read_train_validation_hourly(hourly_path, columns=list(REQUIRED_COLUMNS))
    )
    labeled_data = pd.read_parquet(
        label_path,
        columns=[
            "station_id",
            "hour",
            "partition",
            *p2_train.PREDICTOR_COLUMNS,
            *p2_train.TARGET_COLUMNS,
        ],
        filters=[[("partition", "==", "train")], [("partition", "==", "validation")]],
    )
    experiments = [
        (
            config,
            p2_train.run_validation_experiment(
                labeled_data, {**p2_train.CANDIDATE_PARAMS, **config}
            ),
        )
        for config in configs
    ]

    # Una fila por run hijo; los baselines no dependen de los hiperparámetros.
    children = []
    for position, (config, experiment) in enumerate(experiments):
        for row in experiment.metrics.to_dict(orient="records"):
            is_candidate = row["model"] == p2_train.CANDIDATE_NAME
            if not is_candidate and position > 0:
                continue
            label = (
                f"search-{search_suffix(config)}" if searching and is_candidate else row["model"]
            )
            row = {
                **row,
                **config,
                "run_name": p2_run_name(row["target_name"], label),
                "stage": stage if is_candidate else "baseline",
            }
            children.append((row, experiment.models[row["target"]], config))
    metrics = pd.DataFrame([row for row, _, _ in children])

    # Mismas filas de validation que usó la evaluación, para las curvas PR.
    splits = p2_train.prepare_p2_splits(labeled_data)
    validation_features = p2_train._feature_array(splits["validation"])

    thresholds = provenance["label_pressure_thresholds_train_only"]
    tags = {
        **common_tags("P2", data_version),
        "catalog_version": provenance["catalog_version"],
        "catalog_sha256": provenance["catalog_sha256"],
        "label_artifact_sha256": provenance["label_artifact_sha256"],
        "labels_reproduced_exactly": str(provenance["labels_reproduced_exactly"]),
    }
    shared_params = {
        **period_params(len(splits["train"]), len(splits["validation"])),
        "predictor_columns": ",".join(p2_train.PREDICTOR_COLUMNS),
        "decision_threshold": p2_train.DECISION_THRESHOLD,
        "label_pressure_quantile": provenance["label_pressure_quantile"],
        "bike_pressure_ratio_threshold": thresholds["bike_pressure_ratio"],
        "dock_pressure_ratio_threshold": thresholds["dock_pressure_ratio"],
    }
    dataset = mlflow.data.from_pandas(labeled_data, source=str(label_path), name="p2_targets")
    manifest = data_manifest(
        data_version,
        label_path,
        hourly_file=hourly_path.name,
        **{
            key: tags[key]
            for key in ("catalog_version", "catalog_sha256", "label_artifact_sha256")
        },
        **shared_params,
    )
    validation_table = metrics.rename(columns={"average_precision": "val_average_precision"})
    summaries = {
        target_name: selection_summary(
            validation_table.loc[validation_table["target_name"] == target_name],
            "val_average_precision",
            True,
        )
        for target_name in p2_train.TARGET_NAMES.values()
    }

    with mlflow.start_run():
        mlflow.set_tags(
            {
                **tags,
                "mlflow.runName": "P2-search" if searching else "P2-classification",
                "stage_of_work": stage,
            }
        )
        mlflow.log_params(shared_params)
        mlflow.log_text(metrics.to_csv(index=False), "reports/p2_model_comparison.csv")
        mlflow.log_dict(
            {
                "data_provenance": provenance,
                "tags": tags,
                "params": shared_params,
                "candidate_parameters": [
                    {**p2_train.CANDIDATE_PARAMS, **config} for config in configs
                ],
            },
            "reports/p2_run_report.json",
        )
        log_selection(summaries)
        audit_path = catalog_path.with_name(AUDIT_FILENAME)
        if audit_path.is_file():
            mlflow.log_artifact(str(audit_path), artifact_path="reports")

        for row, objective, config in children:
            if model not in ("all", row["model"]):
                continue
            is_candidate = row["model"] == p2_train.CANDIDATE_NAME
            estimator = objective.candidate if is_candidate else objective.dummy
            title = f"{row['target_name']} · {row['run_name']}"

            with mlflow.start_run(run_name=row["run_name"], nested=True):
                log_child_metadata(
                    child_tags(tags, row["model"], row["stage"], objective=row["target_name"]),
                    {
                        **shared_params,
                        **P2_MODEL_PARAMS[row["model"]],
                        **({**p2_train.CANDIDATE_PARAMS, **config} if is_candidate else {}),
                    },
                    {
                        **{f"val_{name}": row[name] for name in P2_VALIDATION_METRICS},
                        "train_positive_prevalence": objective.training_prevalence,
                        "fit_seconds": row["fit_seconds"],
                        "evaluation_seconds": row["evaluation_seconds"],
                    },
                    dataset,
                    manifest,
                )

                mlflow.log_dict(
                    {name: row[name] for name in ("tn", "fp", "fn", "tp")},
                    "evaluation/confusion_matrix.json",
                )
                mlflow.log_figure(
                    confusion_matrix_figure(row["tn"], row["fp"], row["fn"], row["tp"], title),
                    "evaluation/confusion_matrix.png",
                )

                probabilities = p2_train._positive_probability(estimator, validation_features)
                curve = precision_recall_table(
                    splits["validation"][row["target"]].to_numpy(),
                    probabilities,
                )
                mlflow.log_text(curve.to_csv(index=False), "evaluation/precision_recall.csv")
                mlflow.log_figure(
                    precision_recall_figure(
                        curve, row["average_precision"], row["positive_prevalence"], title
                    ),
                    "evaluation/precision_recall.png",
                )

                log_signed_model(estimator, validation_features[:5], row["model"])

    logger.success("Runs P2 registrados en MLflow.")
    return metrics


def station_map_figure(
    stations: pd.DataFrame,
    labels: np.ndarray,
    catalog: pd.DataFrame,
    title: str,
) -> Figure:
    """Ubicar las estaciones en un mapa coloreadas por grupo."""
    locations = (
        pd.DataFrame({"id": stations.index, "cluster": labels})
        .merge(catalog.loc[:, ["id", "latitude", "longitude"]], on="id", how="inner")
        .dropna(subset=["latitude", "longitude"])
    )

    figure = Figure(figsize=(7, 7), layout="constrained")
    axes = figure.subplots()
    scatter = axes.scatter(
        locations["longitude"],
        locations["latitude"],
        c=locations["cluster"],
        cmap="tab10",
        vmin=0,
        vmax=9,
        s=18,
        alpha=0.85,
    )
    axes.set_aspect("equal")
    axes.set_xlabel("Longitud")
    axes.set_ylabel("Latitud")
    axes.set_title(title)
    axes.legend(*scatter.legend_elements(), title="Grupo")

    return figure


def clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Métricas internas de calidad y tamaño de los grupos."""
    sizes = np.bincount(labels)
    return {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
        "n_clusters": len(sizes),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
    }


def run_p3(
    hourly_path: Path = HOURLY_PATH,
    catalog_path: Path = EXTERNAL_DATA_DIR / CATALOG_FILENAME,
    k: int | None = None,
    max_k: int = P3_MAX_K,
    model: str = "all",
) -> pd.DataFrame:
    """Agrupar estaciones con K-Means y registrar un run por modelo.

    ``k`` fija la cantidad de grupos seleccionada (si no, se elige por
    silhouette entre 2 y ``max_k``) y ``model`` limita los runs hijos al
    baseline o a K-Means (``"all"`` registra ambos).
    """
    hourly_data = read_train_validation_hourly(hourly_path)
    data_version = calculate_data_version(hourly_data)

    features = clustering.build_station_features(hourly_data)
    result = clustering.fit_clusters(features, max_k, k)
    # Mismo espacio transformado y estandarizado para comparar todos los modelos.
    X_scaled = clustering.scaled_features(result.model, features)

    if catalog_path.is_file():
        catalog = load_catalog(catalog_path, strict=False)
    else:
        catalog = None
        logger.warning(f"No se encontró el catálogo {catalog_path}; se omite el mapa.")

    feature_columns = list(features.columns)
    tags = {**common_tags("P3", data_version), "objective": "perfiles de uso por estación"}
    shared_params = {
        "random_state": RANDOM_SEED,
        "data_period": f"{YEAR_START.date()}/{VALIDATION_END.date()}",
        "n_stations": len(features),
        "feature_columns": ",".join(feature_columns),
        "transform": "log1p (con signo en mean_net_flow) + StandardScaler",
        "k_range": f"2-{result.evaluation['k'].max()}",
        "selection_metric": "silhouette_score" if k is None else "fixed_k",
    }
    dataset = mlflow.data.from_pandas(hourly_data, source=str(hourly_path), name=DATASET_NAME)

    candidates = {}
    if model in ("all", P3_BASELINE_NAME):
        baseline = ActivityQuantileModel.fit(features["total_activity"], result.best_k)
        candidates[P3_BASELINE_NAME] = (baseline, baseline.predict(None, features))
    if model in ("all", P3_KMEANS_NAME):
        for n_clusters in result.evaluation["k"]:
            kmeans = clustering.make_model(n_clusters)
            candidates[f"kmeans_k{n_clusters}"] = (kmeans, kmeans.fit_predict(features))

    summary = []
    with mlflow.start_run():
        mlflow.set_tags(
            {
                **tags,
                "mlflow.runName": "P3-clustering",
                "selected_model": f"kmeans_k{result.best_k}",
            }
        )
        mlflow.log_params({**shared_params, "selected_k": result.best_k})
        mlflow.log_text(features.to_csv(), "reports/station_features.csv")
        mlflow.log_text(result.evaluation.to_csv(index=False), "reports/p3_k_evaluation.csv")
        mlflow.log_figure(
            clustering.silhouette_figure(result.evaluation), "evaluation/silhouette_by_k.png"
        )

        for model_name, (estimator, labels) in candidates.items():
            is_baseline = model_name == P3_BASELINE_NAME
            metrics = clustering_metrics(X_scaled, labels)
            title = f"{model_name} ({metrics['n_clusters']} grupos)"

            with mlflow.start_run(
                run_name=f"P3-clustering-{model_name}".replace("_", "-"), nested=True
            ):
                selected = model_name == f"kmeans_k{result.best_k}"
                mlflow.set_tags({**tags, "model_name": model_name, "selected": str(selected)})
                if is_baseline:
                    model_params = {
                        "model_class": "mibici.modeling.mlflow_runs.ActivityQuantileModel",
                        "n_groups": result.best_k,
                        "quantile_column": "total_activity",
                    }
                else:
                    model_params = {
                        "model_class": "sklearn.cluster.KMeans",
                        "n_clusters": int(metrics["n_clusters"]),
                        "n_init": clustering.N_INIT,
                    }
                mlflow.log_params({**shared_params, **model_params})
                mlflow.log_input(dataset, context="training")

                if not is_baseline:
                    metrics["inertia"] = float(estimator[-1].inertia_)
                mlflow.log_metrics(metrics)

                mlflow.log_text(
                    features.assign(cluster=labels).to_csv(), "evaluation/station_clusters.csv"
                )
                mlflow.log_text(
                    clustering.cluster_profiles(features, labels).to_csv(),
                    "evaluation/cluster_profiles.csv",
                )
                mlflow.log_figure(
                    clustering.pca_figure(X_scaled, labels, int(metrics["n_clusters"])),
                    "evaluation/pca.png",
                )
                if catalog is not None:
                    mlflow.log_figure(
                        station_map_figure(features, labels, catalog, title),
                        "evaluation/station_map.png",
                    )

                if is_baseline:
                    mlflow.pyfunc.log_model(
                        name="model", python_model=estimator, input_example=features.head(5)
                    )
                else:
                    mlflow.sklearn.log_model(
                        estimator,
                        name="model",
                        input_example=features.head(5),
                        skops_trusted_types=P3_TRUSTED_TYPES,
                    )

            summary.append({"model": model_name, "selected": selected, **metrics})

    logger.success("Runs P3 registrados en MLflow.")
    return pd.DataFrame(summary)


@app.command("p1")
def p1_command(hourly_path: Path = HOURLY_PATH):
    """Registrar en MLflow los modelos P1 (solo train/validation)."""
    metrics = run_p1(hourly_path)
    logger.info(f"Métricas P1:\n{metrics.to_string(index=False)}")


@app.command("p2")
def p2_command(
    label_path: Path = p2_train.LABEL_DATA_PATH,
    hourly_path: Path = HOURLY_PATH,
    catalog_path: Path = EXTERNAL_DATA_DIR / CATALOG_FILENAME,
):
    """Registrar en MLflow los modelos P2 (solo train/validation)."""
    metrics = run_p2(label_path, hourly_path, catalog_path)
    logger.info(f"Métricas P2:\n{metrics.to_string(index=False)}")


@app.command("p3")
def p3_command(
    hourly_path: Path = HOURLY_PATH,
    catalog_path: Path = EXTERNAL_DATA_DIR / CATALOG_FILENAME,
):
    """Registrar en MLflow el agrupamiento de estaciones P3 (enero-octubre)."""
    metrics = run_p3(hourly_path, catalog_path)
    logger.info(f"Métricas P3:\n{metrics.to_string(index=False)}")


@app.command("train")
def train_command(
    question: str = typer.Option(..., help="Pregunta a entrenar: p1, p2 o p3."),
    model: str = typer.Option("all", help="Modelo a registrar o 'all' para todos."),
    learning_rate: float = HGB_PARAMS["learning_rate"],
    max_iter: int = HGB_PARAMS["max_iter"],
    max_leaf_nodes: int = HGB_PARAMS["max_leaf_nodes"],
    min_samples_leaf: int = HGB_PARAMS["min_samples_leaf"],
    l2_regularization: float = HGB_PARAMS["l2_regularization"],
    k: int = typer.Option(0, help="Grupos de P3; 0 los elige por silhouette."),
    max_k: int = P3_MAX_K,
):
    """Entrenar una pregunta con hiperparámetros elegidos y registrarla en MLflow.

    Los hiperparámetros de HistGradientBoosting aplican a P1 y P2; ``k`` y
    ``max_k`` aplican a P3. Los valores por defecto vienen de ``config.py``.
    """
    question = question.lower()
    if question not in TRAIN_MODELS:
        raise typer.BadParameter(f"Pregunta no reconocida: {question}. Usa p1, p2 o p3.")
    if model not in ("all", *TRAIN_MODELS[question]):
        options = ", ".join(("all", *TRAIN_MODELS[question]))
        raise typer.BadParameter(f"Modelo no reconocido para {question}: {model}. Usa {options}.")

    hgb_params = {
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
    }
    if question == "p1":
        metrics = run_p1(hgb_params=hgb_params, model=model)
    elif question == "p2":
        metrics = run_p2(hgb_params=hgb_params, model=model)
    else:
        metrics = run_p3(k=k or None, max_k=max_k, model=model)
    logger.info(f"Métricas {question.upper()}:\n{metrics.to_string(index=False)}")


@app.command("search")
def search_command(
    question: str = typer.Option(..., help="Pregunta a explorar: p1 o p2."),
    learning_rates: str = typer.Option(
        ",".join(map(str, SEARCH_LEARNING_RATES)), help="Valores de learning_rate, con comas."
    ),
    max_leaf_nodes: str = typer.Option(
        ",".join(map(str, SEARCH_MAX_LEAF_NODES)), help="Valores de max_leaf_nodes, con comas."
    ),
):
    """Buscar hiperparámetros de HistGradientBoosting: un run hijo por combinación."""
    question = question.lower()
    if question not in ("p1", "p2"):
        raise typer.BadParameter(f"Pregunta no reconocida: {question}. Usa p1 o p2.")

    grid = parse_search_grid(learning_rates, max_leaf_nodes)
    run = run_p1 if question == "p1" else run_p2
    metrics = run(search_grid=grid)
    logger.info(f"Métricas {question.upper()}:\n{metrics.to_string(index=False)}")


if __name__ == "__main__":
    app()
