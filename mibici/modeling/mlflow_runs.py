"""Registro en MLflow de los experimentos P1, P2 y P3.

Reutiliza el entrenamiento y la evaluación de ``train.py``, ``p2_train.py`` y
``clustering.py`` sin duplicar su lógica y aplica el contrato de
``references/p1_p2_mlflow_handoff.md``: un run padre por problema y un run
hijo por modelo, con métricas de validación prefijadas con ``val_``. Nunca
lee registros desde noviembre de 2025 (test reservado).

    mlflow run . -e p1_train --env-manager local
    mlflow run . -e p2_train --env-manager local
    mlflow run . -e p3_train --env-manager local
"""

import hashlib
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from loguru import logger
from matplotlib.figure import Figure
import mlflow
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
from mibici.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT, RANDOM_SEED
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


def run_p1(hourly_path: Path = HOURLY_PATH) -> pd.DataFrame:
    """Entrenar y evaluar P1 en train/validation y registrar un run por modelo."""
    hourly_data = read_train_validation_hourly(hourly_path)
    data_version = calculate_data_version(hourly_data)

    supervised_data = prepare_supervised_data(hourly_data)
    splits = split_chronologically(supervised_data, evaluate_test=False)
    baseline_metrics = evaluate_baselines(splits)

    # Entrenar una sola vez para poder registrar el modelo evaluado.
    fit_start = perf_counter()
    candidate = train_candidate_model(splits["train"])
    fit_seconds = perf_counter() - fit_start
    candidate_metrics = evaluate_candidate(splits, model=candidate, fit_seconds=fit_seconds)
    metrics = pd.concat([baseline_metrics, candidate_metrics], ignore_index=True)

    feature_columns = list(candidate.feature_names_in_)

    # Cohorte de evaluación de train.py: filas de validation con rezago semanal.
    cohort = splits["validation"]
    cohort = cohort.loc[cohort[SEASONAL_COLUMN].notna()]
    dummy = train_dummy_regressor(splits["train"])
    seasonal = SeasonalNaiveModel()
    models = {
        "dummy_mean": (dummy, ["hour_of_day"]),
        "seasonal_weekly": (seasonal, [SEASONAL_COLUMN]),
        "hist_gradient_boosting_poisson": (candidate, feature_columns),
    }

    predictors = {
        "dummy_mean": "ninguno (media de train)",
        "seasonal_weekly": SEASONAL_COLUMN,
        "hist_gradient_boosting_poisson": ",".join(feature_columns),
    }
    tags = {**common_tags("P1", data_version), "objective": "departures por estación-hora"}
    shared_params = period_params(len(splits["train"]), len(splits["validation"]))
    dataset = mlflow.data.from_pandas(hourly_data, source=str(hourly_path), name=DATASET_NAME)

    with mlflow.start_run():
        mlflow.set_tags({**tags, "mlflow.runName": "P1-regression"})
        mlflow.log_params(shared_params)
        mlflow.log_text(metrics.to_csv(index=False), "reports/p1_model_comparison.csv")
        mlflow.log_dict(
            {
                "tags": tags,
                "params": shared_params,
                "predictor_columns": feature_columns,
                "excluded_predictor_columns": ["station_id", "hour", "target_departures"],
                "candidate_parameters": CANDIDATE_PARAMS,
            },
            "reports/p1_run_report.json",
        )

        for model_name, model_metrics in metrics.groupby("model", sort=False):
            by_split = model_metrics.set_index("evaluation_set")
            validation = by_split.loc["validation"]
            train = by_split.loc["train"]

            with mlflow.start_run(run_name=P1_RUN_NAMES[model_name], nested=True):
                mlflow.set_tags({**tags, "model_name": model_name})
                mlflow.log_params(
                    {
                        **shared_params,
                        **P1_MODEL_PARAMS[model_name],
                        "predictor_columns": predictors[model_name],
                    }
                )
                mlflow.log_input(dataset, context="training")
                mlflow.log_metrics(
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
                    }
                )

                model, model_columns = models[model_name]
                model_input = cohort.loc[:, model_columns]
                if model is seasonal:
                    predictions = seasonal.predict(None, model_input)
                else:
                    predictions = model.predict(model_input)
                profile = hourly_profile_table(cohort, predictions)
                mlflow.log_text(profile.to_csv(index=False), "evaluation/hourly_profile.csv")
                mlflow.log_figure(
                    hourly_profile_figure(profile, f"Validation · {model_name}"),
                    "evaluation/hourly_profile.png",
                )

                if model is seasonal:
                    mlflow.pyfunc.log_model(
                        name="model", python_model=seasonal, input_example=model_input.head(5)
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        name="model",
                        input_example=model_input.head(5),
                        skops_trusted_types=HGB_TRUSTED_TYPES if model is candidate else None,
                    )

    logger.success("Runs P1 registrados en MLflow.")
    return metrics


def run_p2(
    label_path: Path = p2_train.LABEL_DATA_PATH,
    hourly_path: Path = HOURLY_PATH,
    catalog_path: Path = EXTERNAL_DATA_DIR / CATALOG_FILENAME,
) -> pd.DataFrame:
    """Reproducir etiquetas, entrenar P2 y registrar un run por objetivo y modelo."""
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
    experiment = p2_train.run_validation_experiment(labeled_data)
    metrics = experiment.metrics

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

    with mlflow.start_run():
        mlflow.set_tags({**tags, "mlflow.runName": "P2-classification"})
        mlflow.log_params(shared_params)
        mlflow.log_text(metrics.to_csv(index=False), "reports/p2_model_comparison.csv")
        mlflow.log_dict(
            {
                "data_provenance": provenance,
                "tags": tags,
                "params": shared_params,
                "candidate_parameters": p2_train.CANDIDATE_PARAMS,
            },
            "reports/p2_run_report.json",
        )
        audit_path = catalog_path.with_name(AUDIT_FILENAME)
        if audit_path.is_file():
            mlflow.log_artifact(str(audit_path), artifact_path="reports")

        for row in metrics.to_dict(orient="records"):
            objective = experiment.models[row["target"]]
            is_candidate = row["model"] == p2_train.CANDIDATE_NAME
            model = objective.candidate if is_candidate else objective.dummy
            title = f"{row['target_name']} · {row['model']}"

            with mlflow.start_run(
                run_name=p2_run_name(row["target_name"], row["model"]), nested=True
            ):
                mlflow.set_tags(
                    {**tags, "objective": row["target_name"], "model_name": row["model"]}
                )
                mlflow.log_params({**shared_params, **P2_MODEL_PARAMS[row["model"]]})
                mlflow.log_input(dataset, context="training")
                mlflow.log_metrics(
                    {
                        **{f"val_{name}": row[name] for name in P2_VALIDATION_METRICS},
                        "train_positive_prevalence": objective.training_prevalence,
                        "fit_seconds": row["fit_seconds"],
                        "evaluation_seconds": row["evaluation_seconds"],
                    }
                )

                mlflow.log_dict(
                    {name: row[name] for name in ("tn", "fp", "fn", "tp")},
                    "evaluation/confusion_matrix.json",
                )
                mlflow.log_figure(
                    confusion_matrix_figure(row["tn"], row["fp"], row["fn"], row["tp"], title),
                    "evaluation/confusion_matrix.png",
                )

                probabilities = p2_train._positive_probability(model, validation_features)
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

                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    input_example=validation_features[:5],
                    skops_trusted_types=HGB_TRUSTED_TYPES if is_candidate else None,
                )

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
) -> pd.DataFrame:
    """Agrupar estaciones con K-Means y registrar un run por modelo."""
    hourly_data = read_train_validation_hourly(hourly_path)
    data_version = calculate_data_version(hourly_data)

    features = clustering.build_station_features(hourly_data)
    result = clustering.fit_clusters(features)
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
        "selection_metric": "silhouette_score",
    }
    dataset = mlflow.data.from_pandas(hourly_data, source=str(hourly_path), name=DATASET_NAME)

    baseline = ActivityQuantileModel.fit(features["total_activity"], result.best_k)
    candidates = {P3_BASELINE_NAME: (baseline, baseline.predict(None, features))}
    for k in result.evaluation["k"]:
        model = clustering.make_model(k)
        candidates[f"kmeans_k{k}"] = (model, model.fit_predict(features))

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

        for model_name, (model, labels) in candidates.items():
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
                    metrics["inertia"] = float(model[-1].inertia_)
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
                        name="model", python_model=model, input_example=features.head(5)
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
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


if __name__ == "__main__":
    app()
