"""Entrenamiento y evaluación temporal de clasificadores proxy P2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import typer

from mibici.config import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    START_DATE,
)
from mibici.features import FLOW_COLUMNS, LAG_HOURS, REQUIRED_COLUMNS, ROLLING_WINDOWS
from mibici.modeling.p2_catalog import (
    CATALOG_FILENAME,
    PROVENANCE_FILENAME,
    load_catalog,
)
from mibici.modeling.p2_targets import (
    PRESSURE_QUANTILE,
    TARGET_BIKE_COLUMN,
    TARGET_DOCK_COLUMN,
    TRAIN_END,
    VALIDATION_END,
    YEAR_START,
    apply_pressure_thresholds,
    calculate_pressure_windows,
    fit_pressure_thresholds,
)

app = typer.Typer()

TARGET_COLUMNS = (TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN)
TARGET_NAMES = {
    TARGET_BIKE_COLUMN: "bike_pressure",
    TARGET_DOCK_COLUMN: "dock_pressure",
}
CALENDAR_FEATURES = ("hour_of_day", "day_of_week", "month", "is_weekend")
PREDICTOR_COLUMNS = (
    *CALENDAR_FEATURES,
    *(f"{column}_lag_{lag}h" for column in FLOW_COLUMNS for lag in LAG_HOURS),
    *(f"departures_rolling_mean_{window}h" for window in ROLLING_WINDOWS),
)
DECISION_THRESHOLD = 0.5
BASELINE_NAME = "dummy_prior"
CANDIDATE_NAME = "hist_gradient_boosting_balanced"
CANDIDATE_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}
LABEL_DATA_PATH = PROCESSED_DATA_DIR / "p2_targets_2025_train_validation.parquet"
HOURLY_DATA_PATH = PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet"
CATALOG_PATH = EXTERNAL_DATA_DIR / CATALOG_FILENAME
METRICS_PATH = PROCESSED_DATA_DIR / "p2_validation_metrics.json"
PROVENANCE_DATA_COLUMNS = (
    "station_id",
    "hour",
    "partition",
    *TARGET_COLUMNS,
)


@dataclass(frozen=True)
class TrainedObjective:
    """Dos clasificadores ajustados para una etiqueta binaria."""

    target_column: str
    dummy: DummyClassifier
    candidate: HistGradientBoostingClassifier
    dummy_fit_seconds: float
    candidate_fit_seconds: float
    training_prevalence: float


@dataclass(frozen=True)
class ValidationRun:
    """Métricas y clasificadores en memoria de la ejecución train/validation."""

    metrics: pd.DataFrame
    models: dict[str, TrainedObjective]


def _target_values(data: pd.DataFrame, target_column: str, context: str) -> np.ndarray:
    """Validar una etiqueta binaria y devolverla como int8."""
    if target_column not in data.columns:
        raise ValueError(f"Falta la variable objetivo '{target_column}'.")
    target = data[target_column]
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        raise TypeError(f"'{target_column}' debe ser numérica en {context}.")
    if target.isna().any():
        raise ValueError(f"'{target_column}' contiene etiquetas ausentes en {context}.")

    values = target.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError(f"'{target_column}' solo puede contener etiquetas binarias 0/1.")
    if np.unique(values).size != 2:
        raise ValueError(f"'{target_column}' debe contener ambas clases en {context}.")
    return values.astype(np.int8)


def prepare_p2_splits(labeled_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Validar contrato y seleccionar exclusivamente train y validation.

    Filtra filas ``test`` usando solo la columna de partición antes de leer o
    validar sus etiquetas. Las características se seleccionan por lista
    permitida; IDs, timestamps, partición, etiquetas y columnas extra no se
    pueden convertir accidentalmente en predictores.
    """
    if not isinstance(labeled_data, pd.DataFrame):
        raise TypeError("labeled_data debe ser un pandas.DataFrame.")
    required_columns = {
        "station_id",
        "hour",
        "partition",
        *PREDICTOR_COLUMNS,
        *TARGET_COLUMNS,
    }
    missing_columns = required_columns - set(labeled_data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias de P2: {sorted(missing_columns)}.")
    if labeled_data["partition"].isna().any():
        raise ValueError("'partition' contiene valores nulos.")

    partition = labeled_data["partition"].astype("string")
    known_partitions = {"train", "validation", "test"}
    unknown = set(partition.dropna().unique()) - known_partitions
    if unknown:
        raise ValueError(f"Particiones desconocidas: {sorted(unknown)}.")
    train_validation = labeled_data.loc[partition.isin(("train", "validation"))].copy()

    if not pd.api.types.is_datetime64_any_dtype(train_validation["hour"]):
        raise TypeError("'hour' debe tener tipo datetime en train y validation.")
    if train_validation["hour"].isna().any():
        raise ValueError("'hour' contiene valores nulos en train o validation.")
    if train_validation["hour"].dt.tz is not None:
        raise ValueError("'hour' debe usar timestamps sin zona horaria.")
    if train_validation["station_id"].isna().any():
        raise ValueError("'station_id' contiene valores nulos.")
    if train_validation.duplicated(["station_id", "hour"]).any():
        raise ValueError("Hay claves station_id-hour duplicadas.")

    expected_train = train_validation["hour"].ge(YEAR_START) & train_validation["hour"].lt(
        TRAIN_END
    )
    expected_validation = train_validation["hour"].ge(TRAIN_END) & train_validation["hour"].lt(
        VALIDATION_END
    )
    expected_partition = np.where(expected_train, "train", "validation")
    date_is_in_range = expected_train | expected_validation
    if not date_is_in_range.all():
        raise ValueError("Hay fechas fuera del rango de train/validation P2.")
    if not train_validation["partition"].astype("string").eq(expected_partition).all():
        raise ValueError("La partición registrada no coincide con los límites temporales.")

    for column in PREDICTOR_COLUMNS:
        values = train_validation[column]
        if not pd.api.types.is_numeric_dtype(values):
            raise TypeError(f"El predictor '{column}' debe ser numérico.")
        if np.isinf(values.to_numpy(dtype=np.float64, na_value=np.nan)).any():
            raise ValueError(f"El predictor '{column}' contiene valores infinitos.")

    splits = {
        split_name: train_validation.loc[train_validation["partition"].eq(split_name)].reset_index(
            drop=True
        )
        for split_name in ("train", "validation")
    }
    empty_splits = [name for name, frame in splits.items() if frame.empty]
    if empty_splits:
        raise ValueError(f"Faltan filas en las particiones: {empty_splits}.")
    for split_name, frame in splits.items():
        for target_column in TARGET_COLUMNS:
            _target_values(frame, target_column, split_name)

    return splits


def _feature_array(data: pd.DataFrame) -> np.ndarray:
    """Crear matriz compacta usando solo predictores históricos permitidos."""
    features = data.loc[:, PREDICTOR_COLUMNS]
    return features.to_numpy(dtype=np.float32, na_value=np.nan, copy=True)


def train_objective(
    training_data: pd.DataFrame,
    target_column: str,
    training_features: np.ndarray | None = None,
) -> TrainedObjective:
    """Ajustar Dummy y HGB usando solo filas train y pesos balanceados de train."""
    if training_data.empty:
        raise ValueError("No se puede entrenar sin observaciones train.")
    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"Objetivo P2 no reconocido: {target_column}.")
    y_train = _target_values(training_data, target_column, "train")
    if training_features is None:
        training_features = _feature_array(training_data)
    if training_features.shape[0] != len(training_data):
        raise ValueError("La cantidad de predictores no coincide con las filas train.")
    if np.isinf(training_features).any():
        raise ValueError("Los predictores train contienen valores infinitos.")

    dummy = DummyClassifier(strategy="prior")
    start = time.perf_counter()
    dummy.fit(training_features, y_train)
    dummy_fit_seconds = time.perf_counter() - start

    candidate = HistGradientBoostingClassifier(**CANDIDATE_PARAMS)
    start = time.perf_counter()
    candidate.fit(training_features, y_train)
    candidate_fit_seconds = time.perf_counter() - start

    return TrainedObjective(
        target_column=target_column,
        dummy=dummy,
        candidate=candidate,
        dummy_fit_seconds=dummy_fit_seconds,
        candidate_fit_seconds=candidate_fit_seconds,
        training_prevalence=float(y_train.mean()),
    )


def _positive_probability(
    model: DummyClassifier | HistGradientBoostingClassifier,
    features: np.ndarray,
) -> np.ndarray:
    """Obtener y validar la probabilidad correspondiente a la clase positiva."""
    classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError("El modelo ajustado no expone exactamente una clase positiva 1.")
    probabilities = model.predict_proba(features)[:, positive_indices[0]]
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("El clasificador produjo probabilidades fuera de [0, 1].")
    return probabilities


def evaluate_probabilities(
    y_true: np.ndarray,
    positive_probabilities: np.ndarray,
    decision_threshold: float = DECISION_THRESHOLD,
) -> dict[str, float | int]:
    """Calcular AP, Precision, Recall, F1 y matriz de confusión."""
    if not np.isfinite(decision_threshold) or not 0 < decision_threshold < 1:
        raise ValueError("decision_threshold debe estar estrictamente entre 0 y 1.")
    if y_true.ndim != 1 or positive_probabilities.ndim != 1:
        raise ValueError("y_true y positive_probabilities deben ser vectores.")
    if len(y_true) != len(positive_probabilities) or not len(y_true):
        raise ValueError("y_true y probabilidades deben tener igual longitud no vacía.")
    if not np.isin(y_true, (0, 1)).all() or np.unique(y_true).size != 2:
        raise ValueError("validation debe contener ambas clases binarias.")
    if (
        not np.isfinite(positive_probabilities).all()
        or (positive_probabilities < 0).any()
        or (positive_probabilities > 1).any()
    ):
        raise ValueError("Las probabilidades deben ser finitas y estar en [0, 1].")

    predicted = positive_probabilities >= decision_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, positive_probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_prevalence": float(np.mean(y_true)),
        "n_evaluated": len(y_true),
    }


def evaluate_objective(
    models: TrainedObjective,
    validation_data: pd.DataFrame,
    validation_features: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Evaluar ambos modelos sobre las mismas filas de validation."""
    y_validation = _target_values(
        validation_data,
        models.target_column,
        "validation",
    )
    if validation_features is None:
        validation_features = _feature_array(validation_data)
    if validation_features.shape[0] != len(validation_data):
        raise ValueError("La cantidad de predictores no coincide con validation.")
    if np.isinf(validation_features).any():
        raise ValueError("Los predictores validation contienen valores infinitos.")

    results = []
    for model_name, model, fit_seconds in (
        (BASELINE_NAME, models.dummy, models.dummy_fit_seconds),
        (CANDIDATE_NAME, models.candidate, models.candidate_fit_seconds),
    ):
        start = time.perf_counter()
        probabilities = _positive_probability(model, validation_features)
        metrics = evaluate_probabilities(
            y_validation,
            probabilities,
            decision_threshold=DECISION_THRESHOLD,
        )
        evaluation_seconds = time.perf_counter() - start
        results.append(
            {
                "evaluation_set": "validation",
                "target": models.target_column,
                "target_name": TARGET_NAMES[models.target_column],
                "model": model_name,
                **metrics,
                "decision_threshold": DECISION_THRESHOLD,
                "n_total": len(validation_data),
                "n_excluded": 0,
                "fit_seconds": fit_seconds,
                "evaluation_seconds": evaluation_seconds,
            }
        )
    return results


def run_validation_experiment(labeled_data: pd.DataFrame) -> ValidationRun:
    """Entrenar dos tareas independientes y evaluar solo validation."""
    splits = prepare_p2_splits(labeled_data)
    train_data = splits["train"]
    validation_data = splits["validation"]
    training_features = _feature_array(train_data)
    validation_features = _feature_array(validation_data)

    models = {}
    metrics = []
    for target_column in TARGET_COLUMNS:
        objective_models = train_objective(
            train_data,
            target_column,
            training_features=training_features,
        )
        models[target_column] = objective_models
        metrics.extend(
            evaluate_objective(
                objective_models,
                validation_data,
                validation_features=validation_features,
            )
        )
    return ValidationRun(metrics=pd.DataFrame(metrics), models=models)


def _file_sha256(path: Path) -> str:
    """Calcular SHA-256 de un archivo usando memoria acotada."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_label_reproducibility(
    label_path: Path,
    hourly_path: Path,
    catalog_path: Path,
) -> dict[str, object]:
    """Recrear etiquetas desde enero-octubre y verificar igualdad exacta.

    Esta verificación recupera los percentiles de etiquetado que no se
    serializaron dentro del Parquet de objetivos. Nunca lee registros desde
    noviembre y valida solo claves y etiquetas guardadas.
    """
    required_paths = (label_path, hourly_path, catalog_path)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Faltan archivos para reproducir etiquetas: {missing_paths}.")

    start = pd.Timestamp(START_DATE)
    validation_end = start + pd.DateOffset(months=10)
    saved_labels = pd.read_parquet(
        label_path,
        columns=list(PROVENANCE_DATA_COLUMNS),
        filters=[
            [
                ("partition", "==", "train"),
            ],
            [
                ("partition", "==", "validation"),
            ],
        ],
    )
    hourly_data = pd.read_parquet(
        hourly_path,
        columns=list(REQUIRED_COLUMNS),
        filters=[
            ("hour", ">=", start),
            ("hour", "<", validation_end),
        ],
    )
    station_catalog = load_catalog(catalog_path, strict=False)
    pressure_calculation = calculate_pressure_windows(hourly_data, station_catalog)
    train_windows = pressure_calculation.windows.loc[
        pressure_calculation.windows["partition"].eq("train")
    ]
    thresholds = fit_pressure_thresholds(train_windows)
    reproduced = apply_pressure_thresholds(
        pressure_calculation.windows,
        thresholds,
    ).loc[:, list(PROVENANCE_DATA_COLUMNS)]

    sort_columns = ["station_id", "hour"]
    saved_labels = saved_labels.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    reproduced = reproduced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    saved_labels["partition"] = saved_labels["partition"].astype("string")
    reproduced["partition"] = reproduced["partition"].astype("string")
    keys_match = saved_labels[sort_columns + ["partition"]].equals(
        reproduced[sort_columns + ["partition"]]
    )
    labels_match = all(
        np.array_equal(
            saved_labels[target].to_numpy(),
            reproduced[target].to_numpy(),
        )
        for target in TARGET_COLUMNS
    )
    if len(saved_labels) != len(reproduced) or not keys_match or not labels_match:
        raise ValueError(
            "Las etiquetas guardadas no se reproducen exactamente desde el Parquet horario."
        )

    provenance_path = catalog_path.with_name(PROVENANCE_FILENAME)
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Falta la procedencia del catálogo: {provenance_path}.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    catalog_digest = _file_sha256(catalog_path)
    if provenance.get("sha256") != catalog_digest:
        raise ValueError("El hash del catálogo no coincide con su archivo de procedencia.")

    return {
        "labels_reproduced_exactly": True,
        "label_artifact_sha256": _file_sha256(label_path),
        "catalog_version": provenance.get("catalog_version"),
        "catalog_sha256": catalog_digest,
        "label_pressure_quantile": PRESSURE_QUANTILE,
        "label_pressure_thresholds_train_only": asdict(thresholds),
        "decision_threshold": DECISION_THRESHOLD,
        "decision_threshold_policy": (
            "Predeclarado en 0.5; no se ajustó con validation ni se usó calibración."
        ),
    }


def _json_default(value: Any) -> Any:
    """Convertir valores NumPy a tipos JSON estándar."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo no serializable en el reporte: {type(value).__name__}.")


@app.command()
def main(
    label_path: Path = LABEL_DATA_PATH,
    hourly_path: Path = HOURLY_DATA_PATH,
    catalog_path: Path = CATALOG_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Reproducir las etiquetas y ejecutar solo train/validation de P2."""
    provenance = verify_label_reproducibility(
        label_path=label_path,
        hourly_path=hourly_path,
        catalog_path=catalog_path,
    )
    labeled_data = pd.read_parquet(
        label_path,
        columns=[
            "station_id",
            "hour",
            "partition",
            *PREDICTOR_COLUMNS,
            *TARGET_COLUMNS,
        ],
        filters=[
            [
                ("partition", "==", "train"),
            ],
            [
                ("partition", "==", "validation"),
            ],
        ],
    )
    run = run_validation_experiment(labeled_data)
    report = {
        "data_provenance": provenance,
        "python_version": platform.python_version(),
        "scikit_learn_version": SKLEARN_VERSION,
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "excluded_predictor_columns": [
            "station_id",
            "hour",
            "partition",
            *TARGET_COLUMNS,
            "bike_pressure",
            "dock_pressure",
            "bike_pressure_ratio",
            "dock_pressure_ratio",
        ],
        "baseline": {"name": BASELINE_NAME, "strategy": "prior"},
        "candidate": {
            "name": CANDIDATE_NAME,
            "estimator": "HistGradientBoostingClassifier",
            "parameters": CANDIDATE_PARAMS,
            "class_weight_policy": (
                "class_weight='balanced', computed by scikit-learn from train labels only"
            ),
        },
        "validation_metrics": run.metrics.to_dict(orient="records"),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Resultados de validación P2:\n{}",
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
    )
    logger.success("Reporte de validación guardado en: {}", metrics_path)


if __name__ == "__main__":
    app()
