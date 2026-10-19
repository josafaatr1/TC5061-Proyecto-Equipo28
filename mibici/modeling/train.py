"""Baselines reproducibles de regresión P1 para MiBici."""

from pathlib import Path
from time import perf_counter

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import typer

from mibici.config import (
    END_DATE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    START_DATE,
)
from mibici.features import build_features

app = typer.Typer()

SPLIT_NAMES = ("train", "validation", "test")
TARGET_COLUMN = "target_departures"
SEASONAL_COLUMN = "departures_lag_168h"
PREDICTOR_EXCLUDED_COLUMNS = frozenset({"station_id", "hour", TARGET_COLUMN})
CANDIDATE_PARAMS = {
    "loss": "poisson",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}


def _target_values(data: pd.DataFrame, context: str) -> np.ndarray:
    """Validar la etiqueta y devolverla como un vector numérico finito."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Falta la columna obligatoria '{TARGET_COLUMN}'.")

    target = data[TARGET_COLUMN].infer_objects()
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        raise TypeError(f"'{TARGET_COLUMN}' debe ser numérica en {context}.")
    if target.isna().any():
        raise ValueError(f"'{TARGET_COLUMN}' contiene valores ausentes en {context}.")

    values = target.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"'{TARGET_COLUMN}' contiene valores no finitos en {context}.")
    if (values < 0).any():
        raise ValueError(f"'{TARGET_COLUMN}' no puede ser negativa en {context}.")

    return values


def prepare_supervised_data(hourly_data: pd.DataFrame) -> pd.DataFrame:
    """Alinear las características de la hora objetivo con sus salidas reales.

    ``hour`` representa el inicio de la hora cuya cantidad de salidas se
    pronostica. La predicción se realiza antes de que esa hora comience.
    Las claves estación-hora se usan para alinear el objetivo y se conservan
    para trazabilidad; los conteos del instante objetivo no se añaden como
    predictores.

    Args:
        hourly_data: Dataset estación-hora producido por
            ``complete_station_hourly``.

    Returns:
        Tabla con predictores de ``build_features`` y la etiqueta
        ``target_departures`` alineada por estación y hora.

    Raises:
        ValueError: Si faltan etiquetas, las claves no se pueden alinear o
            alguna etiqueta no representa un conteo válido de viajes.
    """
    features = build_features(hourly_data)
    target = hourly_data["departures"]
    if target.isna().any():
        raise ValueError("'departures' contiene etiquetas ausentes.")

    target_values = target.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(target_values).all():
        raise ValueError("'departures' contiene etiquetas no finitas.")
    if (target_values < 0).any():
        raise ValueError("'departures' no puede contener conteos negativos.")

    source_index = pd.MultiIndex.from_frame(hourly_data[["station_id", "hour"]])
    feature_index = pd.MultiIndex.from_frame(features[["station_id", "hour"]])
    source_targets = pd.Series(target_values, index=source_index)
    aligned_targets = source_targets.reindex(feature_index)

    if len(features) != len(hourly_data) or aligned_targets.isna().any():
        raise ValueError("No fue posible alinear todas las etiquetas por estación y hora.")

    supervised_data = features.copy()
    supervised_data[TARGET_COLUMN] = aligned_targets.to_numpy()

    return supervised_data


def split_chronologically(
    supervised_data: pd.DataFrame,
    evaluate_test: bool = True,
) -> dict[str, pd.DataFrame]:
    """Separar filas por hora objetivo en train, validación y prueba.

    Los intervalos son enero-agosto, septiembre-octubre y noviembre-diciembre
    de 2025. Las fechas de inicio y fin del año se toman de la configuración;
    las fronteras mensuales se calculan desde ``START_DATE``. No se descartan
    silenciosamente observaciones fuera de ese rango en las particiones
    seleccionadas. Cuando ``evaluate_test=False``, ``hour`` es la única
    columna usada para excluir las filas desde el inicio de test; sus claves
    y etiquetas no se validan.

    Args:
        supervised_data: Tabla de ``prepare_supervised_data``.
        evaluate_test: Incluir el split de noviembre-diciembre.

    Returns:
        Diccionario ordenado con los DataFrames ``train``, ``validation`` y
        ``test``, cada uno cronológico por estación.

    Raises:
        ValueError: Si el esquema, las fechas o alguno de los conjuntos no es
            válido.
    """
    if not isinstance(supervised_data, pd.DataFrame):
        raise TypeError("supervised_data debe ser un pandas.DataFrame.")

    required_columns = {"station_id", "hour", TARGET_COLUMN}
    missing_columns = required_columns - set(supervised_data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing_columns)}")
    if not pd.api.types.is_datetime64_any_dtype(supervised_data["hour"]):
        raise TypeError("'hour' debe tener tipo datetime.")
    if supervised_data["hour"].isna().any():
        raise ValueError("'hour' no puede contener valores nulos.")
    if supervised_data["hour"].dt.tz is not None:
        raise ValueError("'hour' debe usar timestamps sin zona horaria.")

    year_start = pd.Timestamp(START_DATE)
    year_end = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)
    train_end = year_start + pd.DateOffset(months=8)
    validation_end = year_start + pd.DateOffset(months=10)
    if not year_start < train_end < validation_end < year_end:
        raise ValueError("Las fechas centrales no permiten los splits de P1.")

    if evaluate_test:
        selected_data = supervised_data
    else:
        selected_data = supervised_data.loc[supervised_data["hour"].lt(validation_end)]

    if selected_data["station_id"].isna().any():
        raise ValueError("'station_id' no puede contener valores nulos.")
    if selected_data.duplicated(["station_id", "hour"]).any():
        raise ValueError("Existen combinaciones estación-hora duplicadas.")

    _target_values(selected_data, "la partición temporal")

    hours = selected_data["hour"]
    outside_year = (hours < year_start) | (hours >= year_end)
    if outside_year.any():
        raise ValueError("Hay horas objetivo fuera del periodo configurado de 2025.")

    split_masks = {
        "train": hours.lt(train_end),
        "validation": hours.ge(train_end) & hours.lt(validation_end),
    }
    if evaluate_test:
        split_masks["test"] = hours.ge(validation_end) & hours.lt(year_end)

    splits = {
        name: selected_data.loc[mask]
        .sort_values(["station_id", "hour"], kind="stable")
        .reset_index(drop=True)
        for name, mask in split_masks.items()
    }
    empty_splits = [name for name, data in splits.items() if data.empty]
    if empty_splits:
        raise ValueError(f"No hay observaciones para los conjuntos: {empty_splits}.")

    return splits


def train_dummy_regressor(training_data: pd.DataFrame) -> DummyRegressor:
    """Ajustar el baseline de media usando exclusivamente el conjunto train."""
    if not isinstance(training_data, pd.DataFrame):
        raise TypeError("training_data debe ser un pandas.DataFrame.")
    if training_data.empty:
        raise ValueError("No se puede ajustar DummyRegressor sin filas train.")
    if "hour_of_day" not in training_data.columns:
        raise ValueError("Falta la característica 'hour_of_day'.")

    targets = _target_values(training_data, "train")
    hour_of_day = training_data["hour_of_day"]
    if (
        not pd.api.types.is_numeric_dtype(hour_of_day)
        or hour_of_day.isna().any()
        or not np.isfinite(hour_of_day.to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("'hour_of_day' debe contener valores numéricos finitos.")

    model = DummyRegressor(strategy="mean")
    model.fit(training_data[["hour_of_day"]], targets)

    return model


def _feature_matrix(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validar y seleccionar predictores numéricos sin claves ni etiquetas."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data debe ser un pandas.DataFrame.")

    columns = feature_columns or tuple(
        column for column in data.columns if column not in PREDICTOR_EXCLUDED_COLUMNS
    )
    if not columns:
        raise ValueError("No hay características predictoras para entrenar.")

    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Faltan características predictoras: {sorted(missing_columns)}")

    features = data.loc[:, columns]
    for column in columns:
        values = features[column]
        if not pd.api.types.is_numeric_dtype(values):
            raise TypeError(f"La característica '{column}' debe ser numérica.")
        if np.isinf(values.to_numpy(dtype=np.float64, na_value=np.nan)).any():
            raise ValueError(f"La característica '{column}' contiene valores infinitos.")

    return features, columns


def train_candidate_model(
    training_data: pd.DataFrame,
) -> HistGradientBoostingRegressor:
    """Ajustar HistGradientBoostingRegressor Poisson usando exclusivamente train.

    Las características faltantes se conservan: HistGradientBoostingRegressor
    las maneja de forma nativa. ``station_id`` se excluye porque su orden
    numérico no representa una relación de magnitud entre estaciones.
    """
    if not isinstance(training_data, pd.DataFrame):
        raise TypeError("training_data debe ser un pandas.DataFrame.")
    if training_data.empty:
        raise ValueError("No se puede ajustar el candidato sin filas train.")

    targets = _target_values(training_data, "train")
    features, _ = _feature_matrix(training_data)
    model = HistGradientBoostingRegressor(**CANDIDATE_PARAMS)
    model.fit(features, targets)

    return model


def _split_names(evaluate_test: bool) -> tuple[str, ...]:
    """Devolver los conjuntos de evaluación autorizados para esta ejecución."""
    return SPLIT_NAMES if evaluate_test else SPLIT_NAMES[:2]


def _validate_evaluation_splits(
    splits: dict[str, pd.DataFrame],
    evaluate_test: bool,
) -> tuple[str, ...]:
    """Validar particiones requeridas sin inspeccionar test por defecto."""
    selected_splits = _split_names(evaluate_test)
    allowed_splits = set(SPLIT_NAMES)
    if not set(selected_splits).issubset(splits) or not set(splits).issubset(allowed_splits):
        raise ValueError(
            f"splits debe contener {selected_splits} y ninguna partición desconocida."
        )
    return selected_splits


def evaluate_baselines(
    splits: dict[str, pd.DataFrame],
    evaluate_test: bool = False,
) -> pd.DataFrame:
    """Calcular métricas de baselines sin evaluar test por defecto.

    Ambos modelos se evalúan sobre una cohorte común con rezago semanal
    disponible. Las etiquetas ausentes o los rezagos no finitos producen un
    error explícito. Test solo se incluye con ``evaluate_test=True``.
    """
    split_names = _validate_evaluation_splits(splits, evaluate_test)

    fit_start = perf_counter()
    dummy_model = train_dummy_regressor(splits["train"])
    dummy_fit_seconds = perf_counter() - fit_start
    results = []

    for split_name in split_names:
        split_data = splits[split_name]
        if split_data.empty:
            raise ValueError(f"El conjunto {split_name} está vacío.")

        targets = _target_values(split_data, split_name)
        required_columns = {"hour_of_day", SEASONAL_COLUMN}
        missing_columns = required_columns - set(split_data.columns)
        if missing_columns:
            raise ValueError(
                f"Faltan columnas para evaluar {split_name}: {sorted(missing_columns)}"
            )

        hour_of_day = split_data["hour_of_day"]
        if (
            not pd.api.types.is_numeric_dtype(hour_of_day)
            or hour_of_day.isna().any()
            or not np.isfinite(hour_of_day.to_numpy(dtype=np.float64)).all()
        ):
            raise ValueError(f"'hour_of_day' debe ser finita en {split_name}.")

        seasonal = split_data[SEASONAL_COLUMN]
        if not pd.api.types.is_numeric_dtype(seasonal) or pd.api.types.is_bool_dtype(seasonal):
            raise TypeError(f"'{SEASONAL_COLUMN}' debe ser numérica.")
        seasonal_values = seasonal.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        if np.isinf(seasonal_values).any():
            raise ValueError(f"'{SEASONAL_COLUMN}' contiene valores infinitos en {split_name}.")

        available = ~np.isnan(seasonal_values)
        evaluated_count = int(available.sum())
        if evaluated_count == 0:
            raise ValueError(f"No hay rezagos semanales disponibles en {split_name}.")

        evaluation_data = split_data.loc[available]
        y_true = targets[available]
        evaluation_start = perf_counter()
        dummy_predictions = dummy_model.predict(evaluation_data[["hour_of_day"]])
        dummy_evaluation_seconds = perf_counter() - evaluation_start
        evaluation_start = perf_counter()
        seasonal_predictions = seasonal_values[available]
        seasonal_evaluation_seconds = perf_counter() - evaluation_start

        predictions_by_model = (
            ("dummy_mean", dummy_predictions, dummy_fit_seconds, dummy_evaluation_seconds),
            ("seasonal_weekly", seasonal_predictions, 0.0, seasonal_evaluation_seconds),
        )
        for model_name, predictions, fit_seconds, evaluation_seconds in predictions_by_model:
            results.append(
                {
                    "evaluation_set": split_name,
                    "model": model_name,
                    "mae": mean_absolute_error(y_true, predictions),
                    "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
                    "n_total": len(split_data),
                    "n_evaluated": evaluated_count,
                    "n_excluded": len(split_data) - evaluated_count,
                    "fit_seconds": fit_seconds,
                    "evaluation_seconds": evaluation_seconds,
                }
            )

    return pd.DataFrame(results)


def evaluate_candidate(
    splits: dict[str, pd.DataFrame],
    evaluate_test: bool = False,
    model: HistGradientBoostingRegressor | None = None,
    fit_seconds: float | None = None,
) -> pd.DataFrame:
    """Ajustar y evaluar el candidato sobre la cohorte de los baselines.

    Si se recibe ``model`` ya ajustado con ``train_candidate_model``, se evalúa
    sin volver a entrenar; ``fit_seconds`` reporta el tiempo de ese ajuste.
    """
    split_names = _validate_evaluation_splits(splits, evaluate_test)

    if model is None:
        fit_start = perf_counter()
        model = train_candidate_model(splits["train"])
        fit_seconds = perf_counter() - fit_start
    elif fit_seconds is None:
        fit_seconds = float("nan")
    feature_columns = tuple(model.feature_names_in_)
    results = []

    for split_name in split_names:
        split_data = splits[split_name]
        if split_data.empty:
            raise ValueError(f"El conjunto {split_name} está vacío.")

        targets = _target_values(split_data, split_name)
        if SEASONAL_COLUMN not in split_data.columns:
            raise ValueError(f"Falta la característica requerida '{SEASONAL_COLUMN}'.")
        seasonal_values = split_data[SEASONAL_COLUMN].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        if np.isinf(seasonal_values).any():
            raise ValueError(f"'{SEASONAL_COLUMN}' contiene valores infinitos en {split_name}.")
        available = ~np.isnan(seasonal_values)
        evaluated_count = int(available.sum())
        if evaluated_count == 0:
            raise ValueError(f"No hay rezagos semanales disponibles en {split_name}.")

        evaluation_data = split_data.loc[available]
        features, _ = _feature_matrix(evaluation_data, feature_columns)
        y_true = targets[available]

        evaluation_start = perf_counter()
        predictions = model.predict(features)
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError(
                f"El candidato produjo predicciones no finitas o negativas en {split_name}."
            )
        evaluation_seconds = perf_counter() - evaluation_start
        results.append(
            {
                "evaluation_set": split_name,
                "model": "hist_gradient_boosting_poisson",
                "mae": mean_absolute_error(y_true, predictions),
                "rmse": np.sqrt(mean_squared_error(y_true, predictions)),
                "n_total": len(split_data),
                "n_evaluated": evaluated_count,
                "n_excluded": len(split_data) - evaluated_count,
                "fit_seconds": fit_seconds,
                "evaluation_seconds": evaluation_seconds,
            }
        )

    return pd.DataFrame(results)


def run_model_comparison(
    hourly_data: pd.DataFrame,
    evaluate_test: bool = False,
) -> pd.DataFrame:
    """Preparar, entrenar y comparar baselines y candidato sin test por defecto."""
    supervised_data = prepare_supervised_data(hourly_data)
    splits = split_chronologically(supervised_data, evaluate_test=evaluate_test)
    baseline_metrics = evaluate_baselines(splits, evaluate_test=evaluate_test)
    candidate_metrics = evaluate_candidate(splits, evaluate_test=evaluate_test)

    return pd.concat([baseline_metrics, candidate_metrics], ignore_index=True)


@app.command()
def main(
    input_path: Path = (PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet"),
    metrics_path: Path = REPORTS_DIR / "p1_baseline_metrics.csv",
    evaluate_test: bool = typer.Option(
        False,
        "--evaluate-test",
        help="Incluir noviembre-diciembre únicamente para la evaluación final.",
    ),
) -> None:
    """Ejecutar entrenamiento y validación; test requiere una opción explícita."""
    logger.info(f"Leyendo datos horarios desde: {input_path}")
    if evaluate_test:
        hourly_data = pd.read_parquet(input_path)
    else:
        validation_end = pd.Timestamp(START_DATE) + pd.DateOffset(months=10)
        hourly_data = pd.read_parquet(
            input_path,
            filters=[("hour", "<", validation_end)],
        )
    metrics = run_model_comparison(hourly_data, evaluate_test=evaluate_test)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    logger.info(f"Métricas P1:\n{metrics.to_string(index=False)}")
    logger.success(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    app()
