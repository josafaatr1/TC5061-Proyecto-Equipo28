"""Construcción de características temporales para la regresión P1."""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer()

REQUIRED_COLUMNS = (
    "station_id",
    "hour",
    "departures",
    "arrivals",
    "net_flow",
    "total_activity",
    "is_observed_hour",
)
FLOW_COLUMNS = ("departures", "arrivals", "net_flow")
LAG_HOURS = (1, 24, 168)
ROLLING_WINDOWS = (24, 168)


def _validate_hourly_data(hourly_data: pd.DataFrame) -> None:
    """Validar esquema, tipos y unicidad del dataset estación-hora."""
    if not isinstance(hourly_data, pd.DataFrame):
        raise TypeError("hourly_data debe ser un pandas.DataFrame.")

    missing_columns = set(REQUIRED_COLUMNS) - set(hourly_data.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing_columns)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(hourly_data["hour"]):
        raise TypeError("'hour' debe tener tipo datetime.")
    if hourly_data["hour"].isna().any():
        raise ValueError("'hour' no puede contener valores nulos.")
    if (hourly_data["hour"] != hourly_data["hour"].dt.floor("h")).any():
        raise ValueError("'hour' debe estar alineada al inicio de cada hora.")

    if hourly_data["station_id"].isna().any():
        raise ValueError("'station_id' no puede contener valores nulos.")
    if hourly_data.duplicated(["station_id", "hour"]).any():
        raise ValueError("Existen combinaciones estación-hora duplicadas.")

    numeric_columns = (*FLOW_COLUMNS, "total_activity")
    for column in numeric_columns:
        if (
            not pd.api.types.is_numeric_dtype(hourly_data[column])
            or pd.api.types.is_bool_dtype(hourly_data[column])
        ):
            raise TypeError(f"'{column}' debe ser una columna numérica.")

    if not pd.api.types.is_bool_dtype(hourly_data["is_observed_hour"]):
        raise TypeError("'is_observed_hour' debe ser una columna booleana.")
    if hourly_data["is_observed_hour"].isna().any():
        raise ValueError("'is_observed_hour' no puede contener valores nulos.")

    unobserved_counts = hourly_data.loc[
        ~hourly_data["is_observed_hour"],
        [*FLOW_COLUMNS, "total_activity"],
    ]
    if (
        unobserved_counts.isna().any().any()
        or not unobserved_counts.eq(0).all().all()
    ):
        raise ValueError(
            "Las horas no observadas deben tener conteos completados en cero."
        )


def _historical_values(
    ordered_data: pd.DataFrame,
    column: str,
    lag_hours: int,
) -> pd.Series:
    """Obtener el valor de la hora exacta anterior, sin imputar brechas."""
    source_index = pd.MultiIndex.from_frame(
        ordered_data[["station_id", "hour"]]
    )
    target_index = pd.MultiIndex.from_arrays(
        [
            ordered_data["station_id"],
            ordered_data["hour"] - pd.Timedelta(hours=lag_hours),
        ],
        names=source_index.names,
    )

    historical_series = pd.Series(
        ordered_data[column].array,
        index=source_index,
    )

    return historical_series.reindex(target_index).reset_index(drop=True)


def _historical_departures_mean(
    ordered_data: pd.DataFrame,
    window_hours: int,
) -> pd.Series:
    """Calcular la media de salidas en la ventana temporal anterior."""
    history = ordered_data[["station_id", "hour"]].copy()
    history["departures"] = ordered_data["departures"]

    rolling_mean = (
        history.groupby("station_id", sort=False)
        .rolling(
            window=f"{window_hours}h",
            on="hour",
            closed="left",
            min_periods=window_hours,
        )["departures"]
        .mean()
    )

    target_index = pd.MultiIndex.from_frame(
        ordered_data[["station_id", "hour"]]
    )

    return rolling_mean.reindex(target_index).reset_index(drop=True)


def build_features(hourly_data: pd.DataFrame) -> pd.DataFrame:
    """Generar predictores disponibles antes del inicio de cada hora objetivo.

    Cada fila de entrada representa una estación y una hora objetivo; se
    asume que la predicción se realiza justo antes de que esa hora comience.
    El resultado conserva ``station_id`` y ``hour`` como claves, pero excluye
    los conteos de la hora objetivo. Los ceros de horas completadas se
    conservan como conteos observados de cero viajes; un timestamp ausente
    permanece desconocido y no se rellena.

    Args:
        hourly_data: Dataset estación-hora con el esquema de
            ``complete_station_hourly``.

    Returns:
        DataFrame ordenado por estación y hora con claves, variables de
        calendario, rezagos exactos y medias móviles históricas.
    """
    _validate_hourly_data(hourly_data)

    ordered_data = hourly_data.sort_values(
        ["station_id", "hour"],
        kind="stable",
    ).reset_index(drop=True)

    features = ordered_data[["station_id", "hour"]].copy()
    features["hour_of_day"] = ordered_data["hour"].dt.hour
    features["day_of_week"] = ordered_data["hour"].dt.dayofweek
    features["month"] = ordered_data["hour"].dt.month
    features["is_weekend"] = ordered_data["hour"].dt.dayofweek >= 5

    for column in FLOW_COLUMNS:
        for lag_hours in LAG_HOURS:
            features[f"{column}_lag_{lag_hours}h"] = _historical_values(
                ordered_data,
                column,
                lag_hours,
            )

    for window_hours in ROLLING_WINDOWS:
        features[f"departures_rolling_mean_{window_hours}h"] = (
            _historical_departures_mean(ordered_data, window_hours)
        )

    return features


@app.command()
def main(
    input_path: Path = (
        PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet"
    ),
    output_path: Path = PROCESSED_DATA_DIR / "features_p1_2025.parquet",
) -> None:
    """Leer el dataset horario y guardar las características de P1."""
    logger.info(f"Leyendo datos horarios desde: {input_path}")
    hourly_data = pd.read_parquet(input_path)
    features = build_features(hourly_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.success(f"Características P1 guardadas en: {output_path}")


if __name__ == "__main__":
    app()
