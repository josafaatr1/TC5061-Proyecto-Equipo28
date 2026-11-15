"""Datos sintéticos compartidos por varias pruebas."""

import numpy as np
import pandas as pd

from mibici.modeling import p2_train
from mibici.modeling.p2_targets import TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN


def make_hourly_data(
    hours: pd.DatetimeIndex,
    departures: list[int] | None = None,
    station_id: int = 1,
) -> pd.DataFrame:
    """Crear registros sintéticos con el esquema estación-hora del pipeline."""
    departure_values = (
        list(departures) if departures is not None else [index % 7 for index in range(len(hours))]
    )
    arrivals = [1] * len(hours)

    return pd.DataFrame(
        {
            "station_id": station_id,
            "hour": hours,
            "departures": departure_values,
            "arrivals": arrivals,
            "net_flow": [
                arrival - departure
                for arrival, departure in zip(
                    arrivals,
                    departure_values,
                    strict=True,
                )
            ],
            "total_activity": [
                arrival + departure
                for arrival, departure in zip(
                    arrivals,
                    departure_values,
                    strict=True,
                )
            ],
            "is_observed_hour": [True] * len(hours),
        }
    )


def make_catalog(
    station_ids: list[int],
    capacities: list[float | None] | None = None,
) -> pd.DataFrame:
    """Crear un catálogo normalizable con capacidades sintéticas."""
    capacity_values = capacities or [10.0] * len(station_ids)
    return pd.DataFrame(
        {
            "id": station_ids,
            "name": [f"station-{station_id}" for station_id in station_ids],
            "latitude": [20.0] * len(station_ids),
            "longitude": [-103.0] * len(station_ids),
            "dpcapacity": capacity_values,
        }
    )


def make_hourly(
    records: list[tuple[int, pd.Timestamp, int, int]],
    observed: list[bool] | None = None,
) -> pd.DataFrame:
    """Crear datos estación-hora consistentes con el esquema procesado."""
    departures = [record[2] for record in records]
    arrivals = [record[3] for record in records]
    observed_values = observed or [True] * len(records)
    return pd.DataFrame(
        {
            "station_id": [record[0] for record in records],
            "hour": [record[1] for record in records],
            "departures": departures,
            "arrivals": arrivals,
            "net_flow": [
                arrival - departure
                for arrival, departure in zip(arrivals, departures, strict=True)
            ],
            "total_activity": [
                arrival + departure
                for arrival, departure in zip(arrivals, departures, strict=True)
            ],
            "is_observed_hour": observed_values,
        }
    )


def make_labeled_data(include_test: bool = True) -> pd.DataFrame:
    """Crear train, validation y opcionalmente test con clases desbalanceadas."""
    records = []
    partitions = (
        ("train", pd.date_range("2025-08-01", periods=240, freq="h")),
        ("validation", pd.date_range("2025-09-01", periods=120, freq="h")),
    )
    predictor_columns = p2_train.PREDICTOR_COLUMNS

    for partition, hours in partitions:
        for index, hour in enumerate(hours):
            row = {
                "station_id": 11,
                "hour": hour,
                "partition": partition,
                TARGET_BIKE_COLUMN: int(index % 10 == 0),
                TARGET_DOCK_COLUMN: int(index % 13 == 0),
            }
            for column in predictor_columns:
                if column == "hour_of_day":
                    row[column] = hour.hour
                elif column == "day_of_week":
                    row[column] = hour.dayofweek
                elif column == "month":
                    row[column] = hour.month
                elif column == "is_weekend":
                    row[column] = hour.dayofweek >= 5
                else:
                    row[column] = float((index % 17) - 8)
            row["departures_lag_168h"] = np.nan if index < 5 else row["departures_lag_168h"]
            row["bike_pressure_ratio"] = 999.0
            row["dock_pressure_ratio"] = 999.0
            row["future_departures"] = 999.0
            records.append(row)

    if include_test:
        test_row = records[-1].copy()
        test_row.update(
            {
                "station_id": 11,
                "hour": pd.Timestamp("2025-11-01"),
                "partition": "test",
                TARGET_BIKE_COLUMN: 9,
                TARGET_DOCK_COLUMN: 9,
                "departures_lag_1h": np.inf,
            }
        )
        records.append(test_row)
    return pd.DataFrame(records)


def make_hourly_for_reproduction() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crear movimientos con presiones positivas en train y validation."""
    hours = pd.date_range("2025-08-25", periods=240, freq="h")
    rows = []
    for station_id, offset in ((1, 0), (2, 2)):
        for index, hour in enumerate(hours):
            phase = (index + offset) % 4
            departures = 2 if phase in (0, 1) else 0
            arrivals = 2 if phase in (2, 3) else 0
            rows.append(
                {
                    "station_id": station_id,
                    "hour": hour,
                    "departures": departures,
                    "arrivals": arrivals,
                    "net_flow": arrivals - departures,
                    "total_activity": arrivals + departures,
                    "is_observed_hour": True,
                }
            )
    hourly = pd.DataFrame(rows)
    catalog = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["One", "Two"],
            "latitude": [20.0, 20.1],
            "longitude": [-103.0, -103.1],
            "dpcapacity": [10, 20],
        }
    )
    return hourly, catalog
