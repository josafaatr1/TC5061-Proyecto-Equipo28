"""Pruebas de la evaluación final en test: sin fugas de datos y con métricas correctas."""

import numpy as np
import pandas as pd
import pytest
from tests.test_modeling import make_hourly_data
from tests.test_p2_targets import make_catalog, make_hourly

from mibici.modeling.evaluate import p1_cohorts, regression_metrics
from mibici.modeling.p2_targets import (
    TARGET_BIKE_COLUMN,
    TARGET_DOCK_COLUMN,
    VALIDATION_END,
    PressureThresholds,
    build_p2_test_dataset,
    calculate_pressure_windows,
)


def test_regression_metrics_match_manual_values():
    """MAE y RMSE de un caso calculado a mano."""
    metrics = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0]))

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["n_evaluated"] == 3


def test_p1_cohorts_split_by_period_and_ignore_test_values():
    """Validation termina antes de noviembre y no depende de los valores de test."""
    hours = pd.date_range("2025-08-25", "2025-12-31 23:00", freq="h")
    hourly = make_hourly_data(hours)
    changed = hourly.copy()
    in_test = changed["hour"] >= VALIDATION_END
    changed.loc[in_test, "departures"] += 5
    changed.loc[in_test, "net_flow"] -= 5
    changed.loc[in_test, "total_activity"] += 5

    cohorts = p1_cohorts(hourly)
    shifted = p1_cohorts(changed)

    assert cohorts["validation"]["hour"].max() < VALIDATION_END
    assert cohorts["test"]["hour"].min() == VALIDATION_END
    pd.testing.assert_frame_equal(cohorts["validation"], shifted["validation"])
    assert not cohorts["test"]["target_departures"].equals(shifted["test"]["target_departures"])


def make_boundary_hourly() -> pd.DataFrame:
    """Dos estaciones con flujos alternos de 28 de octubre a 4 de noviembre."""
    hours = pd.date_range("2025-10-28", "2025-11-04 23:00", freq="h")
    records = []
    for station_id, offset in ((1, 0), (2, 2)):
        for index, hour in enumerate(hours):
            phase = (index + offset) % 4
            records.append((station_id, hour, 2 if phase in (0, 1) else 0, 2 if phase > 1 else 0))
    return make_hourly(records)


def test_p2_test_dataset_uses_given_thresholds_and_only_past_history():
    """Las etiquetas usan los umbrales recibidos y los predictores solo el pasado."""
    hourly = make_boundary_hourly()
    catalog = make_catalog([1, 2], [10.0, 20.0])

    dataset = build_p2_test_dataset(hourly, catalog, PressureThresholds(0.2, 0.2))
    frozen = build_p2_test_dataset(hourly, catalog, PressureThresholds(1e6, 1e6))

    assert dataset["hour"].min() == VALIDATION_END
    assert dataset["hour"].max() == pd.Timestamp("2025-11-04 21:00")
    assert dataset["partition"].eq("test").all()
    assert dataset[[TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN]].sum().gt(0).all()
    assert frozen[[TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN]].sum().eq(0).all()

    first = dataset.loc[dataset["station_id"].eq(1) & dataset["hour"].eq(VALIDATION_END)].iloc[0]
    previous = hourly.loc[hourly["station_id"].eq(1) & hourly["hour"].eq("2025-10-31 23:00")]
    assert first["departures_lag_1h"] == previous["departures"].iloc[0]


def test_p2_test_windows_reject_validation_hours():
    """El modo test no acepta horas de enero-octubre y el modo normal rechaza test."""
    hourly = make_boundary_hourly()
    catalog = make_catalog([1, 2], [10.0, 20.0])

    with pytest.raises(ValueError, match="noviembre-diciembre"):
        calculate_pressure_windows(hourly, catalog, test=True)
    with pytest.raises(ValueError, match="reservado"):
        calculate_pressure_windows(hourly, catalog)
