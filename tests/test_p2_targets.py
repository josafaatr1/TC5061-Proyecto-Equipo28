"""Pruebas de ventanas, umbrales y etiquetas proxy de P2."""

import pandas as pd
import pytest

from mibici.features import build_features
from mibici.modeling.p2_targets import (
    BIKE_PRESSURE_COLUMN,
    BIKE_RATIO_COLUMN,
    DOCK_PRESSURE_COLUMN,
    DOCK_RATIO_COLUMN,
    TARGET_BIKE_COLUMN,
    TARGET_DOCK_COLUMN,
    PressureThresholds,
    apply_pressure_thresholds,
    build_p2_targets,
    calculate_pressure_windows,
    fit_pressure_thresholds,
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


def test_calculate_pressure_windows_handles_bike_dock_and_zero_flows():
    """Calcular presión direccional con flujos acumulados de tres horas."""
    hours = pd.date_range("2025-01-01", periods=3, freq="h")
    records = []
    flows = {
        1: [(3, 0), (0, 0), (0, 2)],
        2: [(0, 2), (0, 3), (5, 0)],
        3: [(0, 0), (0, 0), (0, 0)],
    }
    for station_id, station_flows in flows.items():
        records.extend(
            (station_id, hour, departure, arrival)
            for hour, (departure, arrival) in zip(hours, station_flows, strict=True)
        )
    hourly = make_hourly(records)
    catalog = make_catalog([1, 2, 3])

    result = calculate_pressure_windows(hourly, catalog)
    windows = result.windows.set_index("station_id")

    assert windows.loc[1, BIKE_PRESSURE_COLUMN] == 3
    assert windows.loc[1, DOCK_PRESSURE_COLUMN] == 0
    assert windows.loc[1, BIKE_RATIO_COLUMN] == pytest.approx(0.3)
    assert windows.loc[2, BIKE_PRESSURE_COLUMN] == 0
    assert windows.loc[2, DOCK_PRESSURE_COLUMN] == 5
    assert windows.loc[2, DOCK_RATIO_COLUMN] == pytest.approx(0.5)
    assert windows.loc[3, BIKE_PRESSURE_COLUMN] == 0
    assert windows.loc[3, DOCK_PRESSURE_COLUMN] == 0
    assert len(windows) == 3


def test_calculate_pressure_windows_rejects_discontinuous_hours():
    """No tratar un registro dos horas después como hora consecutiva."""
    hours = pd.to_datetime(
        [
            "2025-01-01 00:00",
            "2025-01-01 01:00",
            "2025-01-01 03:00",
            "2025-01-01 04:00",
            "2025-01-01 05:00",
        ]
    )
    hourly = make_hourly([(1, hour, 1, 0) for hour in hours])

    result = calculate_pressure_windows(hourly, make_catalog([1]))

    assert result.windows["hour"].tolist() == [pd.Timestamp("2025-01-01 03:00")]
    assert result.audit["excluded_incomplete_windows"] == 4
    assert result.audit["valid_windows"] == 1


def test_calculate_pressure_windows_excludes_partition_boundaries():
    """Excluir ventanas exactas que atraviesan de train a validation."""
    hours = pd.date_range("2025-08-31 21:00", periods=7, freq="h")
    hourly = make_hourly([(1, hour, 0, 1) for hour in hours])

    result = calculate_pressure_windows(hourly, make_catalog([1]))

    assert result.windows["hour"].tolist() == [
        pd.Timestamp("2025-08-31 21:00"),
        pd.Timestamp("2025-09-01 00:00"),
        pd.Timestamp("2025-09-01 01:00"),
    ]
    assert result.audit["excluded_cross_partition_windows"] == 2
    assert result.audit["excluded_incomplete_windows"] == 2
    assert result.audit["windows_by_partition"]["train"]["valid_windows"] == 1
    assert result.audit["windows_by_partition"]["validation"]["valid_windows"] == 2


def test_invalid_and_missing_capacities_are_excluded_with_traceability():
    """Excluir capacidad cero y ausente sin dividir ni perder trazabilidad."""
    hours = pd.date_range("2025-01-01", periods=3, freq="h")
    records = [(station_id, hour, 1, 0) for station_id in (1, 2, 3, 4) for hour in hours]
    catalog = make_catalog([1, 2, 3], [10.0, 0.0, None])

    result = calculate_pressure_windows(make_hourly(records), catalog)

    assert result.windows["station_id"].unique().tolist() == [1]
    assert result.audit["records_excluded_without_valid_capacity"] == 9
    assert result.excluded_stations.set_index("station_id")["reason"].to_dict() == {
        2: "missing_or_nonpositive_capacity",
        3: "missing_or_nonpositive_capacity",
        4: "missing_catalog_station",
    }


def test_duplicate_catalog_ids_are_rejected():
    """No permitir IDs duplicados que vuelvan ambigua la capacidad."""
    hours = pd.date_range("2025-01-01", periods=3, freq="h")
    hourly = make_hourly([(1, hour, 1, 0) for hour in hours])
    catalog = make_catalog([1, 1], [10.0, 10.0])

    with pytest.raises(ValueError, match="IDs duplicados"):
        calculate_pressure_windows(hourly, catalog)


def test_fit_pressure_thresholds_uses_only_positive_train_values():
    """Calcular percentiles 90 independientes de las presiones de validación."""
    train = pd.DataFrame(
        {
            "partition": ["train"] * 5,
            BIKE_RATIO_COLUMN: [0.0, 0.1, 0.2, 0.3, 0.4],
            DOCK_RATIO_COLUMN: [0.0, 0.2, 0.4, 0.6, 0.8],
        }
    )
    thresholds = fit_pressure_thresholds(train)
    changed_validation = pd.DataFrame(
        {
            "partition": ["validation"],
            BIKE_RATIO_COLUMN: [100.0],
            DOCK_RATIO_COLUMN: [200.0],
        }
    )

    assert thresholds.bike_pressure_ratio == pytest.approx(0.37)
    assert thresholds.dock_pressure_ratio == pytest.approx(0.74)
    assert fit_pressure_thresholds(train) == thresholds
    with pytest.raises(ValueError, match="solo pueden ajustarse"):
        fit_pressure_thresholds(pd.concat([train, changed_validation], ignore_index=True))


def test_apply_pressure_thresholds_creates_independent_binary_labels():
    """Etiquetar presiones de bicicletas y anclajes por separado."""
    windows = pd.DataFrame(
        {
            "partition": ["train", "validation", "validation"],
            BIKE_RATIO_COLUMN: [0.4, 0.5, 0.1],
            DOCK_RATIO_COLUMN: [0.1, 0.2, 0.5],
        }
    )
    thresholds = PressureThresholds(0.4, 0.3)

    labeled = apply_pressure_thresholds(windows, thresholds)

    assert labeled[TARGET_BIKE_COLUMN].tolist() == [1, 1, 0]
    assert labeled[TARGET_DOCK_COLUMN].tolist() == [0, 0, 1]


def test_threshold_fitting_and_summaries_reject_test():
    """Impedir ajuste o resumen de umbrales con información del test."""
    windows = pd.DataFrame(
        {
            "partition": ["train", "test"],
            BIKE_RATIO_COLUMN: [0.1, 0.2],
            DOCK_RATIO_COLUMN: [0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="solo pueden ajustarse"):
        fit_pressure_thresholds(windows)


def test_build_targets_is_deterministic_causal_and_preserves_inputs():
    """Generar etiquetas reproducibles con predictores estrictamente históricos."""
    hours = pd.date_range("2025-08-25", periods=240, freq="h")
    records = []
    for station_id, offset in ((1, 0), (2, 2)):
        for index, hour in enumerate(hours):
            phase = (index + offset) % 4
            departures = 2 if phase in (0, 1) else 0
            arrivals = 2 if phase in (2, 3) else 0
            records.append((station_id, hour, departures, arrivals))
    hourly = make_hourly(records)
    original_hourly = hourly.copy(deep=True)
    catalog = make_catalog([1, 2], [10.0, 20.0])
    original_catalog = catalog.copy(deep=True)

    result = build_p2_targets(hourly, catalog)
    repeated = build_p2_targets(hourly, catalog)

    assert set(result.data["partition"]) == {"train", "validation"}
    pd.testing.assert_frame_equal(result.data, repeated.data)
    assert result.thresholds == repeated.thresholds
    assert result.data[TARGET_BIKE_COLUMN].isin([0, 1]).all()
    assert result.data[TARGET_DOCK_COLUMN].isin([0, 1]).all()
    assert "future_arrivals" not in result.data.columns
    assert "future_departures" not in result.data.columns
    assert BIKE_RATIO_COLUMN not in result.data.columns
    assert DOCK_RATIO_COLUMN not in result.data.columns
    assert BIKE_RATIO_COLUMN in result.pressure_windows.columns
    assert DOCK_RATIO_COLUMN in result.pressure_windows.columns

    feature_columns = build_features(hourly).columns
    cutoff = pd.Timestamp("2025-09-01")
    modified = hourly.copy(deep=True)
    future = modified["hour"] >= cutoff
    modified.loc[future, "departures"] += 1
    modified.loc[future, "net_flow"] = (
        modified.loc[future, "arrivals"] - modified.loc[future, "departures"]
    )
    modified.loc[future, "total_activity"] = (
        modified.loc[future, "arrivals"] + modified.loc[future, "departures"]
    )
    changed = build_p2_targets(modified, catalog)
    baseline_past = result.data.loc[result.data["hour"] < cutoff, feature_columns]
    changed_past = changed.data.loc[changed.data["hour"] < cutoff, feature_columns]
    pd.testing.assert_frame_equal(
        baseline_past.reset_index(drop=True),
        changed_past.reset_index(drop=True),
    )
    assert result.thresholds == changed.thresholds
    pd.testing.assert_frame_equal(hourly, original_hourly)
    pd.testing.assert_frame_equal(catalog, original_catalog)


def test_test_period_input_is_rejected_before_target_construction():
    """Evitar cargar o calcular ventanas de noviembre-diciembre."""
    hours = pd.date_range("2025-10-31 21:00", periods=4, freq="h")
    hourly = make_hourly([(1, hour, 0, 1) for hour in hours])

    with pytest.raises(ValueError, match="periodo test está reservado"):
        calculate_pressure_windows(hourly, make_catalog([1]))


def test_fit_thresholds_rejects_pressure_without_positive_train_values():
    """No crear etiquetas sin presión positiva con que definir el percentil."""
    train = pd.DataFrame(
        {
            "partition": ["train", "train"],
            BIKE_RATIO_COLUMN: [0.0, 0.0],
            DOCK_RATIO_COLUMN: [0.1, 0.0],
        }
    )

    with pytest.raises(ValueError, match="No hay presiones positivas"):
        fit_pressure_thresholds(train)
