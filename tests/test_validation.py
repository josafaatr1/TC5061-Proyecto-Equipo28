"""Pruebas sintéticas de integridad y drift de P1 y P2."""

import numpy as np
import pandas as pd
import pytest

from mibici.modeling.p2_targets import TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN
from mibici.modeling.p2_train import PREDICTOR_COLUMNS
from mibici.validation import (
    MAX_DEEPCHECKS_SAMPLE_ROWS,
    _validate_partition,
    run_validation,
)


def make_hourly_data() -> pd.DataFrame:
    hours = pd.date_range("2025-08-20", "2025-11-05", freq="h", inclusive="left")
    index = np.arange(len(hours))
    departures = index % 7
    arrivals = (index + 3) % 5
    return pd.DataFrame(
        {
            "station_id": 1,
            "hour": hours,
            "departures": departures,
            "arrivals": arrivals,
            "net_flow": arrivals - departures,
            "total_activity": arrivals + departures,
            "is_observed_hour": True,
        }
    )


def make_p2_data() -> pd.DataFrame:
    records = []
    partitions = (
        ("train", pd.date_range("2025-08-01", periods=240, freq="h")),
        ("validation", pd.date_range("2025-09-01", periods=120, freq="h")),
    )
    for partition, hours in partitions:
        for index, hour in enumerate(hours):
            row = {
                "station_id": 11,
                "hour": hour,
                "partition": partition,
                TARGET_BIKE_COLUMN: int(index % 10 == 0),
                TARGET_DOCK_COLUMN: int(index % 13 == 0),
            }
            for column in PREDICTOR_COLUMNS:
                if column == "hour_of_day":
                    row[column] = hour.hour
                elif column == "day_of_week":
                    row[column] = hour.dayofweek
                elif column == "month":
                    row[column] = hour.month
                elif column == "is_weekend":
                    row[column] = hour.dayofweek >= 5
                else:
                    row[column] = float(index % 17)
            row["departures_lag_168h"] = np.nan if index < 5 else row["departures_lag_168h"]
            records.append(row)
    return pd.DataFrame(records)


def make_partition() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": [1, 1],
            "hour": pd.date_range("2025-01-01", periods=2, freq="h"),
            "departures_lag_1h": [np.nan, 2.0],
            "feature": [3.0, 4.0],
            "target": [0.0, 1.0],
        }
    )


def test_validate_partition_reports_missing_lags_without_failing():
    summary = _validate_partition(
        make_partition(),
        ("departures_lag_1h", "feature"),
        "target",
        "synthetic",
    )

    assert summary["lag_feature_nulls"]["departures_lag_1h"] == {
        "count": 1,
        "percent": 50.0,
    }
    assert summary["null_counts"]["target"] == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.assign(station_id=[1, np.nan]), "station_id.*nulos"),
        (lambda data: data.assign(hour=[pd.NaT, pd.Timestamp("2025-01-01 01:00")]), "hour.*nulos"),
        (lambda data: data.assign(target=[0.0, np.nan]), "etiquetas nulas"),
        (lambda data: data.assign(feature=[np.inf, 4.0]), "infinitos"),
        (
            lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True),
            "claves station_id-hour duplicadas",
        ),
    ],
)
def test_validate_partition_stops_on_invalid_keys_labels_infinities_or_duplicates(
    mutate,
    message,
):
    with pytest.raises(ValueError, match=message):
        _validate_partition(
            mutate(make_partition()),
            ("departures_lag_1h", "feature"),
            "target",
            "synthetic",
        )


def test_run_validation_generates_synthetic_reports_without_using_test_for_models(tmp_path):
    p1_path = tmp_path / "p1_hourly.parquet"
    p2_path = tmp_path / "p2_labels.parquet"
    output_dir = tmp_path / "deepchecks"
    make_hourly_data().to_parquet(p1_path, index=False)
    make_p2_data().to_parquet(p2_path, index=False)

    summary = run_validation(p1_path, p2_path, output_dir)

    assert summary["deepchecks_version"] == "0.19.1"
    assert summary["sample_limit_rows"] == MAX_DEEPCHECKS_SAMPLE_ROWS
    assert summary["test_policy"] == {
        "p1_test_used_for": ["integrity", "distribution_drift"],
        "p1_test_used_for_training": False,
        "p1_test_used_for_model_selection": False,
        "p2_test_loaded": False,
    }
    assert set(summary["partitions"]["p1"]) == {"train", "validation", "test"}
    assert summary["partitions"]["p1"]["test"]["rows"] > 0
    assert summary["partitions"]["p1"]["train"]["lag_feature_nulls"]
    assert set(summary["partitions"]["p2"][TARGET_BIKE_COLUMN]) == {
        "train",
        "validation",
    }
    assert summary["p2_predictors_by_target"][TARGET_BIKE_COLUMN] == list(PREDICTOR_COLUMNS)
    assert summary["p2_predictors_by_target"][TARGET_DOCK_COLUMN] == list(PREDICTOR_COLUMNS)
    assert TARGET_DOCK_COLUMN not in summary["p2_predictors_by_target"][TARGET_BIKE_COLUMN]
    assert TARGET_BIKE_COLUMN not in summary["p2_predictors_by_target"][TARGET_DOCK_COLUMN]

    reports = summary["reports"]
    assert len(reports) == 9
    assert sum(report["suite"] == "data_integrity" for report in reports) == 5
    assert sum(report["suite"] == "train_test_validation" for report in reports) == 4
    assert {report["html"] for report in reports} == {
        "p1_integrity_train.html",
        "p1_integrity_validation.html",
        "p1_integrity_test.html",
        "p1_train_vs_validation.html",
        "p1_train_vs_test.html",
        "p2_integrity_train.html",
        "p2_integrity_validation.html",
        "p2_bike_train_vs_validation.html",
        "p2_dock_train_vs_validation.html",
    }
    assert all(report["checks_errored"] == 0 for report in reports)
    assert all(
        report["checks_not_passed"] == len(report["not_passed_checks"]) for report in reports
    )
    reports_by_name = {report["html"]: report for report in reports}
    assert (
        "New Label Train Test" in reports_by_name["p1_train_vs_validation.html"]["checks_excluded"]
    )
    assert (
        "Train Test Samples Mix"
        in reports_by_name["p1_train_vs_validation.html"]["checks_excluded"]
    )
    assert (
        "New Label Train Test"
        not in reports_by_name["p2_bike_train_vs_validation.html"]["checks_excluded"]
    )
    assert reports_by_name["p1_train_vs_validation.html"]["partitions"] == [
        "train",
        "validation",
    ]
    assert reports_by_name["p1_train_vs_test.html"]["partitions"] == ["train", "test"]
    assert reports_by_name["p2_bike_train_vs_validation.html"]["partitions"] == [
        "train",
        "validation",
    ]
    assert reports_by_name["p2_dock_train_vs_validation.html"]["partitions"] == [
        "train",
        "validation",
    ]
    assert all((output_dir / report["html"]).is_file() for report in reports)
    assert (output_dir / "summary.json").is_file()
