"""Pruebas sintéticas de los reportes de drift y calidad con Evidently."""

import json

import pytest

from mibici.modeling.p2_targets import TARGET_BIKE_COLUMN
from mibici.monitoring.drift_report import (
    CATEGORICAL_COLUMNS,
    MONITORED_COLUMNS,
    NUMERICAL_COLUMNS,
    drift_detected,
    generate_reports,
    load_reference,
)
from tests.factories import make_labeled_data


def test_monitored_columns_are_model_inputs_plus_targets():
    assert set(CATEGORICAL_COLUMNS) | set(NUMERICAL_COLUMNS) == set(MONITORED_COLUMNS)
    assert not set(CATEGORICAL_COLUMNS) & set(NUMERICAL_COLUMNS)
    assert {"month", "hour_of_day", TARGET_BIKE_COLUMN} <= set(CATEGORICAL_COLUMNS)
    assert "station_id" not in MONITORED_COLUMNS


def test_load_reference_keeps_only_train_partition(tmp_path):
    label_path = tmp_path / "labels.parquet"
    labeled = make_labeled_data(include_test=False)
    labeled.to_parquet(label_path)

    reference = load_reference(label_path)

    assert list(reference.columns) == list(MONITORED_COLUMNS)
    assert len(reference) == labeled["partition"].eq("train").sum()
    assert set(reference["month"]) == {8}


@pytest.mark.parametrize(
    ("method", "score", "expected"),
    [
        ("Jensen-Shannon distance", 0.83, True),
        ("Wasserstein distance (normed)", 0.03, False),
        ("K-S p_value", 0.01, True),
        ("chi-square p_value", 0.40, False),
        # Con per_column_method Evidently guarda el id corto de la prueba.
        ("z", 1e-9, True),
        ("z", 0.40, False),
    ],
)
def test_drift_detected_respects_direction_of_each_method(method, score, expected):
    threshold = 0.05 if "p_value" in method or method == "z" else 0.1
    assert drift_detected(method, score, threshold) is expected


def test_generate_reports_writes_html_and_per_column_summary(tmp_path):
    labeled = make_labeled_data(include_test=False)
    reference = labeled.loc[labeled["partition"].eq("train"), list(MONITORED_COLUMNS)]
    current = labeled.loc[labeled["partition"].eq("validation"), list(MONITORED_COLUMNS)]

    summary = generate_reports(reference, current, tmp_path)

    assert (tmp_path / "data_drift_report.html").stat().st_size > 0
    assert (tmp_path / "data_quality_report.html").stat().st_size > 0
    saved = json.loads((tmp_path / "drift_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert [row["column"] for row in summary["columns"]] == list(MONITORED_COLUMNS)
    by_column = {row["column"]: row for row in summary["columns"]}
    # Agosto contra septiembre: el mes cambia por completo y debe marcarse como drift.
    assert by_column["month"]["drift_detected"] is True
    assert by_column[TARGET_BIKE_COLUMN]["method"] == "Z-test p_value"
    assert summary["reference"]["rows"] == len(reference)
    assert summary["current"]["rows"] == len(current)
    assert summary["prevalence"][TARGET_BIKE_COLUMN]["reference"] == pytest.approx(
        reference[TARGET_BIKE_COLUMN].mean()
    )
