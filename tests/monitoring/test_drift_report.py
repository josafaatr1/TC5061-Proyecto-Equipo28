"""Pruebas sintéticas de los reportes de drift y calidad con Evidently."""

import hashlib
import json

import pytest

from mibici.modeling.p2_catalog import CATALOG_FILENAME, PROVENANCE_FILENAME
from mibici.modeling.p2_targets import TARGET_BIKE_COLUMN, build_p2_targets
from mibici.monitoring.drift_report import (
    CATEGORICAL_COLUMNS,
    MONITORED_COLUMNS,
    NUMERICAL_COLUMNS,
    generate_reports,
    load_reference,
    summarize_drift,
    train_thresholds,
)
from tests.factories import make_hourly_for_reproduction, make_labeled_data


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


def _value_drift(metric_id, column, method, score, status):
    metric = {
        "id": metric_id,
        "config": {
            "type": "evidently:metric_v2:ValueDrift",
            "column": column,
            "method": method,
            "threshold": 0.05,
        },
        "value": score,
    }
    test = {"id": "drift", "metric_config": {"metric_id": metric_id}, "status": status}
    return metric, test


def test_summarize_drift_uses_evidently_decision_not_score_direction():
    # Fisher es una prueba de p-value cuyo nombre no contiene "p_value": la decisión
    # debe venir del estado de la prueba de Evidently, no inferirse del puntaje.
    fisher, fisher_test = _value_drift("m1", TARGET_BIKE_COLUMN, "fisher_exact", 1e-9, "FAIL")
    z_test, z_status = _value_drift("m2", "target_dock_pressure", "z", 0.9, "SUCCESS")
    snapshot = {"metrics": [z_test, fisher], "tests": [fisher_test, z_status]}

    rows = {row["column"]: row for row in summarize_drift(snapshot)}

    assert rows[TARGET_BIKE_COLUMN]["drift_detected"] is True
    assert rows["target_dock_pressure"]["drift_detected"] is False
    assert rows["target_dock_pressure"]["method"] == "Z-test p_value"


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


def _write_label_artifacts(tmp_path, alter_label=False):
    hourly, catalog = make_hourly_for_reproduction()
    hourly_path = tmp_path / "hourly.parquet"
    catalog_path = tmp_path / CATALOG_FILENAME
    label_path = tmp_path / "labels.parquet"
    hourly.to_parquet(hourly_path, index=False)
    catalog.to_csv(catalog_path, index=False)
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    catalog_path.with_name(PROVENANCE_FILENAME).write_text(
        json.dumps({"catalog_version": "2026-09", "sha256": digest}), encoding="utf-8"
    )
    result = build_p2_targets(hourly, catalog)
    labels = result.data.copy()
    if alter_label:
        labels.loc[0, TARGET_BIKE_COLUMN] = 1 - labels.loc[0, TARGET_BIKE_COLUMN]
    labels.to_parquet(label_path, index=False)
    return result.thresholds, (label_path, hourly_path, catalog_path)


def test_train_thresholds_match_the_ones_behind_reference_labels(tmp_path):
    expected, paths = _write_label_artifacts(tmp_path)

    assert train_thresholds(*paths) == expected


def test_train_thresholds_reject_labels_from_a_different_build(tmp_path):
    # Si reference y el Parquet horario no vienen de la misma construcción, las etiquetas
    # de current usarían otros umbrales y el drift de las etiquetas sería artificial.
    _, paths = _write_label_artifacts(tmp_path, alter_label=True)

    with pytest.raises(ValueError, match="no se reproducen exactamente"):
        train_thresholds(*paths)


def test_train_thresholds_use_provenance_of_supplied_catalog(tmp_path):
    expected, (label_path, hourly_path, catalog_path) = _write_label_artifacts(tmp_path)
    historic = tmp_path / "nomenclatura_2026_09.csv"
    catalog_path.rename(historic)
    catalog_path.with_name(PROVENANCE_FILENAME).rename(historic.with_suffix(".provenance.json"))

    assert train_thresholds(label_path, hourly_path, historic) == expected
