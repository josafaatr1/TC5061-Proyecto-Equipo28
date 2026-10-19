"""Pruebas de entrenamiento y evaluación temporal de clasificadores P2."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici.features import FLOW_COLUMNS, LAG_HOURS, ROLLING_WINDOWS
from mibici.modeling import p2_train
from mibici.modeling.p2_catalog import CATALOG_FILENAME, PROVENANCE_FILENAME
from mibici.modeling.p2_targets import TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN


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


def test_prepare_splits_whitelists_historical_features_and_ignores_test():
    """Excluir llaves, presión futura, extras y registros de test."""
    labeled_data = make_labeled_data()

    splits = p2_train.prepare_p2_splits(labeled_data)

    assert set(splits) == {"train", "validation"}
    assert len(splits["train"]) == 240
    assert len(splits["validation"]) == 120
    assert set(p2_train.PREDICTOR_COLUMNS).isdisjoint(
        {
            "station_id",
            "hour",
            "partition",
            TARGET_BIKE_COLUMN,
            TARGET_DOCK_COLUMN,
            "bike_pressure_ratio",
            "dock_pressure_ratio",
            "future_departures",
        }
    )
    assert np.isinf(
        labeled_data.loc[labeled_data["partition"].eq("test"), "departures_lag_1h"]
    ).all()


def test_train_objective_uses_balanced_weights_and_explicit_seed():
    """Ajustar el candidato con class weight solo a partir de las filas train."""
    train = p2_train.prepare_p2_splits(make_labeled_data())["train"]

    trained = p2_train.train_objective(train, TARGET_BIKE_COLUMN)

    assert trained.dummy.strategy == "prior"
    assert trained.candidate.class_weight == "balanced"
    assert trained.candidate.loss == "log_loss"
    assert trained.candidate.early_stopping is False
    assert trained.candidate.random_state == 42
    assert trained.training_prevalence == pytest.approx(0.1)


def test_train_objective_rejects_single_class_and_invalid_labels():
    """Fallar explícitamente ante una sola clase o etiquetas fuera de {0,1}."""
    train = p2_train.prepare_p2_splits(make_labeled_data())["train"].copy()
    train[TARGET_BIKE_COLUMN] = 0
    with pytest.raises(ValueError, match="ambas clases"):
        p2_train.train_objective(train, TARGET_BIKE_COLUMN)

    train = p2_train.prepare_p2_splits(make_labeled_data())["train"].copy()
    train.loc[0, TARGET_BIKE_COLUMN] = 2
    with pytest.raises(ValueError, match="binarias 0/1"):
        p2_train.train_objective(train, TARGET_BIKE_COLUMN)


def test_prepare_splits_rejects_invalid_validation_labels():
    """No aceptar etiquetas ausentes o inválidas en validation."""
    labeled_data = make_labeled_data()
    validation_row = labeled_data["partition"].eq("validation")
    first_validation_index = labeled_data.index[validation_row][0]
    labeled_data.loc[first_validation_index, TARGET_DOCK_COLUMN] = np.nan

    with pytest.raises(ValueError, match="etiquetas ausentes"):
        p2_train.prepare_p2_splits(labeled_data)


def test_evaluate_probabilities_calculates_pr_metrics_and_confusion():
    """Calcular AP y medidas de decisión en un ejemplo verificable."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.7, 0.4, 0.9])

    metrics = p2_train.evaluate_probabilities(y_true, probabilities)

    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["positive_prevalence"] == pytest.approx(0.5)
    assert metrics["n_evaluated"] == 4


def test_evaluate_probabilities_rejects_validation_with_single_class():
    """AP no se calcula cuando validation carece de una clase."""
    with pytest.raises(ValueError, match="ambas clases"):
        p2_train.evaluate_probabilities(
            np.zeros(4, dtype=np.int8),
            np.array([0.1, 0.2, 0.3, 0.4]),
        )


def test_run_validation_uses_same_rows_and_is_reproducible():
    """Comparar las dos tareas sin evaluar test ni ajustar con validation."""
    labeled_data = make_labeled_data()
    first = p2_train.run_validation_experiment(labeled_data)
    changed_validation = labeled_data.copy(deep=True)
    validation_mask = changed_validation["partition"].eq("validation")
    changed_validation.loc[validation_mask, TARGET_BIKE_COLUMN] = (
        1 - changed_validation.loc[validation_mask, TARGET_BIKE_COLUMN]
    )
    changed_validation.loc[validation_mask, TARGET_DOCK_COLUMN] = (
        1 - changed_validation.loc[validation_mask, TARGET_DOCK_COLUMN]
    )
    second = p2_train.run_validation_experiment(changed_validation)

    assert len(first.metrics) == 4
    assert set(first.metrics["evaluation_set"]) == {"validation"}
    assert set(first.metrics["target"]) == set(p2_train.TARGET_COLUMNS)
    assert set(first.metrics["model"]) == {
        p2_train.BASELINE_NAME,
        p2_train.CANDIDATE_NAME,
    }
    assert first.metrics["n_evaluated"].eq(120).all()
    assert first.metrics["n_excluded"].eq(0).all()
    assert first.metrics["positive_prevalence"].between(0, 1).all()
    assert first.metrics["average_precision"].between(0, 1).all()
    assert first.metrics["decision_threshold"].eq(0.5).all()

    validation = p2_train.prepare_p2_splits(labeled_data)["validation"]
    features = p2_train._feature_array(validation)
    for target_column in p2_train.TARGET_COLUMNS:
        first_probability = first.models[target_column].candidate.predict_proba(features)
        second_probability = second.models[target_column].candidate.predict_proba(features)
        np.testing.assert_array_equal(first_probability, second_probability)


def test_verify_label_reproducibility_reconstructs_exact_thresholds(tmp_path):
    """Verificar el Parquet sintético de etiquetas contra la reconstrucción."""
    from mibici.modeling.p2_targets import build_p2_targets

    hourly, catalog = make_hourly_for_reproduction()
    hourly_path = tmp_path / "hourly.parquet"
    catalog_path = tmp_path / CATALOG_FILENAME
    labels_path = tmp_path / "labels.parquet"
    hourly.to_parquet(hourly_path, index=False)
    catalog.to_csv(catalog_path, index=False)
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    catalog_path.with_name(PROVENANCE_FILENAME).write_text(
        json.dumps({"catalog_version": "2026-09", "sha256": digest}),
        encoding="utf-8",
    )
    labels = build_p2_targets(hourly, catalog)
    labels.data.to_parquet(labels_path, index=False)

    provenance = p2_train.verify_label_reproducibility(
        labels_path,
        hourly_path,
        catalog_path,
    )

    assert provenance["labels_reproduced_exactly"] is True
    assert provenance["catalog_version"] == "2026-09"
    assert provenance["catalog_sha256"] == digest
    assert provenance["label_pressure_quantile"] == 0.9
    assert provenance["label_pressure_thresholds_train_only"] == {
        "bike_pressure_ratio": labels.thresholds.bike_pressure_ratio,
        "dock_pressure_ratio": labels.thresholds.dock_pressure_ratio,
    }


def test_verify_label_reproducibility_rejects_changed_saved_label(tmp_path):
    """Detectar alteración de una etiqueta almacenada."""
    from mibici.modeling.p2_targets import build_p2_targets

    hourly, catalog = make_hourly_for_reproduction()
    hourly_path = tmp_path / "hourly.parquet"
    catalog_path = tmp_path / CATALOG_FILENAME
    labels_path = tmp_path / "labels.parquet"
    hourly.to_parquet(hourly_path, index=False)
    catalog.to_csv(catalog_path, index=False)
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    catalog_path.with_name(PROVENANCE_FILENAME).write_text(
        json.dumps({"catalog_version": "2026-09", "sha256": digest}),
        encoding="utf-8",
    )
    labels = build_p2_targets(hourly, catalog).data.copy()
    labels.loc[0, TARGET_BIKE_COLUMN] = 1 - labels.loc[0, TARGET_BIKE_COLUMN]
    labels.to_parquet(labels_path, index=False)

    with pytest.raises(ValueError, match="no se reproducen exactamente"):
        p2_train.verify_label_reproducibility(
            labels_path,
            hourly_path,
            catalog_path,
        )


def test_cli_writes_only_validation_metrics_and_uses_safe_filter(tmp_path, monkeypatch):
    """La CLI filtra particiones y nunca expone resultados del conjunto test."""
    labeled_data = make_labeled_data()
    calls = {}

    def fake_verify_label_reproducibility(label_path, hourly_path, catalog_path):
        return {
            "labels_reproduced_exactly": True,
            "catalog_version": "2026-09",
            "catalog_sha256": "catalog-hash",
            "label_pressure_thresholds_train_only": {
                "bike_pressure_ratio": 0.3,
                "dock_pressure_ratio": 0.3,
            },
            "label_pressure_quantile": 0.9,
            "decision_threshold": 0.5,
        }

    def fake_read_parquet(path, columns=None, filters=None):
        calls["filters"] = filters
        return labeled_data.loc[
            :,
            [
                "station_id",
                "hour",
                "partition",
                *p2_train.PREDICTOR_COLUMNS,
                *p2_train.TARGET_COLUMNS,
            ],
        ]

    monkeypatch.setattr(
        p2_train,
        "verify_label_reproducibility",
        fake_verify_label_reproducibility,
    )
    monkeypatch.setattr(p2_train.pd, "read_parquet", fake_read_parquet)
    metrics_path = tmp_path / "p2-metrics.json"
    result = CliRunner().invoke(
        p2_train.app,
        [
            "--label-path",
            str(tmp_path / "labels.parquet"),
            "--hourly-path",
            str(tmp_path / "hourly.parquet"),
            "--catalog-path",
            str(tmp_path / CATALOG_FILENAME),
            "--metrics-path",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert {row["evaluation_set"] for row in report["validation_metrics"]} == {"validation"}
    assert all(row["n_excluded"] == 0 for row in report["validation_metrics"])
    assert calls["filters"] == [
        [("partition", "==", "train")],
        [("partition", "==", "validation")],
    ]


def test_feature_column_list_is_shared_with_existing_feature_constants():
    """Mantener predictores limitados al calendario, rezagos y medias P1."""
    expected = (
        *p2_train.CALENDAR_FEATURES,
        *(f"{column}_lag_{lag}h" for column in FLOW_COLUMNS for lag in LAG_HOURS),
        *(f"departures_rolling_mean_{window}h" for window in ROLLING_WINDOWS),
    )

    assert p2_train.PREDICTOR_COLUMNS == expected
    assert len(p2_train.PREDICTOR_COLUMNS) == 15
