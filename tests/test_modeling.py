import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici.modeling import train as train_module
from mibici.modeling.train import (
    TARGET_COLUMN,
    evaluate_baselines,
    evaluate_candidate,
    prepare_supervised_data,
    run_model_comparison,
    split_chronologically,
    train_candidate_model,
    train_dummy_regressor,
)


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


def make_evaluation_split(
    targets: list[float],
    seasonal_values: list[float],
) -> pd.DataFrame:
    """Crear una partición mínima para evaluar ambos baselines."""
    hours = pd.date_range("2025-01-01", periods=len(targets), freq="h")

    return pd.DataFrame(
        {
            "hour": hours,
            "hour_of_day": hours.hour,
            TARGET_COLUMN: targets,
            "departures_lag_168h": seasonal_values,
        }
    )


def test_prepare_supervised_data_aligns_departures_by_station_and_hour():
    """Alinear la etiqueta de la hora pronosticada tras desordenar la entrada."""
    hours = pd.date_range("2025-01-01", periods=180, freq="h")
    departures = list(range(len(hours)))
    hourly_data = make_hourly_data(hours, departures).sample(
        frac=1,
        random_state=17,
    )

    supervised = prepare_supervised_data(hourly_data)
    target_hour = hours[179]
    target = supervised.loc[supervised["hour"] == target_hour].iloc[0]

    assert len(supervised) == len(hourly_data)
    assert target[TARGET_COLUMN] == departures[179]
    assert target["departures_lag_1h"] == departures[178]
    assert target["departures_lag_168h"] == departures[11]
    assert list(supervised["hour"]) == sorted(supervised["hour"])


def test_prepare_supervised_data_rejects_missing_targets():
    """No descartar silenciosamente observaciones sin etiqueta."""
    hours = pd.date_range("2025-01-01", periods=3, freq="h")
    hourly_data = make_hourly_data(hours)
    hourly_data.loc[1, "departures"] = np.nan

    with pytest.raises(ValueError, match="etiquetas ausentes"):
        prepare_supervised_data(hourly_data)


def test_prepare_supervised_data_keeps_features_causal():
    """Cambios futuros no alteran características de horas anteriores."""
    hours = pd.date_range("2025-01-01", periods=200, freq="h")
    original = make_hourly_data(hours)
    modified = original.copy(deep=True)
    cutoff = hours[169]
    future_rows = modified["hour"] >= cutoff
    modified.loc[future_rows, "departures"] += 1000

    original_supervised = prepare_supervised_data(original)
    modified_supervised = prepare_supervised_data(modified)
    feature_columns = [column for column in original_supervised.columns if column != TARGET_COLUMN]
    original_past = original_supervised.loc[
        original_supervised["hour"] <= cutoff,
        feature_columns,
    ].reset_index(drop=True)
    modified_past = modified_supervised.loc[
        modified_supervised["hour"] <= cutoff,
        feature_columns,
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_past, modified_past)


def test_split_chronologically_uses_expected_2025_boundaries():
    """Asignar las fronteras mensuales al conjunto temporal correcto."""
    hours = pd.to_datetime(
        [
            "2025-08-31 23:00",
            "2025-09-01 00:00",
            "2025-10-31 23:00",
            "2025-11-01 00:00",
            "2025-12-31 23:00",
        ]
    )
    supervised = pd.DataFrame(
        {
            "station_id": [1] * len(hours),
            "hour": hours,
            TARGET_COLUMN: [1, 2, 3, 4, 5],
        }
    )

    splits = split_chronologically(supervised)

    assert len(splits["train"]) == 1
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2
    assert splits["train"]["hour"].max() == pd.Timestamp("2025-08-31 23:00")
    assert splits["validation"]["hour"].min() == pd.Timestamp("2025-09-01 00:00")
    assert splits["validation"]["hour"].max() == pd.Timestamp("2025-10-31 23:00")
    assert splits["test"]["hour"].min() == pd.Timestamp("2025-11-01 00:00")
    assert sum(map(len, splits.values())) == len(supervised)


def test_split_without_test_ignores_invalid_test_labels():
    """No validar etiquetas de noviembre-diciembre si test está reservado."""
    supervised = pd.DataFrame(
        {
            "station_id": [1, 1, 1],
            "hour": pd.to_datetime(
                [
                    "2025-08-31 23:00",
                    "2025-09-01 00:00",
                    "2025-11-01 00:00",
                ]
            ),
            TARGET_COLUMN: [1, 2, "not-a-count"],
        }
    )

    splits = split_chronologically(supervised, evaluate_test=False)

    assert set(splits) == {"train", "validation"}
    assert splits["train"][TARGET_COLUMN].tolist() == [1]
    assert splits["validation"][TARGET_COLUMN].tolist() == [2]


def test_split_without_test_ignores_invalid_test_station_rows():
    """No validar claves o duplicados de noviembre-diciembre si test está reservado."""
    supervised = pd.DataFrame(
        {
            "station_id": [1, 1, None, None],
            "hour": pd.to_datetime(
                [
                    "2025-08-31 23:00",
                    "2025-09-01 00:00",
                    "2025-11-01 00:00",
                    "2025-11-01 00:00",
                ]
            ),
            TARGET_COLUMN: [1, 2, 3, 4],
        }
    )

    splits = split_chronologically(supervised, evaluate_test=False)

    assert set(splits) == {"train", "validation"}
    assert sum(map(len, splits.values())) == 2


def test_split_chronologically_rejects_rows_outside_configured_year():
    """Evitar descartar silenciosamente horas fuera del año evaluado."""
    supervised = pd.DataFrame(
        {
            "station_id": [1, 1, 1, 1],
            "hour": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-09-01 00:00",
                    "2025-11-01 00:00",
                    "2026-01-01 00:00",
                ]
            ),
            TARGET_COLUMN: [1, 2, 3, 4],
        }
    )

    with pytest.raises(ValueError, match="fuera del periodo"):
        split_chronologically(supervised)


def test_dummy_regressor_uses_training_target_mean():
    """Ajustar el baseline estadístico con las etiquetas de train."""
    training_data = pd.DataFrame(
        {
            "hour_of_day": [0, 12, 23],
            TARGET_COLUMN: [2, 4, 6],
        }
    )

    model = train_dummy_regressor(training_data)
    predictions = model.predict(pd.DataFrame({"hour_of_day": [1, 18]}))

    np.testing.assert_allclose(predictions, [4, 4])


def test_evaluate_baselines_reports_metrics_and_missing_seasonal_rows():
    """Comparar MAE/RMSE sobre la misma cohorte y contar exclusiones."""
    splits = {
        "train": make_evaluation_split(
            [10, 20, 30, 40],
            [8, 18, 28, 38],
        ),
        "validation": make_evaluation_split(
            [50, 60, 70],
            [48, 58, np.nan],
        ),
        "test": make_evaluation_split(
            [80, 90],
            [78, 88],
        ),
    }

    metrics = evaluate_baselines(splits, evaluate_test=True)
    validation_metrics = metrics.loc[metrics["evaluation_set"] == "validation"].set_index("model")

    assert set(metrics["evaluation_set"]) == {
        "train",
        "validation",
        "test",
    }
    assert set(metrics["model"]) == {"dummy_mean", "seasonal_weekly"}
    assert validation_metrics.loc["dummy_mean", "mae"] == pytest.approx(30)
    assert validation_metrics.loc["dummy_mean", "rmse"] == pytest.approx(np.sqrt(925))
    assert validation_metrics.loc["seasonal_weekly", "mae"] == pytest.approx(2)
    assert validation_metrics.loc["seasonal_weekly", "rmse"] == pytest.approx(2)
    assert validation_metrics["n_total"].eq(3).all()
    assert validation_metrics["n_evaluated"].eq(2).all()
    assert validation_metrics["n_excluded"].eq(1).all()


def test_validation_and_test_targets_do_not_change_fitted_dummy():
    """Ajustar DummyRegressor solo con train y repetir resultados."""
    splits = {
        "train": make_evaluation_split([10, 20, 30], [8, 18, 28]),
        "validation": make_evaluation_split([40, 50], [38, 48]),
        "test": make_evaluation_split([60, 70], [58, 68]),
    }

    original_metrics = evaluate_baselines(splits)
    repeated_metrics = evaluate_baselines(splits)
    changed_splits = dict(splits)
    changed_splits["validation"] = splits["validation"].copy(deep=True)
    changed_splits["validation"][TARGET_COLUMN] += 1000
    changed_metrics = evaluate_baselines(changed_splits)

    pd.testing.assert_frame_equal(
        original_metrics.drop(columns=["fit_seconds", "evaluation_seconds"]),
        repeated_metrics.drop(columns=["fit_seconds", "evaluation_seconds"]),
    )
    original_train = original_metrics.loc[
        original_metrics["evaluation_set"] == "train"
    ].reset_index(drop=True)
    changed_train = changed_metrics.loc[changed_metrics["evaluation_set"] == "train"].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        original_train.drop(columns=["fit_seconds", "evaluation_seconds"]),
        changed_train.drop(columns=["fit_seconds", "evaluation_seconds"]),
    )
    assert not np.isclose(
        original_metrics.loc[
            original_metrics["evaluation_set"] == "validation",
            "mae",
        ].to_numpy(),
        changed_metrics.loc[
            changed_metrics["evaluation_set"] == "validation",
            "mae",
        ].to_numpy(),
    ).all()


def test_default_baseline_evaluation_does_not_include_test_metrics():
    """La evaluación predeterminada devuelve solo entrenamiento y validación."""
    splits = {
        "train": make_evaluation_split([10, 20], [8, 18]),
        "validation": make_evaluation_split([30, 40], [28, 38]),
        "test": make_evaluation_split([50, 60], [48, 58]),
    }

    metrics = evaluate_baselines(splits)

    assert set(metrics["evaluation_set"]) == {"train", "validation"}


def test_candidate_uses_poisson_and_excludes_station_identity():
    """Entrenar con predictores temporales y rezagos, no con claves."""
    hours = pd.date_range("2025-01-01", periods=240, freq="h")
    supervised = prepare_supervised_data(make_hourly_data(hours))

    model = train_candidate_model(supervised)

    assert model.loss == "poisson"
    assert model.random_state == train_module.RANDOM_SEED
    assert "station_id" not in model.feature_names_in_
    assert "hour" not in model.feature_names_in_
    assert TARGET_COLUMN not in model.feature_names_in_
    assert "departures_rolling_mean_24h" in model.feature_names_in_
    assert np.isnan(supervised["departures_lag_168h"].iloc[0])
    predictions = model.predict(supervised.loc[:, model.feature_names_in_])
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_candidate_evaluation_is_deterministic_and_ignores_test_by_default():
    """El candidato usa train y no devuelve métricas del test reservado."""
    hours = pd.date_range("2025-01-01", "2026-01-01", inclusive="left", freq="h")
    supervised = prepare_supervised_data(make_hourly_data(hours))
    splits = split_chronologically(supervised)

    first = evaluate_candidate(splits)
    second = evaluate_candidate(splits)

    assert set(first["evaluation_set"]) == {"train", "validation"}
    pd.testing.assert_frame_equal(
        first.drop(columns=["fit_seconds", "evaluation_seconds"]),
        second.drop(columns=["fit_seconds", "evaluation_seconds"]),
    )
    assert first["n_evaluated"].eq(first["n_total"] - first["n_excluded"]).all()
    assert first["mae"].notna().all()
    assert first["rmse"].notna().all()


def test_default_cli_reads_only_through_validation_and_hides_test_metrics(
    tmp_path,
    monkeypatch,
):
    """La CLI por defecto filtra test y no publica sus métricas."""
    calls = {}

    def fake_read_parquet(path, **kwargs):
        calls["read_filters"] = kwargs.get("filters")
        return pd.DataFrame({"synthetic": [1]})

    def fake_run_model_comparison(hourly_data, evaluate_test=False):
        calls["evaluate_test"] = evaluate_test
        return pd.DataFrame(
            {
                "evaluation_set": ["train", "validation"],
                "model": ["dummy_mean", "dummy_mean"],
                "mae": [1.0, 2.0],
                "rmse": [1.0, 2.0],
            }
        )

    monkeypatch.setattr(train_module.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(train_module, "run_model_comparison", fake_run_model_comparison)
    metrics_path = tmp_path / "metrics.csv"
    runner = CliRunner()

    result = runner.invoke(
        train_module.app,
        [
            "--input-path",
            str(tmp_path / "input.parquet"),
            "--metrics-path",
            str(metrics_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["evaluate_test"] is False
    assert calls["read_filters"] == [("hour", "<", pd.Timestamp("2025-11-01 00:00:00"))]
    assert set(pd.read_csv(metrics_path)["evaluation_set"]) == {"train", "validation"}


def test_evaluate_baselines_rejects_missing_targets():
    """Fallar explícitamente ante etiquetas ausentes en una partición."""
    splits = {
        "train": make_evaluation_split([1, 2], [0, 1]),
        "validation": make_evaluation_split([3, np.nan], [2, 1]),
        "test": make_evaluation_split([4, 5], [3, 4]),
    }

    with pytest.raises(ValueError, match="valores ausentes"):
        evaluate_baselines(splits)


def test_run_baselines_integrates_synthetic_2025_data():
    """Comparar tres modelos en entrenamiento y validación sintéticos."""
    hours = pd.date_range("2025-01-01", "2026-01-01", inclusive="left", freq="h")
    hourly_data = make_hourly_data(hours)

    metrics = run_model_comparison(hourly_data)

    assert len(metrics) == 6
    assert set(metrics["evaluation_set"]) == {
        "train",
        "validation",
    }
    assert set(metrics["model"]) == {
        "dummy_mean",
        "seasonal_weekly",
        "hist_gradient_boosting_poisson",
    }
    assert metrics["n_evaluated"].gt(0).all()
    assert metrics[["mae", "rmse"]].notna().all().all()
