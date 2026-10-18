import numpy as np
import pandas as pd
import pytest

from mibici.modeling.train import (
    TARGET_COLUMN,
    evaluate_baselines,
    prepare_supervised_data,
    run_baselines,
    split_chronologically,
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

    metrics = evaluate_baselines(splits)
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

    pd.testing.assert_frame_equal(original_metrics, repeated_metrics)
    original_train = original_metrics.loc[
        original_metrics["evaluation_set"] == "train"
    ].reset_index(drop=True)
    changed_train = changed_metrics.loc[changed_metrics["evaluation_set"] == "train"].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(original_train, changed_train)
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
    """Ejecutar preparación, splits y ambos baselines de extremo a extremo."""
    hours = pd.date_range("2025-01-01", "2026-01-01", inclusive="left", freq="h")
    hourly_data = make_hourly_data(hours)

    metrics = run_baselines(hourly_data)

    assert len(metrics) == 6
    assert set(metrics["evaluation_set"]) == {
        "train",
        "validation",
        "test",
    }
    assert set(metrics["model"]) == {"dummy_mean", "seasonal_weekly"}
    assert metrics["n_evaluated"].gt(0).all()
    assert metrics[["mae", "rmse"]].notna().all().all()
