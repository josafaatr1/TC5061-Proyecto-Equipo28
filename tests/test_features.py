import pandas as pd
import pytest

from mibici.features import build_features


def make_hourly_data(
    hours: pd.DatetimeIndex,
    station_id: int = 1,
    departures: list[int] | None = None,
    arrivals: list[int] | None = None,
    is_observed_hour: list[bool] | None = None,
) -> pd.DataFrame:
    """Crear un dataset sintético con el esquema horario del proyecto."""
    row_count = len(hours)
    departure_values = list(departures) if departures is not None else list(range(row_count))
    arrival_values = (
        list(arrivals) if arrivals is not None else [10 + 2 * index for index in range(row_count)]
    )
    observed_values = (
        list(is_observed_hour) if is_observed_hour is not None else [True] * row_count
    )

    for index, is_observed in enumerate(observed_values):
        if not is_observed:
            departure_values[index] = 0
            arrival_values[index] = 0

    return pd.DataFrame(
        {
            "station_id": station_id,
            "hour": hours,
            "departures": departure_values,
            "arrivals": arrival_values,
            "net_flow": [
                arrival - departure
                for arrival, departure in zip(
                    arrival_values,
                    departure_values,
                    strict=True,
                )
            ],
            "total_activity": [
                arrival + departure
                for arrival, departure in zip(
                    arrival_values,
                    departure_values,
                    strict=True,
                )
            ],
            "is_observed_hour": observed_values,
        }
    )


def test_builds_exact_lags_rolling_means_and_calendar_features():
    """Calcular rezagos exactos y medias de horas anteriores."""
    hours = pd.date_range("2025-01-01", periods=170, freq="h")
    hourly_data = make_hourly_data(hours)

    features = build_features(hourly_data)
    target = features.loc[features["hour"] == hours[169]].iloc[0]

    assert target["hour_of_day"] == 1
    assert target["day_of_week"] == 2
    assert target["month"] == 1
    assert not target["is_weekend"]
    assert features.loc[features["hour"] == hours[72], "is_weekend"].item()

    assert target["departures_lag_1h"] == 168
    assert target["departures_lag_24h"] == 145
    assert target["departures_lag_168h"] == 1
    assert target["arrivals_lag_1h"] == 346
    assert target["arrivals_lag_24h"] == 300
    assert target["arrivals_lag_168h"] == 12
    assert target["net_flow_lag_1h"] == 178
    assert target["net_flow_lag_24h"] == 155
    assert target["net_flow_lag_168h"] == 11

    assert target["departures_rolling_mean_24h"] == 156.5
    assert target["departures_rolling_mean_168h"] == 84.5


def test_completed_zero_counts_contribute_to_historical_features():
    """Usar ceros completados como cero viajes en lags y medias."""
    hours = pd.date_range("2025-01-01", periods=170, freq="h")
    observed = [True] * len(hours)
    observed[168] = False
    hourly_data = make_hourly_data(
        hours,
        departures=[2] * len(hours),
        arrivals=[3] * len(hours),
        is_observed_hour=observed,
    )

    target = build_features(hourly_data).iloc[169]

    assert target["departures_lag_1h"] == 0
    assert target["arrivals_lag_1h"] == 0
    assert target["net_flow_lag_1h"] == 0
    assert target["departures_rolling_mean_24h"] == pytest.approx(46 / 24)
    assert target["departures_rolling_mean_168h"] == pytest.approx(334 / 168)


def test_future_values_do_not_change_features_at_or_before_their_hour():
    """Mantener invariables las características ante cambios futuros."""
    hours = pd.date_range("2025-01-01", periods=200, freq="h")
    hourly_data = make_hourly_data(hours)
    changed_data = hourly_data.copy(deep=True)
    target_hour = hours[169]
    future_rows = changed_data["hour"] >= target_hour

    changed_data.loc[future_rows, "departures"] += 10_000
    changed_data.loc[future_rows, "arrivals"] += 20_000
    changed_data.loc[future_rows, "net_flow"] += 10_000
    changed_data.loc[future_rows, "total_activity"] += 30_000

    original_features = build_features(hourly_data)
    changed_features = build_features(changed_data)
    original_past = original_features.loc[original_features["hour"] <= target_hour].reset_index(
        drop=True
    )
    changed_past = changed_features.loc[changed_features["hour"] <= target_hour].reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(original_past, changed_past)
    assert "departures" not in original_features.columns
    assert "arrivals" not in original_features.columns
    assert "net_flow" not in original_features.columns
    assert "total_activity" not in original_features.columns


def test_station_history_is_independent():
    """No compartir los rezagos ni las medias entre estaciones."""
    hours = pd.date_range("2025-01-01", periods=170, freq="h")
    station_one = make_hourly_data(hours, station_id=1)
    station_two = make_hourly_data(
        hours,
        station_id=2,
        departures=[value + 10_000 for value in range(len(hours))],
    )

    combined_features = build_features(pd.concat([station_one, station_two], ignore_index=True))
    station_one_features = combined_features.loc[combined_features["station_id"] == 1].reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        station_one_features,
        build_features(station_one),
    )


def test_unsorted_input_returns_chronological_rows_per_station():
    """Ordenar cronológicamente por estación aunque la entrada esté mezclada."""
    hours = pd.date_range("2025-01-01", periods=6, freq="h")
    station_one = make_hourly_data(hours, station_id=1)
    station_two = make_hourly_data(hours, station_id=2)
    hourly_data = pd.concat([station_two, station_one], ignore_index=True)
    shuffled_data = hourly_data.sample(frac=1, random_state=12)

    features = build_features(shuffled_data)
    expected = build_features(hourly_data)

    pd.testing.assert_frame_equal(features, expected)
    assert features[["station_id", "hour"]].equals(
        features.sort_values(["station_id", "hour"])[["station_id", "hour"]].reset_index(drop=True)
    )


def test_missing_hours_remain_unavailable_but_completed_zeros_are_used():
    """Distinguir una hora ausente del cero completado de una hora interna."""
    all_hours = pd.date_range("2025-01-01", periods=6, freq="h")
    included_hours = all_hours.delete(4)
    hourly_data = make_hourly_data(
        included_hours,
        is_observed_hour=[True, True, False, True, True],
    )

    features = build_features(hourly_data)
    target_after_unobserved_hour = features.loc[features["hour"] == all_hours[3]].iloc[0]
    target_after_missing_hour = features.loc[features["hour"] == all_hours[5]].iloc[0]

    assert target_after_unobserved_hour["departures_lag_1h"] == 0
    assert target_after_unobserved_hour["arrivals_lag_1h"] == 0
    assert target_after_unobserved_hour["net_flow_lag_1h"] == 0
    assert pd.isna(target_after_missing_hour["departures_lag_1h"])
    assert pd.isna(target_after_missing_hour["arrivals_lag_1h"])
    assert pd.isna(target_after_missing_hour["net_flow_lag_1h"])
    assert pd.isna(target_after_missing_hour["departures_lag_24h"])
    assert pd.isna(target_after_missing_hour["departures_rolling_mean_24h"])
    assert len(features) == len(hourly_data)
    assert all_hours[4] not in set(features["hour"])


def test_missing_hour_invalidates_full_elapsed_window():
    """No calcular una media 24h como si una hora ausente no existiera."""
    complete_hours = pd.date_range("2025-01-01", periods=26, freq="h")
    hourly_data = make_hourly_data(complete_hours.delete(10))

    target = (
        build_features(hourly_data).loc[lambda frame: frame["hour"] == complete_hours[25]].iloc[0]
    )

    assert target["departures_lag_24h"] == 1
    assert pd.isna(target["departures_rolling_mean_24h"])


def test_rejects_missing_required_columns():
    """Informar qué columna obligatoria falta en la entrada."""
    hours = pd.date_range("2025-01-01", periods=2, freq="h")
    hourly_data = make_hourly_data(hours).drop(columns="arrivals")

    with pytest.raises(ValueError, match="arrivals"):
        build_features(hourly_data)


@pytest.mark.parametrize(
    ("column", "values", "error_type"),
    [
        ("hour", ["2025-01-01 00:00:00"], TypeError),
        ("departures", ["one"], TypeError),
        ("is_observed_hour", [1], TypeError),
    ],
)
def test_rejects_invalid_column_types(
    column: str,
    values: list[object],
    error_type: type[Exception],
):
    """Rechazar tipos incompatibles con el esquema horario."""
    hours = pd.date_range("2025-01-01", periods=1, freq="h")
    hourly_data = make_hourly_data(hours)
    hourly_data[column] = values

    with pytest.raises(error_type):
        build_features(hourly_data)


def test_is_deterministic_and_does_not_modify_input():
    """Conservar la entrada y repetir exactamente el mismo resultado."""
    hours = pd.date_range("2025-01-01", periods=170, freq="h")
    hourly_data = make_hourly_data(hours)
    original = hourly_data.copy(deep=True)

    first_result = build_features(hourly_data)
    second_result = build_features(hourly_data)

    pd.testing.assert_frame_equal(hourly_data, original)
    pd.testing.assert_frame_equal(first_result, second_result)


def test_rejects_duplicate_station_hours():
    """Evitar búsquedas históricas ambiguas por estación y hora."""
    hours = pd.date_range("2025-01-01", periods=2, freq="h")
    hourly_data = make_hourly_data(hours)
    duplicated_data = pd.concat(
        [hourly_data, hourly_data.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicadas"):
        build_features(duplicated_data)


def test_rejects_nonzero_counts_for_unobserved_hours():
    """Rechazar filas que contradicen la semántica del dataset completado."""
    hours = pd.date_range("2025-01-01", periods=1, freq="h")
    hourly_data = make_hourly_data(hours, is_observed_hour=[False])
    hourly_data.loc[0, "departures"] = 1

    with pytest.raises(ValueError, match="completados en cero"):
        build_features(hourly_data)
