import pandas as pd
import pytest

from mibici.clustering import build_station_features


def make_hourly_data():
    """Crear datos sintéticos de tres estaciones para probar P3."""
    hours = pd.date_range("2025-01-01", periods=48, freq="h")
    stations = []

    for station_id, departures, arrivals in [
        (1, 2, 3),
        (2, 5, 4),
        (3, 8, 6),
    ]:
        stations.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "hour": hours,
                    "departures": departures,
                    "arrivals": arrivals,
                    "net_flow": arrivals - departures,
                    "total_activity": departures + arrivals,
                }
            )
        )

    return pd.concat(stations, ignore_index=True)


def test_build_station_features_creates_one_row_per_station():
    """Cada estación debe tener una fila de indicadores."""
    features = build_station_features(make_hourly_data())

    assert len(features) == 3
    assert set(features.index) == {1, 2, 3}
    assert features.index.name == "station_id"


def test_build_station_features_calculates_activity_totals():
    """Los totales deben corresponder a la suma de los registros horarios."""
    features = build_station_features(make_hourly_data())

    assert features.loc[1, "total_departures"] == 96
    assert features.loc[2, "total_departures"] == 240
    assert features.loc[3, "total_departures"] == 384

    assert features.loc[1, "total_activity"] == 240
    assert features.loc[2, "total_activity"] == 432
    assert features.loc[3, "total_activity"] == 672


def test_build_station_features_rejects_missing_columns():
    """Debe informar si faltan columnas requeridas."""
    data = make_hourly_data().drop(columns=["departures"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        build_station_features(data)
