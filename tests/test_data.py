
from pathlib import Path

import pandas as pd
import pytest

from mibici.dataset import (
    STATION_COLUMNS,
    TRIP_COLUMNS,
    load_stations,
    load_trips,
    prepare_trips,
    save_interim_trips,
    validate_columns,
    aggregate_station_hourly,
    complete_station_hourly,
)

def test_validate_columns_success():
    """Verificar que se acepten todas las columnas obligatorias."""
    df = pd.DataFrame(columns=sorted(TRIP_COLUMNS))

    validate_columns(df, TRIP_COLUMNS)


def test_validate_columns_missing():
    """Verificar que se detecten columnas faltantes."""
    df = pd.DataFrame({"Viaje_Id": [1]})

    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_columns(df, TRIP_COLUMNS)


def test_load_trips(tmp_path: Path):
    """Verificar la lectura y conversión de fechas de los viajes."""
    file_path = tmp_path / "trips.csv"

    # Crear un archivo temporal con la estructura original de MiBici.
    df = pd.DataFrame(
        {
            "Viaje_Id": [1, 2],
            "Usuario_Id": [100, 101],
            "Genero": ["M", "F"],
            "Año_de_nacimiento": [1990, 1995],
            "Inicio_del_viaje": [
                "2025-01-01 08:00:00",
                "fecha_invalida",
            ],
            "Fin_del_viaje": [
                "2025-01-01 08:15:00",
                "2025-01-01 09:30:00",
            ],
            "Origen_Id": [10, 20],
            "Destino_Id": [20, 30],
        }
    )

    df.to_csv(file_path, index=False, encoding="latin-1")

    result = load_trips(file_path)

    assert len(result) == 2
    assert set(result.columns) == TRIP_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(result["Inicio_del_viaje"])
    assert pd.isna(result.loc[1, "Inicio_del_viaje"])
    assert result.loc[0, "Origen_Id"] == 10


def test_load_stations(tmp_path: Path):
    """Verificar la lectura del catálogo de estaciones."""
    file_path = tmp_path / "stations.csv"

    df = pd.DataFrame(
        {
            "id": [10, 20],
            "name": ["Estación Centro", "Estación Norte"],
            "latitude": [20.67, 20.69],
            "longitude": [-103.35, -103.36],
            "dpcapacity": [15, 20],
        }
    )

    df.to_csv(file_path, index=False, encoding="utf-8")

    result = load_stations(file_path)

    assert len(result) == 2
    assert set(result.columns) == STATION_COLUMNS
    assert result.loc[0, "dpcapacity"] == 15


def test_load_trips_missing_file(tmp_path: Path):
    """Verificar que se informe cuando no existe el archivo."""
    file_path = tmp_path / "nonexistent.csv"

    with pytest.raises(FileNotFoundError):
        load_trips(file_path)


def test_load_stations_missing_columns(tmp_path: Path):
    """Verificar que se rechace un catálogo incompleto."""
    file_path = tmp_path / "incomplete.csv"

    df = pd.DataFrame({"id": [1], "name": ["Estación Centro"]})
    df.to_csv(file_path, index=False, encoding="utf-8")

    with pytest.raises(ValueError, match="Faltan columnas"):
        load_stations(file_path)


def test_prepare_trips():
    """Verificar la transformación y los indicadores de calidad."""
    df = pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 3],
            "Usuario_Id": [100, 101, 102],
            "Genero": ["M", "F", "M"],
            "Año_de_nacimiento": [1990, 1995, 2000],
            "Inicio_del_viaje": [
                "2025-01-01 08:00:00",
                "2025-01-01 09:00:00",
                "2025-01-01 10:00:00",
            ],
            "Fin_del_viaje": [
                "2025-01-01 08:15:00",
                "2025-01-01 09:00:00",
                "2025-01-01 11:30:00",
            ],
            "Origen_Id": [10, 20, 30],
            "Destino_Id": [20, 30, 40],
        }
    )

    result = prepare_trips(df)

    # Verificar que no se eliminen registros.
    assert len(result) == 3

    # Verificar las duraciones calculadas.
    assert result["duration_seconds"].tolist() == [
        900.0,
        0.0,
        5400.0,
    ]

    # Verificar los indicadores de calidad.
    assert result["has_nonpositive_duration"].tolist() == [
        False,
        True,
        False,
    ]
    assert result["is_long_trip"].tolist() == [
        False,
        False,
        True,
    ]

    # Verificar que no se propaguen datos personales innecesarios.
    assert "Usuario_Id" not in result.columns
    assert "Genero" not in result.columns
    assert "Año_de_nacimiento" not in result.columns


def test_prepare_trips_preserves_input():
    """Verificar que la transformación no modifique los datos originales."""
    df = pd.DataFrame(
        {
            "Viaje_Id": [1],
            "Usuario_Id": [100],
            "Genero": ["M"],
            "Año_de_nacimiento": [1990],
            "Inicio_del_viaje": ["2025-01-01 08:00:00"],
            "Fin_del_viaje": ["2025-01-01 08:15:00"],
            "Origen_Id": [10],
            "Destino_Id": [20],
        }
    )

    original = df.copy(deep=True)

    prepare_trips(df)

    pd.testing.assert_frame_equal(df, original)


def test_save_interim_trips(tmp_path: Path):
    """Verificar la persistencia y recuperación del dataset intermedio."""
    output_path = tmp_path / "interim" / "trips.parquet"

    # Crear un conjunto de datos pequeño con el esquema intermedio.
    df = pd.DataFrame(
        {
            "trip_id": [1, 2],
            "started_at": pd.to_datetime(
                ["2025-01-01 08:00:00", "2025-01-01 09:00:00"]
            ),
            "ended_at": pd.to_datetime(
                ["2025-01-01 08:15:00", "2025-01-01 09:00:00"]
            ),
            "origin_station_id": [10, 20],
            "destination_station_id": [20, 30],
            "duration_seconds": [900.0, 0.0],
            "has_invalid_timestamp": [False, False],
            "has_nonpositive_duration": [False, True],
            "is_long_trip": [False, False],
        }
    )

    # Guardar y recuperar los datos desde un directorio temporal.
    save_interim_trips(df, output_path)

    assert output_path.is_file()

    restored = pd.read_parquet(output_path)

    # Verificar que no se pierdan registros, valores ni tipos.
    pd.testing.assert_frame_equal(df, restored)


import pytest


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_load_trips_multiple_encodings(tmp_path, encoding):
    """Verificar la lectura de archivos con distintas codificaciones."""
    file_path = tmp_path / f"trips_{encoding}.csv"

    # Crear un archivo de prueba con el esquema oficial de MiBici.
    df = pd.DataFrame(
        {
            "Viaje_Id": [1],
            "Usuario_Id": [100],
            "Genero": ["M"],
            "Año_de_nacimiento": [1995],
            "Inicio_del_viaje": ["2025-03-01 08:00:00"],
            "Fin_del_viaje": ["2025-03-01 08:15:00"],
            "Origen_Id": [10],
            "Destino_Id": [20],
        }
    )

    df.to_csv(file_path, index=False, encoding=encoding)

    # Comprobar que el lector interpreta correctamente el archivo.
    result = load_trips(file_path)

    assert len(result) == 1
    assert "Año_de_nacimiento" in result.columns
    assert result["Viaje_Id"].iloc[0] == 1
    assert result["Inicio_del_viaje"].iloc[0] == pd.Timestamp(
        "2025-03-01 08:00:00"
    )


def test_load_trips_invalid_schema(tmp_path):
    """Rechazar archivos que no contienen el esquema obligatorio."""
    file_path = tmp_path / "invalid_trips.csv"

    # Crear un archivo sin las columnas requeridas.
    pd.DataFrame(
        {
            "Viaje_Id": [1],
            "Origen_Id": [10],
        }
    ).to_csv(file_path, index=False, encoding="utf-8")

    with pytest.raises(ValueError, match="codificación compatible"):
        load_trips(file_path)


def test_aggregate_station_hourly():
    """Verificar la agregación de viajes por estación y hora."""
    trips = pd.DataFrame(
        {
            "origin_station_id": [2, 2, 3],
            "destination_station_id": [3, 3, 2],
            "started_at": pd.to_datetime([
                "2025-01-01 08:10:00",
                "2025-01-01 08:30:00",
                "2025-01-01 09:05:00",
            ]),
            "ended_at": pd.to_datetime([
                "2025-01-01 08:25:00",
                "2025-01-01 09:10:00",
                "2025-01-01 09:20:00",
            ]),
        }
    )

    result = aggregate_station_hourly(trips)

    # Verificar que cada viaje produzca una salida y una llegada.
    assert result["departures"].sum() == 3
    assert result["arrivals"].sum() == 3

    # Comprobar las salidas de la estación 2 a las 08:00.
    station_2_hour_8 = result[
        (result["station_id"] == 2)
        & (result["hour"] == pd.Timestamp("2025-01-01 08:00:00"))
    ].iloc[0]

    assert station_2_hour_8["departures"] == 2
    assert station_2_hour_8["arrivals"] == 0
    assert station_2_hour_8["net_flow"] == -2
    assert station_2_hour_8["total_activity"] == 2

    # La segunda llegada ocurre a las 09:00, no a las 08:00.
    station_3_hour_9 = result[
        (result["station_id"] == 3)
        & (result["hour"] == pd.Timestamp("2025-01-01 09:00:00"))
    ].iloc[0]

    assert station_3_hour_9["arrivals"] == 1
    assert station_3_hour_9["departures"] == 1


def test_aggregate_station_hourly_preserves_input():
    """Comprobar que la agregación no modifique los viajes originales."""
    trips = pd.DataFrame(
        {
            "origin_station_id": [2],
            "destination_station_id": [3],
            "started_at": pd.to_datetime(["2025-01-01 08:00:00"]),
            "ended_at": pd.to_datetime(["2025-01-01 08:15:00"]),
        }
    )

    original = trips.copy(deep=True)

    aggregate_station_hourly(trips)

    pd.testing.assert_frame_equal(trips, original)


def test_aggregate_station_hourly_missing_columns():
    """Rechazar viajes sin las columnas necesarias para agregar."""
    trips = pd.DataFrame(
        {
            "origin_station_id": [2],
            "started_at": pd.to_datetime(["2025-01-01 08:00:00"]),
        }
    )

    with pytest.raises(ValueError, match="Faltan columnas"):
        aggregate_station_hourly(trips)


def test_complete_station_hourly():
    """Completar horas faltantes sin modificar los conteos originales."""
    hourly = pd.DataFrame(
        {
            "station_id": [2, 2],
            "hour": pd.to_datetime([
                "2025-01-01 08:00:00",
                "2025-01-01 10:00:00",
            ]),
            "departures": [3, 2],
            "arrivals": [1, 4],
            "net_flow": [-2, 2],
            "total_activity": [4, 6],
        }
    )

    result = complete_station_hourly(hourly)

    # Deben existir tres horas consecutivas.
    assert len(result) == 3

    # La hora intermedia no tenía actividad registrada.
    missing_hour = result[
        result["hour"] == pd.Timestamp("2025-01-01 09:00:00")
    ].iloc[0]

    assert missing_hour["departures"] == 0
    assert missing_hour["arrivals"] == 0
    assert not missing_hour["is_observed_hour"]

    # Los conteos originales deben conservarse.
    assert result["departures"].sum() == 5
    assert result["arrivals"].sum() == 5
    assert result["is_observed_hour"].sum() == 2


def test_complete_station_hourly_respects_boundaries():
    """Excluir horas fuera del calendario sin inventar actividad."""
    hourly = pd.DataFrame(
        {
            "station_id": [2, 2, 2],
            "hour": pd.to_datetime([
                "2025-12-31 22:00:00",
                "2025-12-31 23:00:00",
                "2026-01-01 00:00:00",
            ]),
            "departures": [2, 1, 0],
            "arrivals": [1, 2, 1],
            "net_flow": [-1, 1, 1],
            "total_activity": [3, 3, 1],
        }
    )

    result = complete_station_hourly(hourly)

    assert len(result) == 2
    assert result["hour"].max() == pd.Timestamp(
        "2025-12-31 23:00:00"
    )
    assert result["departures"].sum() == 3
    assert result["arrivals"].sum() == 3


def test_complete_station_hourly_rejects_duplicates():
    """Rechazar combinaciones duplicadas de estación y hora."""
    hourly = pd.DataFrame(
        {
            "station_id": [2, 2],
            "hour": pd.to_datetime([
                "2025-01-01 08:00:00",
                "2025-01-01 08:00:00",
            ]),
            "departures": [1, 1],
            "arrivals": [1, 1],
            "net_flow": [0, 0],
            "total_activity": [2, 2],
        }
    )

    with pytest.raises(ValueError, match="duplicadas"):
        complete_station_hourly(hourly)


def test_build_annual_hourly_dataset(tmp_path, monkeypatch):
    """Validar el pipeline anual con archivos temporales."""
    import mibici.dataset as dataset_module

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"

    interim_dir.mkdir()
    processed_dir.mkdir()

    # Redirigir las rutas para no modificar los datos reales.
    monkeypatch.setattr(
        dataset_module, "INTERIM_DATA_DIR", interim_dir
    )
    monkeypatch.setattr(
        dataset_module, "PROCESSED_DATA_DIR", processed_dir
    )

    # Crear un viaje por mes para verificar la integración anual.
    for month in range(1, 13):
        trips = pd.DataFrame(
            {
                "trip_id": [month],
                "origin_station_id": [2],
                "destination_station_id": [3],
                "started_at": [
                    pd.Timestamp(2025, month, 15, 8)
                ],
                "ended_at": [
                    pd.Timestamp(2025, month, 15, 9)
                ],
            }
        )

        trips.to_parquet(
            interim_dir / f"trips_2025_{month:02d}.parquet",
            index=False,
        )

    # Ejecutar el pipeline completo.
    observed_path, completed_path = (
        dataset_module.build_annual_hourly_dataset(2025)
    )

    assert observed_path.is_file()
    assert completed_path.is_file()

    observed = pd.read_parquet(observed_path)
    completed = pd.read_parquet(completed_path)

    # Los 12 viajes deben conservar sus salidas y llegadas.
    assert observed["departures"].sum() == 12
    assert observed["arrivals"].sum() == 12

    assert completed["departures"].sum() == 12
    assert completed["arrivals"].sum() == 12

    # Ambas estaciones deben aparecer en el resultado.
    assert set(completed["station_id"]) == {2, 3}

    # El calendario completado debe contener horas sin movimientos.
    assert (~completed["is_observed_hour"]).any()

    # No deben existir combinaciones duplicadas.
    assert not completed.duplicated(
        ["station_id", "hour"]
    ).any()


def test_build_annual_hourly_dataset_missing_month(
    tmp_path, monkeypatch
):
    """Rechazar la construcción anual si falta un archivo mensual."""
    import mibici.dataset as dataset_module

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"

    interim_dir.mkdir()
    processed_dir.mkdir()

    # Utilizar directorios temporales para proteger los datos reales.
    monkeypatch.setattr(
        dataset_module, "INTERIM_DATA_DIR", interim_dir
    )
    monkeypatch.setattr(
        dataset_module, "PROCESSED_DATA_DIR", processed_dir
    )

    # Crear únicamente once meses; diciembre estará ausente.
    for month in range(1, 12):
        trips = pd.DataFrame(
            {
                "trip_id": [month],
                "origin_station_id": [2],
                "destination_station_id": [3],
                "started_at": [
                    pd.Timestamp(2025, month, 15, 8)
                ],
                "ended_at": [
                    pd.Timestamp(2025, month, 15, 9)
                ],
            }
        )

        trips.to_parquet(
            interim_dir / f"trips_2025_{month:02d}.parquet",
            index=False,
        )

    # El pipeline debe identificar el archivo faltante.
    with pytest.raises(
        FileNotFoundError,
        match="trips_2025_12.parquet",
    ):
        dataset_module.build_annual_hourly_dataset(2025)

    # No deben generarse archivos procesados incompletos.
    assert not list(processed_dir.glob("*.parquet"))


def test_build_annual_hourly_dataset_invalid_month(
    tmp_path, monkeypatch
):
    """Rechazar viajes iniciados fuera del mes del archivo."""
    import mibici.dataset as dataset_module

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"

    interim_dir.mkdir()
    processed_dir.mkdir()

    # Redirigir el pipeline a directorios temporales.
    monkeypatch.setattr(
        dataset_module, "INTERIM_DATA_DIR", interim_dir
    )
    monkeypatch.setattr(
        dataset_module, "PROCESSED_DATA_DIR", processed_dir
    )

    for month in range(1, 13):
        # Introducir deliberadamente un viaje de enero en febrero.
        trip_month = 1 if month == 2 else month

        trips = pd.DataFrame(
            {
                "trip_id": [month],
                "origin_station_id": [2],
                "destination_station_id": [3],
                "started_at": [
                    pd.Timestamp(2025, trip_month, 15, 8)
                ],
                "ended_at": [
                    pd.Timestamp(2025, trip_month, 15, 9)
                ],
            }
        )

        trips.to_parquet(
            interim_dir / f"trips_2025_{month:02d}.parquet",
            index=False,
        )

    # Detectar el viaje de enero dentro del archivo de febrero.
    with pytest.raises(
        ValueError,
        match="fuera del mes correspondiente",
    ):
        dataset_module.build_annual_hourly_dataset(2025)

    # No generar resultados si la validación falla.
    assert not list(processed_dir.glob("*.parquet"))


def test_build_annual_hourly_dataset_null_stations(
    tmp_path, monkeypatch
):
    """Rechazar viajes con estaciones de origen o destino nulas."""
    import mibici.dataset as dataset_module

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"

    interim_dir.mkdir()
    processed_dir.mkdir()

    # Aislar la prueba de los datos reales.
    monkeypatch.setattr(
        dataset_module, "INTERIM_DATA_DIR", interim_dir
    )
    monkeypatch.setattr(
        dataset_module, "PROCESSED_DATA_DIR", processed_dir
    )

    for month in range(1, 13):
        trips = pd.DataFrame(
            {
                "trip_id": [month],
                "origin_station_id": [
                    None if month == 3 else 2
                ],
                "destination_station_id": [3],
                "started_at": [
                    pd.Timestamp(2025, month, 15, 8)
                ],
                "ended_at": [
                    pd.Timestamp(2025, month, 15, 9)
                ],
            }
        )

        trips.to_parquet(
            interim_dir / f"trips_2025_{month:02d}.parquet",
            index=False,
        )

    # Detectar el identificador nulo en marzo.
    with pytest.raises(
        ValueError,
        match="identificadores de estación nulos",
    ):
        dataset_module.build_annual_hourly_dataset(2025)

    # No producir datasets anuales incompletos.
    assert not list(processed_dir.glob("*.parquet"))


def test_build_annual_hourly_dataset_duplicate_trips(
    tmp_path, monkeypatch
):
    """Detectar identificadores repetidos entre archivos mensuales."""
    import mibici.dataset as dataset_module

    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"

    interim_dir.mkdir()
    processed_dir.mkdir()

    # Utilizar directorios temporales.
    monkeypatch.setattr(
        dataset_module, "INTERIM_DATA_DIR", interim_dir
    )
    monkeypatch.setattr(
        dataset_module, "PROCESSED_DATA_DIR", processed_dir
    )

    for month in range(1, 13):
        trips = pd.DataFrame(
            {
                # Enero y febrero comparten deliberadamente el mismo ID.
                "trip_id": [1 if month == 2 else month],
                "origin_station_id": [2],
                "destination_station_id": [3],
                "started_at": [
                    pd.Timestamp(2025, month, 15, 8)
                ],
                "ended_at": [
                    pd.Timestamp(2025, month, 15, 9)
                ],
            }
        )

        trips.to_parquet(
            interim_dir / f"trips_2025_{month:02d}.parquet",
            index=False,
        )

    # El pipeline debe rechazar los identificadores duplicados.
    with pytest.raises(
        ValueError,
        match="identificadores de viaje duplicados",
    ):
        dataset_module.build_annual_hourly_dataset(2025)

    # No deben producirse datasets procesados.
    assert not list(processed_dir.glob("*.parquet"))


def test_run_pipeline_executes_all_stages_in_order(
    monkeypatch, tmp_path
):
    """Verificar el orden del pipeline sin procesar datos reales."""
    from mibici import dataset

    executed_stages = []
    manifest_path = tmp_path / "manifest.csv"
    observed_path = tmp_path / "observed.parquet"
    completed_path = tmp_path / "completed.parquet"

    # Simular la descarga anual.
    def fake_download_year(path):
        assert path == manifest_path
        executed_stages.append("download")
        return pd.DataFrame({"month": range(1, 13)})

    # Simular la preparación de los doce meses.
    def fake_prepare_year(year):
        assert year == 2025
        executed_stages.append("prepare")
        return [
            tmp_path / f"trips_2025_{month:02d}.parquet"
            for month in range(1, 13)
        ]

    # Simular la generación de los datasets horarios.
    def fake_build_hourly(year):
        assert year == 2025
        executed_stages.append("build")
        return observed_path, completed_path

    monkeypatch.setattr(
        dataset, "download_year", fake_download_year
    )
    monkeypatch.setattr(
        dataset, "prepare_year", fake_prepare_year
    )
    monkeypatch.setattr(
        dataset,
        "build_annual_hourly_dataset",
        fake_build_hourly,
    )

    result = dataset.run_pipeline(
        year=2025,
        manifest_path=manifest_path,
    )

    assert executed_stages == ["download", "prepare", "build"]
    assert result == (observed_path, completed_path)
 

def test_prepare_trips_exactly_one_hour_is_not_long():
    """Un viaje de exactamente una hora no debe clasificarse como largo."""
    df = pd.DataFrame(
        {
            "Viaje_Id": [1],
            "Usuario_Id": [100],
            "Genero": ["M"],
            "Año_de_nacimiento": [1990],
            "Inicio_del_viaje": ["2025-01-01 08:00:00"],
            "Fin_del_viaje": ["2025-01-01 09:00:00"],
            "Origen_Id": [10],
            "Destino_Id": [20],
        }
    )

    result = prepare_trips(df)

    assert result.loc[0, "duration_seconds"] == 3600
    assert not result.loc[0, "is_long_trip"]


def test_prepare_trips_invalid_timestamp():
    """Identificar fechas inválidas y duraciones que no pueden calcularse."""
    df = pd.DataFrame(
        {
            "Viaje_Id": [1],
            "Usuario_Id": [100],
            "Genero": ["M"],
            "Año_de_nacimiento": [1990],
            "Inicio_del_viaje": ["fecha_invalida"],
            "Fin_del_viaje": ["2025-01-01 09:00:00"],
            "Origen_Id": [10],
            "Destino_Id": [20],
        }
    )

    result = prepare_trips(df)

    assert result.loc[0, "has_invalid_timestamp"]
    assert pd.isna(result.loc[0, "duration_seconds"])
    assert not result.loc[0, "has_nonpositive_duration"]
    assert not result.loc[0, "is_long_trip"]


def test_prepare_trips_missing_required_column():
    """Rechazar datos que no contienen todas las columnas obligatorias."""
    df = pd.DataFrame(
        {
            "Viaje_Id": [1],
            "Inicio_del_viaje": ["2025-01-01 08:00:00"],
        }
    )

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        prepare_trips(df)


def test_aggregate_station_hourly_ignores_invalid_timestamps():
    """Ignorar fechas inválidas sin perder los movimientos válidos."""
    trips = pd.DataFrame(
        {
            "origin_station_id": [2, 3],
            "destination_station_id": [3, 2],
            "started_at": pd.to_datetime(
                ["2025-01-01 08:10:00", None]
            ),
            "ended_at": pd.to_datetime(
                [None, "2025-01-01 09:20:00"]
            ),
        }
    )

    result = aggregate_station_hourly(trips)

    assert result["departures"].sum() == 1
    assert result["arrivals"].sum() == 1
    assert result.loc[result["station_id"] == 2, "departures"].sum() == 1
    assert result.loc[result["station_id"] == 2, "arrivals"].sum() == 1