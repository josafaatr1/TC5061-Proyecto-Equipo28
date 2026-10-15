
from pathlib import Path

import pandas as pd

import typer

import hashlib

import time

import requests

from loguru import logger

from mibici.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    PROJ_ROOT,
)

app = typer.Typer()

# Columnas obligatorias de los archivos originales.
TRIP_COLUMNS = {
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
}

STATION_COLUMNS = {
    "id",
    "name",
    "latitude",
    "longitude",
    "dpcapacity",
}


def validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Verificar que el archivo contenga las columnas obligatorias."""
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing_columns)}"
        )

def load_trips(file_path: Path) -> pd.DataFrame:
    """Leer viajes de MiBici con validación de esquema y codificación."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de viajes: {file_path}"
        )

    # Probar primero UTF-8 y después CP1252.
    for encoding in ["utf-8", "cp1252"]:
        try:
            # Comprobar el esquema antes de cargar el archivo completo.
            header = pd.read_csv(
                file_path,
                encoding=encoding,
                nrows=0,
            )

            if not TRIP_COLUMNS.issubset(header.columns):
                continue

            # Leer el archivo completo con la codificación seleccionada.
            df = pd.read_csv(
                file_path,
                encoding=encoding,
            )

            validate_columns(df, TRIP_COLUMNS)

            # Convertir las fechas al tipo datetime.
            for column in ["Inicio_del_viaje", "Fin_del_viaje"]:
                df[column] = pd.to_datetime(
                    df[column],
                    format="%Y-%m-%d %H:%M:%S",
                    errors="coerce",
                )

            logger.info(
                f"Archivo leído con codificación {encoding}: "
                f"{file_path.name}"
            )

            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"No se encontró una codificación compatible "
        f"con el esquema esperado: {file_path}"
    )

def load_stations(file_path: Path) -> pd.DataFrame:
    """Leer el catálogo de estaciones de MiBici."""
    if not file_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    # El catálogo inspeccionado utiliza codificación UTF-8.
    df = pd.read_csv(file_path, encoding="utf-8")

    validate_columns(df, STATION_COLUMNS)

    return df


def prepare_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Estandarizar viajes y agregar indicadores de calidad."""
    validate_columns(df, TRIP_COLUMNS)

    # Crear una copia para preservar los datos de entrada.
    result = df.copy()

    # Conservar únicamente las columnas necesarias para el análisis.
    result = result[
        [
            "Viaje_Id",
            "Inicio_del_viaje",
            "Fin_del_viaje",
            "Origen_Id",
            "Destino_Id",
        ]
    ]

    # Estandarizar los nombres de las columnas.
    result = result.rename(
        columns={
            "Viaje_Id": "trip_id",
            "Inicio_del_viaje": "started_at",
            "Fin_del_viaje": "ended_at",
            "Origen_Id": "origin_station_id",
            "Destino_Id": "destination_station_id",
        }
    )

    # Convertir fechas cuando la función recibe texto.
    for column in ["started_at", "ended_at"]:
        result[column] = pd.to_datetime(
            result[column],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )

    # Calcular la duración en segundos.
    result["duration_seconds"] = (
        result["ended_at"] - result["started_at"]
    ).dt.total_seconds()

    # Identificar situaciones que requieren revisión.
    result["has_invalid_timestamp"] = (
        result["started_at"].isna() | result["ended_at"].isna()
    )
    result["has_nonpositive_duration"] = (
        result["duration_seconds"] <= 0
    )
    result["is_long_trip"] = result["duration_seconds"] > 3600

    return result


def save_interim_trips(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guardar los viajes estandarizados en formato Parquet."""
    # Crear el directorio de destino cuando sea necesario.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar los datos con compresión sin modificar el DataFrame.
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    logger.success(f"Dataset intermedio guardado en: {output_path}")

@app.command()


def download_file(
    url: str,
    output_path: Path,
    max_retries: int = 3,
) -> str:
    """Descargar un archivo sin sobrescribir datos existentes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reutilizar archivos existentes y calcular su huella digital.
    if output_path.exists():
        logger.info(f"Archivo existente, se omite descarga: {output_path}")
        return calculate_sha256(output_path)

    temporary_path = output_path.with_name(output_path.name + ".part")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    for attempt in range(1, max_retries + 1):
        try:
            # Descargar por bloques para limitar el uso de memoria.
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()

                with temporary_path.open("wb") as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            file.write(chunk)

            # Comprobar que la descarga no esté vacía.
            if temporary_path.stat().st_size == 0:
                raise ValueError("El archivo descargado está vacío.")

            # Mover el archivo completo a su ubicación definitiva.
            temporary_path.replace(output_path)

            file_hash = calculate_sha256(output_path)
            logger.success(f"Descarga completada: {output_path.name}")

            return file_hash

        except (requests.RequestException, OSError, ValueError) as error:
            temporary_path.unlink(missing_ok=True)

            if attempt == max_retries:
                raise RuntimeError(
                    f"No fue posible descargar {url}"
                ) from error

            logger.warning(
                f"Intento {attempt}/{max_retries} fallido: {error}"
            )
            time.sleep(attempt * 2)

    raise RuntimeError("No se completó la descarga.")


def calculate_sha256(file_path: Path) -> str:
    """Calcular la huella SHA-256 de un archivo."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def aggregate_station_hourly(trips: pd.DataFrame) -> pd.DataFrame:
    """Agregar salidas y llegadas por estación y hora."""
    required_columns = {
        "origin_station_id",
        "destination_station_id",
        "started_at",
        "ended_at",
    }
    validate_columns(trips, required_columns)

    # Contabilizar las salidas según la hora de inicio.
    departures = (
        trips.dropna(subset=["origin_station_id", "started_at"])
        .assign(hour=lambda df: df["started_at"].dt.floor("h"))
        .groupby(["origin_station_id", "hour"])
        .size()
        .rename("departures")
        .rename_axis(index={"origin_station_id": "station_id"})
    )

    # Contabilizar las llegadas según la hora de finalización.
    arrivals = (
        trips.dropna(subset=["destination_station_id", "ended_at"])
        .assign(hour=lambda df: df["ended_at"].dt.floor("h"))
        .groupby(["destination_station_id", "hour"])
        .size()
        .rename("arrivals")
        .rename_axis(index={"destination_station_id": "station_id"})
    )

    # Combinar los movimientos sin perder horas con un solo tipo de actividad.
    hourly = pd.concat([departures, arrivals], axis=1).fillna(0)

    hourly[["departures", "arrivals"]] = hourly[
        ["departures", "arrivals"]
    ].astype("int64")

    # Calcular el balance neto y la actividad total.
    hourly["net_flow"] = hourly["arrivals"] - hourly["departures"]
    hourly["total_activity"] = (
        hourly["arrivals"] + hourly["departures"]
    )

    return hourly.reset_index().sort_values(
        ["station_id", "hour"]
    ).reset_index(drop=True)


def complete_station_hourly(
    hourly: pd.DataFrame,
    start_date: str = "2025-01-01",
    end_date: str = "2026-01-01",
) -> pd.DataFrame:
    """Completar horas dentro del periodo observado de cada estación."""
    required_columns = {
        "station_id",
        "hour",
        "departures",
        "arrivals",
        "net_flow",
        "total_activity",
    }
    validate_columns(hourly, required_columns)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start >= end:
        raise ValueError("El inicio debe ser anterior al final.")

    # Conservar exclusivamente las horas del calendario solicitado.
    observed = hourly.loc[
        (hourly["hour"] >= start) & (hourly["hour"] < end)
    ].copy()

    if observed.empty:
        return observed.assign(is_observed_hour=pd.Series(dtype="bool"))

    if observed.duplicated(["station_id", "hour"]).any():
        raise ValueError("Existen combinaciones estación-hora duplicadas.")

    completed = []

    for station_id, station_data in observed.groupby("station_id"):
        station_data = station_data.sort_values("hour")

        # Delimitar la ventana con la primera y última actividad observada.
        first_hour = station_data["hour"].min()
        last_hour = station_data["hour"].max()

        hourly_index = pd.date_range(
            start=first_hour,
            end=last_hour,
            freq="h",
        )

        station_result = (
            station_data.set_index("hour")
            .reindex(hourly_index)
            .rename_axis("hour")
        )

        station_result["station_id"] = station_id

        # Distinguir horas con actividad de horas completadas.
        station_result["is_observed_hour"] = (
            station_result["total_activity"].notna()
        )

        # Completar conteos ausentes sin inferir inventarios.
        count_columns = [
            "departures",
            "arrivals",
            "net_flow",
            "total_activity",
        ]

        station_result[count_columns] = (
            station_result[count_columns]
            .fillna(0)
            .astype("int64")
        )

        completed.append(station_result.reset_index())

    result = pd.concat(completed, ignore_index=True)

    return result[
        [
            "station_id",
            "hour",
            "departures",
            "arrivals",
            "net_flow",
            "total_activity",
            "is_observed_hour",
        ]
    ].sort_values(
        ["station_id", "hour"]
    ).reset_index(drop=True)


def build_annual_hourly_dataset(
    year: int = 2025,
) -> tuple[Path, Path]:

    """Construir datasets horarios observados y completados de un año."""
    monthly_frames = []

    for month in range(1, 13):
        file_path = (
            INTERIM_DATA_DIR / f"trips_{year}_{month:02d}.parquet"
        )

        if not file_path.is_file():
            raise FileNotFoundError(
                f"No existe el archivo mensual: {file_path}"
            )

        # Cargar únicamente los campos necesarios para la agregación.
        monthly_trips = pd.read_parquet(
            file_path,
            columns=[
                "trip_id",
                "origin_station_id",
                "destination_station_id",
                "started_at",
                "ended_at",
            ],
        )


        # Verificar que los viajes comiencen en el mes correspondiente.
        expected_period = pd.Period(
            year=year,
            month=month,
            freq="M",
        )

        invalid_dates = (
            monthly_trips["started_at"].isna()
            | (
                monthly_trips["started_at"].dt.to_period("M")
                != expected_period
            )
        )

        if invalid_dates.any():
            raise ValueError(
                f"El archivo {file_path.name} contiene "
                f"{invalid_dates.sum()} viajes con fechas "
                "de inicio fuera del mes correspondiente."
            )


        # Rechazar viajes sin estaciones de origen o destino.
        station_columns = [
            "origin_station_id",
            "destination_station_id",
        ]

        invalid_stations = (
            monthly_trips[station_columns]
            .isna()
            .any(axis=1)
        )

        if invalid_stations.any():
            raise ValueError(
                f"El archivo {file_path.name} contiene "
                f"{invalid_stations.sum()} viajes "
                "con identificadores de estación nulos."
            )

        monthly_frames.append(monthly_trips)

    # Unificar los viajes antes de agregar para respetar cruces de mes.
    annual_trips = pd.concat(monthly_frames, ignore_index=True)

    # Detectar identificadores de viaje nulos o duplicados.
    if annual_trips["trip_id"].isna().any():
        raise ValueError(
            "El dataset anual contiene identificadores de viaje nulos."
        )

    duplicated_trips = annual_trips["trip_id"].duplicated()

    if duplicated_trips.any():
        raise ValueError(
            f"El dataset anual contiene "
            f"{duplicated_trips.sum()} identificadores "
            "de viaje duplicados."
        )

    del monthly_frames

    # Agregar los movimientos por estación y hora.
    observed = aggregate_station_hourly(annual_trips)

    # Verificar que no se pierdan salidas ni llegadas.
    if (
        observed["departures"].sum() != len(annual_trips)
        or observed["arrivals"].sum() != len(annual_trips)
    ):
        raise ValueError("La agregación no conserva los viajes.")

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    # Completar únicamente las ventanas observadas dentro del año.
    completed = complete_station_hourly(
        observed,
        start_date=start_date,
        end_date=end_date,
    )

    observed_in_year = observed.loc[
        (observed["hour"] >= pd.Timestamp(start_date))
        & (observed["hour"] < pd.Timestamp(end_date))
    ]

    # Comprobar que el completado conserve los movimientos del año.
    for column in ("departures", "arrivals"):
        if (
            completed[column].sum()
            != observed_in_year[column].sum()
        ):
            raise ValueError(
                f"El completado modificó los conteos de {column}."
            )

    # Guardar los dos niveles de datos procesados.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    observed_path = (
        PROCESSED_DATA_DIR / f"station_hourly_{year}.parquet"
    )
    completed_path = (
        PROCESSED_DATA_DIR
        / f"station_hourly_{year}_complete.parquet"
    )

    observed.to_parquet(
        observed_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    completed.to_parquet(
        completed_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    return observed_path, completed_path

def main(
    trips_path: Path = RAW_DATA_DIR / "datos_abiertos_2025_01.csv",
    stations_path: Path = RAW_DATA_DIR / "nomenclatura_2026_09.csv",
    output_path: Path = INTERIM_DATA_DIR / "trips_2025_01.parquet",
):
    """Ejecutar la ingesta y preparación inicial de datos de MiBici."""
    logger.info("Iniciando pipeline de datos de MiBici.")

    # Leer y validar los archivos originales.
    trips = load_trips(trips_path)
    stations = load_stations(stations_path)

    logger.info(f"Viajes originales: {len(trips):,}")
    logger.info(f"Estaciones del catálogo: {len(stations):,}")

    # Estandarizar los viajes y calcular indicadores de calidad.
    prepared = prepare_trips(trips)

    logger.info(
        f"Fechas inválidas: "
        f"{prepared['has_invalid_timestamp'].sum():,}"
    )
    logger.info(
        f"Duraciones no positivas: "
        f"{prepared['has_nonpositive_duration'].sum():,}"
    )
    logger.info(
        f"Viajes largos: "
        f"{prepared['is_long_trip'].sum():,}"
    )

    # Guardar el dataset intermedio sin modificar los CSV originales.
    save_interim_trips(prepared, output_path)

    logger.success("Pipeline de datos completado correctamente.")


def download_year(
    manifest_path: Path,
    output_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Descargar los archivos mensuales definidos en el manifiesto."""
    manifest = pd.read_csv(
        manifest_path,
        dtype={"year": str, "month": str, "url": str},
    )

    # Validar que el manifiesto contenga los doce meses esperados.
    expected_months = {f"{month:02d}" for month in range(1, 13)}

    if (
        len(manifest) != 12
        or set(manifest["month"]) != expected_months
        or manifest["url"].isna().any()
        or not manifest["url"].is_unique
        or not manifest["year"].eq("2025").all()
    ):
        raise ValueError("El manifiesto de 2025 es inválido.")

    results = []

    for row in manifest.sort_values("month").itertuples(index=False):
        # Construir el nombre local a partir del año y mes.
        filename = f"datos_abiertos_{row.year}_{row.month}.csv"
        output_path = output_dir / filename

        # Verificar que la URL corresponda al archivo esperado.
        if (
            not row.url.startswith(
                "https://www.mibici.net/site/assets/files/"
            )
            or not row.url.endswith(f"/{filename}")
        ):
            raise ValueError(f"URL inesperada para {filename}")

        logger.info(f"Procesando descarga: {filename}")

        # La función reutiliza los archivos que ya existen.
        file_hash = download_file(row.url, output_path)

        results.append(
            {
                "year": row.year,
                "month": row.month,
                "filename": filename,
                "source_url": row.url,
                "size_bytes": output_path.stat().st_size,
                "sha256": file_hash,
            }
        )

    return pd.DataFrame(results)

if __name__ == "__main__":
    app()
