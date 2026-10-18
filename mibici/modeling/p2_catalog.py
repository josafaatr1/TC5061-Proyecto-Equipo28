"""Descarga, validación y auditoría del catálogo de estaciones para P2."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
import numpy as np
import pandas as pd
import requests
import typer

from mibici.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, START_DATE

app = typer.Typer()

OFFICIAL_SOURCE_PAGE = "https://www.mibici.net/es/datos-abiertos/"
OFFICIAL_CATALOG_URL = "https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
CATALOG_FILENAME = "nomenclatura_2026_09.csv"
PROVENANCE_FILENAME = "nomenclatura_2026_09.provenance.json"
AUDIT_FILENAME = "p2_capacity_audit_2025_train_validation.json"
REQUIRED_COLUMNS = ("id", "name", "latitude", "longitude", "dpcapacity")


def _external_directory(output_dir: Path | None) -> Path:
    """Validar y crear el directorio externo permitido."""
    external_root = EXTERNAL_DATA_DIR.resolve()
    directory = (output_dir or external_root).resolve()
    if not directory.is_relative_to(external_root):
        raise ValueError(f"El catálogo solo puede guardarse dentro de {external_root}.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sha256(content: bytes) -> str:
    """Calcular SHA-256 del contenido original descargado."""
    return hashlib.sha256(content).hexdigest()


def download_catalog(
    output_dir: Path | None = None,
    url: str = OFFICIAL_CATALOG_URL,
) -> tuple[Path, str]:
    """Descargar el CSV oficial y guardar su huella y procedencia.

    Args:
        output_dir: Directorio destino dentro de ``data/external``.
        url: URL oficial del archivo de nomenclatura.

    Returns:
        Ruta del CSV descargado y su hash SHA-256.

    Raises:
        ValueError: Si la URL no usa HTTPS del dominio oficial.
        RuntimeError: Si falla la descarga o no se recibe contenido.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "www.mibici.net":
        raise ValueError("La descarga debe provenir de una URL HTTPS oficial de www.mibici.net.")

    directory = _external_directory(output_dir)
    catalog_path = directory / CATALOG_FILENAME
    temporary_path = catalog_path.with_suffix(".csv.part")

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "*/*",
            },
            timeout=(15, 120),
        )
        response.raise_for_status()
        content = response.content
        if not content:
            raise RuntimeError("La respuesta oficial del catálogo está vacía.")
        temporary_path.write_bytes(content)
        temporary_path.replace(catalog_path)
    except requests.RequestException as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"No fue posible descargar el catálogo oficial: {url}") from error
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"No fue posible guardar el catálogo en {directory}.") from error

    digest = _sha256(content)
    provenance = {
        "catalog_filename": CATALOG_FILENAME,
        "catalog_version": "2026-09",
        "source_page": OFFICIAL_SOURCE_PAGE,
        "source_url": url,
        "sha256": digest,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
    }
    provenance_path = directory / PROVENANCE_FILENAME
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return catalog_path, digest


def normalize_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Normalizar identificadores y capacidades sin ocultar valores inválidos."""
    missing_columns = set(REQUIRED_COLUMNS) - set(catalog.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing_columns)}.")

    normalized = catalog.loc[:, REQUIRED_COLUMNS].copy()
    numeric_ids = pd.to_numeric(normalized["id"], errors="coerce")
    valid_ids = (
        numeric_ids.notna()
        & np.isfinite(numeric_ids.to_numpy(dtype=np.float64, na_value=np.nan))
        & numeric_ids.mod(1).eq(0)
    )
    normalized["id"] = numeric_ids.where(valid_ids).astype("Int64")
    normalized["name"] = normalized["name"].astype("string").str.strip()
    normalized["latitude"] = pd.to_numeric(normalized["latitude"], errors="coerce")
    normalized["longitude"] = pd.to_numeric(normalized["longitude"], errors="coerce")
    normalized["dpcapacity"] = pd.to_numeric(
        normalized["dpcapacity"],
        errors="coerce",
    ).astype("float64")

    return normalized


def load_catalog(catalog_path: Path, strict: bool = True) -> pd.DataFrame:
    """Leer y normalizar un catálogo CSV, validándolo opcionalmente.

    Con ``strict=False`` se preservan filas inválidas para que una auditoría
    pueda reportarlas sin descartarlas.
    """
    if not catalog_path.is_file():
        raise FileNotFoundError(f"No se encontró el catálogo: {catalog_path}")
    try:
        raw_catalog = pd.read_csv(catalog_path, encoding="utf-8")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError(f"No fue posible leer el catálogo CSV: {catalog_path}") from error

    catalog = normalize_catalog(raw_catalog)
    if strict:
        validate_catalog(catalog)
    return catalog


def catalog_quality(catalog: pd.DataFrame) -> dict[str, object]:
    """Contar problemas de identificadores, nulos y capacidades."""
    normalized = normalize_catalog(catalog)
    duplicate_ids = normalized.loc[
        normalized["id"].notna() & normalized["id"].duplicated(keep=False),
        "id",
    ]
    capacity_values = normalized["dpcapacity"].to_numpy(dtype=np.float64)
    invalid_capacity = (
        normalized["dpcapacity"].isna().to_numpy()
        | ~np.isfinite(capacity_values)
        | (capacity_values <= 0)
    )
    valid_capacity = ~invalid_capacity
    valid_values = normalized.loc[valid_capacity, "dpcapacity"]
    invalid_capacity_ids = normalized.loc[invalid_capacity, "id"].dropna()

    return {
        "row_count": len(normalized),
        "null_id_rows": int(normalized["id"].isna().sum()),
        "duplicate_id_values": int(duplicate_ids.nunique()),
        "duplicate_id_rows": int(duplicate_ids.size),
        "null_capacity_rows": int(normalized["dpcapacity"].isna().sum()),
        "invalid_capacity_rows": int(invalid_capacity.sum()),
        "invalid_capacity_station_ids": sorted(
            invalid_capacity_ids.astype("int64").unique().tolist()
        ),
        "capacity_min": float(valid_values.min()) if not valid_values.empty else None,
        "capacity_max": float(valid_values.max()) if not valid_values.empty else None,
        "capacity_distribution": {
            str(capacity): int(count)
            for capacity, count in valid_values.value_counts().sort_index().items()
        },
    }


def validate_catalog(catalog: pd.DataFrame) -> None:
    """Rechazar identificadores duplicados/nulos y capacidades inválidas."""
    quality = catalog_quality(catalog)
    errors = []
    if quality["null_id_rows"]:
        errors.append(f"{quality['null_id_rows']} identificadores nulos o no enteros")
    if quality["duplicate_id_values"]:
        errors.append(f"{quality['duplicate_id_values']} identificadores duplicados")
    if quality["invalid_capacity_rows"]:
        errors.append(f"{quality['invalid_capacity_rows']} capacidades nulas o no positivas")
    if errors:
        raise ValueError("Catálogo inválido: " + "; ".join(errors) + ".")


def audit_station_coverage(
    hourly_path: Path,
    catalog: pd.DataFrame,
) -> dict[str, object]:
    """Auditar coincidencias y capacidad solo entre enero y octubre de 2025."""
    start = pd.Timestamp(START_DATE)
    validation_end = start + pd.DateOffset(months=10)
    try:
        hourly = pd.read_parquet(
            hourly_path,
            columns=["station_id", "hour"],
            filters=[("hour", ">=", start), ("hour", "<", validation_end)],
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"No fue posible leer datos horarios de train/validación: {hourly_path}"
        ) from error

    if hourly.empty:
        raise ValueError("No hay registros estación-hora entre enero y octubre de 2025.")
    if not pd.api.types.is_datetime64_any_dtype(hourly["hour"]):
        raise TypeError("'hour' debe tener tipo datetime en el Parquet.")
    if hourly["station_id"].isna().any():
        raise ValueError("'station_id' contiene valores nulos en el Parquet.")
    if (hourly["hour"] < start).any() or (hourly["hour"] >= validation_end).any():
        raise ValueError("La lectura de cobertura incluyó horas fuera de enero-octubre de 2025.")

    normalized = normalize_catalog(catalog)
    quality = catalog_quality(normalized)
    catalog_ids = normalized.loc[normalized["id"].notna(), "id"].astype("int64")
    hourly_ids = set(hourly["station_id"].astype("int64").unique())
    catalog_id_set = set(catalog_ids)
    matched_ids = hourly_ids & catalog_id_set
    unmatched_ids = hourly_ids - catalog_id_set

    id_counts = normalized.groupby("id", dropna=True)["id"].size()
    unique_ids = set(id_counts.loc[id_counts.eq(1)].index.astype("int64"))
    capacity_values = normalized["dpcapacity"].to_numpy(dtype=np.float64)
    capacity_is_valid = (
        normalized["dpcapacity"].notna().to_numpy()
        & np.isfinite(capacity_values)
        & (capacity_values > 0)
    )
    valid_capacity_ids = (
        set(normalized.loc[capacity_is_valid, "id"].dropna().astype("int64")) & unique_ids
    )

    record_has_capacity = hourly["station_id"].astype("int64").isin(valid_capacity_ids)
    valid_record_count = int(record_has_capacity.sum())
    record_count = len(hourly)

    return {
        "period_start_inclusive": start.isoformat(),
        "period_end_exclusive": validation_end.isoformat(),
        "hourly_records": record_count,
        "used_station_count": len(hourly_ids),
        "catalog_row_count": quality["row_count"],
        "catalog_unique_id_count": len(catalog_id_set),
        "stations_with_catalog_match": len(matched_ids),
        "stations_without_catalog_match": len(unmatched_ids),
        "unmatched_station_ids": sorted(unmatched_ids),
        "station_coverage_percent": (
            100.0 * len(matched_ids) / len(hourly_ids) if hourly_ids else 0.0
        ),
        "stations_with_unique_valid_capacity": len(hourly_ids & valid_capacity_ids),
        "stations_without_valid_capacity_ids": sorted(hourly_ids - valid_capacity_ids),
        "records_with_valid_capacity": valid_record_count,
        "records_without_valid_capacity": record_count - valid_record_count,
        "record_capacity_coverage_percent": (
            100.0 * valid_record_count / record_count if record_count else 0.0
        ),
        "catalog_quality": quality,
    }


@app.command()
def main() -> None:
    """Descargar el catálogo oficial provisional y auditar cobertura P2."""
    catalog_path, digest = download_catalog()
    catalog = load_catalog(catalog_path, strict=False)
    quality = catalog_quality(catalog)
    if quality["invalid_capacity_rows"] or quality["duplicate_id_values"]:
        logger.warning(
            "El catálogo conserva filas problemáticas para auditoría: {}",
            json.dumps(quality, ensure_ascii=False),
        )
    coverage = audit_station_coverage(
        PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet",
        catalog,
    )
    report = {
        "catalog_version": "2026-09",
        "source_page": OFFICIAL_SOURCE_PAGE,
        "source_url": OFFICIAL_CATALOG_URL,
        "catalog_path": str(catalog_path),
        "sha256": digest,
        "temporal_limitation": (
            "Este catálogo es de 2026-09; sus capacidades no están verificadas para el año 2025."
        ),
        "catalog_quality": quality,
        "coverage": coverage,
    }
    report_path = catalog_path.parent / AUDIT_FILENAME
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Auditoría de capacidades P2:\n{}", json.dumps(report, indent=2))
    logger.success("Auditoría guardada en: {}", report_path)


if __name__ == "__main__":
    app()
