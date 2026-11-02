"""Pruebas de carga, validación y auditoría del catálogo P2."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from typer.testing import CliRunner

from mibici.modeling import p2_catalog


def make_catalog() -> pd.DataFrame:
    """Construir un catálogo mínimo con el esquema oficial."""
    return pd.DataFrame(
        {
            "id": ["10", "20"],
            "name": ["Centro", "Norte"],
            "latitude": ["20.67", "20.69"],
            "longitude": ["-103.35", "-103.36"],
            "dpcapacity": ["15", "20"],
        }
    )


def test_normalize_catalog_converts_ids_and_capacity_types():
    """Normalizar texto numérico a tipos utilizables en joins."""
    catalog = p2_catalog.normalize_catalog(make_catalog())

    assert str(catalog["id"].dtype) == "Int64"
    assert pd.api.types.is_numeric_dtype(catalog["dpcapacity"])
    assert catalog["id"].tolist() == [10, 20]
    assert catalog["dpcapacity"].tolist() == [15.0, 20.0]


def test_catalog_quality_reports_invalid_values_without_silently_dropping():
    """Exponer nulos, duplicados y capacidad no positiva en el resumen."""
    catalog = make_catalog()
    catalog.loc[1, "id"] = "10"
    catalog.loc[1, "dpcapacity"] = "0"
    catalog.loc[0, "name"] = None

    quality = p2_catalog.catalog_quality(catalog)

    assert quality["row_count"] == 2
    assert quality["duplicate_id_values"] == 1
    assert quality["duplicate_id_rows"] == 2
    assert quality["invalid_capacity_rows"] == 1
    assert quality["invalid_capacity_station_ids"] == [10]


@pytest.mark.parametrize(
    ("column", "values", "error_text"),
    [
        ("id", ["10", None], "identificadores nulos"),
        ("id", ["10", "10"], "identificadores duplicados"),
        ("dpcapacity", ["15", "0"], "capacidades nulas o no positivas"),
        ("dpcapacity", ["15", "bad"], "capacidades nulas o no positivas"),
    ],
)
def test_validate_catalog_rejects_invalid_schema_values(column, values, error_text):
    """Rechazar identificadores ambiguos y capacidades inválidas."""
    catalog = make_catalog()
    catalog[column] = values

    with pytest.raises(ValueError, match=error_text):
        p2_catalog.validate_catalog(p2_catalog.normalize_catalog(catalog))


def test_load_catalog_rejects_missing_columns(tmp_path: Path):
    """Informar claramente cuando el CSV no contiene el esquema oficial."""
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"id": [10], "dpcapacity": [15]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        p2_catalog.load_catalog(path)


def test_load_catalog_can_preserve_invalid_rows_for_audit(tmp_path: Path):
    """Mantener capacidades inválidas disponibles para el reporte de auditoría."""
    path = tmp_path / "catalog.csv"
    catalog = make_catalog()
    catalog.loc[1, "dpcapacity"] = "0"
    catalog.to_csv(path, index=False)

    loaded = p2_catalog.load_catalog(path, strict=False)

    assert len(loaded) == 2
    assert p2_catalog.catalog_quality(loaded)["invalid_capacity_rows"] == 1


def test_download_catalog_saves_csv_hash_and_provenance(tmp_path, monkeypatch):
    """Descargar con respuesta simulada y persistir trazabilidad."""
    monkeypatch.setattr(p2_catalog, "EXTERNAL_DATA_DIR", tmp_path)
    content = make_catalog().to_csv(index=False).encode("utf-8")
    response = Mock()
    response.content = content
    response.raise_for_status.return_value = None
    get = Mock(return_value=response)
    monkeypatch.setattr(p2_catalog.requests, "get", get)

    catalog_path, digest = p2_catalog.download_catalog()

    provenance_path = tmp_path / p2_catalog.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert catalog_path == tmp_path / p2_catalog.CATALOG_FILENAME
    assert catalog_path.read_bytes() == content
    assert digest == hashlib.sha256(content).hexdigest()
    assert provenance["sha256"] == digest
    assert provenance["catalog_version"] == "2026-09"
    assert provenance["source_url"] == p2_catalog.OFFICIAL_CATALOG_URL
    get.assert_called_once_with(
        p2_catalog.OFFICIAL_CATALOG_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        },
        timeout=(15, 120),
    )


def test_download_catalog_reports_request_errors(tmp_path, monkeypatch):
    """Convertir fallos HTTP en un error accionable."""
    monkeypatch.setattr(p2_catalog, "EXTERNAL_DATA_DIR", tmp_path)
    get = Mock(side_effect=requests.Timeout("timed out"))
    monkeypatch.setattr(p2_catalog.requests, "get", get)

    with pytest.raises(RuntimeError, match="No fue posible descargar"):
        p2_catalog.download_catalog()


def test_download_catalog_rejects_non_official_url(tmp_path, monkeypatch):
    """No permitir descargas del catálogo desde otros dominios."""
    monkeypatch.setattr(p2_catalog, "EXTERNAL_DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match="URL HTTPS oficial"):
        p2_catalog.download_catalog(url="https://example.com/catalog.csv")


def test_audit_coverage_uses_only_january_through_october(tmp_path: Path):
    """Excluir noviembre y reportar explícitamente estaciones sin match."""
    hourly_path = tmp_path / "hourly.parquet"
    pd.DataFrame(
        {
            "station_id": [10, 10, 30, 20],
            "hour": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-10-31 23:00",
                    "2025-09-01 00:00",
                    "2025-11-01 00:00",
                ]
            ),
        }
    ).to_parquet(hourly_path, index=False)
    catalog = p2_catalog.normalize_catalog(make_catalog())

    coverage = p2_catalog.audit_station_coverage(hourly_path, catalog)

    assert coverage["hourly_records"] == 3
    assert coverage["used_station_count"] == 2
    assert coverage["catalog_row_count"] == 2
    assert coverage["stations_with_catalog_match"] == 1
    assert coverage["stations_without_catalog_match"] == 1
    assert coverage["unmatched_station_ids"] == [30]
    assert coverage["stations_without_valid_capacity_ids"] == [30]
    assert coverage["station_coverage_percent"] == pytest.approx(50.0)
    assert coverage["records_with_valid_capacity"] == 2
    assert coverage["record_capacity_coverage_percent"] == pytest.approx(200 / 3)
    assert coverage["period_end_exclusive"] == "2025-11-01T00:00:00"


def test_audit_coverage_distinguishes_invalid_capacity_from_missing_station(
    tmp_path: Path,
):
    """No contar estaciones emparejadas con capacidad cero como válidas."""
    hourly_path = tmp_path / "hourly.parquet"
    hourly = pd.DataFrame(
        {
            "station_id": [10, 30],
            "hour": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
        }
    )
    hourly.to_parquet(hourly_path, index=False)
    catalog = make_catalog()
    catalog.loc[0, "dpcapacity"] = "0"

    coverage = p2_catalog.audit_station_coverage(
        hourly_path,
        p2_catalog.normalize_catalog(catalog),
    )

    assert coverage["stations_with_catalog_match"] == 1
    assert coverage["stations_without_catalog_match"] == 1
    assert coverage["stations_without_valid_capacity_ids"] == [10, 30]
    assert coverage["records_with_valid_capacity"] == 0
    assert coverage["records_without_valid_capacity"] == 2


def test_cli_is_available_as_a_reproducible_command():
    """Exponer un comando invocable para reconstruir la auditoría."""
    result = CliRunner().invoke(p2_catalog.app, ["--help"])

    assert result.exit_code == 0
    assert "main" in result.output
