from pathlib import Path

import pandas as pd

from mibici.tracking import (
    calculate_data_version,
    summarize_hourly,
    summarize_quality,
    summarize_raw_files,
)


def write_raw_files(raw_dir: Path) -> None:
    """Crear doce CSV mensuales mínimos."""
    raw_dir.mkdir()

    for month in range(1, 13):
        file_path = raw_dir / f"datos_abiertos_2025_{month:02d}.csv"
        file_path.write_text(f"mes,{month}\n")


def test_summarize_raw_files(tmp_path: Path):
    """Verificar el inventario de los CSV originales."""
    raw_dir = tmp_path / "raw"
    write_raw_files(raw_dir)

    inventory = summarize_raw_files(2025, raw_dir)

    assert len(inventory) == 12
    assert list(inventory["month"]) == list(range(1, 13))
    assert inventory["sha256"].str.len().eq(64).all()
    assert inventory["sha256"].is_unique


def test_calculate_data_version_detects_changes(tmp_path: Path):
    """La huella cambia solo cuando cambian los datos de entrada."""
    raw_dir = tmp_path / "raw"
    write_raw_files(raw_dir)

    first = calculate_data_version(summarize_raw_files(2025, raw_dir))
    second = calculate_data_version(summarize_raw_files(2025, raw_dir))

    (raw_dir / "datos_abiertos_2025_03.csv").write_text("mes,cambiado\n")
    changed = calculate_data_version(summarize_raw_files(2025, raw_dir))

    assert first == second
    assert first != changed


def test_summarize_quality(tmp_path: Path):
    """Verificar los conteos mensuales de viajes e indicadores."""
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()

    for month in range(1, 13):
        pd.DataFrame(
            {
                "has_invalid_timestamp": [False, True, False],
                "has_nonpositive_duration": [False, False, False],
                "is_long_trip": [True, True, False],
            }
        ).to_parquet(interim_dir / f"trips_2025_{month:02d}.parquet")

    quality = summarize_quality(2025, interim_dir)

    assert len(quality) == 12
    assert quality["trips"].eq(3).all()
    assert quality["invalid_timestamp"].eq(1).all()
    assert quality["nonpositive_duration"].eq(0).all()
    assert quality["long_trips"].eq(2).all()


def test_summarize_hourly(tmp_path: Path):
    """Verificar el tamaño y la cobertura de los datasets horarios."""
    observed_path = tmp_path / "observed.parquet"
    completed_path = tmp_path / "completed.parquet"

    pd.DataFrame({"station_id": [1, 1, 2]}).to_parquet(observed_path)
    pd.DataFrame({"total_activity": [3, 0, 0, 5]}).to_parquet(completed_path)

    summary = summarize_hourly(observed_path, completed_path)

    assert summary["observed_rows"] == 3
    assert summary["completed_rows"] == 4
    assert summary["stations_observed"] == 2
    assert summary["zero_activity_pct"] == 50

def test_log_pipeline_run_registra_manifiesto(tmp_path, monkeypatch):
    """Verificar que el manifiesto se registre como artefacto de MLflow."""
    from unittest.mock import Mock

    import mibici.tracking as tracking

    manifest_path = tmp_path / "mibici_sources_2025.csv"
    manifest_path.write_text("month,url\n1,https://example.com/data.csv\n")

    observed_path = tmp_path / "observed.parquet"
    completed_path = tmp_path / "completed.parquet"

    # Sustituir los resúmenes para no depender del pipeline anual.
    monkeypatch.setattr(
        tracking,
        "summarize_raw_files",
        lambda year: pd.DataFrame(
            {"month": [1], "sha256": ["a" * 64], "size_bytes": [100]}
        ),
    )
    monkeypatch.setattr(
        tracking,
        "summarize_quality",
        lambda year: pd.DataFrame(
            {
                "month": [1],
                "trips": [10],
                "invalid_timestamp": [0],
                "nonpositive_duration": [0],
                "long_trips": [0],
            }
        ),
    )
    monkeypatch.setattr(
        tracking,
        "summarize_hourly",
        lambda observed, completed: {
            "observed_rows": 5,
            "completed_rows": 8,
            "stations_observed": 2,
            "zero_activity_pct": 37.5,
        },
    )

    # Evitar conexiones reales a MLflow.
    log_artifact = Mock()
    monkeypatch.setattr(tracking.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(tracking.mlflow, "log_params", Mock())
    monkeypatch.setattr(tracking.mlflow, "log_metrics", Mock())
    monkeypatch.setattr(tracking.mlflow, "log_metric", Mock())
    monkeypatch.setattr(tracking.mlflow, "set_tag", Mock())

    tracking.log_pipeline_run(
        year=2025,
        manifest_path=manifest_path,
        observed_path=observed_path,
        completed_path=completed_path,
        duration_seconds=1.0,
    )

    # Comprobar que el manifiesto se registra en la carpeta esperada.
    log_artifact.assert_any_call(
        str(manifest_path),
        artifact_path="manifests",
    )