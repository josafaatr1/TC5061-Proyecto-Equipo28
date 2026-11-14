"""Reportes de Data Drift y Data Quality con Evidently para el modelo P2.

Compara las variables de entrada del clasificador P2 (presión de bicicletas y de
anclajes) y sus etiquetas entre dos periodos:

- Reference: enero-agosto de 2025, la partición train con la que se entrenó el champion.
- Current: noviembre-diciembre de 2025, el periodo posterior que hace las veces de
  producción. Sus etiquetas usan los umbrales de presión ajustados solo con train.

    python -m mibici.monitoring.drift_report
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from evidently import DataDefinition, Dataset, Report
from evidently.legacy.calculations import stattests
from evidently.legacy.calculations.stattests.registry import StatTest
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import pandas as pd
import typer

from mibici.config import REPORTS_DIR
from mibici.features import REQUIRED_COLUMNS
from mibici.modeling.p2_catalog import load_catalog
from mibici.modeling.p2_targets import (
    TARGET_BIKE_COLUMN,
    TARGET_DOCK_COLUMN,
    VALIDATION_END,
    PressureThresholds,
    build_p2_test_dataset,
    calculate_pressure_windows,
    fit_pressure_thresholds,
)
from mibici.modeling.p2_train import (
    CALENDAR_FEATURES,
    CATALOG_PATH,
    HOURLY_DATA_PATH,
    LABEL_DATA_PATH,
    PREDICTOR_COLUMNS,
)

app = typer.Typer()

OUTPUT_DIR = REPORTS_DIR / "evidently"
REFERENCE_PERIOD = "2025-01-01/2025-08-31"
CURRENT_PERIOD = "2025-11-01/2025-12-31"

TARGET_COLUMNS = (TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN)
MONITORED_COLUMNS = (*PREDICTOR_COLUMNS, *TARGET_COLUMNS)
# Calendario y etiquetas son categorías discretas; rezagos y medias móviles son numéricos.
CATEGORICAL_COLUMNS = (*CALENDAR_FEATURES, *TARGET_COLUMNS)
NUMERICAL_COLUMNS = tuple(c for c in MONITORED_COLUMNS if c not in CATEGORICAL_COLUMNS)
# Jensen-Shannon casi no reacciona a cambios de prevalencia en etiquetas binarias muy
# desbalanceadas (≈4 % de positivos); la prueba Z de proporciones sí los detecta.
PER_COLUMN_METHOD = {column: "z" for column in TARGET_COLUMNS}
DATA_DEFINITION = DataDefinition(
    numerical_columns=list(NUMERICAL_COLUMNS),
    categorical_columns=list(CATEGORICAL_COLUMNS),
)


def load_reference(label_path: Path = LABEL_DATA_PATH) -> pd.DataFrame:
    """Partición train (enero-agosto) de las etiquetas P2 ya construidas."""
    return pd.read_parquet(
        label_path,
        columns=list(MONITORED_COLUMNS),
        filters=[("partition", "==", "train")],
    )


def fit_train_thresholds(hourly: pd.DataFrame, catalog: pd.DataFrame) -> PressureThresholds:
    """Recalcular los umbrales de presión solo con ventanas train, como en ``p2_targets``."""
    calculation = calculate_pressure_windows(hourly.loc[hourly["hour"] < VALIDATION_END], catalog)
    train_windows = calculation.windows.loc[calculation.windows["partition"].eq("train")]
    return fit_pressure_thresholds(train_windows)


def load_current(
    hourly_path: Path = HOURLY_DATA_PATH,
    catalog_path: Path = CATALOG_PATH,
) -> tuple[pd.DataFrame, PressureThresholds]:
    """Predictores y etiquetas de noviembre-diciembre con umbrales congelados de train."""
    hourly = pd.read_parquet(hourly_path, columns=list(REQUIRED_COLUMNS))
    catalog = load_catalog(catalog_path, strict=False)
    thresholds = fit_train_thresholds(hourly, catalog)
    current = build_p2_test_dataset(hourly, catalog, thresholds)
    return current.loc[:, list(MONITORED_COLUMNS)], thresholds


def _as_dataset(frame: pd.DataFrame) -> Dataset:
    # Evidently trata como categóricas las columnas de DATA_DEFINITION; bool e int8 se
    # llevan a int64 para que las categorías (y su orden en las gráficas) coincidan.
    frame = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("int64")
    return Dataset.from_pandas(frame, data_definition=DATA_DEFINITION)


def method_display_name(method: str) -> str:
    """Nombre legible de una prueba de Evidently (``"z"`` -> ``"Z-test p_value"``)."""
    names = {
        test.name: test.display_name
        for test in vars(stattests).values()
        if isinstance(test, StatTest)
    }
    return names.get(method, method)


def drift_detected(method: str, score: float, threshold: float) -> bool:
    """Las pruebas de p-value detectan drift por debajo del umbral; las distancias, por encima."""
    if "p_value" in method_display_name(method):
        return score < threshold
    return score >= threshold


def summarize_drift(snapshot_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Resultado por variable de los ``ValueDrift`` de un reporte de Evidently."""
    rows = []
    for metric in snapshot_dict["metrics"]:
        config = metric["config"]
        if not config["type"].endswith("ValueDrift"):
            continue
        score = float(metric["value"])
        rows.append(
            {
                "column": config["column"],
                "method": method_display_name(config["method"]),
                "threshold": config["threshold"],
                "score": score,
                "drift_detected": drift_detected(config["method"], score, config["threshold"]),
            }
        )
    return sorted(rows, key=lambda row: MONITORED_COLUMNS.index(row["column"]))


def generate_reports(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    thresholds: PressureThresholds | None = None,
) -> dict[str, Any]:
    """Escribir los HTML de drift y calidad y un ``drift_summary.json`` por variable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_data, current_data = _as_dataset(reference), _as_dataset(current)

    drift = Report(
        [DataDriftPreset(columns=list(MONITORED_COLUMNS), per_column_method=PER_COLUMN_METHOD)]
    ).run(current_data=current_data, reference_data=reference_data)
    drift.save_html(str(output_dir / "data_drift_report.html"))

    quality = Report([DataSummaryPreset(columns=list(MONITORED_COLUMNS))]).run(
        current_data=current_data, reference_data=reference_data
    )
    quality.save_html(str(output_dir / "data_quality_report.html"))

    columns = summarize_drift(drift.dict())
    summary = {
        "model": "P2 (mibici-p2-bike-pressure / mibici-p2-dock-pressure)",
        "reference": {"period": REFERENCE_PERIOD, "partition": "train", "rows": len(reference)},
        "current": {"period": CURRENT_PERIOD, "partition": "test", "rows": len(current)},
        "pressure_thresholds_train_only": asdict(thresholds) if thresholds else None,
        "drifted_columns": sum(row["drift_detected"] for row in columns),
        "monitored_columns": len(columns),
        "prevalence": {
            column: {
                "reference": float(reference[column].mean()),
                "current": float(current[column].mean()),
            }
            for column in TARGET_COLUMNS
        },
        "columns": columns,
    }
    (output_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


@app.command()
def main(
    label_path: Path = LABEL_DATA_PATH,
    hourly_path: Path = HOURLY_DATA_PATH,
    catalog_path: Path = CATALOG_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generar los reportes de Data Drift y Data Quality de P2."""
    reference = load_reference(label_path)
    current, thresholds = load_current(hourly_path, catalog_path)
    logger.info(f"Reference {REFERENCE_PERIOD}: {len(reference):,} filas")
    logger.info(f"Current {CURRENT_PERIOD}: {len(current):,} filas")

    summary = generate_reports(reference, current, output_dir, thresholds)
    for row in summary["columns"]:
        flag = "DRIFT" if row["drift_detected"] else "ok"
        logger.info(f"{row['column']:<30} {row['method']:<32} {row['score']:.4f} {flag}")
    logger.success(
        f"{summary['drifted_columns']}/{summary['monitored_columns']} variables con drift. "
        f"Reportes en {output_dir}"
    )


if __name__ == "__main__":
    app()
