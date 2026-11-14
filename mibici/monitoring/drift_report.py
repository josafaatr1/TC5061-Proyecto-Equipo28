"""Generación de reportes de Data Drift y Data Quality para MiBici."""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "station_hourly_2025_complete.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports" / "evidently"



def load_and_split_data():
    """Carga el dataset y separa reference y current."""

    df = pd.read_parquet(DATA_PATH)

    # Crear variable categórica para identificar las estaciones
    df["station_id"] = df["station_id"].astype(str)

    # Extraer la hora del día (0-23)
    df["hour_of_day"] = df["hour"].dt.hour

    # Reference: enero a agosto de 2025
    reference = df[df["hour"] < "2025-09-01"].copy()

    # Current: septiembre y octubre de 2025
    current = df[
        (df["hour"] >= "2025-09-01") &
        (df["hour"] < "2025-11-01")
    ].copy()

    return reference, current





def generate_drift_report(reference, current):
    """Genera el reporte de Data Drift."""

    report = Report(
        metrics=[
            DataDriftPreset(
                columns=[
                    "station_id",
                    "hour_of_day",
                    "departures",
                    "arrivals",
                    "net_flow",
                    "total_activity",
                    "is_observed_hour",
                ]
            )
        ]
    )

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "data_drift_report_v2.html"
    snapshot.save_html(str(output_path))

    print(f"Reporte de Data Drift guardado en: {output_path}")




def generate_quality_report(current):
    """Genera el reporte de Data Quality."""

    report = Report(
        metrics=[
            DataSummaryPreset()
        ]
    )

    snapshot = report.run(
        current_data=current,
    )

    output_path = REPORTS_DIR / "data_quality_report.html"
    snapshot.save_html(str(output_path))

    print(f"Reporte de Data Quality guardado en: {output_path}")



def main():
    """Ejecuta el proceso completo de monitoreo."""

    print("Iniciando análisis de Data Drift y Data Quality...")

    reference, current = load_and_split_data()

    print(f"Registros de referencia: {len(reference):,}")
    print(f"Registros actuales: {len(current):,}")

    generate_drift_report(reference, current)
    generate_quality_report(current)

    print("Análisis completado correctamente.")


if __name__ == "__main__":
    main()
