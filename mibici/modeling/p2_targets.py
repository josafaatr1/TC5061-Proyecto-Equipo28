"""Construcción de etiquetas proxy de presión operacional para P2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    END_DATE,
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    START_DATE,
)
from mibici.features import REQUIRED_COLUMNS, build_features
from mibici.modeling.p2_catalog import (
    CATALOG_FILENAME,
    catalog_quality,
    load_catalog,
    normalize_catalog,
)

app = typer.Typer()

TARGET_BIKE_COLUMN = "target_bike_pressure"
TARGET_DOCK_COLUMN = "target_dock_pressure"
BIKE_PRESSURE_COLUMN = "bike_pressure"
DOCK_PRESSURE_COLUMN = "dock_pressure"
BIKE_RATIO_COLUMN = "bike_pressure_ratio"
DOCK_RATIO_COLUMN = "dock_pressure_ratio"
PRESSURE_QUANTILE = 0.90
YEAR_START = pd.Timestamp(START_DATE)
TRAIN_END = YEAR_START + pd.DateOffset(months=8)
VALIDATION_END = YEAR_START + pd.DateOffset(months=10)
YEAR_END = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)
OUTPUT_PATH = PROCESSED_DATA_DIR / "p2_targets_2025_train_validation.parquet"


@dataclass(frozen=True)
class PressureThresholds:
    """Percentiles de presión relativa calculados solo en ventanas train."""

    bike_pressure_ratio: float
    dock_pressure_ratio: float


@dataclass(frozen=True)
class PressureWindowCalculation:
    """Ventanas válidas, orden de anclas y auditoría de exclusiones."""

    windows: pd.DataFrame
    sorted_hourly: pd.DataFrame
    anchor_positions: np.ndarray
    excluded_stations: pd.DataFrame
    audit: dict[str, object]


@dataclass(frozen=True)
class P2TargetResult:
    """Dataset de predictores históricos, proxies y trazabilidad."""

    data: pd.DataFrame
    pressure_windows: pd.DataFrame
    thresholds: PressureThresholds
    excluded_stations: pd.DataFrame
    audit: dict[str, object]


def _validate_hourly_data(hourly_data: pd.DataFrame) -> pd.DataFrame:
    """Validar y ordenar registros solo de enero a octubre de 2025."""
    if not isinstance(hourly_data, pd.DataFrame):
        raise TypeError("hourly_data debe ser un pandas.DataFrame.")

    missing_columns = set(REQUIRED_COLUMNS) - set(hourly_data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas horarias obligatorias: {sorted(missing_columns)}.")
    if not pd.api.types.is_datetime64_any_dtype(hourly_data["hour"]):
        raise TypeError("'hour' debe tener tipo datetime.")
    if hourly_data["hour"].isna().any():
        raise ValueError("'hour' no puede contener valores nulos.")
    if hourly_data["hour"].dt.tz is not None:
        raise ValueError("'hour' debe usar timestamps sin zona horaria.")
    if (hourly_data["hour"] != hourly_data["hour"].dt.floor("h")).any():
        raise ValueError("'hour' debe estar alineada al inicio de cada hora.")
    if ((hourly_data["hour"] < YEAR_START) | (hourly_data["hour"] >= VALIDATION_END)).any():
        raise ValueError(
            "La entrada debe contener únicamente enero-octubre de 2025; "
            "el periodo test está reservado."
        )
    if hourly_data["station_id"].isna().any():
        raise ValueError("'station_id' no puede contener valores nulos.")
    if hourly_data.duplicated(["station_id", "hour"]).any():
        raise ValueError("Existen combinaciones estación-hora duplicadas.")

    station_ids = pd.to_numeric(hourly_data["station_id"], errors="coerce")
    station_id_values = station_ids.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(station_id_values).all() or (station_id_values % 1 != 0).any():
        raise ValueError("'station_id' debe contener identificadores enteros válidos.")

    numeric_columns = ("departures", "arrivals", "net_flow", "total_activity")
    for column in numeric_columns:
        values = hourly_data[column]
        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or values.isna().any()
            or not np.isfinite(values.to_numpy(dtype=np.float64)).all()
        ):
            raise ValueError(f"'{column}' debe contener valores numéricos finitos.")

    if (hourly_data[["departures", "arrivals"]] < 0).any().any():
        raise ValueError("'departures' y 'arrivals' no pueden ser negativas.")
    if not pd.api.types.is_bool_dtype(hourly_data["is_observed_hour"]):
        raise TypeError("'is_observed_hour' debe ser booleana.")
    if hourly_data["is_observed_hour"].isna().any():
        raise ValueError("'is_observed_hour' no puede contener valores nulos.")

    if not hourly_data["net_flow"].eq(hourly_data["arrivals"] - hourly_data["departures"]).all():
        raise ValueError("'net_flow' debe ser igual a arrivals - departures.")
    if (
        not hourly_data["total_activity"]
        .eq(hourly_data["arrivals"] + hourly_data["departures"])
        .all()
    ):
        raise ValueError("'total_activity' debe ser igual a arrivals + departures.")

    unobserved = hourly_data.loc[
        ~hourly_data["is_observed_hour"],
        ["departures", "arrivals", "net_flow", "total_activity"],
    ]
    if not unobserved.eq(0).all().all():
        raise ValueError("Las horas no observadas deben conservar conteos cero.")

    ordered = hourly_data.copy()
    ordered["station_id"] = station_ids.astype("int64")
    return ordered.sort_values(["station_id", "hour"], kind="stable").reset_index(drop=True)


def _capacity_selection(
    ordered_hourly: pd.DataFrame,
    station_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Seleccionar capacidades positivas únicas y registrar estaciones excluidas."""
    catalog = normalize_catalog(station_catalog)
    if catalog["id"].isna().any():
        raise ValueError("El catálogo contiene identificadores nulos o no enteros.")

    duplicated_ids = catalog.loc[catalog["id"].duplicated(keep=False), "id"]
    if not duplicated_ids.empty:
        duplicate_values = sorted(duplicated_ids.astype("int64").unique().tolist())
        raise ValueError(
            f"El catálogo contiene IDs duplicados potencialmente conflictivos: {duplicate_values}."
        )

    capacity_values = catalog["dpcapacity"].to_numpy(dtype=np.float64)
    valid_capacity = (
        catalog["dpcapacity"].notna().to_numpy()
        & np.isfinite(capacity_values)
        & (capacity_values > 0)
    )
    valid_catalog = catalog.loc[valid_capacity, ["id", "dpcapacity"]].copy()
    valid_catalog["id"] = valid_catalog["id"].astype("int64")
    capacity_by_station = valid_catalog.set_index("id")["dpcapacity"]
    catalog_id_set = set(catalog["id"].astype("int64"))
    valid_id_set = set(capacity_by_station.index)

    hourly_counts = ordered_hourly["station_id"].value_counts().sort_index()
    observed_ids = set(hourly_counts.index.astype("int64"))
    excluded_ids = sorted(observed_ids - valid_id_set)
    invalid_catalog_ids = catalog.loc[~valid_capacity, "id"].dropna().astype("int64")
    invalid_id_set = set(invalid_catalog_ids)

    exclusion_rows = []
    for station_id in excluded_ids:
        if station_id not in catalog_id_set:
            reason = "missing_catalog_station"
        elif station_id in invalid_id_set:
            reason = "missing_or_nonpositive_capacity"
        else:
            reason = "invalid_capacity"
        exclusion_rows.append(
            {
                "station_id": station_id,
                "reason": reason,
                "hourly_records_excluded": int(hourly_counts.loc[station_id]),
            }
        )
    excluded_stations = pd.DataFrame(
        exclusion_rows,
        columns=["station_id", "reason", "hourly_records_excluded"],
    )

    eligible_mask = ordered_hourly["station_id"].isin(valid_id_set)
    selected_hourly = ordered_hourly.loc[eligible_mask].reset_index(drop=True)
    selected_hourly["station_capacity"] = selected_hourly["station_id"].map(capacity_by_station)
    if selected_hourly["station_capacity"].isna().any():
        raise ValueError("Hay estaciones seleccionadas sin capacidad válida.")
    if (selected_hourly["station_capacity"] <= 0).any():
        raise ValueError("Se detectó capacidad no positiva antes de normalizar presiones.")

    quality = catalog_quality(catalog)
    audit = {
        "input_hourly_records": len(ordered_hourly),
        "eligible_hourly_records": len(selected_hourly),
        "records_excluded_without_valid_capacity": len(ordered_hourly) - len(selected_hourly),
        "used_station_count": len(observed_ids),
        "stations_with_valid_capacity": len(observed_ids & valid_id_set),
        "stations_excluded_without_valid_capacity": len(excluded_ids),
        "excluded_station_ids": excluded_ids,
        "catalog_invalid_capacity_station_ids": quality["invalid_capacity_station_ids"],
        "capacity_coverage_percent": (
            100.0 * len(selected_hourly) / len(ordered_hourly) if len(ordered_hourly) else 0.0
        ),
        "capacity_validity_policy": (
            "Only stations with one catalog row and finite dpcapacity > 0 are "
            "included; excluded stations and record counts are retained in the audit."
        ),
    }
    return selected_hourly, excluded_stations, audit


def _partition_for_hours(hours: pd.Series) -> pd.Series:
    """Asignar train o validation usando la hora de inicio de ventana."""
    return pd.Series(
        np.where(hours < TRAIN_END, "train", "validation"),
        index=hours.index,
        dtype="string",
    )


def calculate_pressure_windows(
    hourly_data: pd.DataFrame,
    station_catalog: pd.DataFrame,
) -> PressureWindowCalculation:
    """Calcular presiones acumuladas en ventanas exactas y dentro del split.

    Una ventana anclada en ``t`` usa los flujos en ``t``, ``t+1`` y ``t+2``.
    Las horas ausentes no se imputan; se excluyen anclas incompletas. Las
    ventanas exactas que crucen de train a validation también se excluyen.
    """
    ordered = _validate_hourly_data(hourly_data)
    selected, excluded_stations, audit = _capacity_selection(ordered, station_catalog)

    if selected.empty:
        raise ValueError("No hay estaciones con capacidad válida para construir P2.")

    selected["partition"] = _partition_for_hours(selected["hour"])
    grouped = selected.groupby("station_id", sort=False, observed=True)
    future_hour_1 = grouped["hour"].shift(-1)
    future_hour_2 = grouped["hour"].shift(-2)
    exact_window = future_hour_1.eq(selected["hour"] + pd.Timedelta(hours=1)) & future_hour_2.eq(
        selected["hour"] + pd.Timedelta(hours=2)
    )
    future_partition_1 = grouped["partition"].shift(-1)
    future_partition_2 = grouped["partition"].shift(-2)
    same_partition = selected["partition"].eq(future_partition_1) & selected["partition"].eq(
        future_partition_2
    )
    crosses_partition = exact_window & ~same_partition
    valid_window = exact_window & same_partition
    incomplete_window = ~exact_window

    future_arrivals_1 = grouped["arrivals"].shift(-1)
    future_arrivals_2 = grouped["arrivals"].shift(-2)
    future_departures_1 = grouped["departures"].shift(-1)
    future_departures_2 = grouped["departures"].shift(-2)
    flow_0 = selected["arrivals"] - selected["departures"]
    flow_1 = future_arrivals_1 - future_departures_1
    flow_2 = future_arrivals_2 - future_departures_2
    cumulative_1 = flow_0
    cumulative_2 = flow_0 + flow_1
    cumulative_3 = cumulative_2 + flow_2
    lowest_cumulative = np.minimum(
        np.minimum(cumulative_1.to_numpy(), cumulative_2.to_numpy()),
        cumulative_3.to_numpy(),
    )
    highest_cumulative = np.maximum(
        np.maximum(cumulative_1.to_numpy(), cumulative_2.to_numpy()),
        cumulative_3.to_numpy(),
    )
    bike_pressure = np.where(lowest_cumulative < 0, -lowest_cumulative, 0.0)
    dock_pressure = np.where(highest_cumulative > 0, highest_cumulative, 0.0)
    capacities = selected["station_capacity"].to_numpy(dtype=np.float64)
    if not np.isfinite(capacities).all() or (capacities <= 0).any():
        raise ValueError("La normalización requiere capacidades positivas y finitas.")

    anchor_positions = np.flatnonzero(valid_window.to_numpy())
    windows = selected.loc[
        valid_window,
        ["station_id", "hour", "partition", "station_capacity"],
    ].copy()
    windows[BIKE_PRESSURE_COLUMN] = bike_pressure[anchor_positions]
    windows[DOCK_PRESSURE_COLUMN] = dock_pressure[anchor_positions]
    windows[BIKE_RATIO_COLUMN] = (
        windows[BIKE_PRESSURE_COLUMN].to_numpy(dtype=np.float64) / capacities[anchor_positions]
    )
    windows[DOCK_RATIO_COLUMN] = (
        windows[DOCK_PRESSURE_COLUMN].to_numpy(dtype=np.float64) / capacities[anchor_positions]
    )
    windows = windows.reset_index(drop=True)

    partition_audit = {}
    for partition in ("train", "validation"):
        anchor_mask = selected["partition"].eq(partition)
        partition_audit[partition] = {
            "eligible_anchors": int(anchor_mask.sum()),
            "valid_windows": int((valid_window & anchor_mask).sum()),
            "excluded_incomplete_windows": int((incomplete_window & anchor_mask).sum()),
            "excluded_cross_partition_windows": int((crosses_partition & anchor_mask).sum()),
        }

    audit.update(
        {
            "window_definition": "future hours t, t+1, and t+2, inclusive",
            "train_end_exclusive": TRAIN_END.isoformat(),
            "validation_end_exclusive": VALIDATION_END.isoformat(),
            "valid_windows": len(windows),
            "excluded_incomplete_windows": int(incomplete_window.sum()),
            "excluded_cross_partition_windows": int(crosses_partition.sum()),
            "windows_by_partition": partition_audit,
        }
    )

    return PressureWindowCalculation(
        windows=windows,
        sorted_hourly=selected.drop(columns=["partition", "station_capacity"]),
        anchor_positions=anchor_positions,
        excluded_stations=excluded_stations,
        audit=audit,
    )


def fit_pressure_thresholds(train_windows: pd.DataFrame) -> PressureThresholds:
    """Ajustar percentiles 90 de presiones positivas exclusivamente en train."""
    if not isinstance(train_windows, pd.DataFrame):
        raise TypeError("train_windows debe ser un pandas.DataFrame.")
    if train_windows.empty:
        raise ValueError("No hay ventanas completas de train para ajustar umbrales.")
    if "partition" not in train_windows.columns:
        raise ValueError("Falta la columna 'partition'.")
    if not train_windows["partition"].eq("train").all():
        raise ValueError("Los umbrales solo pueden ajustarse con ventanas de train.")

    thresholds = {}
    for column in (BIKE_RATIO_COLUMN, DOCK_RATIO_COLUMN):
        if column not in train_windows.columns:
            raise ValueError(f"Falta la presión normalizada '{column}'.")
        values = pd.to_numeric(train_windows[column], errors="coerce")
        array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"'{column}' debe contener valores finitos no negativos.")
        positive_values = values.loc[values.gt(0)]
        if positive_values.empty:
            raise ValueError(f"No hay presiones positivas en train para '{column}'.")
        thresholds[column] = float(positive_values.quantile(PRESSURE_QUANTILE))

    return PressureThresholds(
        bike_pressure_ratio=thresholds[BIKE_RATIO_COLUMN],
        dock_pressure_ratio=thresholds[DOCK_RATIO_COLUMN],
    )


def apply_pressure_thresholds(
    windows: pd.DataFrame,
    thresholds: PressureThresholds,
) -> pd.DataFrame:
    """Etiquetar presiones con umbrales congelados y sin usar datos de test."""
    required_columns = {
        "partition",
        BIKE_RATIO_COLUMN,
        DOCK_RATIO_COLUMN,
    }
    missing_columns = required_columns - set(windows.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas para etiquetar: {sorted(missing_columns)}.")
    if windows["partition"].isin(["test"]).any():
        raise ValueError("No se permite construir ni inspeccionar etiquetas de test.")
    if not windows["partition"].isin(["train", "validation"]).all():
        raise ValueError("Las ventanas deben pertenecer a train o validation.")
    for threshold in asdict(thresholds).values():
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Los umbrales deben ser positivos y finitos.")

    labeled = windows.copy()
    bike_values = pd.to_numeric(labeled[BIKE_RATIO_COLUMN], errors="coerce")
    dock_values = pd.to_numeric(labeled[DOCK_RATIO_COLUMN], errors="coerce")
    for column, values in (
        (BIKE_RATIO_COLUMN, bike_values),
        (DOCK_RATIO_COLUMN, dock_values),
    ):
        numeric = values.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError(f"'{column}' debe contener valores finitos no negativos.")

    labeled[TARGET_BIKE_COLUMN] = bike_values.ge(thresholds.bike_pressure_ratio).astype("int8")
    labeled[TARGET_DOCK_COLUMN] = dock_values.ge(thresholds.dock_pressure_ratio).astype("int8")
    return labeled


def summarize_targets(dataset: pd.DataFrame) -> dict[str, object]:
    """Resumir distribución de presiones y clases para train y validation."""
    if "partition" not in dataset.columns:
        raise ValueError("Falta la columna 'partition'.")
    if dataset["partition"].isin(["test"]).any():
        raise ValueError("No se permite resumir distribuciones del conjunto test.")

    summary: dict[str, object] = {}
    for partition in ("train", "validation"):
        rows = dataset.loc[dataset["partition"].eq(partition)]
        partition_summary: dict[str, object] = {
            "windows": len(rows),
            "targets": {},
            "pressures": {},
        }
        for target in (TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN):
            positives = int(rows[target].sum())
            negatives = len(rows) - positives
            partition_summary["targets"][target] = {
                "positive": positives,
                "positive_percent": 100.0 * positives / len(rows) if len(rows) else 0.0,
                "negative": negatives,
                "negative_percent": 100.0 * negatives / len(rows) if len(rows) else 0.0,
            }
        for pressure in (
            BIKE_PRESSURE_COLUMN,
            DOCK_PRESSURE_COLUMN,
            BIKE_RATIO_COLUMN,
            DOCK_RATIO_COLUMN,
        ):
            values = rows[pressure]
            partition_summary["pressures"][pressure] = {
                "count": len(values),
                "positive_count": int(values.gt(0).sum()),
                "min": float(values.min()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "p90": float(values.quantile(0.90)) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
            }
        summary[partition] = partition_summary
    return summary


def build_p2_targets(
    hourly_data: pd.DataFrame,
    station_catalog: pd.DataFrame,
) -> P2TargetResult:
    """Construir predictores causales, presiones y etiquetas para train/validación.

    Las etiquetas describen un proxy de presión acumulada durante las tres
    horas futuras; no representan inventarios reales de bicicletas o anclajes.
    El catálogo de 2026-09 es una aproximación retrospectiva no validada para
    2025. Solo IDs con una capacidad única, positiva y finita se incluyen.
    """
    calculation = calculate_pressure_windows(hourly_data, station_catalog)
    train_windows = calculation.windows.loc[calculation.windows["partition"].eq("train")]
    thresholds = fit_pressure_thresholds(train_windows)
    labeled_windows = apply_pressure_thresholds(calculation.windows, thresholds)

    features = build_features(calculation.sorted_hourly)
    feature_rows = features.iloc[calculation.anchor_positions].reset_index(drop=True)
    label_columns = ["partition", TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN]
    if not feature_rows[["station_id", "hour"]].equals(
        labeled_windows[["station_id", "hour"]].reset_index(drop=True)
    ):
        raise ValueError("Las características y las ventanas objetivo no quedaron alineadas.")
    dataset = pd.concat(
        [
            feature_rows,
            labeled_windows.loc[:, label_columns].reset_index(drop=True),
        ],
        axis=1,
    )

    audit = dict(calculation.audit)
    audit["threshold_quantile"] = PRESSURE_QUANTILE
    audit["thresholds_train_only"] = asdict(thresholds)
    audit["class_and_pressure_distribution"] = summarize_targets(labeled_windows)
    audit["catalog_capacity_temporal_warning"] = (
        "Las capacidades de nomenclatura_2026_09.csv son aproximaciones "
        "retrospectivas no validadas para 2025."
    )
    return P2TargetResult(
        data=dataset,
        pressure_windows=labeled_windows,
        thresholds=thresholds,
        excluded_stations=calculation.excluded_stations,
        audit=audit,
    )


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet",
    catalog_path: Path = EXTERNAL_DATA_DIR / CATALOG_FILENAME,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Construir y guardar etiquetas proxy únicamente para train/validación."""
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"No existe el catálogo local {catalog_path}; ejecutar primero "
            "'python -m mibici.modeling.p2_catalog'."
        )

    validation_end = pd.Timestamp(START_DATE) + pd.DateOffset(months=10)
    hourly_data = pd.read_parquet(
        input_path,
        columns=list(REQUIRED_COLUMNS),
        filters=[
            ("hour", ">=", pd.Timestamp(START_DATE)),
            ("hour", "<", validation_end),
        ],
    )
    station_catalog = load_catalog(catalog_path, strict=False)

    started_at = time.perf_counter()
    result = build_p2_targets(hourly_data, station_catalog)
    elapsed_seconds = time.perf_counter() - started_at

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_parquet(output_path, index=False)
    report = dict(result.audit)
    report["elapsed_seconds"] = elapsed_seconds
    report["output_path"] = str(output_path)
    logger.info("Auditoría de etiquetas proxy P2:\n{}", report)
    logger.success("Etiquetas de train/validación guardadas en: {}", output_path)


if __name__ == "__main__":
    app()
