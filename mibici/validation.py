"""Validación de integridad y drift de los datasets P1 y P2 con Deepchecks."""

from __future__ import annotations

import importlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import PROCESSED_DATA_DIR, RANDOM_SEED, REPORTS_DIR
from mibici.modeling.p2_targets import TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN
from mibici.modeling.p2_train import (
    PREDICTOR_COLUMNS as P2_PREDICTOR_COLUMNS,
)
from mibici.modeling.p2_train import (
    prepare_p2_splits,
)
from mibici.modeling.train import (
    TARGET_COLUMN as P1_TARGET_COLUMN,
)
from mibici.modeling.train import (
    prepare_supervised_data,
    split_chronologically,
)

app = typer.Typer()

P1_INPUT_PATH = PROCESSED_DATA_DIR / "station_hourly_2025_complete.parquet"
P2_INPUT_PATH = PROCESSED_DATA_DIR / "p2_targets_2025_train_validation.parquet"
DEFAULT_OUTPUT_DIR = REPORTS_DIR / "deepchecks"
KEY_COLUMNS = ("station_id", "hour")
P2_TARGET_COLUMNS = (TARGET_BIKE_COLUMN, TARGET_DOCK_COLUMN)
MAX_DEEPCHECKS_SAMPLE_ROWS = 100_000
UNSUPPORTED_DATA_INTEGRITY_CHECKS = {
    "OutlierSampleDetection",
    "FeatureFeatureCorrelation",
}
EXCLUDED_TRAIN_TEST_CHECKS = {
    "IndexTrainTestLeakage",
    "TrainTestSamplesMix",
    "MultivariateDrift",
}


def _load_deepchecks() -> tuple[Any, Any, Any, str]:
    """Cargar Deepchecks con los alias retirados por las versiones fijadas."""
    import numpy

    # Deepchecks 0.19.1 aún referencia np.Inf y el alias de scorer max_error.
    numpy.__dict__.setdefault("Inf", numpy.inf)

    scorer_registry = getattr(
        importlib.import_module("sklearn.metrics._scorer"),
        "_SCORERS",
        None,
    )
    if not isinstance(scorer_registry, dict):
        raise TypeError("El registro de scorers de scikit-learn tiene un tipo inesperado.")
    if "max_error" not in scorer_registry:
        if "neg_max_error" not in scorer_registry:
            raise RuntimeError("scikit-learn no expone el scorer requerido por Deepchecks.")
        scorer_registry["max_error"] = scorer_registry["neg_max_error"]

    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity, train_test_validation

    return Dataset, data_integrity, train_test_validation, version("deepchecks")


def _validate_partition(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    partition_name: str,
) -> dict[str, object]:
    """Detenerse ante claves, etiquetas o valores no finitos inválidos."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{partition_name} debe ser un pandas.DataFrame.")
    if not data.columns.is_unique:
        raise ValueError(f"{partition_name} contiene nombres de columna duplicados.")
    required_columns = {*KEY_COLUMNS, *feature_columns, target_column}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas en {partition_name}: {sorted(missing_columns)}.")
    if data.empty:
        raise ValueError(f"{partition_name} está vacío.")

    for column in KEY_COLUMNS:
        if data[column].isna().any():
            raise ValueError(f"'{column}' contiene valores nulos en {partition_name}.")
    if not pd.api.types.is_datetime64_any_dtype(data["hour"]):
        raise TypeError(f"'hour' debe ser datetime en {partition_name}.")
    if data["hour"].dt.tz is not None:
        raise ValueError(f"'hour' debe usar timestamps sin zona horaria en {partition_name}.")
    if data.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"Hay claves station_id-hour duplicadas en {partition_name}.")

    target = data[target_column]
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        raise TypeError(f"'{target_column}' debe ser numérica en {partition_name}.")
    if target.isna().any():
        raise ValueError(f"'{target_column}' contiene etiquetas nulas en {partition_name}.")

    checked_columns = (*KEY_COLUMNS[:1], *feature_columns, target_column)
    for column in checked_columns:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        if np.isinf(values).any():
            raise ValueError(f"'{column}' contiene valores infinitos en {partition_name}.")

    columns = (*KEY_COLUMNS, *feature_columns, target_column)
    null_counts = {column: int(data[column].isna().sum()) for column in columns}
    lag_columns = tuple(column for column in feature_columns if "_lag_" in column)
    lag_nulls = {
        column: {
            "count": null_counts[column],
            "percent": round(null_counts[column] / len(data) * 100, 4),
        }
        for column in lag_columns
    }

    return {
        "rows": len(data),
        "column_types": {column: str(data[column].dtype) for column in columns},
        "null_counts": null_counts,
        "lag_feature_nulls": lag_nulls,
    }


def _sample_frame(data: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    if len(data) <= sample_rows:
        return data
    return data.sample(n=sample_rows, random_state=RANDOM_SEED)


def _categorical_features(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> list[str]:
    return [column for column in feature_columns if pd.api.types.is_bool_dtype(data[column])]


def _make_labeled_dataset(
    Dataset: Any,
    data: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    label_type: str,
    dataset_name: str,
    datetime_column: str | None = None,
) -> Any:
    columns = (*feature_columns, target_column)
    if datetime_column is not None:
        columns = (datetime_column, *columns)
    dataset = _sample_frame(data.loc[:, list(columns)], MAX_DEEPCHECKS_SAMPLE_ROWS)
    return Dataset(
        dataset,
        label=target_column,
        features=list(feature_columns),
        cat_features=_categorical_features(data, feature_columns),
        datetime_name=datetime_column,
        label_type=label_type,
        dataset_name=dataset_name,
    )


def _make_train_test_validation_suite(
    suite_factory: Any,
    label_type: str,
    sample_rows: int,
) -> tuple[Any, list[str]]:
    suite = suite_factory(n_samples=sample_rows, random_state=RANDOM_SEED)
    excluded_checks = set(EXCLUDED_TRAIN_TEST_CHECKS)
    if label_type == "regression":
        excluded_checks.add("NewLabelTrainTest")

    excluded_names = []
    for index, check in tuple(suite.checks.items()):
        if type(check).__name__ in excluded_checks:
            excluded_names.append(str(check.name()))
            suite.remove(index)

    return suite, excluded_names


def _make_data_integrity_suite(
    suite_factory: Any,
    label_type: str | None,
    sample_rows: int,
) -> tuple[Any, list[str]]:
    suite = suite_factory(n_samples=sample_rows, random_state=RANDOM_SEED)
    excluded_checks = set(UNSUPPORTED_DATA_INTEGRITY_CHECKS)
    if label_type == "regression":
        excluded_checks.add("ConflictingLabels")
    elif label_type is None:
        excluded_checks.update(
            {
                "ConflictingLabels",
                "FeatureLabelCorrelation",
                "IdentifierLabelCorrelation",
            }
        )

    excluded_names = []
    for index, check in tuple(suite.checks.items()):
        if type(check).__name__ in excluded_checks:
            excluded_names.append(str(check.name()))
            suite.remove(index)

    return suite, excluded_names


def _check_name(check_result: Any) -> str:
    header = getattr(check_result, "header", None)
    if header:
        return str(header)

    check = getattr(check_result, "check", None)
    name = getattr(check, "name", None)
    return str(name()) if callable(name) else type(check).__name__


def _condition_value(condition: Any, name: str) -> Any:
    if isinstance(condition, dict):
        return condition.get(name)
    return getattr(condition, name, None)


def _suite_findings(result: Any) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    not_passed_checks = []
    for check_result in result.get_not_passed_checks():
        conditions = []
        for condition in getattr(check_result, "conditions_results", []):
            category = _condition_value(condition, "category")
            status = getattr(category, "value", str(category))
            if status != "PASS":
                conditions.append(
                    {
                        "condition": _condition_value(condition, "name"),
                        "status": status,
                        "details": _condition_value(condition, "details"),
                    }
                )
        not_passed_checks.append({"check": _check_name(check_result), "conditions": conditions})

    errored_checks = []
    for check_result in result.results:
        error = getattr(check_result, "exception", None)
        if error is not None:
            errored_checks.append({"check": _check_name(check_result), "error": str(error)})

    return not_passed_checks, errored_checks


def _write_suite_report(
    result: Any,
    output_dir: Path,
    report_name: str,
    suite_name: str,
    sampled_rows: dict[str, int],
    excluded_checks: list[str],
) -> dict[str, object]:
    report_path = output_dir / report_name
    result.save_as_html(file=str(report_path), requirejs=False)
    not_passed_checks, errored_checks = _suite_findings(result)
    return {
        "suite": suite_name,
        "html": report_path.name,
        "partitions": list(sampled_rows),
        "sampled_rows": sampled_rows,
        "checks_excluded": excluded_checks,
        "checks_run": len(result.results),
        "checks_not_passed": len(not_passed_checks),
        "not_passed_checks": not_passed_checks,
        "checks_errored": len(errored_checks),
        "errored_checks": errored_checks,
    }


def run_validation(
    p1_input: Path = P1_INPUT_PATH,
    p2_input: Path = P2_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Generar reportes de integridad y drift sin entrenar ni seleccionar modelos."""
    for input_path, description in (
        (p1_input, "dataset horario P1"),
        (p2_input, "etiquetas train/validation P2"),
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"No se encontró el {description}: {input_path}")

    hourly_data = pd.read_parquet(p1_input)
    p1_data = prepare_supervised_data(hourly_data)
    p1_splits = split_chronologically(p1_data, evaluate_test=True)
    p1_feature_columns = tuple(
        column for column in p1_data.columns if column not in {*KEY_COLUMNS, P1_TARGET_COLUMN}
    )

    labeled_p2 = pd.read_parquet(p2_input)
    p2_splits = prepare_p2_splits(labeled_p2)

    p1_partition_summaries = {
        name: _validate_partition(
            frame,
            p1_feature_columns,
            P1_TARGET_COLUMN,
            f"P1 {name}",
        )
        for name, frame in p1_splits.items()
    }
    p2_partition_summaries: dict[str, dict[str, dict[str, object]]] = {}
    p2_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for target_column in P2_TARGET_COLUMNS:
        p2_frames[target_column] = {}
        p2_partition_summaries[target_column] = {}
        for split_name, split_data in p2_splits.items():
            columns = (*KEY_COLUMNS, *P2_PREDICTOR_COLUMNS, target_column)
            target_data = split_data.loc[:, list(columns)]
            p2_frames[target_column][split_name] = target_data
            p2_partition_summaries[target_column][split_name] = _validate_partition(
                target_data,
                P2_PREDICTOR_COLUMNS,
                target_column,
                f"P2 {target_column} {split_name}",
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset, data_integrity, train_test_validation, deepchecks_version = _load_deepchecks()
    reports = []

    for split_name, split_data in p1_splits.items():
        dataset = _make_labeled_dataset(
            Dataset,
            split_data,
            p1_feature_columns,
            P1_TARGET_COLUMN,
            "regression",
            f"P1 {split_name}",
            datetime_column="hour",
        )
        sampled_rows = min(len(split_data), MAX_DEEPCHECKS_SAMPLE_ROWS)
        suite, excluded_checks = _make_data_integrity_suite(
            data_integrity,
            "regression",
            sampled_rows,
        )
        result = suite.run(dataset, with_display=False)
        reports.append(
            _write_suite_report(
                result,
                output_dir,
                f"p1_integrity_{split_name}.html",
                "data_integrity",
                {split_name: sampled_rows},
                excluded_checks,
            )
        )

    p1_train = p1_splits["train"]
    for split_name in ("validation", "test"):
        comparison_data = p1_splits[split_name]
        sample_rows = min(
            len(p1_train),
            len(comparison_data),
            MAX_DEEPCHECKS_SAMPLE_ROWS,
        )
        train_dataset = _make_labeled_dataset(
            Dataset,
            p1_train,
            p1_feature_columns,
            P1_TARGET_COLUMN,
            "regression",
            "P1 train",
            datetime_column="hour",
        )
        comparison_dataset = _make_labeled_dataset(
            Dataset,
            comparison_data,
            p1_feature_columns,
            P1_TARGET_COLUMN,
            "regression",
            f"P1 {split_name}",
            datetime_column="hour",
        )
        suite, excluded_checks = _make_train_test_validation_suite(
            train_test_validation,
            "regression",
            sample_rows,
        )
        result = suite.run(train_dataset, comparison_dataset, with_display=False)
        reports.append(
            _write_suite_report(
                result,
                output_dir,
                f"p1_train_vs_{split_name}.html",
                "train_test_validation",
                {"train": sample_rows, split_name: sample_rows},
                excluded_checks,
            )
        )

    p2_integrity_columns = (*P2_PREDICTOR_COLUMNS, *P2_TARGET_COLUMNS)
    p2_categorical = [
        column
        for column in P2_PREDICTOR_COLUMNS
        if pd.api.types.is_bool_dtype(p2_splits["train"][column])
    ]
    p2_categorical.extend(P2_TARGET_COLUMNS)
    for split_name, split_data in p2_splits.items():
        integrity_data = _sample_frame(
            split_data.loc[:, ["hour", *p2_integrity_columns]],
            MAX_DEEPCHECKS_SAMPLE_ROWS,
        )
        dataset = Dataset(
            integrity_data,
            features=list(p2_integrity_columns),
            cat_features=p2_categorical,
            datetime_name="hour",
            dataset_name=f"P2 {split_name}",
        )
        sampled_rows = len(integrity_data)
        suite, excluded_checks = _make_data_integrity_suite(
            data_integrity,
            None,
            sampled_rows,
        )
        result = suite.run(dataset, with_display=False)
        reports.append(
            _write_suite_report(
                result,
                output_dir,
                f"p2_integrity_{split_name}.html",
                "data_integrity",
                {split_name: sampled_rows},
                excluded_checks,
            )
        )

    for target_column in P2_TARGET_COLUMNS:
        train_data = p2_frames[target_column]["train"]
        validation_data = p2_frames[target_column]["validation"]
        sample_rows = min(
            len(train_data),
            len(validation_data),
            MAX_DEEPCHECKS_SAMPLE_ROWS,
        )
        train_dataset = _make_labeled_dataset(
            Dataset,
            train_data,
            P2_PREDICTOR_COLUMNS,
            target_column,
            "binary",
            f"P2 {target_column} train",
            datetime_column="hour",
        )
        validation_dataset = _make_labeled_dataset(
            Dataset,
            validation_data,
            P2_PREDICTOR_COLUMNS,
            target_column,
            "binary",
            f"P2 {target_column} validation",
            datetime_column="hour",
        )
        suite, excluded_checks = _make_train_test_validation_suite(
            train_test_validation,
            "binary",
            sample_rows,
        )
        result = suite.run(train_dataset, validation_dataset, with_display=False)
        target_name = "bike" if target_column == TARGET_BIKE_COLUMN else "dock"
        reports.append(
            _write_suite_report(
                result,
                output_dir,
                f"p2_{target_name}_train_vs_validation.html",
                "train_test_validation",
                {"train": sample_rows, "validation": sample_rows},
                excluded_checks,
            )
        )

    summary: dict[str, object] = {
        "deepchecks_version": deepchecks_version,
        "sample_limit_rows": MAX_DEEPCHECKS_SAMPLE_ROWS,
        "test_policy": {
            "p1_test_used_for": ["integrity", "distribution_drift"],
            "p1_test_used_for_training": False,
            "p1_test_used_for_model_selection": False,
            "p2_test_loaded": False,
        },
        "partitions": {
            "p1": p1_partition_summaries,
            "p2": p2_partition_summaries,
        },
        "p2_predictors_by_target": {
            target_column: list(P2_PREDICTOR_COLUMNS) for target_column in P2_TARGET_COLUMNS
        },
        "reports": reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.success(f"Reportes HTML y resumen JSON guardados en: {output_dir}")
    return summary


@app.command()
def main(
    p1_input: Annotated[
        Path,
        typer.Option("--p1-input", help="Parquet horario completo de P1."),
    ] = P1_INPUT_PATH,
    p2_input: Annotated[
        Path,
        typer.Option("--p2-input", help="Parquet de etiquetas train/validation de P2."),
    ] = P2_INPUT_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directorio de reportes HTML y summary.json."),
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    run_validation(p1_input=p1_input, p2_input=p2_input, output_dir=output_dir)


if __name__ == "__main__":
    app()
