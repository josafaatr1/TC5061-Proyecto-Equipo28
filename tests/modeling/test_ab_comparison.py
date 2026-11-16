"""Controles de procedencia para comparar variantes P2."""

from copy import deepcopy

import pytest

from mibici.modeling.ab_comparison import validate_configs
from mibici.modeling.p2_train import PREDICTOR_COLUMNS


def _config():
    return {
        "tags": {
            "question": "P2",
            "objective": "bike_pressure",
            "model_family": "hist_gradient_boosting",
            "train_validation_data_version": "datos",
            "catalog_sha256": "catalogo",
            "label_artifact_sha256": "etiquetas",
        },
        "params": {
            "train_period": "train",
            "validation_period": "validation",
            "predictor_columns": ",".join(PREDICTOR_COLUMNS),
            "decision_threshold": 0.5,
            "bike_pressure_ratio_threshold": 0.3,
            "dock_pressure_ratio_threshold": 0.4,
        },
    }


def _provenance():
    return {
        "train_validation_data_version": "datos",
        "catalog_sha256": "catalogo",
        "label_artifact_sha256": "etiquetas",
        "label_pressure_thresholds_train_only": {
            "bike_pressure_ratio": 0.3,
            "dock_pressure_ratio": 0.4,
        },
    }


def test_validate_configs_accepts_matching_provenance():
    config = _config()
    thresholds = validate_configs([config, deepcopy(config)], "bike_pressure", _provenance())
    assert thresholds.bike_pressure_ratio == 0.3
    assert thresholds.dock_pressure_ratio == 0.4


def test_validate_configs_rejects_mixed_labels():
    config = _config()
    other = deepcopy(config)
    other["tags"]["label_artifact_sha256"] = "otras"
    with pytest.raises(ValueError, match="label_artifact_sha256"):
        validate_configs([config, other], "bike_pressure", _provenance())


def test_validate_configs_rejects_changed_local_thresholds():
    config = _config()
    provenance = _provenance()
    provenance["label_pressure_thresholds_train_only"]["bike_pressure_ratio"] = 0.8
    with pytest.raises(ValueError, match="umbrales"):
        validate_configs([config, deepcopy(config)], "bike_pressure", provenance)
