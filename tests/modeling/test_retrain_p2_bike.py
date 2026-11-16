"""Pruebas del reentrenamiento temporal y comparación pareada P2."""

from types import SimpleNamespace

import numpy as np
import pytest

from mibici.modeling import retrain_p2_bike
from mibici.modeling.p2_targets import VALIDATION_END
from tests.factories import make_labeled_data


def test_training_rows_excludes_test_before_reading_its_invalid_values():
    labeled = make_labeled_data(include_test=True)
    training = retrain_p2_bike.training_rows(labeled)

    assert set(training["partition"]) == {"train", "validation"}
    assert len(training) == len(labeled) - 1
    assert training["hour"].max() < VALIDATION_END


def test_training_rows_rejects_mislabeled_test_date():
    labeled = make_labeled_data(include_test=False)
    labeled.loc[0, "hour"] = VALIDATION_END
    with pytest.raises(ValueError, match="fuera del rango"):
        retrain_p2_bike.training_rows(labeled)


def test_comparison_uses_same_rows_and_reports_deltas(monkeypatch):
    current = make_labeled_data(include_test=False).iloc[:4].copy()
    current["target_bike_pressure"] = [0, 1, 0, 1]
    seen = []

    class Model:
        classes_ = np.array([0, 1])

        def __init__(self, probabilities):
            self.probabilities = probabilities

        def predict_proba(self, features):
            seen.append(features.copy())
            return np.column_stack((1 - self.probabilities, self.probabilities))

    a = Model(np.array([0.1, 0.6, 0.8, 0.4]))
    b = Model(np.array([0.1, 0.8, 0.2, 0.7]))
    monkeypatch.setattr(
        retrain_p2_bike.mlflow,
        "sklearn",
        SimpleNamespace(load_model=lambda uri: a if uri.endswith("/1") else b),
    )

    rows = retrain_p2_bike.comparison_rows("2", "run-b", current)

    np.testing.assert_array_equal(seen[0], seen[1])
    assert rows[0]["n_evaluated"] == rows[1]["n_evaluated"] == 4
    assert rows[1]["delta_recall"] == pytest.approx(0.5)
    assert rows[1]["relative_recall"] == pytest.approx(1.0)
