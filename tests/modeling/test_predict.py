"""Unit tests for MLflow model loading and inference."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mibici.modeling.predict import DEFAULT_MODEL_URI, load_model, predict


@patch("mibici.modeling.predict.mlflow.pyfunc.load_model")
def test_load_model_uses_champion_alias(mock_load_model):
    """The default model URI must reference the champion alias."""
    expected_model = MagicMock()
    mock_load_model.return_value = expected_model

    result = load_model()

    mock_load_model.assert_called_once_with(DEFAULT_MODEL_URI)
    assert result is expected_model


@patch("mibici.modeling.predict.mlflow.pyfunc.load_model")
def test_load_model_accepts_custom_uri(mock_load_model):
    """A specific model version can be loaded explicitly."""
    model_uri = "models:/mibici-p1-departures/1"

    load_model(model_uri)

    mock_load_model.assert_called_once_with(model_uri)


def test_predict_returns_python_floats():
    """Predictions must be JSON-serializable Python floats."""
    model = MagicMock()
    model.predict.return_value = np.array([0.5, 1.25])

    features = pd.DataFrame({"hour_of_day": [8, 9]})

    result = predict(model, features)

    assert result == [0.5, 1.25]
    assert all(isinstance(value, float) for value in result)
    model.predict.assert_called_once_with(features)


def test_predict_rejects_non_dataframe():
    """Inference requires a pandas DataFrame."""
    model = MagicMock()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        predict(model, {"hour_of_day": 8})

    model.predict.assert_not_called()


def test_predict_propagates_model_errors():
    """Model inference errors must not be silently suppressed."""
    model = MagicMock()
    model.predict.side_effect = ValueError("Invalid model input")

    features = pd.DataFrame({"hour_of_day": [8]})

    with pytest.raises(ValueError, match="Invalid model input"):
        predict(model, features)
