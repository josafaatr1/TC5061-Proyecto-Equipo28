"""Utilities for loading and serving registered MLflow models."""

import os

from loguru import logger
import mlflow
import mlflow.pyfunc
import pandas as pd

DEFAULT_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/mibici-p1-departures@champion",
)


def load_model(model_uri: str = DEFAULT_MODEL_URI):
    """Load a registered MLflow model for inference.

    Args:
        model_uri: MLflow model URI, including its version or alias.

    Returns:
        Loaded MLflow PyFunc model.

    Raises:
        mlflow.exceptions.MlflowException: If the model cannot be loaded.
    """
    logger.info(f"Loading MLflow model: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    logger.success(f"MLflow model loaded successfully: {model_uri}")
    return model


def predict(model, features: pd.DataFrame) -> list[float]:
    """Generate predictions using a loaded MLflow model.

    Args:
        model: MLflow PyFunc model returned by load_model().
        features: DataFrame containing the model's input features.

    Returns:
        Predicted departures as a list of floats.

    Raises:
        TypeError: If features is not a pandas DataFrame.
        mlflow.exceptions.MlflowException: If the input violates
            the registered model's signature.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    predictions = model.predict(features)

    return [float(value) for value in predictions]
