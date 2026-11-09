from contextlib import asynccontextmanager
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mibici.modeling.p2_train import DECISION_THRESHOLD, PREDICTOR_COLUMNS

# Configuración de MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

# Ubicación de los modelos. Por defecto se cargan del Model Registry (alias champion);
# la imagen de contenedor apunta estas variables a los modelos empaquetados en /app/models.
BIKE_MODEL_URI = os.getenv("BIKE_MODEL_URI", "models:/mibici-p2-bike-pressure@champion")
DOCK_MODEL_URI = os.getenv("DOCK_MODEL_URI", "models:/mibici-p2-dock-pressure@champion")

models: dict[str, Any] = {}


# Variables de entrada
class PredictionInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: bool

    departures_lag_1h: float = Field(ge=0)
    departures_lag_24h: float = Field(ge=0)
    departures_lag_168h: float = Field(ge=0)

    arrivals_lag_1h: float = Field(ge=0)
    arrivals_lag_24h: float = Field(ge=0)
    arrivals_lag_168h: float = Field(ge=0)

    net_flow_lag_1h: float
    net_flow_lag_24h: float
    net_flow_lag_168h: float

    departures_rolling_mean_24h: float = Field(ge=0)
    departures_rolling_mean_168h: float = Field(ge=0)

    @field_validator(*PREDICTOR_COLUMNS[4:])
    @classmethod
    def validate_float32_range(cls, value: float) -> float:
        if abs(value) > float(np.finfo(np.float32).max):
            raise ValueError("El valor debe ser representable como float32.")
        return value


class ModelPrediction(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    prediction: Literal[0, 1]
    probability: float = Field(ge=0, le=1)


class Predictions(BaseModel):
    bike: ModelPrediction
    dock: ModelPrediction


class PredictionResponse(BaseModel):
    predictions: Predictions
    decision_threshold: Literal[0.5] = DECISION_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.clear()
    try:
        models.update(
            bike=mlflow.sklearn.load_model(BIKE_MODEL_URI),
            dock=mlflow.sklearn.load_model(DOCK_MODEL_URI),
        )
        yield
    finally:
        models.clear()


app = FastAPI(
    title="MiBici Pressure Prediction API",
    description="API para predecir presión de bicicletas y docks.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    errors = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


def require_models() -> None:
    if any(models.get(name) is None for name in ("bike", "dock")):
        raise HTTPException(status_code=503, detail="Los modelos no están disponibles.")


@app.get("/health", responses={503: {"description": "Modelos no disponibles"}})
def health():
    require_models()
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"description": "Modelos no disponibles"}},
)
def predict(data: PredictionInput):
    require_models()

    features = pd.DataFrame(
        [data.model_dump()],
        columns=PREDICTOR_COLUMNS,
    )

    predictions = {}

    for name in ("bike", "dock"):
        model = models[name]
        model_features = (
            features
            if hasattr(model, "feature_names_in_")
            else features.to_numpy(dtype=np.float32)
        )
        probabilities = model.predict_proba(model_features)[0]
        positive_index = list(model.classes_).index(1)

        predictions[name] = {
            "prediction": int(model.predict(model_features)[0]),
            "probability": float(probabilities[positive_index]),
        }

    return {
        "predictions": predictions,
        "decision_threshold": DECISION_THRESHOLD,
    }
