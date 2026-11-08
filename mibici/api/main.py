
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mibici.modeling.p2_train import PREDICTOR_COLUMNS


# Configuración de MLflow
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

BIKE_MODEL_URI = "models:/mibici-p2-bike-pressure@champion"
DOCK_MODEL_URI = "models:/mibici-p2-dock-pressure@champion"

models = {}


# Variables de entrada
class PredictionInput(BaseModel):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["bike"] = mlflow.sklearn.load_model(BIKE_MODEL_URI)
    models["dock"] = mlflow.sklearn.load_model(DOCK_MODEL_URI)
    yield
    models.clear()


app = FastAPI(
    title="MiBici Pressure Prediction API",
    description="API para predecir presión de bicicletas y docks.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Los modelos no están disponibles.",
        )

    features = pd.DataFrame(
        [data.model_dump()],
        columns=PREDICTOR_COLUMNS,
    )

    predictions = {}

    for name, model in models.items():
        probabilities = model.predict_proba(features)[0]
        positive_index = list(model.classes_).index(1)

        predictions[name] = {
            "prediction": int(model.predict(features)[0]),
            "probability": float(probabilities[positive_index]),
        }

    return {
        "predictions": predictions,
        "decision_threshold": 0.5,
    }
