import json
import os
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import warnings

from fastapi.testclient import TestClient
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from mibici.api import main
from mibici.modeling.p2_train import PREDICTOR_COLUMNS


@pytest.fixture
def payload():
    return {
        "hour_of_day": 8,
        "day_of_week": 2,
        "month": 11,
        "is_weekend": False,
        "departures_lag_1h": 1.0,
        "departures_lag_24h": 2.0,
        "departures_lag_168h": 1.0,
        "arrivals_lag_1h": 1.0,
        "arrivals_lag_24h": 2.0,
        "arrivals_lag_168h": 1.0,
        "net_flow_lag_1h": -1.0,
        "net_flow_lag_24h": 0.0,
        "net_flow_lag_168h": 1.0,
        "departures_rolling_mean_24h": 1.5,
        "departures_rolling_mean_168h": 1.2,
    }


@pytest.fixture
def client(monkeypatch):
    bike = MagicMock(spec=["classes_", "predict", "predict_proba"])
    bike.classes_ = np.array([1, 0])
    bike.predict_proba.return_value = np.array([[0.7, 0.3]])
    bike.predict.return_value = np.array([1])
    dock = MagicMock(spec=["classes_", "predict", "predict_proba"])
    dock.classes_ = np.array([0, 1])
    dock.predict_proba.return_value = np.array([[0.8, 0.2]])
    dock.predict.return_value = np.array([0])
    loader = MagicMock(side_effect=[bike, dock])
    monkeypatch.setattr(main.mlflow.sklearn, "load_model", loader)
    main.models.clear()
    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        yield test_client, loader, bike, dock
    assert not main.models


def test_health(client):
    response = client[0].get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "models_loaded": ["bike", "dock"]}


def test_predict_contract_and_positive_class(client, payload):
    response = client[0].post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "predictions": {
            "bike": {"prediction": 1, "probability": 0.7},
            "dock": {"prediction": 0, "probability": 0.2},
        },
        "decision_threshold": 0.5,
    }
    for prediction in response.json()["predictions"].values():
        assert type(prediction["prediction"]) is int
        assert type(prediction["probability"]) is float
        assert 0 <= prediction["probability"] <= 1


def test_input_schema_matches_training_columns():
    assert tuple(main.PredictionInput.model_fields) == tuple(PREDICTOR_COLUMNS)
    assert len(PREDICTOR_COLUMNS) == 15


def test_predict_uses_training_order_and_float32(client, payload):
    response = client[0].post("/predict", json=dict(reversed(list(payload.items()))))
    assert response.status_code == 200
    expected = np.array([[payload[column] for column in PREDICTOR_COLUMNS]], dtype=np.float32)
    for model in client[2:]:
        features = model.predict_proba.call_args.args[0]
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        np.testing.assert_array_equal(features, expected)
        np.testing.assert_array_equal(model.predict.call_args.args[0], expected)


@pytest.mark.parametrize("hour,day,month", [(0, 0, 1), (23, 6, 12)])
def test_valid_boundaries(client, payload, hour, day, month):
    payload.update(hour_of_day=hour, day_of_week=day, month=month, is_weekend=day >= 5)
    for column in PREDICTOR_COLUMNS[4:]:
        payload[column] = -10.0 if column.startswith("net_flow") else 0.0
    assert client[0].post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("hour_of_day", -1),
        ("hour_of_day", 24),
        ("day_of_week", -1),
        ("day_of_week", 7),
        ("month", 0),
        ("month", 13),
        ("hour_of_day", "8"),
        ("hour_of_day", True),
        ("hour_of_day", 8.5),
        ("is_weekend", "false"),
        ("is_weekend", 1),
        ("departures_lag_1h", "1.0"),
        ("arrivals_lag_1h", True),
        ("net_flow_lag_1h", []),
    ],
)
def test_rejects_invalid_ranges_and_types(client, payload, field, value):
    payload[field] = value
    response = client[0].post("/predict", json=payload)
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])
    client[2].predict_proba.assert_not_called()


@pytest.mark.parametrize(
    "field", [column for column in PREDICTOR_COLUMNS[4:] if not column.startswith("net_flow")]
)
def test_rejects_negative_counts_and_means(client, payload, field):
    payload[field] = -0.1
    assert client[0].post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", PREDICTOR_COLUMNS)
def test_rejects_missing_required_field(client, payload, field):
    del payload[field]
    assert client[0].post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", PREDICTOR_COLUMNS)
def test_rejects_null(client, payload, field):
    payload[field] = None
    assert client[0].post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", PREDICTOR_COLUMNS[4:])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), "NaN", "Infinity", "-Infinity"]
)
def test_rejects_nonfinite_values(client, payload, field, value):
    payload[field] = value
    response = client[0].post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    client[2].predict_proba.assert_not_called()


def test_rejects_unknown_fields(client, payload):
    payload["station_id"] = 42
    assert client[0].post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("value", [1e100, -1e100])
def test_rejects_float32_overflow(client, payload, value):
    payload["net_flow_lag_1h"] = value
    assert client[0].post("/predict", json=payload).status_code == 422
    client[2].predict_proba.assert_not_called()


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_probability_boundaries(client, payload, probability):
    client[2].predict_proba.return_value = np.array([[probability, 1 - probability]])
    client[2].predict.return_value = np.array([int(probability)])
    response = client[0].post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predictions"]["bike"]["probability"] == probability


@pytest.mark.parametrize("missing", [("bike",), ("dock",), ("bike", "dock")])
def test_missing_models_return_503(client, payload, missing):
    for name in missing:
        main.models.pop(name)
    assert client[0].post("/predict", json=payload).status_code == 503
    assert client[0].get("/health").status_code == 503


def test_models_are_loaded_once_per_lifespan(client, payload):
    for _ in range(3):
        assert client[0].post("/predict", json=payload).status_code == 200
        assert client[0].get("/health").status_code == 200
    assert client[1].call_count == 2
    assert [call.args[0] for call in client[1].call_args_list] == [
        main.BIKE_MODEL_URI,
        main.DOCK_MODEL_URI,
    ]


def test_openapi_and_docs(client):
    response = client[0].get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    request_schema = schema["components"]["schemas"]["PredictionInput"]
    assert set(request_schema["required"]) == set(PREDICTOR_COLUMNS)
    assert request_schema["additionalProperties"] is False
    output = schema["paths"]["/predict"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert output["$ref"].endswith("/PredictionResponse")
    assert client[0].get("/docs").status_code == 200


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_model_output_is_not_returned(client, payload, probability):
    client[2].predict_proba.return_value = np.array([[probability, 0.0]])
    response = client[0].post("/predict", json=payload)
    assert response.status_code == 500
    assert "Traceback" not in response.text


def test_inference_error_does_not_expose_internal_details(client, payload):
    client[2].predict_proba.side_effect = ValueError("private-artifact-path")
    response = client[0].post("/predict", json=payload)
    assert response.status_code == 500
    assert "private-artifact-path" not in response.text


def test_preserves_sklearn_tie_prediction(client, payload):
    client[2].predict_proba.return_value = np.array([[0.5, 0.5]])
    client[2].predict.return_value = np.array([0])
    response = client[0].post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predictions"]["bike"] == {"prediction": 0, "probability": 0.5}


def test_named_estimators_receive_ordered_dataframe(client, payload):
    client[2].feature_names_in_ = np.array(PREDICTOR_COLUMNS)
    assert client[0].post("/predict", json=payload).status_code == 200
    features = client[2].predict_proba.call_args.args[0]
    assert isinstance(features, pd.DataFrame)
    assert tuple(features.columns) == tuple(PREDICTOR_COLUMNS)


def test_failed_startup_clears_partial_models(monkeypatch):
    loader = MagicMock(side_effect=[object(), FileNotFoundError("missing model")])
    monkeypatch.setattr(main.mlflow.sklearn, "load_model", loader)
    with pytest.raises(FileNotFoundError, match="missing model"), TestClient(main.app):
        pass
    assert not main.models


def test_documented_request_is_valid():
    example = Path(__file__).resolve().parents[2] / "docs" / "examples" / "p2-request.json"
    assert tuple(
        main.PredictionInput.model_validate_json(example.read_text()).model_dump()
    ) == tuple(PREDICTOR_COLUMNS)


@pytest.mark.skipif(
    os.getenv("MIBICI_API_CONTAINER") != "1",
    reason="Requires the API container listening on MIBICI_API_URL (default localhost:8000)",
)
def test_live_container_http_contract(payload):
    base_url = os.getenv("MIBICI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    with urlopen(f"{base_url}/health", timeout=30) as response:
        assert response.status == 200
        assert json.load(response) == {"status": "healthy", "models_loaded": ["bike", "dock"]}
    request = Request(
        f"{base_url}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        assert response.status == 200
        result = main.PredictionResponse.model_validate_json(response.read())
        assert result.decision_threshold == 0.5
    with urlopen(f"{base_url}/openapi.json", timeout=30) as response:
        assert "/predict" in json.load(response)["paths"]
    for changed in ({"hour_of_day": 24}, {"unknown": 1}, {"net_flow_lag_1h": float("nan")}):
        request.data = json.dumps({**payload, **changed}).encode()
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=30)
        assert error.value.code == 422
        assert json.load(error.value)["detail"]


@pytest.mark.skipif(
    os.getenv("MIBICI_API_INTEGRATION") != "1",
    reason="Requires the real Registry and models exported with make export-models",
)
def test_real_registry_export_and_api_agree(payload, monkeypatch):
    root = Path(__file__).resolve().parents[2] / "models"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    registry = mlflow.MlflowClient()
    monkeypatch.setattr(main, "BIKE_MODEL_URI", "models:/mibici-p2-bike-pressure@champion")
    monkeypatch.setattr(main, "DOCK_MODEL_URI", "models:/mibici-p2-dock-pressure@champion")
    expected_names = {"mibici-p2-bike-pressure", "mibici-p2-dock-pressure"}
    assert {entry["registered_model"] for entry in metadata["models"]} == expected_names
    features = np.array([[payload[column] for column in PREDICTOR_COLUMNS]], dtype=np.float32)
    with warnings.catch_warnings(), TestClient(main.app) as real_client:
        warnings.filterwarnings("error", message="X has feature names")
        response = real_client.post("/predict", json=payload)
        assert response.status_code == 200
        assert real_client.get("/health").status_code == 200
        for entry in metadata["models"]:
            name = entry["registered_model"]
            version = registry.get_model_version_by_alias(name, "champion")
            assert str(entry["version"]) == str(version.version)
            assert entry["run_id"] == version.run_id
            run = registry.get_run(version.run_id)
            assert float(run.data.params["decision_threshold"]) == 0.5
            assert run.data.params["predictor_columns"].split(",") == list(PREDICTOR_COLUMNS)
            local_path = root / entry["path"]
            signature = mlflow.models.Model.load(str(local_path / "MLmodel")).signature
            assert signature.inputs.to_dict() == [
                {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": (-1, 15)}}
            ]
            local_model = mlflow.sklearn.load_model(str(local_path))
            with warnings.catch_warnings(record=True) as legacy_warnings:
                warnings.simplefilter("always")
                legacy_probabilities = local_model.predict_proba(
                    pd.DataFrame([payload], columns=PREDICTOR_COLUMNS)
                )
            assert any("feature names" in str(warning.message) for warning in legacy_warnings)
            np.testing.assert_array_equal(
                legacy_probabilities, local_model.predict_proba(features)
            )
            key = "bike" if name == "mibici-p2-bike-pressure" else "dock"
            registered_model = main.models[key]
            np.testing.assert_array_equal(
                local_model.predict_proba(features), registered_model.predict_proba(features)
            )
            positive_index = list(local_model.classes_).index(1)
            result = response.json()["predictions"][key]
            assert result["probability"] == float(
                local_model.predict_proba(features)[0, positive_index]
            )
            assert result["prediction"] == int(local_model.predict(features)[0])
