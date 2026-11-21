
"""Load testing for the MiBici prediction API using Locust."""

from locust import HttpUser, between, task


PREDICTION_PAYLOAD = {
    "hour_of_day": 8,
    "day_of_week": 2,
    "month": 11,
    "is_weekend": False,
    "departures_lag_1h": 12.0,
    "departures_lag_24h": 15.0,
    "departures_lag_168h": 10.0,
    "arrivals_lag_1h": 9.0,
    "arrivals_lag_24h": 11.0,
    "arrivals_lag_168h": 8.0,
    "net_flow_lag_1h": 3.0,
    "net_flow_lag_24h": 4.0,
    "net_flow_lag_168h": 2.0,
    "departures_rolling_mean_24h": 13.5,
    "departures_rolling_mean_168h": 11.2,
}


class MiBiciUser(HttpUser):
    """Simulates clients requesting bike and dock pressure predictions."""

    wait_time = between(1, 2)

    @task
    def predict(self):
        with self.client.post(
            "/predict",
            json=PREDICTION_PAYLOAD,
            name="POST /predict",
            catch_response=True,
            timeout=30,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Unexpected HTTP status: {response.status_code}"
                )
                return

            try:
                body = response.json()
                predictions = body["predictions"]

                for model_name in ("bike", "dock"):
                    result = predictions[model_name]
                    prediction = result["prediction"]
                    probability = result["probability"]

                    if type(prediction) is not int or prediction not in (0, 1):
                        raise ValueError(
                            f"Invalid prediction for {model_name}: {prediction}"
                        )

                    if (
                        type(probability) not in (int, float)
                        or not 0 <= probability <= 1
                    ):
                        raise ValueError(
                            f"Invalid probability for {model_name}: {probability}"
                        )

                if body["decision_threshold"] != 0.5:
                    raise ValueError("Unexpected decision threshold")

            except (ValueError, KeyError, TypeError) as error:
                response.failure(f"Invalid prediction response: {error}")
