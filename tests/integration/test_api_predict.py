"""Integration tests for the FastAPI pricing API."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    ModelState,
    get_model_state,
)
from src.api.main import app
from src.utils.config import load_config

CLEAN_FEATURES = [
    "price",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "season",
    "days_from_start",
    "temperature_mean",
    "precipitation_sum",
    "wind_speed_max",
    "is_hot_day",
    "is_cold_day",
    "is_rainy_day",
    "is_holiday",
    "days_until_holiday",
    "near_holiday",
    "room_type_encoded",
    "beds",
    "bathrooms",
    "amenity_score",
    "review_score",
    "location_cluster",
]


class FakeDemandForecaster:
    """Minimal deterministic forecaster for API contract tests."""

    def __init__(self):
        self.model = object()
        self.feature_names = CLEAN_FEATURES
        self.split_metadata = {
            "train_start": "2023-03-06",
        }
        self.metrics = {
            "holdout_auc": 0.689724,
            "holdout_pr_auc": 0.788364,
            "holdout_log_loss": 0.604016,
            "holdout_brier": 0.209294,
            "cv_auc_mean": 0.726742,
        }
        self.last_features = None

    def predict_single(self, features):
        self.last_features = features
        return 0.65

    def get_feature_importance(self):
        return pd.DataFrame(
            {
                "feature": CLEAN_FEATURES,
                "importance": [
                    1.0 / len(CLEAN_FEATURES)
                ] * len(CLEAN_FEATURES),
            }
        )


@pytest.fixture
def model_state():
    state = ModelState(
        demand_forecaster=FakeDemandForecaster(),
        config=load_config(),
        model_version="test-v0.2.0",
        is_loaded=True,
        neighborhood_prices={
            "Williamsburg": 180.0,
            "Harlem": 140.0,
        },
        neighborhood_location_clusters={
            "Williamsburg": 2,
            "Harlem": 4,
        },
        global_reference_price=160.0,
    )
    return state


@pytest.fixture
def client(model_state):
    """Use dependency overrides so API tests do not need model artifacts."""
    app.dependency_overrides[
        get_model_state
    ] = lambda: model_state

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


VALID_PAYLOAD = {
    "room_type": "Entire home/apt",
    "beds": 2,
    "bathrooms": 1.0,
    "neighborhood": "Williamsburg",
    "checkin_date": "2024-07-15",
    "checkout_date": "2024-07-18",
    "amenity_score": 0.7,
    "review_score": 4.5,
}


class TestPredictEndpoint:
    def test_valid_prediction(self, client):
        response = client.post(
            "/predict",
            json=VALID_PAYLOAD,
        )
        assert response.status_code == 200

        data = response.json()
        assert "recommended_price" in data
        assert "scenario_results" in data
        assert "unavailability_probability" in data
        assert data["recommended_price"] > 0
        assert (
            0
            <= data["unavailability_probability"]
            <= 1
        )
        assert data["selected_scenario"] == "moderate"

    def test_reference_price_is_used_by_model(
        self,
        client,
        model_state,
    ):
        payload = {
            **VALID_PAYLOAD,
            "reference_price": 225.0,
        }
        response = client.post(
            "/predict",
            json=payload,
        )
        assert response.status_code == 200
        assert response.json()["reference_price"] == 225.0
        assert (
            model_state
            .demand_forecaster
            .last_features["price"]
            == 225.0
        )

    def test_neighborhood_median_used_when_price_missing(
        self,
        client,
    ):
        response = client.post(
            "/predict",
            json=VALID_PAYLOAD,
        )
        data = response.json()
        assert data["reference_price"] == 180.0
        assert (
            data["reference_price_source"]
            == "neighborhood_median"
        )

    def test_response_has_no_elasticity_claim(
        self,
        client,
    ):
        response = client.post(
            "/predict",
            json=VALID_PAYLOAD,
        )
        data = response.json()
        assert "elasticity_coeff" not in data
        assert "expected_revenue" not in data
        assert "price_range" not in data

    def test_all_sensitivity_scenarios_returned(
        self,
        client,
    ):
        response = client.post(
            "/predict",
            json=VALID_PAYLOAD,
        )
        names = {
            row["scenario_name"]
            for row in response.json()[
                "scenario_results"
            ]
        }
        assert names == {
            "low",
            "moderate",
            "high",
        }

    def test_missing_fields_returns_422(
        self,
        client,
    ):
        response = client.post(
            "/predict",
            json={
                "room_type": "Private room",
            },
        )
        assert response.status_code == 422

    def test_invalid_beds_returns_422(
        self,
        client,
    ):
        payload = {
            **VALID_PAYLOAD,
            "beds": 0,
        }
        response = client.post(
            "/predict",
            json=payload,
        )
        assert response.status_code == 422

    def test_invalid_review_score_returns_422(
        self,
        client,
    ):
        payload = {
            **VALID_PAYLOAD,
            "review_score": 10.0,
        }
        response = client.post(
            "/predict",
            json=payload,
        )
        assert response.status_code == 422

    def test_different_room_types(
        self,
        client,
    ):
        for room_type in [
            "Entire home/apt",
            "Private room",
            "Shared room",
        ]:
            payload = {
                **VALID_PAYLOAD,
                "room_type": room_type,
            }
            response = client.post(
                "/predict",
                json=payload,
            )
            assert response.status_code == 200


class TestHealthEndpoint:
    def test_health_returns_200(
        self,
        client,
    ):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True


class TestMetricsEndpoint:
    def test_metrics_returns_real_holdout_metrics(
        self,
        client,
    ):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        assert data["holdout_auc"] == pytest.approx(
            0.689724
        )
        assert "holdout_pr_auc" in data
        assert "holdout_log_loss" in data
        assert "holdout_brier" in data
        assert "total_predictions" in data
        assert "elasticity_coeff" not in data


class TestExplainEndpoint:
    def test_explain_returns_truthful_schema(
        self,
        client,
    ):
        payload = {
            "room_type": "Private room",
            "beds": 1,
            "bathrooms": 1.0,
            "neighborhood": "Harlem",
            "checkin_date": "2024-08-01",
            "amenity_score": 0.5,
            "review_score": 4.0,
            "reference_price": 150.0,
        }

        response = client.post(
            "/explain",
            json=payload,
        )

        assert response.status_code == 200
        data = response.json()
        assert "shap_values" in data
        assert "predicted_unavailability" in data
        assert data["reference_price"] == 150.0


class TestDriftEndpoint:
    def test_drift_report_returns_200(
        self,
        client,
    ):
        response = client.get(
            "/drift-report"
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "drift_score" in data
