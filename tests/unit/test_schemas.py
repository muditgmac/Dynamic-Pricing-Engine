"""Unit tests for Pydantic API schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    HealthResponse,
    PricingRequest,
    PricingResponse,
    ScenarioSummary,
)


class TestPricingRequest:
    def test_valid_request(self):
        req = PricingRequest(
            room_type="Entire home/apt",
            beds=2,
            bathrooms=1.0,
            neighborhood="Williamsburg",
            checkin_date="2024-07-15",
            checkout_date="2024-07-18",
        )
        assert req.beds == 2
        assert req.amenity_score == 0.5
        assert req.sensitivity_scenario == "moderate"
        assert req.reference_price is None

    def test_rejects_zero_beds(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=0,
                bathrooms=1.0,
                neighborhood="SoHo",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
            )

    def test_rejects_negative_bathrooms(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Shared room",
                beds=1,
                bathrooms=-1,
                neighborhood="Harlem",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
            )

    def test_rejects_review_score_out_of_range(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=1,
                bathrooms=1.0,
                neighborhood="Chelsea",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
                review_score=6.0,
            )

    def test_rejects_amenity_score_out_of_range(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=1,
                bathrooms=1.0,
                neighborhood="Chelsea",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
                amenity_score=1.5,
            )

    def test_rejects_invalid_sensitivity_scenario(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=1,
                bathrooms=1.0,
                neighborhood="Chelsea",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
                sensitivity_scenario="extreme",
            )

    def test_rejects_checkout_before_checkin(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=1,
                bathrooms=1.0,
                neighborhood="Chelsea",
                checkin_date="2024-07-18",
                checkout_date="2024-07-15",
            )

    def test_rejects_non_positive_reference_price(self):
        with pytest.raises(ValidationError):
            PricingRequest(
                room_type="Private room",
                beds=1,
                bathrooms=1.0,
                neighborhood="Chelsea",
                checkin_date="2024-07-15",
                checkout_date="2024-07-18",
                reference_price=0,
            )


class TestPricingResponse:
    def test_valid_response(self):
        scenario = ScenarioSummary(
            scenario_name="moderate",
            sensitivity=1.0,
            recommended_price=175.0,
            demand_proxy_at_recommended=0.65,
            revenue_proxy_at_reference=113.75,
            revenue_proxy_at_recommended=113.75,
            revenue_proxy_change_pct=0.0,
            price_bounds=[87.5, 262.5],
        )

        response = PricingResponse(
            recommended_price=175.0,
            reference_price=175.0,
            reference_price_source="request",
            unavailability_probability=0.65,
            selected_scenario="moderate",
            assumed_sensitivity=1.0,
            demand_proxy_at_recommended=0.65,
            revenue_proxy_at_reference=113.75,
            revenue_proxy_at_recommended=113.75,
            revenue_proxy_change_pct=0.0,
            price_bounds=[87.5, 262.5],
            scenario_results=[scenario],
            shap_top_features=[],
            is_anomaly=False,
            model_version="v0.2.0",
            methodology_note="Assumption-based scenario.",
        )

        assert response.recommended_price == 175.0
        assert response.unavailability_probability == 0.65

    def test_rejects_invalid_probability(self):
        with pytest.raises(ValidationError):
            PricingResponse(
                recommended_price=175.0,
                reference_price=175.0,
                reference_price_source="request",
                unavailability_probability=1.5,
                selected_scenario="moderate",
                assumed_sensitivity=1.0,
                demand_proxy_at_recommended=0.65,
                revenue_proxy_at_reference=113.75,
                revenue_proxy_at_recommended=113.75,
                revenue_proxy_change_pct=0.0,
                price_bounds=[87.5, 262.5],
                scenario_results=[],
                shap_top_features=[],
                is_anomaly=False,
                model_version="v0.2.0",
                methodology_note="Assumption-based scenario.",
            )


class TestHealthResponse:
    def test_healthy(self):
        response = HealthResponse(
            status="healthy",
            models_loaded=True,
            model_version="v0.2.0",
        )
        assert response.status == "healthy"
