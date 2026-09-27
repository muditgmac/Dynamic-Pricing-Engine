"""Pydantic v2 request/response models for the Pricing API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


SensitivityScenario = Literal[
    "low",
    "moderate",
    "high",
]


class PricingRequest(BaseModel):
    """Request body for POST /predict."""

    room_type: str = Field(
        ...,
        description=(
            "Room type: 'Entire home/apt', "
            "'Private room', 'Hotel room', or 'Shared room'"
        ),
        examples=["Entire home/apt"],
    )
    beds: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of beds",
    )
    bathrooms: float = Field(
        ...,
        ge=0.5,
        le=10,
        description="Number of bathrooms",
    )
    neighborhood: str = Field(
        ...,
        description="Neighborhood name from the dataset taxonomy",
        examples=["Williamsburg"],
    )
    checkin_date: str = Field(
        ...,
        description="Check-in date in YYYY-MM-DD format",
        examples=["2024-07-15"],
    )
    checkout_date: str = Field(
        ...,
        description="Check-out date in YYYY-MM-DD format",
        examples=["2024-07-18"],
    )
    amenity_score: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Fraction of top amenities present (0-1)",
    )
    review_score: float = Field(
        4.0,
        ge=0,
        le=5,
        description="Average review rating (0-5)",
    )
    reference_price: float | None = Field(
        None,
        gt=0,
        description=(
            "Current/reference nightly price. If omitted, "
            "the neighborhood median is used when available."
        ),
    )
    sensitivity_scenario: SensitivityScenario = Field(
        "moderate",
        description=(
            "Explicit price-sensitivity assumption used by "
            "the scenario engine."
        ),
    )

    @model_validator(mode="after")
    def validate_stay_dates(self):
        """Ensure both dates parse and checkout follows check-in."""
        try:
            from pandas import Timestamp

            checkin = Timestamp(self.checkin_date)
            checkout = Timestamp(self.checkout_date)
        except Exception as exc:
            raise ValueError(
                "checkin_date and checkout_date must be valid dates"
            ) from exc

        if checkout <= checkin:
            raise ValueError(
                "checkout_date must be after checkin_date"
            )

        return self


class ShapFeature(BaseModel):
    """A genuine TreeSHAP feature contribution."""

    feature: str = Field(
        ...,
        description="Model feature name",
    )
    contribution: float = Field(
        ...,
        description=(
            "TreeSHAP contribution to the XGBoost raw margin. "
            "Positive values push toward higher unavailability."
        ),
    )


class ScenarioSummary(BaseModel):
    """One explicit price-sensitivity scenario."""

    scenario_name: SensitivityScenario
    sensitivity: float = Field(..., ge=0)
    recommended_price: float = Field(..., gt=0)
    demand_proxy_at_recommended: float = Field(
        ...,
        ge=0,
        le=1,
    )
    revenue_proxy_at_reference: float = Field(
        ...,
        ge=0,
    )
    revenue_proxy_at_recommended: float = Field(
        ...,
        ge=0,
    )
    revenue_proxy_change_pct: float
    price_bounds: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Scenario search bounds [floor, ceiling]",
    )


class PricingResponse(BaseModel):
    """Response body for POST /predict."""

    recommended_price: float = Field(
        ...,
        gt=0,
        description=(
            "Scenario-based nightly price recommendation "
            "under the selected sensitivity assumption"
        ),
    )
    reference_price: float = Field(
        ...,
        gt=0,
        description="Price used for the baseline demand prediction",
    )
    reference_price_source: str = Field(
        ...,
        description=(
            "'request', 'neighborhood_median', or 'global_median'"
        ),
    )
    unavailability_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Predicted calendar-unavailability probability. "
            "This is not a confirmed-booking probability."
        ),
    )
    selected_scenario: SensitivityScenario
    assumed_sensitivity: float = Field(
        ...,
        ge=0,
        description=(
            "Explicit scenario assumption; not an estimated causal elasticity"
        ),
    )
    demand_proxy_at_recommended: float = Field(
        ...,
        ge=0,
        le=1,
    )
    revenue_proxy_at_reference: float = Field(
        ...,
        ge=0,
    )
    revenue_proxy_at_recommended: float = Field(
        ...,
        ge=0,
    )
    revenue_proxy_change_pct: float
    price_bounds: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Scenario search bounds [floor, ceiling]",
    )
    scenario_results: list[ScenarioSummary]
    shap_top_features: list[ShapFeature] = Field(
        default_factory=list,
        description=(
            "Top TreeSHAP contributions to the baseline demand model"
        ),
    )
    is_anomaly: bool
    model_version: str
    methodology_note: str = Field(
        ...,
        description=(
            "Clarifies that pricing outputs are assumption-based "
            "decision-support scenarios, not causal revenue forecasts."
        ),
    )


class ExplainRequest(BaseModel):
    """Request body for POST /explain."""

    room_type: str = Field(..., description="Room type")
    beds: int = Field(..., ge=1, le=20)
    bathrooms: float = Field(..., ge=0.5, le=10)
    neighborhood: str
    checkin_date: str
    amenity_score: float = Field(0.5, ge=0, le=1)
    review_score: float = Field(4.0, ge=0, le=5)
    reference_price: float | None = Field(
        None,
        gt=0,
        description=(
            "Price at which to explain the baseline "
            "unavailability prediction"
        ),
    )


class ExplainResponse(BaseModel):
    """Response body for POST /explain."""

    shap_values: list[ShapFeature]
    base_value: float | None = Field(
        None,
        description="TreeSHAP base value in raw-margin space",
    )
    predicted_unavailability: float = Field(
        ...,
        ge=0,
        le=1,
    )
    reference_price: float = Field(..., gt=0)


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    models_loaded: bool
    model_version: str = ""


class MetricsResponse(BaseModel):
    """Response body for GET /metrics."""

    holdout_auc: float = Field(..., ge=0, le=1)
    holdout_pr_auc: float = Field(..., ge=0, le=1)
    holdout_log_loss: float = Field(..., ge=0)
    holdout_brier: float = Field(..., ge=0)
    cv_auc_mean: float = Field(..., ge=0, le=1)
    total_predictions: int = Field(..., ge=0)


class DriftResponse(BaseModel):
    """Response body for GET /drift-report."""

    status: str = Field(
        ...,
        description="'healthy', 'degraded', or 'critical'",
    )
    drift_score: float = Field(..., description="Overall drift score (0-1)")
    features_drifted: list[str] = Field(
        ...,
        description="List of features that have drifted",
    )
    report_url: str = Field(
        ...,
        description="URL to the Evidently HTML report",
    )
