"""API route definitions for the Dynamic Pricing Engine.

Endpoints:
- POST /predict — baseline unavailability + pricing sensitivity scenarios
- GET  /health — liveness check
- GET  /metrics — leakage-safe holdout metrics
- POST /explain — genuine TreeSHAP explanations when available
- GET  /drift-report — data drift monitoring
"""

import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    ModelState,
    build_features_from_request,
    get_model_state,
)
from src.api.schemas import (
    DriftResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    PricingRequest,
    PricingResponse,
    ScenarioSummary,
    ShapFeature,
)
from src.models.scenario_engine import (
    SENSITIVITY_PRESETS,
    evaluate_sensitivity_presets,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

METHODOLOGY_NOTE = (
    "Pricing outputs are assumption-based scenario results. "
    "The Airbnb calendar target is a calendar-unavailability proxy, "
    "not confirmed bookings, and the sensitivity assumptions are not "
    "causal elasticity estimates."
)


def _reference_price(
    requested_price: float | None,
    neighborhood: str,
    state: ModelState,
) -> tuple[float, str]:
    """Resolve the baseline price and disclose where it came from."""
    if requested_price is not None:
        return float(requested_price), "request"

    if neighborhood in state.neighborhood_prices:
        return (
            float(
                state.neighborhood_prices[
                    neighborhood
                ]
            ),
            "neighborhood_median",
        )

    return (
        float(state.global_reference_price),
        "global_median",
    )


def _training_start_date(state: ModelState) -> str | None:
    """Return the start date used for the trained demand artifact."""
    if state.demand_forecaster is None:
        return None

    return state.demand_forecaster.split_metadata.get(
        "train_start"
    )


def _features_for_request(
    *,
    room_type: str,
    beds: int,
    bathrooms: float,
    neighborhood: str,
    checkin_date: str,
    amenity_score: float,
    review_score: float,
    reference_price: float,
    state: ModelState,
) -> dict:
    """Build the exact leakage-safe feature vector used for inference."""
    location_cluster = (
        state.neighborhood_location_clusters.get(
            neighborhood,
            0,
        )
    )

    return build_features_from_request(
        room_type=room_type,
        beds=beds,
        bathrooms=bathrooms,
        neighborhood=neighborhood,
        checkin_date=checkin_date,
        amenity_score=amenity_score,
        review_score=review_score,
        reference_price=reference_price,
        location_cluster=location_cluster,
        training_start_date=_training_start_date(
            state
        ),
    )


def _tree_shap(
    state: ModelState,
    features: dict,
    limit: int | None = None,
) -> tuple[list[ShapFeature], float | None]:
    """Return genuine TreeSHAP contributions in raw-margin space."""
    if (
        state.shap_explainer is None
        or state.demand_forecaster is None
        or not state.demand_forecaster.feature_names
    ):
        return [], None

    feature_names = (
        state.demand_forecaster.feature_names
    )

    feature_frame = (
        pd.DataFrame([features])
        .reindex(
            columns=feature_names,
            fill_value=0,
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
        .astype(float)
    )

    try:
        explanation = state.shap_explainer(
            feature_frame
        )
        values = np.asarray(
            explanation.values
        )

        if values.ndim == 3:
            values = values[..., -1]

        row_values = values[0]
        pairs = sorted(
            zip(
                feature_names,
                row_values,
                strict=True,
            ),
            key=lambda item: abs(
                float(item[1])
            ),
            reverse=True,
        )

        if limit is not None:
            pairs = pairs[:limit]

        shap_features = [
            ShapFeature(
                feature=name,
                contribution=round(
                    float(value),
                    6,
                ),
            )
            for name, value in pairs
        ]

        base_values = np.asarray(
            explanation.base_values
        ).reshape(-1)
        base_value = (
            float(base_values[0])
            if base_values.size
            else None
        )

        return shap_features, base_value

    except Exception as exc:
        logger.warning(
            "TreeSHAP explanation failed: %s",
            exc,
        )
        return [], None


@router.post(
    "/predict",
    response_model=PricingResponse,
)
async def predict_price(
    request: PricingRequest,
    state: ModelState = Depends(
        get_model_state
    ),
):
    """Return a baseline prediction and explicit pricing scenarios."""
    start_time = time.time()

    if (
        not state.is_loaded
        or state.demand_forecaster is None
        or state.demand_forecaster.model is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Demand model not loaded",
        )

    reference_price, price_source = (
        _reference_price(
            request.reference_price,
            request.neighborhood,
            state,
        )
    )

    features = _features_for_request(
        room_type=request.room_type,
        beds=request.beds,
        bathrooms=request.bathrooms,
        neighborhood=request.neighborhood,
        checkin_date=request.checkin_date,
        amenity_score=request.amenity_score,
        review_score=request.review_score,
        reference_price=reference_price,
        state=state,
    )

    try:
        unavailability_probability = (
            state.demand_forecaster
            .predict_single(features)
        )
    except Exception as exc:
        logger.exception(
            "Demand prediction failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Demand prediction failed",
        ) from exc

    scenario_results = (
        evaluate_sensitivity_presets(
            base_demand_proxy=(
                unavailability_probability
            ),
            reference_price=reference_price,
            config=state.config,
        )
    )

    selected = scenario_results[
        request.sensitivity_scenario
    ]

    scenario_payload = [
        ScenarioSummary(
            scenario_name=name,
            sensitivity=result.sensitivity,
            recommended_price=(
                result.recommended_price
            ),
            demand_proxy_at_recommended=(
                result.demand_proxy_at_recommended
            ),
            revenue_proxy_at_reference=(
                result.revenue_proxy_at_reference
            ),
            revenue_proxy_at_recommended=(
                result.revenue_proxy_at_recommended
            ),
            revenue_proxy_change_pct=(
                result.revenue_proxy_change_pct
            ),
            price_bounds=list(
                result.price_bounds
            ),
        )
        for name, result
        in scenario_results.items()
    ]

    is_anomaly = False
    if (
        state.anomaly_detector
        and state.anomaly_detector.is_fitted
    ):
        try:
            feature_df = pd.DataFrame(
                [features]
            )
            anomaly_pred = (
                state.anomaly_detector.predict(
                    feature_df
                )
            )
            is_anomaly = bool(
                anomaly_pred[0]
            )
        except Exception as exc:
            logger.warning(
                "Anomaly check failed: %s",
                exc,
            )

    shap_features, _ = _tree_shap(
        state,
        features,
        limit=5,
    )

    state.total_predictions += 1
    latency_ms = (
        time.time() - start_time
    ) * 1000

    logger.info(
        "Prediction served in %.1fms",
        latency_ms,
    )

    return PricingResponse(
        recommended_price=(
            selected.recommended_price
        ),
        reference_price=round(
            reference_price,
            2,
        ),
        reference_price_source=price_source,
        unavailability_probability=round(
            float(
                unavailability_probability
            ),
            6,
        ),
        selected_scenario=(
            request.sensitivity_scenario
        ),
        assumed_sensitivity=(
            selected.sensitivity
        ),
        demand_proxy_at_recommended=(
            selected.demand_proxy_at_recommended
        ),
        revenue_proxy_at_reference=(
            selected.revenue_proxy_at_reference
        ),
        revenue_proxy_at_recommended=(
            selected.revenue_proxy_at_recommended
        ),
        revenue_proxy_change_pct=(
            selected.revenue_proxy_change_pct
        ),
        price_bounds=list(
            selected.price_bounds
        ),
        scenario_results=scenario_payload,
        shap_top_features=shap_features,
        is_anomaly=is_anomaly,
        model_version=state.model_version,
        methodology_note=METHODOLOGY_NOTE,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check(
    state: ModelState = Depends(
        get_model_state
    ),
):
    """Liveness/readiness check for CI/CD and load balancers."""
    if not state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded",
        )

    return HealthResponse(
        status="healthy",
        models_loaded=True,
        model_version=state.model_version,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
)
async def get_metrics(
    state: ModelState = Depends(
        get_model_state
    ),
):
    """Return the production demand model's leakage-safe metrics."""
    if (
        not state.is_loaded
        or state.demand_forecaster is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded",
        )

    metrics = (
        state.demand_forecaster.metrics
        or {}
    )

    return MetricsResponse(
        holdout_auc=round(
            metrics.get(
                "holdout_auc",
                0.0,
            ),
            6,
        ),
        holdout_pr_auc=round(
            metrics.get(
                "holdout_pr_auc",
                0.0,
            ),
            6,
        ),
        holdout_log_loss=round(
            metrics.get(
                "holdout_log_loss",
                0.0,
            ),
            6,
        ),
        holdout_brier=round(
            metrics.get(
                "holdout_brier",
                0.0,
            ),
            6,
        ),
        cv_auc_mean=round(
            metrics.get(
                "cv_auc_mean",
                0.0,
            ),
            6,
        ),
        total_predictions=(
            state.total_predictions
        ),
    )


@router.post(
    "/explain",
    response_model=ExplainResponse,
)
async def explain_prediction(
    request: ExplainRequest,
    state: ModelState = Depends(
        get_model_state
    ),
):
    """Return TreeSHAP contributions for a baseline prediction."""
    if (
        not state.is_loaded
        or state.demand_forecaster is None
        or state.demand_forecaster.model is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Demand model not loaded",
        )

    reference_price, _ = (
        _reference_price(
            request.reference_price,
            request.neighborhood,
            state,
        )
    )

    features = _features_for_request(
        room_type=request.room_type,
        beds=request.beds,
        bathrooms=request.bathrooms,
        neighborhood=request.neighborhood,
        checkin_date=request.checkin_date,
        amenity_score=request.amenity_score,
        review_score=request.review_score,
        reference_price=reference_price,
        state=state,
    )

    try:
        prediction = (
            state.demand_forecaster
            .predict_single(features)
        )
    except Exception as exc:
        logger.exception(
            "Demand prediction failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Demand prediction failed",
        ) from exc

    shap_values, base_value = (
        _tree_shap(
            state,
            features,
            limit=None,
        )
    )

    return ExplainResponse(
        shap_values=shap_values,
        base_value=base_value,
        predicted_unavailability=round(
            float(prediction),
            6,
        ),
        reference_price=round(
            reference_price,
            2,
        ),
    )


@router.get(
    "/drift-report",
    response_model=DriftResponse,
)
async def get_drift_report(
    state: ModelState = Depends(
        get_model_state
    ),
):
    """Trigger drift monitoring and return a compact summary."""
    if not state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded",
        )

    try:
        from src.monitoring.drift_detector import (
            run_drift_check,
        )

        drift_result = run_drift_check(
            state.config
        )
        return DriftResponse(
            **drift_result
        )
    except ImportError:
        logger.warning(
            "Evidently not installed. "
            "Returning placeholder drift report."
        )
    except Exception as exc:
        logger.warning(
            "Drift check failed: %s",
            exc,
        )

    return DriftResponse(
        status="healthy",
        drift_score=0.0,
        features_drifted=[],
        report_url=(
            "/static/drift_report.html"
        ),
    )
