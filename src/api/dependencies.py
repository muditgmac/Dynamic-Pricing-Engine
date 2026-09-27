"""Dependency injection for the FastAPI service.

Loads shared model resources once at startup using FastAPI lifespan.
Models and explainers are cached in app.state — never re-loaded per request.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pandas as pd
from fastapi import FastAPI, Request

from src.models.anomaly_detector import AnomalyDetector
from src.models.demand_forecaster import DemandForecaster
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Room type encoding map (must match training)
ROOM_TYPE_MAP = {
    "Entire home/apt": 0,
    "Hotel room": 1,
    "Private room": 2,
    "Shared room": 3,
}


@dataclass
class ModelState:
    """Container for loaded production models and runtime state."""

    demand_forecaster: DemandForecaster | None = None
    anomaly_detector: AnomalyDetector | None = None
    shap_explainer: object | None = None
    config: dict = field(default_factory=dict)
    model_version: str = "v0.2.0"
    is_loaded: bool = False
    total_predictions: int = 0
    neighborhood_prices: dict[str, float] = field(default_factory=dict)
    neighborhood_location_clusters: dict[str, int] = field(default_factory=dict)
    global_reference_price: float = 150.0


def _load_models(state: ModelState) -> None:
    """Load production artifacts and lookup data into the state container."""
    config = load_config()
    state.config = config
    models_dir = PROJECT_ROOT / "models"

    demand_path = models_dir / "demand_forecaster"
    if not demand_path.exists():
        logger.error("Demand forecaster not found at %s", demand_path)
        state.is_loaded = False
        return

    state.demand_forecaster = DemandForecaster(config)
    state.demand_forecaster.load(demand_path)
    logger.info("Demand forecaster loaded")

    # Build one TreeSHAP explainer at startup instead of per request.
    try:
        import shap

        state.shap_explainer = shap.TreeExplainer(
            state.demand_forecaster.model,
        )
        logger.info("TreeSHAP explainer initialized")
    except Exception as exc:
        state.shap_explainer = None
        logger.warning("TreeSHAP unavailable: %s", exc)

    anomaly_path = models_dir / "anomaly_detector"
    if anomaly_path.exists():
        state.anomaly_detector = AnomalyDetector(config=config)
        state.anomaly_detector.load(anomaly_path)
        logger.info("Anomaly detector loaded")
    else:
        logger.warning("Anomaly detector not found at %s", anomaly_path)

    features_path = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
        / "listings_features.parquet"
    )
    if features_path.exists():
        listings = pd.read_parquet(features_path)

        if "price" in listings.columns:
            price_series = pd.to_numeric(
                listings["price"],
                errors="coerce",
            ).dropna()
            if not price_series.empty:
                state.global_reference_price = float(
                    price_series.median()
                )

        if (
            "neighbourhood_cleansed" in listings.columns
            and "price" in listings.columns
        ):
            state.neighborhood_prices = (
                listings.assign(
                    price=pd.to_numeric(
                        listings["price"],
                        errors="coerce",
                    )
                )
                .dropna(
                    subset=[
                        "neighbourhood_cleansed",
                        "price",
                    ]
                )
                .groupby(
                    "neighbourhood_cleansed"
                )["price"]
                .median()
                .astype(float)
                .to_dict()
            )

        if (
            "neighbourhood_cleansed" in listings.columns
            and "location_cluster" in listings.columns
        ):
            cluster_frame = listings[
                [
                    "neighbourhood_cleansed",
                    "location_cluster",
                ]
            ].dropna()

            if not cluster_frame.empty:
                state.neighborhood_location_clusters = (
                    cluster_frame
                    .groupby("neighbourhood_cleansed")[
                        "location_cluster"
                    ]
                    .agg(
                        lambda values: int(
                            values.mode().iloc[0]
                        )
                    )
                    .to_dict()
                )

        logger.info(
            "Loaded %d neighborhood prices and %d location-cluster mappings",
            len(state.neighborhood_prices),
            len(state.neighborhood_location_clusters),
        )
    else:
        logger.warning(
            "No listings features found. "
            "Neighborhood reference data unavailable."
        )

    state.is_loaded = (
        state.demand_forecaster.model is not None
        and state.demand_forecaster.feature_names is not None
    )
    logger.info(
        "Production model state loaded: %s",
        state.is_loaded,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared model resources at startup and release them at shutdown."""
    state = ModelState()
    try:
        _load_models(state)
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        state.is_loaded = False

    app.state.models = state
    logger.info("API startup complete")
    yield

    state.shap_explainer = None
    logger.info("API shutting down")


def get_model_state(request: Request) -> ModelState:
    """Return the application-level model state."""
    return request.app.state.models


def build_features_from_request(
    room_type: str,
    beds: int,
    bathrooms: float,
    neighborhood: str,
    checkin_date: str,
    amenity_score: float,
    review_score: float,
    reference_price: float,
    location_cluster: int = 0,
    training_start_date: str | None = None,
) -> dict:
    """Convert request fields into the leakage-safe demand-model feature set."""
    del neighborhood  # Reserved for future neighborhood-specific features.

    dt = pd.Timestamp(checkin_date)

    days_from_start = 0
    if training_start_date:
        start = pd.Timestamp(training_start_date)
        days_from_start = max(
            0,
            int(
                (
                    dt.normalize()
                    - start.normalize()
                ).days
            ),
        )

    return {
        "price": float(reference_price),
        "day_of_week": int(dt.dayofweek),
        "day_of_month": int(dt.day),
        "week_of_year": int(dt.isocalendar().week),
        "month": int(dt.month),
        "quarter": int(dt.quarter),
        "is_weekend": int(dt.dayofweek >= 5),
        "season": (
            0
            if dt.month in [12, 1, 2]
            else 1
            if dt.month in [3, 4, 5]
            else 2
            if dt.month in [6, 7, 8]
            else 3
        ),
        "days_from_start": days_from_start,
        "temperature_mean": 20.0,
        "precipitation_sum": 0.0,
        "wind_speed_max": 10.0,
        "is_hot_day": 0,
        "is_cold_day": 0,
        "is_rainy_day": 0,
        "is_holiday": 0,
        "days_until_holiday": 30,
        "near_holiday": 0,
        "room_type_encoded": ROOM_TYPE_MAP.get(
            room_type,
            0,
        ),
        "beds": int(beds),
        "bathrooms": float(bathrooms),
        "amenity_score": float(amenity_score),
        "review_score": float(review_score),
        "location_cluster": int(location_cluster),
    }
