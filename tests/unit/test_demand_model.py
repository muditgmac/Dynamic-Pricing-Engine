"""Unit tests for the leakage-safe demand forecaster."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.models.demand_forecaster import DemandForecaster
from src.utils.config import load_config


@pytest.fixture
def trained_forecaster(tmp_path):
    """Train a small model on synthetic panel data."""
    rng = np.random.default_rng(42)

    dates = pd.date_range("2024-01-01", periods=140, freq="D")
    listing_ids = np.arange(1, 11)

    calendar_df = pd.MultiIndex.from_product(
        [dates, listing_ids],
        names=["date", "listing_id"],
    ).to_frame(index=False)

    n = len(calendar_df)
    calendar_df["price"] = rng.uniform(80, 300, n)
    calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek
    calendar_df["day_of_month"] = calendar_df["date"].dt.day
    calendar_df["week_of_year"] = (
        calendar_df["date"].dt.isocalendar().week.astype(int)
    )
    calendar_df["month"] = calendar_df["date"].dt.month
    calendar_df["quarter"] = calendar_df["date"].dt.quarter
    calendar_df["is_weekend"] = (
        calendar_df["day_of_week"] >= 5
    ).astype(int)
    calendar_df["season"] = (
        (calendar_df["month"] % 12) // 3
    ).astype(int)
    calendar_df["days_from_start"] = (
        calendar_df["date"] - calendar_df["date"].min()
    ).dt.days
    calendar_df["temperature_mean"] = rng.normal(18, 7, n)
    calendar_df["precipitation_sum"] = rng.gamma(1.5, 1.0, n)
    calendar_df["wind_speed_max"] = rng.uniform(2, 20, n)
    calendar_df["is_hot_day"] = (
        calendar_df["temperature_mean"] > 28
    ).astype(int)
    calendar_df["is_cold_day"] = (
        calendar_df["temperature_mean"] < 8
    ).astype(int)
    calendar_df["is_rainy_day"] = (
        calendar_df["precipitation_sum"] > 1
    ).astype(int)
    calendar_df["is_holiday"] = 0
    calendar_df["days_until_holiday"] = 10
    calendar_df["near_holiday"] = 0

    # Synthetic signal with noise. This is an unavailability proxy, not a
    # causal price-response simulation.
    score = (
        0.8 * calendar_df["is_weekend"].to_numpy()
        + 0.3 * (calendar_df["month"].to_numpy() >= 6)
        - 0.002 * calendar_df["price"].to_numpy()
        + rng.normal(0, 0.8, n)
    )
    probability = 1 / (1 + np.exp(-score))
    calendar_df["was_booked"] = (
        rng.uniform(0, 1, n) < probability
    ).astype(int)

    listings_df = pd.DataFrame(
        {
            "id": listing_ids,
            "room_type_encoded": rng.integers(0, 4, len(listing_ids)),
            "beds": rng.integers(1, 5, len(listing_ids)),
            "bathrooms": rng.uniform(1, 3, len(listing_ids)),
            "amenity_score": rng.uniform(0.3, 1.0, len(listing_ids)),
            "review_score": rng.uniform(3.5, 5.0, len(listing_ids)),
            "location_cluster": rng.integers(0, 5, len(listing_ids)),
        }
    )

    config = deepcopy(load_config())
    config["mlflow"]["tracking_uri"] = (
        f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    config["model"]["cv_splits"] = 3
    config["model"]["test_size"] = 0.2
    config["model"]["xgboost"]["n_estimators"] = 40
    config["model"]["xgboost"]["early_stopping_rounds"] = 5

    forecaster = DemandForecaster(config=config)
    forecaster.train(calendar_df, listings_df)
    return forecaster


class TestDemandForecaster:
    def test_loads_without_error(self):
        forecaster = DemandForecaster()
        assert forecaster.model is None

    def test_predict_returns_probabilities(self, trained_forecaster):
        X = pd.DataFrame(
            {
                "price": [150.0],
                "day_of_week": [5],
                "month": [7],
                "is_weekend": [1],
                "season": [2],
            }
        )
        pred = trained_forecaster.predict(X)
        assert len(pred) == 1
        assert 0 <= pred[0] <= 1

    def test_predict_single(self, trained_forecaster):
        pred = trained_forecaster.predict_single(
            {
                "price": 180.0,
                "day_of_week": 3,
                "month": 6,
                "is_weekend": 0,
                "season": 2,
            }
        )
        assert isinstance(pred, float)
        assert 0 <= pred <= 1

    def test_feature_importance(self, trained_forecaster):
        importance = trained_forecaster.get_feature_importance()
        assert list(importance.columns) == ["feature", "importance"]
        assert len(importance) > 0
        assert (importance["importance"] >= 0).all()

    def test_target_history_features_are_excluded(self, trained_forecaster):
        forbidden = {
            "rolling_7d_occupancy",
            "rolling_30d_occupancy",
            "occupancy_rate",
        }
        assert forbidden.isdisjoint(trained_forecaster.feature_names)

    def test_price_and_listing_features_are_used(self, trained_forecaster):
        assert "price" in trained_forecaster.feature_names
        assert "room_type_encoded" in trained_forecaster.feature_names
        assert "location_cluster" in trained_forecaster.feature_names

    def test_holdout_is_strictly_future(self, trained_forecaster):
        metadata = trained_forecaster.split_metadata
        assert pd.Timestamp(metadata["train_end"]) < pd.Timestamp(
            metadata["holdout_start"]
        )

    def test_probability_metrics_are_recorded(self, trained_forecaster):
        expected = {
            "holdout_auc",
            "holdout_pr_auc",
            "holdout_log_loss",
            "holdout_brier",
            "holdout_positive_rate",
            "cv_auc_mean",
        }
        assert expected.issubset(trained_forecaster.metrics)

    def test_handles_extra_and_missing_features(self, trained_forecaster):
        X = pd.DataFrame(
            {
                "price": [170.0],
                "day_of_week": [3],
                "month": [6],
                "extra_col": [999],
            }
        )
        pred = trained_forecaster.predict(X)
        assert len(pred) == 1

    def test_raises_without_training(self):
        forecaster = DemandForecaster()
        with pytest.raises(RuntimeError, match="not trained"):
            forecaster.predict(pd.DataFrame({"price": [100.0]}))
