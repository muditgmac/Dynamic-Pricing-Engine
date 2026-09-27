"""Unit tests for the API dependencies module."""

from src.api.dependencies import (
    ROOM_TYPE_MAP,
    ModelState,
    build_features_from_request,
)


def _features(**overrides):
    params = {
        "room_type": "Entire home/apt",
        "beds": 2,
        "bathrooms": 1.0,
        "neighborhood": "Harlem",
        "checkin_date": "2024-07-15",
        "amenity_score": 0.7,
        "review_score": 4.5,
        "reference_price": 175.0,
        "location_cluster": 3,
        "training_start_date": "2023-03-06",
    }
    params.update(overrides)
    return build_features_from_request(
        **params
    )


class TestBuildFeaturesFromRequest:
    def test_returns_clean_model_features(self):
        features = _features()

        assert isinstance(features, dict)
        assert features["price"] == 175.0
        assert features["location_cluster"] == 3
        assert "rolling_7d_occupancy" not in features
        assert "rolling_30d_occupancy" not in features
        assert "occupancy_rate" not in features

    def test_correct_room_type_encoding(self):
        features = _features(
            room_type="Private room"
        )
        assert (
            features["room_type_encoded"]
            == ROOM_TYPE_MAP["Private room"]
        )

    def test_weekend_detection(self):
        features = _features(
            checkin_date="2024-07-13"
        )
        assert features["is_weekend"] == 1

    def test_weekday_detection(self):
        features = _features(
            checkin_date="2024-07-10"
        )
        assert features["is_weekend"] == 0

    def test_summer_season(self):
        features = _features(
            checkin_date="2024-07-15"
        )
        assert features["season"] == 2

    def test_winter_season(self):
        features = _features(
            checkin_date="2024-01-15"
        )
        assert features["season"] == 0

    def test_unknown_room_type_defaults_to_zero(self):
        features = _features(
            room_type="Unknown type"
        )
        assert (
            features["room_type_encoded"]
            == 0
        )

    def test_preserves_numeric_inputs(self):
        features = _features(
            beds=3,
            bathrooms=2.5,
            amenity_score=0.9,
            review_score=4.7,
            reference_price=210.0,
        )
        assert features["beds"] == 3
        assert features["bathrooms"] == 2.5
        assert features["amenity_score"] == 0.9
        assert features["review_score"] == 4.7
        assert features["price"] == 210.0

    def test_days_from_start_uses_training_reference(self):
        features = _features(
            checkin_date="2023-03-16",
            training_start_date="2023-03-06",
        )
        assert features["days_from_start"] == 10


class TestModelState:
    def test_default_state(self):
        state = ModelState()
        assert state.demand_forecaster is None
        assert state.anomaly_detector is None
        assert state.shap_explainer is None
        assert state.is_loaded is False
        assert state.total_predictions == 0
        assert state.model_version == "v0.2.0"
        assert state.global_reference_price == 150.0

    def test_room_type_map_complete(self):
        assert "Entire home/apt" in ROOM_TYPE_MAP
        assert "Private room" in ROOM_TYPE_MAP
        assert "Shared room" in ROOM_TYPE_MAP
        assert "Hotel room" in ROOM_TYPE_MAP
