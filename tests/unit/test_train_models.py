"""Tests for the production model-training orchestration."""

import json
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.models.train_models as tm


def _write_parquet(path, df, row_group_size=None):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        row_group_size=row_group_size,
    )


class TestJsonSafe:
    def test_converts_numpy_scalars(self):
        result = tm._json_safe(
            {
                "integer": np.int64(7),
                "floating": np.float64(0.75),
                "ordinary": "value",
            }
        )

        assert result == {
            "integer": 7,
            "floating": 0.75,
            "ordinary": "value",
        }
        assert isinstance(result["integer"], int)
        assert isinstance(result["floating"], float)


class TestEvenParquetSample:
    def test_reads_all_rows_when_below_limit(self, tmp_path):
        path = tmp_path / "sample.parquet"

        source = pd.DataFrame(
            {
                "row_id": np.arange(5),
                "value": np.arange(10, 15),
            }
        )
        _write_parquet(path, source)

        result = tm._even_parquet_sample(
            path,
            columns=["row_id", "value"],
            max_rows=10,
        )

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            source,
        )

    def test_skips_missing_columns(self, tmp_path):
        path = tmp_path / "sample.parquet"

        source = pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "value": [10, 20, 30],
            }
        )
        _write_parquet(path, source)

        result = tm._even_parquet_sample(
            path,
            columns=["row_id", "not_present"],
            max_rows=10,
        )

        assert result.columns.tolist() == ["row_id"]
        assert result["row_id"].tolist() == [1, 2, 3]

    def test_returns_even_deterministic_sample(self, tmp_path, monkeypatch):
        path = tmp_path / "sample.parquet"

        source = pd.DataFrame(
            {
                "row_id": np.arange(10),
                "value": np.arange(100, 110),
            }
        )

        _write_parquet(
            path,
            source,
            row_group_size=3,
        )

        monkeypatch.setattr(
            tm,
            "PARQUET_BATCH_SIZE",
            4,
        )

        result = tm._even_parquet_sample(
            path,
            columns=["row_id", "value"],
            max_rows=4,
        )

        assert len(result) == 4
        assert result["row_id"].tolist() == [0, 3, 6, 9]
        assert result["row_id"].is_monotonic_increasing

    def test_rejects_nonpositive_max_rows(self, tmp_path):
        path = tmp_path / "sample.parquet"
        _write_parquet(
            path,
            pd.DataFrame({"row_id": [1]}),
        )

        with pytest.raises(
            ValueError,
            match="max_rows must be positive",
        ):
            tm._even_parquet_sample(
                path,
                columns=["row_id"],
                max_rows=0,
            )


class TestTrainAll:
    def test_missing_calendar_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            tm,
            "load_config",
            lambda: {},
        )

        with pytest.raises(
            FileNotFoundError,
            match="calendar_features.parquet",
        ):
            tm.train_all(max_rows=100)

    def test_missing_listings_raises(self, tmp_path, monkeypatch):
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)

        (
            processed / "calendar_features.parquet"
        ).touch()

        monkeypatch.setattr(tm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            tm,
            "load_config",
            lambda: {},
        )

        with pytest.raises(
            FileNotFoundError,
            match="listings_features.parquet",
        ):
            tm.train_all(max_rows=100)

    def test_happy_path_trains_and_persists_summary(
        self,
        tmp_path,
        monkeypatch,
    ):
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)

        calendar_path = (
            processed / "calendar_features.parquet"
        )
        listings_path = (
            processed / "listings_features.parquet"
        )

        calendar_path.touch()
        listings_path.touch()

        raw_calendar = pd.DataFrame(
            {
                "listing_id": [2, 1, 3],
                "date": [
                    "2026-01-02",
                    "not-a-date",
                    "2026-01-01",
                ],
                "price": [200.0, 100.0, 150.0],
            }
        )

        listings = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "room_type_encoded": [0, 1, 2],
                "beds": [1.0, 2.0, 1.0],
                "bathrooms": [1.0, 1.0, 2.0],
                "amenity_score": [0.5, 0.6, 0.7],
                "review_score": [4.5, 4.6, 4.7],
                "location_cluster": [0, 1, 1],
            }
        )

        config = {
            "mlflow": {
                "tracking_uri": "sqlite:///test_mlflow.db",
                "experiment_name": "unit-test",
            }
        }

        monkeypatch.setattr(tm, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            tm,
            "load_config",
            lambda: config,
        )
        monkeypatch.setattr(
            tm,
            "_even_parquet_sample",
            lambda *args, **kwargs: raw_calendar.copy(),
        )
        monkeypatch.setattr(
            tm.pd,
            "read_parquet",
            lambda *args, **kwargs: listings.copy(),
        )

        class FakeDemandForecaster:
            seen_listing_ids = None
            saved = False

            def __init__(self, config):
                self.config = config

            def train(
                self,
                calendar_df,
                listings_df,
                tune_hyperparams=False,
            ):
                type(self).seen_listing_ids = (
                    calendar_df["listing_id"].tolist()
                )

                assert tune_hyperparams is False
                assert len(listings_df) == 3

                return {
                    "roc_auc": np.float64(0.71),
                    "training_rows": np.int64(
                        len(calendar_df)
                    ),
                }

            def save(self):
                type(self).saved = True

        class FakeAnomalyDetector:
            saved = False

            def __init__(self, config):
                self.config = config
                self.contamination = 0.05
                self.model = object()

            def fit(self, df):
                self.fitted_rows = len(df)
                return self

            def predict(self, df):
                assert self.fitted_rows == len(df)
                return np.array(
                    [False, True],
                    dtype=bool,
                )

            def save(self):
                type(self).saved = True

        monkeypatch.setattr(
            tm,
            "DemandForecaster",
            FakeDemandForecaster,
        )
        monkeypatch.setattr(
            tm,
            "AnomalyDetector",
            FakeAnomalyDetector,
        )

        tracking_uris = []
        experiments = []
        logged_params = {}
        logged_metrics = {}

        monkeypatch.setattr(
            tm.mlflow,
            "set_tracking_uri",
            lambda uri: tracking_uris.append(uri),
        )
        monkeypatch.setattr(
            tm.mlflow,
            "set_experiment",
            lambda name: experiments.append(name),
        )
        monkeypatch.setattr(
            tm.mlflow,
            "start_run",
            lambda **kwargs: nullcontext(),
        )
        monkeypatch.setattr(
            tm.mlflow,
            "log_param",
            lambda key, value: logged_params.__setitem__(
                key,
                value,
            ),
        )
        monkeypatch.setattr(
            tm.mlflow,
            "log_metric",
            lambda key, value: logged_metrics.__setitem__(
                key,
                value,
            ),
        )
        monkeypatch.setattr(
            tm.mlflow.sklearn,
            "log_model",
            lambda *args, **kwargs: None,
        )

        summary = tm.train_all(max_rows=100)

        # Invalid date removed; remaining observations sorted
        # chronologically.
        assert (
            FakeDemandForecaster.seen_listing_ids
            == [3, 2]
        )

        assert FakeDemandForecaster.saved is True
        assert FakeAnomalyDetector.saved is True

        assert summary["training_rows"] == 2
        assert summary["demand"]["roc_auc"] == 0.71
        assert (
            summary["demand"]["training_rows"]
            == 2
        )

        assert (
            summary["anomaly"]["training_anomaly_rate"]
            == 0.5
        )
        assert (
            summary["anomaly"]["contamination"]
            == 0.05
        )

        assert tracking_uris == [
            "sqlite:///test_mlflow.db"
        ]
        assert experiments == ["unit-test"]
        assert logged_params["contamination"] == 0.05
        assert logged_params["training_rows"] == 2
        assert (
            logged_metrics["training_anomaly_rate"]
            == 0.5
        )

        summary_path = (
            tmp_path
            / "models"
            / "training_summary.json"
        )

        assert summary_path.exists()

        persisted = json.loads(
            summary_path.read_text()
        )
        assert persisted == summary
