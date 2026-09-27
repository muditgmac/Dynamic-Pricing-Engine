"""Leakage-safe XGBoost model for Airbnb calendar unavailability.

Inside Airbnb calendar availability is used as a demand proxy. An unavailable
night is not treated as a confirmed booking. The model therefore predicts the
probability that a listing-night is unavailable, using price, temporal,
weather, holiday, and listing-level attributes.

Evaluation is strictly chronological and split on whole stay dates so that a
single date can never appear in both training and validation data.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Only contemporaneous/exogenous or listing-level predictors.
# Target-derived occupancy features are intentionally excluded.
DEMAND_FEATURES = [
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

LISTING_FEATURES = [
    "id",
    "room_type_encoded",
    "beds",
    "bathrooms",
    "amenity_score",
    "review_score",
    "location_cluster",
]

TARGET_COLUMNS = ("is_unavailable", "was_booked")

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
}


class DemandForecaster:
    """XGBoost unavailability-proxy forecaster with temporal evaluation."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        self.config = config
        self.model_cfg = config["model"]
        self.model = None
        self.feature_names: list[str] | None = None
        self.metrics: dict[str, float] = {}
        self.split_metadata: dict[str, str | int] = {}

    def _merge_inputs(
        self,
        calendar_df: pd.DataFrame,
        listings_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Merge listing attributes onto calendar observations."""
        df = calendar_df.copy()

        if "date" not in df.columns:
            raise ValueError("Calendar data must contain a 'date' column.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        if listings_df is not None and "listing_id" in df.columns:
            merge_cols = [
                col for col in LISTING_FEATURES if col in listings_df.columns
            ]
            if "id" in merge_cols:
                # Avoid duplicate columns if a prior pipeline step already
                # attached one or more listing-level features.
                feature_cols = [c for c in merge_cols if c != "id"]
                overlap = [c for c in feature_cols if c in df.columns]
                if overlap:
                    df = df.drop(columns=overlap)

                df = df.merge(
                    listings_df[merge_cols],
                    left_on="listing_id",
                    right_on="id",
                    how="left",
                    validate="many_to_one",
                )

        sort_cols = ["date"]
        if "listing_id" in df.columns:
            sort_cols.append("listing_id")

        return (
            df.sort_values(sort_cols, kind="mergesort")
            .reset_index(drop=True)
        )

    def _target_column(self, df: pd.DataFrame) -> str:
        """Return the available target column.

        `was_booked` is retained as a legacy processed-data column name, but it
        is interpreted only as calendar unavailability, not a confirmed booking.
        """
        for column in TARGET_COLUMNS:
            if column in df.columns:
                return column
        raise ValueError(
            "No unavailability target found. Expected one of "
            f"{TARGET_COLUMNS}."
        )

    def _get_available_features(self, df: pd.DataFrame) -> list[str]:
        """Return clean demand features that are present in the input."""
        available = [f for f in DEMAND_FEATURES if f in df.columns]
        if not available:
            raise ValueError(
                "No demand features found in DataFrame. "
                f"Expected some of: {DEMAND_FEATURES}."
            )
        return available

    def _prepare_xy(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build numeric feature matrix and binary target."""
        target_col = self._target_column(df)

        if feature_names is None:
            feature_names = self._get_available_features(df)
            self.feature_names = feature_names

        X = (
            df.reindex(columns=feature_names)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        y = df[target_col].astype(int)

        logger.info(
            "Prepared %s samples with %s features",
            f"{len(X):,}",
            X.shape[1],
        )
        return X, y

    @staticmethod
    def _date_cv_splits(
        dates: pd.Series,
        n_splits: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create expanding-window CV splits using whole unique dates."""
        normalized = pd.to_datetime(dates).reset_index(drop=True)
        unique_dates = np.sort(normalized.unique())

        if len(unique_dates) < n_splits + 1:
            raise ValueError(
                f"Need at least {n_splits + 1} unique dates for "
                f"{n_splits} temporal folds; got {len(unique_dates)}."
            )

        blocks = np.array_split(unique_dates, n_splits + 1)
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for fold in range(1, len(blocks)):
            train_dates = np.concatenate(blocks[:fold])
            val_dates = blocks[fold]

            train_idx = np.flatnonzero(normalized.isin(train_dates).to_numpy())
            val_idx = np.flatnonzero(normalized.isin(val_dates).to_numpy())

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            splits.append((train_idx, val_idx))

        if len(splits) != n_splits:
            raise ValueError(
                f"Could only construct {len(splits)} of {n_splits} folds."
            )
        return splits

    @staticmethod
    def _probability_metrics(
        y_true: pd.Series,
        probabilities: np.ndarray,
    ) -> dict[str, float]:
        """Return discrimination and probability-quality metrics."""
        y_array = np.asarray(y_true, dtype=int)
        p = np.asarray(probabilities, dtype=float)

        if np.unique(y_array).size < 2:
            auc = float("nan")
            pr_auc = float("nan")
        else:
            auc = float(roc_auc_score(y_array, p))
            pr_auc = float(average_precision_score(y_array, p))

        return {
            "auc": auc,
            "pr_auc": pr_auc,
            "log_loss": float(log_loss(y_array, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y_array, p)),
        }

    def train(
        self,
        calendar_df: pd.DataFrame,
        listings_df: pd.DataFrame | None = None,
        tune_hyperparams: bool = False,
        n_iter: int = 20,
    ) -> dict:
        """Train with date-based CV and an untouched future holdout."""
        df = self._merge_inputs(calendar_df, listings_df)

        self.feature_names = self._get_available_features(df)
        X, y = self._prepare_xy(df, self.feature_names)

        unique_dates = np.sort(df["date"].unique())
        test_fraction = float(self.model_cfg.get("test_size", 0.2))

        if not 0 < test_fraction < 1:
            raise ValueError("model.test_size must be between 0 and 1.")
        if len(unique_dates) < 3:
            raise ValueError("Need at least 3 unique dates for temporal holdout.")

        split_at = int(np.floor(len(unique_dates) * (1 - test_fraction)))
        split_at = min(max(split_at, 1), len(unique_dates) - 1)

        train_dates = unique_dates[:split_at]
        holdout_dates = unique_dates[split_at:]

        train_mask = df["date"].isin(train_dates).to_numpy()
        holdout_mask = df["date"].isin(holdout_dates).to_numpy()

        X_train_pool = X.loc[train_mask].reset_index(drop=True)
        y_train_pool = y.loc[train_mask].reset_index(drop=True)
        dates_train_pool = df.loc[train_mask, "date"].reset_index(drop=True)

        X_holdout = X.loc[holdout_mask].reset_index(drop=True)
        y_holdout = y.loc[holdout_mask].reset_index(drop=True)

        self.split_metadata = {
            "train_start": str(pd.Timestamp(train_dates.min()).date()),
            "train_end": str(pd.Timestamp(train_dates.max()).date()),
            "holdout_start": str(pd.Timestamp(holdout_dates.min()).date()),
            "holdout_end": str(pd.Timestamp(holdout_dates.max()).date()),
            "training_rows": int(len(X_train_pool)),
            "holdout_rows": int(len(X_holdout)),
            "training_dates": int(len(train_dates)),
            "holdout_dates": int(len(holdout_dates)),
        }

        if pd.Timestamp(train_dates.max()) >= pd.Timestamp(holdout_dates.min()):
            raise RuntimeError("Temporal holdout overlaps training dates.")

        n_splits = int(self.model_cfg["cv_splits"])
        cv_splits = self._date_cv_splits(dates_train_pool, n_splits)

        xgb_cfg = self.model_cfg["xgboost"]
        base_params = {
            "n_estimators": int(xgb_cfg["n_estimators"]),
            "max_depth": int(xgb_cfg["max_depth"]),
            "learning_rate": float(xgb_cfg["learning_rate"]),
            "subsample": float(xgb_cfg["subsample"]),
            "colsample_bytree": float(xgb_cfg["colsample_bytree"]),
            "random_state": int(self.model_cfg["random_state"]),
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name="demand_forecaster"):
            selected_params = dict(base_params)

            if tune_hyperparams:
                logger.info(
                    "Running leakage-safe randomized search with %s iterations",
                    n_iter,
                )
                search_model = XGBClassifier(
                    random_state=int(self.model_cfg["random_state"]),
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                )
                search = RandomizedSearchCV(
                    search_model,
                    PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    cv=cv_splits,
                    scoring="roc_auc",
                    random_state=int(self.model_cfg["random_state"]),
                    n_jobs=-1,
                    verbose=1,
                )
                search.fit(X_train_pool, y_train_pool)
                selected_params.update(search.best_params_)
                logger.info("Best params: %s", search.best_params_)

            mlflow.log_params(selected_params)
            mlflow.log_param("target_semantics", "calendar_unavailability_proxy")
            mlflow.log_param("feature_count", len(self.feature_names))
            mlflow.log_param("holdout_start", self.split_metadata["holdout_start"])

            fold_metrics: list[dict[str, float]] = []
            best_iterations: list[int] = []

            for fold_i, (train_idx, val_idx) in enumerate(cv_splits, start=1):
                X_fold_train = X_train_pool.iloc[train_idx]
                y_fold_train = y_train_pool.iloc[train_idx]
                X_val = X_train_pool.iloc[val_idx]
                y_val = y_train_pool.iloc[val_idx]

                fold_params = dict(selected_params)
                fold_params["early_stopping_rounds"] = int(
                    xgb_cfg["early_stopping_rounds"]
                )
                fold_model = XGBClassifier(**fold_params)
                fold_model.fit(
                    X_fold_train,
                    y_fold_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                p_val = fold_model.predict_proba(X_val)[:, 1]
                metrics = self._probability_metrics(y_val, p_val)
                metrics["fold"] = float(fold_i)
                fold_metrics.append(metrics)

                best_iteration = getattr(fold_model, "best_iteration", None)
                n_trees = (
                    int(selected_params["n_estimators"])
                    if best_iteration is None
                    else int(best_iteration) + 1
                )
                best_iterations.append(n_trees)

                logger.info(
                    "Fold %s: AUC=%.4f PR-AUC=%.4f LogLoss=%.4f Brier=%.4f",
                    fold_i,
                    metrics["auc"],
                    metrics["pr_auc"],
                    metrics["log_loss"],
                    metrics["brier"],
                )
                for name in ("auc", "pr_auc", "log_loss", "brier"):
                    if np.isfinite(metrics[name]):
                        mlflow.log_metric(
                            f"cv_fold_{fold_i}_{name}",
                            metrics[name],
                        )

            def finite_mean(name: str) -> float:
                values = np.asarray(
                    [m[name] for m in fold_metrics],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                return float(values.mean()) if len(values) else float("nan")

            def finite_std(name: str) -> float:
                values = np.asarray(
                    [m[name] for m in fold_metrics],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                return float(values.std()) if len(values) else float("nan")

            final_n_estimators = max(1, int(np.median(best_iterations)))
            evaluation_params = dict(selected_params)
            evaluation_params["n_estimators"] = final_n_estimators
            evaluation_params.pop("early_stopping_rounds", None)

            # Fit only on pre-holdout dates to obtain an honest future score.
            evaluation_model = XGBClassifier(**evaluation_params)
            evaluation_model.fit(X_train_pool, y_train_pool)
            p_holdout = evaluation_model.predict_proba(X_holdout)[:, 1]
            holdout_metrics = self._probability_metrics(
                y_holdout,
                p_holdout,
            )

            self.metrics = {
                "cv_auc_mean": finite_mean("auc"),
                "cv_auc_std": finite_std("auc"),
                "cv_pr_auc_mean": finite_mean("pr_auc"),
                "cv_log_loss_mean": finite_mean("log_loss"),
                "cv_brier_mean": finite_mean("brier"),
                "holdout_auc": holdout_metrics["auc"],
                "holdout_pr_auc": holdout_metrics["pr_auc"],
                "holdout_log_loss": holdout_metrics["log_loss"],
                "holdout_brier": holdout_metrics["brier"],
                "holdout_positive_rate": float(y_holdout.mean()),
            }

            for name, value in self.metrics.items():
                if np.isfinite(value):
                    mlflow.log_metric(name, value)

            mlflow.log_param("final_n_estimators", final_n_estimators)

            logger.info(
                "Future holdout: AUC=%.4f PR-AUC=%.4f "
                "LogLoss=%.4f Brier=%.4f",
                self.metrics["holdout_auc"],
                self.metrics["holdout_pr_auc"],
                self.metrics["holdout_log_loss"],
                self.metrics["holdout_brier"],
            )

            # Refit the deployable artifact on all supplied observations only
            # after the untouched holdout metrics have been recorded.
            self.model = XGBClassifier(**evaluation_params)
            self.model.fit(X, y)

            input_example = X.iloc[[0]].copy()
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="demand_model",
                input_example=input_example,
            )

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calendar-unavailability probability."""
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_input = (
            X.reindex(columns=self.feature_names, fill_value=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        return self.model.predict_proba(X_input)[:, 1]

    def predict_single(self, features: dict) -> float:
        """Predict unavailability probability for one observation."""
        return float(self.predict(pd.DataFrame([features]))[0])

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model not trained.")

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path | None = None):
        """Save model and metadata to disk."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        if path is None:
            path = PROJECT_ROOT / "models" / "demand_forecaster"

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        joblib.dump(self.metrics, path / "metrics.joblib")
        joblib.dump(self.split_metadata, path / "split_metadata.joblib")
        logger.info("Demand forecaster saved to %s", path)

    def load(self, path: Path | None = None):
        """Load model and metadata from disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "demand_forecaster"

        self.model = joblib.load(path / "model.joblib")
        self.feature_names = joblib.load(path / "feature_names.joblib")
        self.metrics = joblib.load(path / "metrics.joblib")

        split_path = path / "split_metadata.joblib"
        self.split_metadata = (
            joblib.load(split_path) if split_path.exists() else {}
        )
        logger.info("Demand forecaster loaded from %s", path)
