"""Demand Forecaster using XGBoost.

Predicts occupancy rate (probability of booking) for a listing
on a given date, using temporal, weather, holiday, and listing features.

Uses TimeSeriesSplit for cross-validation to prevent data leakage.
All runs are tracked with MLflow.
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Features used for demand prediction
DEMAND_FEATURES = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "season", "days_from_start",
    "temperature_mean", "precipitation_sum", "wind_speed_max",
    "is_hot_day", "is_cold_day", "is_rainy_day",
    "is_holiday", "days_until_holiday", "near_holiday",
    "rolling_7d_occupancy", "rolling_30d_occupancy",
]

# Hyperparameter search space for RandomizedSearchCV
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
    """XGBoost-based demand forecaster with MLflow tracking."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.config = config
        self.model_cfg = config["model"]
        self.model = None
        self.feature_names = None
        self.metrics = {}

    def _get_available_features(self, df: pd.DataFrame) -> list[str]:
        """Return the intersection of expected features and available columns."""
        available = [f for f in DEMAND_FEATURES if f in df.columns]
        if not available:
            raise ValueError(
                f"No demand features found in DataFrame. "
                f"Expected some of: {DEMAND_FEATURES}. "
                f"Got columns: {df.columns.tolist()}"
            )
        return available

    def _prepare_data(
        self, calendar_df: pd.DataFrame, listings_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target y from calendar data."""
        df = calendar_df.copy()

        # Merge listing-level features if provided
        if listings_df is not None and "listing_id" in df.columns:
            listing_cols = ["id", "room_type_encoded", "beds", "bathrooms",
                           "amenity_score", "review_score", "location_cluster",
                           "occupancy_rate", "price_rank_in_neighborhood",
                           "price_vs_neighborhood"]
            merge_cols = [c for c in listing_cols if c in listings_df.columns]
            if merge_cols:
                df = df.merge(
                    listings_df[merge_cols],
                    left_on="listing_id",
                    right_on="id",
                    how="left",
                )

        # Target
        if "was_booked" not in df.columns:
            raise ValueError("Target column 'was_booked' not found in calendar data")
        y = df["was_booked"].astype(int)

        # Features
        self.feature_names = self._get_available_features(df)
        X = df[self.feature_names].copy()

        # Fill NaN with 0 for features
        X = X.fillna(0)

        logger.info(f"Prepared {X.shape[0]:,} samples with {X.shape[1]} features")
        return X, y

    def train(
        self,
        calendar_df: pd.DataFrame,
        listings_df: pd.DataFrame | None = None,
        tune_hyperparams: bool = False,
        n_iter: int = 20,
    ) -> dict:
        """Train with genuine temporal cross-validation.

        A separate XGBoost model is fitted for every TimeSeriesSplit fold.
        The validation fold is therefore always scored by a model trained
        only on observations preceding that fold.

        After cross-validation, a final production model is fitted on all
        supplied observations.
        """
        df = calendar_df.copy()

        # Ensure temporal ordering whenever dates are available.
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

            sort_cols = ["date"]
            if "listing_id" in df.columns:
                sort_cols.append("listing_id")

            df = (
                df.sort_values(sort_cols, kind="mergesort")
                .reset_index(drop=True)
            )

        X, y = self._prepare_data(df, listings_df)

        n_splits = self.model_cfg["cv_splits"]
        if len(X) <= n_splits:
            raise ValueError(
                f"Need more than {n_splits} rows for TimeSeriesSplit; "
                f"got {len(X)}."
            )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        xgb_cfg = self.model_cfg["xgboost"]

        base_params = {
            "n_estimators": xgb_cfg["n_estimators"],
            "max_depth": xgb_cfg["max_depth"],
            "learning_rate": xgb_cfg["learning_rate"],
            "subsample": xgb_cfg["subsample"],
            "colsample_bytree": xgb_cfg["colsample_bytree"],
            "random_state": self.model_cfg["random_state"],
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }

        mlflow.set_tracking_uri(
            self.config["mlflow"]["tracking_uri"]
        )
        mlflow.set_experiment(
            self.config["mlflow"]["experiment_name"]
        )

        with mlflow.start_run(run_name="demand_forecaster"):
            selected_params = dict(base_params)

            if tune_hyperparams:
                logger.info(
                    f"Running RandomizedSearchCV "
                    f"with {n_iter} iterations..."
                )

                search_model = XGBClassifier(
                    random_state=self.model_cfg["random_state"],
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                )

                search = RandomizedSearchCV(
                    search_model,
                    PARAM_DISTRIBUTIONS,
                    n_iter=n_iter,
                    cv=tscv,
                    scoring="roc_auc",
                    random_state=self.model_cfg["random_state"],
                    n_jobs=-1,
                    verbose=1,
                )

                search.fit(X, y)
                selected_params.update(search.best_params_)

                logger.info(
                    f"Best params: {search.best_params_}"
                )

            mlflow.log_params(selected_params)

            fold_metrics = []
            best_iterations = []

            # IMPORTANT:
            # fit an independent model for every temporal fold.
            for fold_i, (train_idx, val_idx) in enumerate(
                tscv.split(X),
                start=1,
            ):
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]

                fold_params = dict(selected_params)
                fold_params["early_stopping_rounds"] = (
                    xgb_cfg["early_stopping_rounds"]
                )

                fold_model = XGBClassifier(**fold_params)

                fold_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                y_pred_proba = (
                    fold_model.predict_proba(X_val)[:, 1]
                )
                y_pred = fold_model.predict(X_val)

                fold_auc = roc_auc_score(
                    y_val,
                    y_pred_proba,
                )
                fold_acc = accuracy_score(
                    y_val,
                    y_pred,
                )
                fold_f1 = f1_score(
                    y_val,
                    y_pred,
                )

                best_iteration = getattr(
                    fold_model,
                    "best_iteration",
                    None,
                )

                if best_iteration is None:
                    n_trees = int(
                        selected_params["n_estimators"]
                    )
                else:
                    n_trees = int(best_iteration) + 1

                best_iterations.append(n_trees)

                fold_metrics.append(
                    {
                        "fold": fold_i,
                        "auc": float(fold_auc),
                        "accuracy": float(fold_acc),
                        "f1": float(fold_f1),
                        "best_n_estimators": n_trees,
                    }
                )

                logger.info(
                    f"Fold {fold_i}: "
                    f"AUC={fold_auc:.4f}, "
                    f"Acc={fold_acc:.4f}, "
                    f"F1={fold_f1:.4f}, "
                    f"trees={n_trees}"
                )

                mlflow.log_metric(
                    f"fold_{fold_i}_auc",
                    float(fold_auc),
                )
                mlflow.log_metric(
                    f"fold_{fold_i}_accuracy",
                    float(fold_acc),
                )
                mlflow.log_metric(
                    f"fold_{fold_i}_f1",
                    float(fold_f1),
                )

            self.metrics = {
                "auc": float(
                    np.mean(
                        [m["auc"] for m in fold_metrics]
                    )
                ),
                "accuracy": float(
                    np.mean(
                        [m["accuracy"] for m in fold_metrics]
                    )
                ),
                "f1": float(
                    np.mean(
                        [m["f1"] for m in fold_metrics]
                    )
                ),
                "auc_std": float(
                    np.std(
                        [m["auc"] for m in fold_metrics]
                    )
                ),
            }

            logger.info(
                f"Mean AUC: {self.metrics['auc']:.4f} "
                f"+/- {self.metrics['auc_std']:.4f}"
            )

            mlflow.log_metrics(self.metrics)

            # Use the typical early-stopping result for the final model.
            final_n_estimators = max(
                1,
                int(np.median(best_iterations)),
            )

            mlflow.log_param(
                "final_n_estimators",
                final_n_estimators,
            )

            final_params = dict(selected_params)
            final_params["n_estimators"] = (
                final_n_estimators
            )

            # Final artifact is trained on all supplied data.
            # No eval set is used here.
            final_params.pop(
                "early_stopping_rounds",
                None,
            )

            self.model = XGBClassifier(**final_params)
            self.model.fit(X, y)

            mlflow.sklearn.log_model(
                self.model,
                "demand_model",
            )

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict booking probability for given features.

        Args:
            X: DataFrame with demand features.

        Returns:
            Array of booking probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        features = [f for f in self.feature_names if f in X.columns]
        X_input = X[features].fillna(0)
        return self.model.predict_proba(X_input)[:, 1]

    def predict_single(self, features: dict) -> float:
        """Predict booking probability for a single observation."""
        df = pd.DataFrame([features])
        return float(self.predict(df)[0])

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance

    def save(self, path: Path | None = None):
        """Save model and metadata to disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "demand_forecaster"
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        joblib.dump(self.metrics, path / "metrics.joblib")
        logger.info(f"Demand forecaster saved to {path}")

    def load(self, path: Path | None = None):
        """Load model and metadata from disk."""
        if path is None:
            path = PROJECT_ROOT / "models" / "demand_forecaster"
        self.model = joblib.load(path / "model.joblib")
        self.feature_names = joblib.load(path / "feature_names.joblib")
        self.metrics = joblib.load(path / "metrics.joblib")
        logger.info(f"Demand forecaster loaded from {path}")
