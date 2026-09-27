"""Train and persist production models used by the pricing engine."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.models.anomaly_detector import (
    ANOMALY_FEATURES,
    AnomalyDetector,
)
from src.models.demand_forecaster import (
    DEMAND_FEATURES,
    DemandForecaster,
)
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_MAX_ROWS = 500_000
PARQUET_BATCH_SIZE = 250_000


def _even_parquet_sample(
    path: Path,
    columns: list[str],
    max_rows: int,
) -> pd.DataFrame:
    """Read an evenly distributed deterministic sample from a Parquet file.

    This avoids materializing the full multi-million-row calendar table
    merely to obtain a development/training sample.
    """
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows

    available_columns = [
        column for column in dict.fromkeys(columns)
        if column in parquet.schema.names
    ]

    missing = sorted(set(columns) - set(available_columns))
    if missing:
        logger.warning(f"Columns not present and skipped: {missing}")

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    if total_rows <= max_rows:
        logger.info(
            f"Reading all {total_rows:,} rows from {path.name}"
        )
        return pd.read_parquet(path, columns=available_columns)

    selected_rows = np.linspace(
        0,
        total_rows - 1,
        num=max_rows,
        dtype=np.int64,
    )

    pieces = []
    global_offset = 0

    for batch in parquet.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=available_columns,
    ):
        batch_end = global_offset + batch.num_rows

        left = np.searchsorted(
            selected_rows, global_offset, side="left"
        )
        right = np.searchsorted(
            selected_rows, batch_end, side="left"
        )

        if right > left:
            local_indices = (
                selected_rows[left:right] - global_offset
            )
            table = pa.Table.from_batches([batch])
            selected = table.take(
                pa.array(local_indices, type=pa.int64())
            )
            pieces.append(selected.to_pandas())

        global_offset = batch_end

    result = pd.concat(pieces, ignore_index=True)

    logger.info(
        f"Loaded deterministic sample: "
        f"{len(result):,} / {total_rows:,} rows"
    )
    return result


def _json_safe(mapping: dict) -> dict:
    """Convert NumPy scalar values to ordinary Python scalars."""
    converted = {}
    for key, value in mapping.items():
        if isinstance(value, np.integer):
            converted[key] = int(value)
        elif isinstance(value, np.floating):
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted


def train_all(max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    """Train demand and anomaly models used in production."""
    config = load_config()

    calendar_path = (
        PROJECT_ROOT / "data" / "processed"
        / "calendar_features.parquet"
    )
    listings_path = (
        PROJECT_ROOT / "data" / "processed"
        / "listings_features.parquet"
    )

    if not calendar_path.exists():
        raise FileNotFoundError(
            f"{calendar_path} not found. "
            "Run feature engineering first."
        )

    if not listings_path.exists():
        raise FileNotFoundError(
            f"{listings_path} not found. "
            "Run feature engineering first."
        )

    calendar_columns = list(
        dict.fromkeys(
            [
                "listing_id",
                "date",
                "price",
                "was_booked",
                "is_rainy_day",
            ]
            + DEMAND_FEATURES
            + ANOMALY_FEATURES
        )
    )

    listings_columns = [
        "id",
        "room_type_encoded",
        "beds",
        "bathrooms",
        "amenity_score",
        "review_score",
        "location_cluster",
    ]

    logger.info(
        f"Loading calendar training sample "
        f"(maximum {max_rows:,} rows)..."
    )
    calendar_df = _even_parquet_sample(
        calendar_path,
        calendar_columns,
        max_rows,
    )

    logger.info("Loading listing-level features...")
    listings_df = pd.read_parquet(
        listings_path,
        columns=listings_columns,
    )

    # TimeSeriesSplit assumes observations are chronologically ordered.
    calendar_df["date"] = pd.to_datetime(
        calendar_df["date"],
        errors="coerce",
    )
    calendar_df = (
        calendar_df
        .dropna(subset=["date"])
        .sort_values(
            ["date", "listing_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Training XGBoost demand forecaster "
        "with temporal cross-validation..."
    )
    demand = DemandForecaster(config=config)
    demand_metrics = demand.train(
        calendar_df,
        listings_df,
        tune_hyperparams=False,
    )
    demand.save()

    logger.info("Training Isolation Forest anomaly detector...")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    anomaly = AnomalyDetector(config=config)

    with mlflow.start_run(run_name="anomaly_detector"):
        anomaly.fit(calendar_df)

        anomaly_flags = anomaly.predict(calendar_df)
        anomaly_rate = float(np.mean(anomaly_flags))

        mlflow.log_param(
            "contamination",
            anomaly.contamination,
        )
        mlflow.log_param(
            "training_rows",
            len(calendar_df),
        )
        mlflow.log_metric(
            "training_anomaly_rate",
            anomaly_rate,
        )

        mlflow.sklearn.log_model(
            anomaly.model,
            "anomaly_model",
        )

    anomaly.save()

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "training_rows": int(len(calendar_df)),
        "demand": _json_safe(demand_metrics),
        "anomaly": {
            "training_anomaly_rate": anomaly_rate,
            "contamination": float(anomaly.contamination),
        },
    }

    summary_path = models_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2)
    )

    logger.info("=== Training complete ===")
    logger.info(
        f"Demand metrics: {summary['demand']}"
    )
    logger.info(
        f"Anomaly metrics: {summary['anomaly']}"
    )
    logger.info(
        f"Summary saved to {summary_path}"
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train production demand and anomaly models."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            "Maximum calendar rows used for training. "
            f"Default: {DEFAULT_MAX_ROWS:,}"
        ),
    )
    args = parser.parse_args()

    train_all(max_rows=args.max_rows)


if __name__ == "__main__":
    main()
