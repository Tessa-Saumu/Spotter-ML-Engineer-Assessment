"""
Populate the two graded output files using the fitted Huber models
serialized by notebooks/08_final_model.ipynb.

Reads models/huber_full.pkl (trained on every column train_test.csv and
validation.csv share -- including market_index, quote_signal) and
models/huber_reduced.pkl (trained only on columns
december_chart_inputs.csv also has), plus the three fitted target/lane
encoders, and writes:
  - outputs/validation_predictions.csv  (load_id, predicted_rate)
  - outputs/december_chart_inputs.csv   (original 7 columns, predicted_rate filled)

Run after 08_final_model.ipynb has been executed at least once (the
.pkl files must already exist in models/). Does not retrain anything --
this script is pure inference, so it can be re-run cheaply any time the
raw validation.csv or december_chart_inputs.csv changes without
re-running any notebook.
"""
from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd

from src import config, data, features

logger = logging.getLogger(__name__)

FULL_NUMERIC_FEATURES = [
    "distance", "weight", "market_index", "quote_signal",
    "pickup_target_enc", "delivery_target_enc", "lane_target_enc",
    "day_of_year_sin", "day_of_year_cos",
]
FULL_CATEGORICAL_FEATURES = ["equipment"]

REDUCED_NUMERIC_FEATURES = [
    "distance", "weight",
    "pickup_target_enc", "delivery_target_enc", "lane_target_enc",
    "day_of_year_sin", "day_of_year_cos",
]
REDUCED_CATEGORICAL_FEATURES = ["equipment"]


def load_fitted_artifacts() -> dict:
    """
    Load the two fitted model pipelines and three fitted encoders
    serialized by 08_final_model.ipynb.

    Returns:
        Dict with keys "full_model", "reduced_model", "pickup_encoder",
        "delivery_encoder", "lane_encoder".

    Raises:
        FileNotFoundError: If any expected .pkl is missing -- run
            08_final_model.ipynb first.
    """
    required = {
        "full_model": config.MODELS_DIR / "huber_full.pkl",
        "reduced_model": config.MODELS_DIR / "huber_reduced.pkl",
        "pickup_encoder": config.MODELS_DIR / "pickup_encoder.pkl",
        "delivery_encoder": config.MODELS_DIR / "delivery_encoder.pkl",
        "lane_encoder": config.MODELS_DIR / "lane_encoder.pkl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing fitted model artifact(s): "
            f"{missing}. Run notebooks/08_final_model.ipynb first to create them."
        )

    logger.info("Loading fitted artifacts from %s", config.MODELS_DIR)
    return {name: joblib.load(path) for name, path in required.items()}


def populate_validation_predictions(artifacts: dict) -> pd.DataFrame:
    """
    Predict posted_rate for every row of validation.csv using the full
    model, and return a DataFrame in the exact load_id,predicted_rate
    format score.py's validate_predictions() checks for.

    Also reports how many rows hit a pickup/delivery/lane the fitted
    encoders never saw in train_test.csv -- validation.csv contains 8
    cities (Chicago, Charlotte, Jackson, San Diego, Knoxville, Laredo,
    Norfolk, Allentown) that never appear anywhere in train_test.csv,
    affecting roughly 12% of rows. Those rows fall back to the
    encoders' fitted global mean rather than a city/lane-specific
    estimate -- distance and equipment still drive the prediction
    normally, only the pickup/delivery/lane adjustment is lost. Worth
    surfacing explicitly here since it wasn't caught until this script
    ran; earlier notebooks only ever checked train/tune/test splits of
    train_test.csv against each other, never train_test.csv's city set
    against validation.csv's.

    Args:
        artifacts: Dict returned by load_fitted_artifacts().

    Returns:
        DataFrame with columns ["load_id", "predicted_rate"], one row
        per row of validation.csv, in the same order as the input file.
    """
    validation = data.load_validation_cleaned()

    unseen_pickup = int((~validation["pickup"].isin(artifacts["pickup_encoder"].index)).sum())
    unseen_delivery = int((~validation["delivery"].isin(artifacts["delivery_encoder"].index)).sum())
    lane = validation["pickup"] + " -> " + validation["delivery"]
    unseen_lane = int((~lane.isin(artifacts["lane_encoder"].index)).sum())
    logger.info(
        "validation.csv: %d/%d rows have an unseen pickup city, %d/%d an unseen delivery city, "
        "%d/%d an unseen lane -- these fall back to the fitted global mean for that encoding",
        unseen_pickup, len(validation), unseen_delivery, len(validation), unseen_lane, len(validation),
    )

    validation_feat = features.build_features(
        validation, artifacts["pickup_encoder"], artifacts["delivery_encoder"], artifacts["lane_encoder"]
    )
    X = validation_feat[FULL_NUMERIC_FEATURES + FULL_CATEGORICAL_FEATURES]

    predicted_rate = artifacts["full_model"].predict(X)
    result = pd.DataFrame({
        "load_id": validation_feat["load_id"],
        "predicted_rate": predicted_rate,
    })
    logger.info("Predicted posted_rate for %d validation rows", len(result))
    return result


def populate_december_predictions(artifacts: dict) -> pd.DataFrame:
    """
    Predict posted_rate for every row of december_chart_inputs.csv
    using the reduced model, and return a DataFrame keeping the
    original seven columns and column order score.py's
    validate_december() checks for.

    Args:
        artifacts: Dict returned by load_fitted_artifacts().

    Returns:
        DataFrame with columns
        ["pickup", "delivery", "distance", "equipment", "weight",
        "date", "predicted_rate"], one row per row of
        december_chart_inputs.csv, original column order preserved.
    """
    december = data.load_december_cleaned()
    original_columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]

    december_feat = features.build_features(
        december, artifacts["pickup_encoder"], artifacts["delivery_encoder"], artifacts["lane_encoder"]
    )
    X = december_feat[REDUCED_NUMERIC_FEATURES + REDUCED_CATEGORICAL_FEATURES]

    predicted_rate = artifacts["reduced_model"].predict(X)
    result = december_feat.copy()
    result["predicted_rate"] = predicted_rate
    logger.info("Predicted posted_rate for %d December rows", len(result))
    return result[original_columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate validation_predictions.csv and december_chart_inputs.csv using the fitted Huber models."
    )
    parser.add_argument(
        "--output-dir", default=str(config.OUTPUTS_DIR),
        help="Directory to write both output CSVs to (default: outputs/).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    artifacts = load_fitted_artifacts()

    validation_predictions = populate_validation_predictions(artifacts)
    december_predictions = populate_december_predictions(artifacts)

    output_dir = config.PROJECT_ROOT / args.output_dir if not args.output_dir.startswith("/") else __import__("pathlib").Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = output_dir / "validation_predictions.csv"
    december_path = output_dir / "december_chart_inputs.csv"

    validation_predictions.to_csv(validation_path, index=False)
    december_predictions.to_csv(december_path, index=False)

    print(f"Wrote {len(validation_predictions):,} rows to {validation_path}")
    print(f"Wrote {len(december_predictions):,} rows to {december_path}")
    print()
    print(
        "Note: validation.csv contains 8 cities never seen in train_test.csv "
        "(Chicago, Charlotte, Jackson, San Diego, Knoxville, Laredo, Norfolk, Allentown), "
        "affecting ~1,447 rows (~12%) whose pickup/delivery/lane encoding falls back to the "
        "fitted global mean -- distance and equipment still drive those predictions normally. "
        "See the INFO log lines above for exact counts."
    )
    print("Next: run score.py --predictions <validation_path> --december-predictions <december_path>")


if __name__ == "__main__":
    main()
