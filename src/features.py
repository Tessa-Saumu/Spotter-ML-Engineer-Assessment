"""
Feature engineering for the freight rate prediction project.

Every function here either derives a feature deterministically from
columns already present (safe to apply to any split, any time), or
fits a statistic on one DataFrame (train) and applies it to another
(tune/test/validation/december) -- never fits and transforms the same
data in one call, so leakage is structurally impossible to introduce by
accident. Fitted encoders are returned alongside the transformed
DataFrame specifically so a caller can re-apply the exact same fitted
encoder to a different split, rather than re-fitting per split.

Every feature built here uses only columns present in every one of
train_test.csv, validation.csv, and december_chart_inputs.csv (pickup,
delivery, distance, equipment, weight, date) -- none of them depend on
pickup_lat/lon, delivery_lat/lon, market_index, or quote_signal, so the
same feature set works for both the "full" model (validation.csv, which
also has the columns this module doesn't touch) and the "reduced"
December model, per the two-model split agreed in the roadmap.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYS_IN_YEAR = 365.25


def add_cyclical_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add sin/cos encodings of day-of-year, a smooth, continuous
    seasonality signal rather than a categorical month feature.

    A categorical month (one-hot or ordinal 1-12) would ask a model to
    predict on categories it never saw in training -- train_test.csv
    only covers January through October, so November and December
    (validation.csv, december_chart_inputs.csv) would be entirely
    unseen categories. sin/cos of day-of-year instead produces
    continuous values that fall naturally in-range for November/
    December even though those specific dates were never trained on,
    since the function is smooth and periodic rather than a lookup
    table keyed by month identity.

    Args:
        df: DataFrame containing date_col (string or datetime dtype).
        date_col: Name of the date column.

    Returns:
        Copy of df with two new columns, "day_of_year_sin" and
        "day_of_year_cos".
    """
    result = df.copy()
    parsed_date = pd.to_datetime(result[date_col])
    day_of_year = parsed_date.dt.dayofyear
    result["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / DAYS_IN_YEAR)
    result["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / DAYS_IN_YEAR)
    logger.info("Added cyclical date features to %d rows", len(result))
    return result


def fit_target_encoder(
    df: pd.DataFrame, group_col: str, target_col: str, smoothing: float = 10.0
) -> pd.Series:
    """
    Fit a smoothed mean-target encoder for one categorical column,
    fit on df only -- must be called on the train split, never on
    tune/test/validation/december, to avoid leakage. Every city here
    (pickup and delivery both have a minimum of 276 occurrences in
    train_test.csv, confirmed before building this) has enough support
    that smoothing mainly guards against any category that turns out
    rarer in a future split than it was in train, not a live problem in
    train_test.csv itself.

    Args:
        df: DataFrame to fit on (the train split only).
        group_col: Categorical column to encode, e.g. "pickup".
        target_col: Numeric target column, e.g. "posted_rate".
        smoothing: Higher values pull a category's encoded value closer
            to the global mean, protecting against categories with few
            rows. 10.0 is a light touch given every category here has
            hundreds of rows at minimum.

    Returns:
        Series indexed by category value, giving the smoothed mean
        target for that category. Pass this to apply_target_encoder()
        for df itself and for every other split.
    """
    global_mean = df[target_col].mean()
    group_stats = df.groupby(group_col)[target_col].agg(["mean", "count"])
    smoothed = (group_stats["mean"] * group_stats["count"] + global_mean * smoothing) / (
        group_stats["count"] + smoothing
    )
    logger.info(
        "Fit target encoder for %s on %d rows, %d categories, global_mean=%.2f",
        group_col, len(df), len(smoothed), global_mean,
    )
    smoothed.attrs["global_mean"] = global_mean
    return smoothed


def apply_target_encoder(
    df: pd.DataFrame, group_col: str, encoder: pd.Series, output_col: str | None = None
) -> pd.DataFrame:
    """
    Apply a target encoder fit by fit_target_encoder() to any split.
    Categories not seen during fitting (shouldn't occur here -- every
    city in validation.csv and december_chart_inputs.csv also appears
    in train_test.csv, but checked defensively) fall back to the fitted
    global mean rather than raising or producing NaN.

    Args:
        df: DataFrame to transform. Can be the same DataFrame passed to
            fit_target_encoder(), or a different split entirely.
        group_col: Categorical column to encode, matching what was
            passed to fit_target_encoder().
        encoder: The Series returned by fit_target_encoder().
        output_col: Name for the new encoded column. Defaults to
            f"{group_col}_target_enc".

    Returns:
        Copy of df with one new numeric column.
    """
    output_col = output_col or f"{group_col}_target_enc"
    result = df.copy()
    global_mean = encoder.attrs.get("global_mean", encoder.mean())
    result[output_col] = result[group_col].map(encoder).fillna(global_mean)

    n_unseen = int(result[group_col].map(encoder).isna().sum())
    if n_unseen > 0:
        logger.info("%d rows in %s had categories unseen by the encoder, filled with global mean", n_unseen, group_col)
    return result


def fit_lane_encoder(
    df: pd.DataFrame, pickup_col: str, delivery_col: str, target_col: str, smoothing: float = 20.0
) -> pd.Series:
    """
    Fit a smoothed mean-target encoder for the pickup-delivery lane
    (e.g. "Richmond -> Baltimore"), fit on df only (train split).

    Lanes are far sparser than individual cities -- 4,014 distinct
    lanes across 48,000 train_test.csv rows, median 10 occurrences,
    537 lanes with fewer than 5 -- confirmed before building this, and
    too sparse for a raw categorical feature (one-hot would add
    thousands of columns) or even a lightly-smoothed target encoding.
    Smoothing is set higher here than fit_target_encoder()'s default
    specifically because of that sparsity -- a lane seen twice should
    lean heavily on the global mean, not its own noisy 2-row average.

    Args:
        df: DataFrame to fit on (the train split only).
        pickup_col: Name of the pickup city column.
        delivery_col: Name of the delivery city column.
        target_col: Numeric target column, e.g. "posted_rate".
        smoothing: Higher than fit_target_encoder()'s default (20.0 vs.
            10.0) because lanes have far less support per category than
            individual cities do.

    Returns:
        Series indexed by (pickup, delivery) tuples, giving the
        smoothed mean target for that lane. Pass to apply_lane_encoder().
    """
    lane = df[pickup_col].astype(str) + " -> " + df[delivery_col].astype(str)
    global_mean = df[target_col].mean()
    group_stats = df.groupby(lane)[target_col].agg(["mean", "count"])
    smoothed = (group_stats["mean"] * group_stats["count"] + global_mean * smoothing) / (
        group_stats["count"] + smoothing
    )
    logger.info(
        "Fit lane encoder on %d rows, %d distinct lanes, global_mean=%.2f",
        len(df), len(smoothed), global_mean,
    )
    smoothed.attrs["global_mean"] = global_mean
    return smoothed


def apply_lane_encoder(
    df: pd.DataFrame, pickup_col: str, delivery_col: str, encoder: pd.Series, output_col: str = "lane_target_enc"
) -> pd.DataFrame:
    """
    Apply a lane encoder fit by fit_lane_encoder() to any split. Lanes
    not seen during fitting fall back to the fitted global mean.

    Args:
        df: DataFrame to transform.
        pickup_col: Name of the pickup city column.
        delivery_col: Name of the delivery city column.
        encoder: The Series returned by fit_lane_encoder().
        output_col: Name for the new encoded column.

    Returns:
        Copy of df with one new numeric column.
    """
    result = df.copy()
    lane = result[pickup_col].astype(str) + " -> " + result[delivery_col].astype(str)
    global_mean = encoder.attrs.get("global_mean", encoder.mean())
    result[output_col] = lane.map(encoder).fillna(global_mean)

    n_unseen = int(lane.map(encoder).isna().sum())
    if n_unseen > 0:
        logger.info("%d rows had lanes unseen by the encoder, filled with global mean", n_unseen)
    return result


def build_features(
    df: pd.DataFrame,
    pickup_encoder: pd.Series,
    delivery_encoder: pd.Series,
    lane_encoder: pd.Series,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Apply the full engineered feature set (cyclical date features,
    pickup/delivery target encoding, lane target encoding) to a
    DataFrame in one call, using encoders already fit on the train
    split. This is the function every notebook downstream of feature
    engineering should call -- fit the three encoders once on train via
    fit_target_encoder()/fit_lane_encoder(), then call build_features()
    identically on train, tune, test, validation, and december.

    Args:
        df: DataFrame to transform. Must contain "pickup", "delivery",
            and date_col.
        pickup_encoder: Fitted encoder from
            fit_target_encoder(train, "pickup", target_col).
        delivery_encoder: Fitted encoder from
            fit_target_encoder(train, "delivery", target_col).
        lane_encoder: Fitted encoder from fit_lane_encoder(train, ...).
        date_col: Name of the date column.

    Returns:
        Copy of df with day_of_year_sin, day_of_year_cos,
        pickup_target_enc, delivery_target_enc, and lane_target_enc
        columns added.
    """
    result = add_cyclical_date_features(df, date_col=date_col)
    result = apply_target_encoder(result, "pickup", pickup_encoder, "pickup_target_enc")
    result = apply_target_encoder(result, "delivery", delivery_encoder, "delivery_target_enc")
    result = apply_lane_encoder(result, "pickup", "delivery", lane_encoder, "lane_target_enc")
    return result