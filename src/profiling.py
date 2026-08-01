"""
Data profiling functions for the freight rate prediction project.

Profiling answers "what exactly are we looking at" -- the structure,
completeness, uniqueness, validity, and internal consistency of the raw
data. It deliberately does NOT look for relationships with the target or
between predictors (that is exploratory data analysis, a separate, later
step). Every function here either describes a column on its own terms,
or checks whether two columns that should structurally agree actually do
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def describe_structure(df: pd.DataFrame, name: str, n_samples: int = 5) -> dict:
    """
    Report basic shape, column names, dtypes, and a sample of rows.

    Args:
        df: DataFrame to describe.
        name: Label for this dataset, used in log messages and the
            returned dict.
        n_samples: Number of sample rows to include in the result.

    Returns:
        Dict with keys "name", "n_rows", "n_columns", "columns",
        "dtypes" (column -> dtype string), and "sample" (DataFrame of
        the first n_samples rows).
    """
    logger.info("Describing structure of %s", name)
    return {
        "name": name,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(n_samples),
    }


def check_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count and percentage of missing values per column.

    Args:
        df: DataFrame to check.

    Returns:
        DataFrame indexed by column name with columns "n_missing" and
        "pct_missing", sorted descending by pct_missing. Columns with
        zero missing values are still included, for completeness.
    """
    logger.info("Checking missingness across %d columns", df.shape[1])
    n_missing = df.isna().sum()
    pct_missing = (df.isna().mean() * 100).round(3)
    result = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    result = result.sort_values("pct_missing", ascending=False)
    logger.info("Columns with missing values: %d", int((n_missing > 0).sum()))
    return result


def check_uniqueness(df: pd.DataFrame, id_col: str) -> dict:
    """
    Check whether an identifier column is unique, and count exact
    duplicate rows across all columns.

    Args:
        df: DataFrame to check.
        id_col: Name of the column expected to be a unique identifier.

    Returns:
        Dict with keys "id_col", "n_rows", "n_unique_ids", "is_unique",
        "n_duplicate_ids", "n_duplicate_rows".
    """
    logger.info("Checking uniqueness for id_col=%s", id_col)
    result = {
        "id_col": id_col,
        "n_rows": len(df),
        "n_unique_ids": int(df[id_col].nunique()),
        "is_unique": bool(df[id_col].is_unique),
        "n_duplicate_ids": int(df[id_col].duplicated().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
    }
    logger.info("Uniqueness result: %s", result)
    return result


def check_domain_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for physically impossible values in whichever columns are
    present: non-positive distance/weight/posted_rate, out-of-range
    latitude/longitude, and pickup equal to delivery.

    Only checks columns that actually exist in df, so this is safe to
    call on train_test.csv, validation.csv, or december_chart_inputs.csv
    even though they don't all share the same columns.

    Args:
        df: DataFrame to check.

    Returns:
        DataFrame with one row per check performed, columns "check",
        "n_flagged", and "description".
    """
    logger.info("Running domain/range checks")
    checks = []

    for col in ("distance", "weight", "posted_rate"):
        if col in df.columns:
            n_flagged = int((df[col] <= 0).sum())
            checks.append({
                "check": f"{col}_non_positive",
                "n_flagged": n_flagged,
                "description": f"Rows where {col} <= 0 (physically impossible)",
            })

    for lat_col in ("pickup_lat", "delivery_lat"):
        if lat_col in df.columns:
            n_flagged = int((~df[lat_col].between(-90, 90)).sum())
            checks.append({
                "check": f"{lat_col}_out_of_range",
                "n_flagged": n_flagged,
                "description": f"Rows where {lat_col} is outside [-90, 90]",
            })

    for lon_col in ("pickup_lon", "delivery_lon"):
        if lon_col in df.columns:
            n_flagged = int((~df[lon_col].between(-180, 180)).sum())
            checks.append({
                "check": f"{lon_col}_out_of_range",
                "n_flagged": n_flagged,
                "description": f"Rows where {lon_col} is outside [-180, 180]",
            })

    if "pickup" in df.columns and "delivery" in df.columns:
        n_flagged = int((df["pickup"] == df["delivery"]).sum())
        checks.append({
            "check": "pickup_equals_delivery",
            "n_flagged": n_flagged,
            "description": "Rows where pickup and delivery are the same city",
        })

    result = pd.DataFrame(checks)
    logger.info("Domain/range checks complete: %d checks run", len(checks))
    return result


def check_cardinality(df: pd.DataFrame, categorical_cols: list[str], top_n: int = 10) -> dict:
    """
    Report the number of distinct values and the most frequent values
    for each categorical column.

    Args:
        df: DataFrame to check.
        categorical_cols: Names of categorical columns to profile.
        top_n: Number of most-frequent values to include per column.

    Returns:
        Dict keyed by column name; each value is a dict with "n_unique"
        (int) and "top_values" (pandas Series of value -> count).
    """
    logger.info("Checking cardinality for columns: %s", categorical_cols)
    result = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        result[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(top_n),
        }
    return result


def describe_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Plain descriptive statistics (count, mean, std, min, quartiles, max)
    for the given numeric columns. Purely tabular -- no comment on
    shape/skew and no visualization; whether any of this warrants a
    transform is an EDA question, not a profiling one.

    Args:
        df: DataFrame to describe.
        numeric_cols: Names of numeric columns to describe.

    Returns:
        The standard pandas .describe() output, transposed so each row
        corresponds to one column of df.
    """
    present = [c for c in numeric_cols if c in df.columns]
    logger.info("Describing numeric columns: %s", present)
    return df[present].describe().T


def _haversine_miles(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """
    Great-circle distance in miles between two lat/lon points (vectorized).

    Args:
        lat1: Latitude of the first point(s), in degrees.
        lon1: Longitude of the first point(s), in degrees.
        lat2: Latitude of the second point(s), in degrees.
        lon2: Longitude of the second point(s), in degrees.

    Returns:
        Series of great-circle distances, in miles.
    """
    radius_miles = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return 2 * radius_miles * np.arcsin(np.sqrt(a))


def check_distance_consistency(
    df: pd.DataFrame,
    distance_col: str = "distance",
    pickup_lat_col: str = "pickup_lat",
    pickup_lon_col: str = "pickup_lon",
    delivery_lat_col: str = "delivery_lat",
    delivery_lon_col: str = "delivery_lon",
) -> dict:
    """
    Check whether `distance_col` is internally consistent with the
    great-circle distance implied by the pickup/delivery coordinates.
    This is a data-quality question -- do two columns that should
    structurally relate actually agree -- not a question about what
    predicts the target.

    Args:
        df: DataFrame containing distance_col and the four coordinate
            columns.
        distance_col: Name of the given distance column.
        pickup_lat_col: Name of the pickup latitude column.
        pickup_lon_col: Name of the pickup longitude column.
        delivery_lat_col: Name of the delivery latitude column.
        delivery_lon_col: Name of the delivery longitude column.

    Returns:
        Dict with "skipped" (bool, True if required columns are
        missing), and if not skipped: "correlation" (Pearson
        correlation between distance_col and the computed great-circle
        distance), "ratio_describe" (descriptive stats of
        distance_col / great_circle_distance), and "flagged_rows" (rows
        where that ratio exceeds 2x).
    """
    required = [distance_col, pickup_lat_col, pickup_lon_col, delivery_lat_col, delivery_lon_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.info("Skipping distance consistency check, missing columns: %s", missing)
        return {"skipped": True, "missing_columns": missing}

    logger.info("Checking distance-vs-coordinates consistency")
    great_circle = _haversine_miles(
        df[pickup_lat_col], df[pickup_lon_col], df[delivery_lat_col], df[delivery_lon_col]
    )
    ratio = df[distance_col] / great_circle
    flagged = df.loc[ratio > 2].copy()
    flagged["great_circle_distance"] = great_circle[ratio > 2]
    flagged["ratio"] = ratio[ratio > 2]

    result = {
        "skipped": False,
        "correlation": float(df[distance_col].corr(great_circle)),
        "ratio_describe": ratio.describe(),
        "flagged_rows": flagged,
    }
    logger.info(
        "Distance consistency: correlation=%.4f, %d rows flagged (ratio > 2x)",
        result["correlation"], len(flagged),
    )
    return result


def check_coordinate_stability(
    df: pd.DataFrame, city_col: str, lat_col: str, lon_col: str, min_occurrences: int = 2
) -> dict:
    """
    Check whether lat/lon is a stable, deterministic function of city
    name -- does every occurrence of a given city have the same
    coordinates, or does it vary row to row.

    Args:
        df: DataFrame containing city_col, lat_col, lon_col.
        city_col: Name of the city column (e.g. "pickup").
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        min_occurrences: Only cities appearing at least this many times
            are included -- a city seen once trivially has zero
            variation, which isn't informative.

    Returns:
        Dict with "skipped" (bool), and if not skipped: "n_cities_checked",
        "max_lat_std", "max_lon_std", and "is_fully_stable" (True if both
        max standard deviations round to 0.0).
    """
    if not all(c in df.columns for c in (city_col, lat_col, lon_col)):
        logger.info("Skipping coordinate stability check for %s, columns missing", city_col)
        return {"skipped": True}

    logger.info("Checking coordinate stability for %s", city_col)
    counts = df[city_col].value_counts()
    repeated = counts[counts >= min_occurrences].index
    subset = df[df[city_col].isin(repeated)]
    stability = subset.groupby(city_col)[[lat_col, lon_col]].std()

    max_lat_std = float(stability[lat_col].max())
    max_lon_std = float(stability[lon_col].max())
    result = {
        "skipped": False,
        "n_cities_checked": len(repeated),
        "max_lat_std": max_lat_std,
        "max_lon_std": max_lon_std,
        "is_fully_stable": round(max_lat_std, 6) == 0.0 and round(max_lon_std, 6) == 0.0,
    }
    logger.info("Coordinate stability (%s): %s", city_col, result)
    return result


def check_date_coverage(df: pd.DataFrame, date_col: str, name: str) -> dict:
    """
    Parse a date column and report its coverage.

    Args:
        df: DataFrame containing date_col.
        date_col: Name of the date column.
        name: Label for this dataset, used in log messages and the
            returned dict.

    Returns:
        Dict with "name", "n_rows", "n_unparseable" (values that failed
        to parse as dates), "min_date", "max_date", and "n_unique_dates".
    """
    logger.info("Checking date coverage for %s", name)
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    n_unparseable = int(parsed.isna().sum())
    result = {
        "name": name,
        "n_rows": len(df),
        "n_unparseable": n_unparseable,
        "min_date": parsed.min(),
        "max_date": parsed.max(),
        "n_unique_dates": int(parsed.nunique()),
    }
    logger.info("Date coverage (%s): %s", name, result)
    return result
