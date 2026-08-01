"""
Raw data loading for the freight rate prediction project.

Deliberately does nothing beyond reading each CSV as-is: no cleaning, no
type coercion beyond what pandas infers by default, no imputation. The
whole point of the profiling step (src/profiling.py) is to inspect and
report on data quality
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Read a CSV file exactly as it is on disk.

    Args:
        path: Path to the CSV file.

    Returns:
        The raw DataFrame, with no cleaning or type coercion applied.

    Raises:
        FileNotFoundError: If path does not point to an existing file.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected data file not found at {path}. "
            f"Place the assessment's data files under {config.DATA_DIR}."
        )
    logger.info("Loading %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows, %d columns", path.name, df.shape[0], df.shape[1])
    return df


def load_train_test() -> pd.DataFrame:
    """Load train_test.csv, unmodified."""
    return _read_csv(config.TRAIN_TEST_PATH)


def load_validation() -> pd.DataFrame:
    """Load validation.csv, unmodified."""
    return _read_csv(config.VALIDATION_PATH)


def load_validation_template() -> pd.DataFrame:
    """Load validation_predictions_template.csv, unmodified."""
    return _read_csv(config.VALIDATION_TEMPLATE_PATH)


def load_december() -> pd.DataFrame:
    """Load december_chart_inputs.csv, unmodified."""
    return _read_csv(config.DECEMBER_PATH)


def load_train_test_cleaned() -> pd.DataFrame:
    """Load data/processed/train_test.csv, written by 02_cleaning.ipynb."""
    return _read_csv(config.TRAIN_TEST_CLEANED_PATH)


def load_validation_cleaned() -> pd.DataFrame:
    """Load data/processed/validation.csv, written by 02_cleaning.ipynb."""
    return _read_csv(config.VALIDATION_CLEANED_PATH)


def load_december_cleaned() -> pd.DataFrame:
    """Load data/processed/december.csv, written by 02_cleaning.ipynb."""
    return _read_csv(config.DECEMBER_CLEANED_PATH)