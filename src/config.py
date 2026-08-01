"""
Central configuration for the freight rate prediction project.
"""
from pathlib import Path

# src/config.py -> src/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Raw files, exactly as delivered for the assessment -- never overwritten.
TRAIN_TEST_PATH = RAW_DATA_DIR / "train_test.csv"
VALIDATION_PATH = RAW_DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = RAW_DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = RAW_DATA_DIR / "december_chart_inputs.csv"

# Processed files written by 02_cleaning.ipynb -- deterministic fixes
# only, no statistical treatments (see that notebook for what "cleaned"
# means here). Downstream notebooks (EDA onward) read from these.
TRAIN_TEST_CLEANED_PATH = PROCESSED_DATA_DIR / "train_test.csv"
VALIDATION_CLEANED_PATH = PROCESSED_DATA_DIR / "validation.csv"
DECEMBER_CLEANED_PATH = PROCESSED_DATA_DIR / "december.csv"

ID_COL = "load_id"
TARGET_COL = "posted_rate"
PREDICTION_COL = "predicted_rate"
DATE_COL = "date"

# Chronological train/tune/test cutoffs for train_test.csv, fixed here
# once so every model in the comparison is evaluated on identical
# splits. Decided in 03_eda.ipynb: October is the most recent month
# available and the closest analog to validation.csv's November, so it
# is reserved as the untouched final test slice; August-September is
# used to compare model/hyperparameter choices; January-July is the
# only data any model actually trains on. Inclusive on both ends.
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-07-31"
TUNE_START = "2025-08-01"
TUNE_END = "2025-09-30"
TEST_START = "2025-10-01"
TEST_END = "2025-10-31"