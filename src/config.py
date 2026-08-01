"""
Central configuration for the freight rate prediction project.
"""
from pathlib import Path

# src/config.py -> src/ -> repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = PROJECT_ROOT / "figures"

TRAIN_TEST_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

ID_COL = "load_id"
TARGET_COL = "posted_rate"
PREDICTION_COL = "predicted_rate"
DATE_COL = "date"