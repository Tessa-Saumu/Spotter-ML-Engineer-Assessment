"""
Evaluation metrics for the freight rate prediction project.

Reports RMSE, MAE, MAPE, and R^2 together rather than optimizing for
one silently -- agreed in the roadmap because each metric answers a
different question (RMSE: how costly are the worst misses; MAE:
typical-case dollar accuracy; MAPE: accuracy relative to a load's own
rate, useful given the wide $200-$1800+ range; R^2: quick cross-model
sanity check, not a stakeholder-facing number on its own). Every
function here takes true/predicted arrays and returns numbers -- no
plotting, no model fitting, no decision about which model is "best".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict:
    """
    Compute RMSE, MAE, MAPE, and R^2 for a set of predictions, reported
    together rather than one at a time, so no single metric is silently
    prioritized when comparing models.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values, same length/order as y_true.

    Returns:
        Dict with keys "rmse", "mae", "mape" (as a percentage, e.g. 8.2
        means 8.2%), "r2", and "n" (number of rows scored).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_pred - y_true
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / y_true)) * 100)

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    result = {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "n": len(y_true)}
    logger.info(
        "Metrics (n=%d): RMSE=%.2f, MAE=%.2f, MAPE=%.2f%%, R2=%.4f",
        result["n"], rmse, mae, mape, r2,
    )
    return result


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Combine several models' regression_metrics() outputs into a single
    comparison table, one row per model.

    Args:
        results: Dict mapping a model name to the dict returned by
            regression_metrics() for that model.

    Returns:
        DataFrame indexed by model name, columns rmse/mae/mape/r2/n,
        sorted ascending by rmse (best RMSE first -- a display
        convenience only, not an endorsement that RMSE is the metric
        that should decide the winner).
    """
    table = pd.DataFrame(results).T
    table = table.sort_values("rmse")
    return table


def residuals(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> np.ndarray:
    """
    Compute residuals (predicted - actual) for residual-plot diagnostics.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values, same length/order as y_true.

    Returns:
        Array of residuals, same length as y_true. Positive values mean
        the model over-predicted; negative means it under-predicted.
    """
    return np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)