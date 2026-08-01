"""
Shared plotting functions for the freight rate prediction project.

Every function here saves its figure to disk under figures/<stage>/ (via
_savefig) and returns the Matplotlib Figure so a notebook can also
display it inline. Nothing in this module makes a modeling or cleaning
decision -- it only renders whatever data it is given. Deciding what a
plot means belongs in the notebook that calls it, not here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

DEFAULT_DPI = 150


def _savefig(fig: plt.Figure, stage: str, filename: str, dpi: int = DEFAULT_DPI) -> Path:
    """
    Save a figure under figures/<stage>/<filename>, creating the stage
    subfolder if it doesn't exist yet.

    Args:
        fig: Matplotlib Figure to save.
        stage: Pipeline stage this figure belongs to, e.g. "cleaning" or
            "eda". Used as the subfolder name under config.FIGURES_DIR.
        filename: Output filename, e.g. "weight_sign_flip_check.png".
            Should include the extension.
        dpi: Resolution to save at.

    Returns:
        Path the figure was saved to.
    """
    stage_dir = config.FIGURES_DIR / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_path = stage_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", output_path)
    return output_path


def plot_distribution_comparison(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str,
    label_b: str,
    stage: str,
    filename: str,
    title: str = "",
    xlabel: str = "",
    bins: int = 40,
) -> plt.Figure:
    """
    Compare two numeric distributions side by side: an overlapping
    histogram and a paired boxplot. Intended for "do these two groups of
    values look like they belong to the same population" questions, e.g.
    comparing abs() of the negative weight values against the already
    valid positive weight values to decide whether a sign-flip
    correction is defensible.

    Args:
        series_a: First group of numeric values.
        series_b: Second group of numeric values.
        label_a: Legend/axis label for series_a.
        label_b: Legend/axis label for series_b.
        stage: Pipeline stage subfolder to save under, e.g. "cleaning".
        filename: Output filename, e.g. "weight_sign_flip_check.png".
        title: Overall figure title.
        xlabel: X-axis label for the histogram panel.
        bins: Number of histogram bins.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    logger.info(
        "Plotting distribution comparison: %s (n=%d) vs %s (n=%d)",
        label_a, len(series_a), label_b, len(series_b),
    )
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(12, 4.5))

    color_a, color_b = "#064A56", "#C97B2E"

    ax_hist.hist(series_a, bins=bins, alpha=0.55, label=label_a, color=color_a, density=True)
    ax_hist.hist(series_b, bins=bins, alpha=0.55, label=label_b, color=color_b, density=True)
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    ax_hist.spines[["top", "right"]].set_visible(False)

    ax_box.boxplot(
        [series_a.dropna(), series_b.dropna()],
        tick_labels=[label_a, label_b],
        patch_artist=True,
        boxprops=dict(facecolor=color_a, alpha=0.5),
        medianprops=dict(color="black"),
    )
    ax_box.spines[["top", "right"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig