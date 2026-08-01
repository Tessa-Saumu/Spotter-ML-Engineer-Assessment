"""
Shared plotting functions for the freight rate prediction project.

Every function here saves its figure to disk under figures/<stage>/ (via
_savefig) and returns the Matplotlib Figure so a notebook can also
display it inline. Nothing in this module makes a modeling or cleaning
decision -- it only renders whatever data it is given. Deciding what a
plot means belongs in the notebook that calls it, not here.

Visual language is deliberately modeled on Spotter's own market
intelligence dashboard (dark navy, teal-forward, warm orange accent) --
see the THEME constants below -- so figures produced here read as an
extension of the product this model would actually feed, not a generic
notebook plot.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config

logger = logging.getLogger(__name__)

DEFAULT_DPI = 150

# --------------------------------------------------------------------
# Spotter brand theme
# --------------------------------------------------------------------
# Pulled directly from the Spotter Lens dashboard: deep navy background,
# a slightly lighter navy for panels/cards, low-opacity gridlines, teal
# as the primary/positive color, soft red for negative, warm orange as
# the sole accent. Reused as named constants (not hardcoded hex strings
# scattered through each function) so the palette only has one place to
# change.
THEME = {
    "background": "#0B1F2A",
    "panel": "#102A38",
    "gridline": "#1C3A4A",
    "primary": "#2EC4B6",
    "positive": "#2EC4B6",
    "negative": "#FF6B6B",
    "accent": "#F4A261",
    "text": "#E6F1F5",
    "text_muted": "#A0B3C0",
}

# Cool-to-warm sequence for heatmaps / choropleth-style gradients,
# matching the "hot" legend on Spotter's own map view. The midpoint is
# a dark slate, not a pale neutral -- a light beige midpoint (tried
# first) measured at ~1.1:1 contrast against light text, effectively
# invisible; #4A5A61 measures ~6.2:1, comfortably above the 4.5:1
# readability floor, while still reading as a neutral "between cool and
# warm" color rather than injecting its own hue into the scale.
_HEATMAP_COLORS = ["#2E86AB", "#2EC4B6", "#4A5A61", "#F4A261", "#FF6B6B"]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("spotter_heat", _HEATMAP_COLORS)

# A small qualitative sequence for categorical series (e.g. equipment
# type), chosen so each color is clearly separable from the panel
# background (all measure at least ~5:1 luminance contrast against
# #102A38) and from each other by hue, not just brightness -- teal,
# warm orange, and a soft yellow give three colors a reader can
# distinguish at a glance even with overlapping scatter points. The
# two grey-blues used in an earlier version measured close to the
# panel's own luminance and were dropped for exactly that reason.
QUALITATIVE_PALETTE = ["#2EC4B6", "#F4A261", "#F4D35E", "#FF6B6B", "#5DA9E9"]


def _relative_luminance(hex_color: str) -> float:
    """
    WCAG relative luminance of a hex color, used to choose readable text
    color against an arbitrary background (e.g. a heatmap cell whose
    color depends on its data value, not a fixed theme color).

    Args:
        hex_color: Color as a "#RRGGBB" hex string.

    Returns:
        Relative luminance in [0, 1], where 0 is black and 1 is white.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def _linearize(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = _linearize(r), _linearize(g), _linearize(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _readable_text_color(background_hex: str) -> str:
    """
    Pick THEME["text"] or THEME["background"] as the text color,
    whichever has higher contrast against the given background color.
    Replaces a fixed abs(value) > threshold rule, which only holds for
    colormaps where every non-extreme cell is dark -- not true once the
    heatmap's own endpoints (teal, orange) are themselves too bright for
    light text.

    Args:
        background_hex: The cell/background color as a "#RRGGBB" hex
            string.

    Returns:
        THEME["text"] (light) or THEME["background"] (dark), whichever
        contrasts more strongly with background_hex.
    """
    bg_luminance = _relative_luminance(background_hex)
    light_contrast = (max(bg_luminance, 0.863) + 0.05) / (min(bg_luminance, 0.863) + 0.05)
    dark_contrast = (max(bg_luminance, 0.012) + 0.05) / (min(bg_luminance, 0.012) + 0.05)
    return THEME["text"] if light_contrast >= dark_contrast else THEME["background"]


def apply_theme() -> None:
    """
    Apply the Spotter-branded dark theme as Matplotlib rcParams for the
    current session. Call once per notebook, near the top, before any
    plotting happens. Every function in this module also sets colors
    explicitly per-figure, so this is a safety net (e.g. for any ad hoc
    plotting a notebook does outside these helper functions) rather than
    the only thing standing between a plot and the default style.
    """
    plt.rcParams.update({
        "figure.facecolor": THEME["background"],
        "axes.facecolor": THEME["panel"],
        "savefig.facecolor": THEME["background"],
        "axes.edgecolor": THEME["gridline"],
        "axes.labelcolor": THEME["text"],
        "axes.titlecolor": THEME["text"],
        "text.color": THEME["text"],
        "xtick.color": THEME["text_muted"],
        "ytick.color": THEME["text_muted"],
        "grid.color": THEME["gridline"],
        "grid.alpha": 0.5,
        "font.size": 10.5,
        "legend.facecolor": THEME["panel"],
        "legend.edgecolor": THEME["gridline"],
        "legend.labelcolor": THEME["text"],
    })
    logger.info("Applied Spotter brand theme to matplotlib rcParams")


def _new_figure(figsize: tuple[float, float], ncols: int = 1, nrows: int = 1):
    """
    Create a themed Figure/Axes pair (or grid of Axes) with the Spotter
    background/panel colors applied directly, independent of whatever
    apply_theme() has or hasn't set globally.

    Args:
        figsize: (width, height) in inches.
        ncols: Number of subplot columns.
        nrows: Number of subplot rows.

    Returns:
        (fig, ax) tuple. ax is a single Axes if nrows == ncols == 1,
        otherwise a numpy array of Axes as returned by plt.subplots.
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(THEME["background"])
    axes = np.atleast_1d(ax).ravel()
    for single_ax in axes:
        single_ax.set_facecolor(THEME["panel"])
        for spine in single_ax.spines.values():
            spine.set_color(THEME["gridline"])
        single_ax.tick_params(colors=THEME["text_muted"])
        single_ax.xaxis.label.set_color(THEME["text"])
        single_ax.yaxis.label.set_color(THEME["text"])
        single_ax.title.set_color(THEME["text"])
    return fig, ax


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
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
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
    fig, (ax_hist, ax_box) = _new_figure((12, 4.5), ncols=2)

    color_a, color_b = THEME["primary"], THEME["accent"]

    ax_hist.hist(series_a, bins=bins, alpha=0.6, label=label_a, color=color_a, density=True)
    ax_hist.hist(series_b, bins=bins, alpha=0.6, label=label_b, color=color_b, density=True)
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("Density")
    ax_hist.legend()
    ax_hist.grid(axis="y", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
    ax_hist.spines[["top", "right"]].set_visible(False)

    box = ax_box.boxplot(
        [series_a.dropna(), series_b.dropna()],
        tick_labels=[label_a, label_b],
        patch_artist=True,
        medianprops=dict(color=THEME["background"], linewidth=1.6),
        whiskerprops=dict(color=THEME["text_muted"]),
        capprops=dict(color=THEME["text_muted"]),
        flierprops=dict(
            markerfacecolor=THEME["text_muted"], markeredgecolor=THEME["text_muted"], markersize=4
        ),
    )
    for patch, color in zip(box["boxes"], [color_a, color_b]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)
    ax_box.grid(axis="y", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
    ax_box.spines[["top", "right"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=THEME["text"])
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig


def plot_trend_line(
    x: pd.Series,
    y: pd.Series,
    stage: str,
    filename: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    annotate_last: bool = True,
    figsize: tuple[float, float] = (10.5, 4.8),
) -> plt.Figure:
    """
    Smooth line chart for a trend over an ordered axis (typically time),
    styled after Spotter's own rate-history chart: teal line, subtle
    fill beneath it, muted gridlines, and the current/last value
    called out with a small label.

    Args:
        x: Ordered x-axis values (e.g. dates or months).
        y: Values to plot against x.
        stage: Pipeline stage subfolder to save under, e.g. "eda".
        filename: Output filename, including extension.
        title: Axis title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        annotate_last: If True, label the final point with its value.
        figsize: (width, height) in inches.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    logger.info("Plotting trend line: %s (n=%d)", title or filename, len(y))
    fig, ax = _new_figure(figsize)

    ax.plot(x, y, color=THEME["primary"], linewidth=2.2, marker="o", markersize=3.5, zorder=3)
    ax.fill_between(x, y, float(np.min(y)) - max(1.0, float(np.ptp(y)) * 0.05), color=THEME["primary"], alpha=0.12)

    if annotate_last:
        last_x, last_y = list(x)[-1], list(y)[-1]
        ax.annotate(
            f"{last_y:,.2f}",
            xy=(last_x, last_y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=THEME["background"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=THEME["primary"], edgecolor="none"),
        )

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig


def plot_ranked_bars(
    categories: pd.Series,
    values: pd.Series,
    stage: str,
    filename: str,
    title: str = "",
    xlabel: str = "",
    top_n: int | None = None,
    highlight_top: int = 0,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart for rankings (e.g. top cities by rate, top
    lanes by volume), matching Spotter's own ranked-list dashboard
    convention. Bars are sorted descending top-to-bottom; optionally the
    top few can be highlighted in the accent color to draw the eye,
    mirroring the "hot" emphasis on Spotter's own leaderboard.

    Args:
        categories: Category labels (e.g. city names).
        values: Numeric value per category, same length/order as
            categories.
        stage: Pipeline stage subfolder to save under.
        filename: Output filename, including extension.
        title: Axis title.
        xlabel: X-axis label.
        top_n: If given, only the top_n largest values are plotted.
        highlight_top: Number of leading bars (largest values) to draw
            in the accent color instead of primary teal.
        figsize: (width, height) in inches. Defaults to a height scaled
            to the number of bars plotted.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    frame = pd.DataFrame({"category": categories.values, "value": values.values})
    frame = frame.sort_values("value", ascending=False)
    if top_n is not None:
        frame = frame.head(top_n)
    frame = frame.iloc[::-1]  # largest at top when plotted horizontally

    logger.info("Plotting ranked bars: %s (%d bars)", title or filename, len(frame))

    if figsize is None:
        figsize = (9.5, max(3.0, 0.38 * len(frame) + 1.0))
    fig, ax = _new_figure(figsize)

    n = len(frame)
    colors = [
        THEME["accent"] if (n - 1 - i) < highlight_top else THEME["primary"]
        for i in range(n)
    ]
    ax.barh(frame["category"], frame["value"], color=colors, height=0.65)

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    stage: str,
    filename: str,
    title: str = "",
    figsize: tuple[float, float] = (7.5, 6.5),
) -> plt.Figure:
    """
    Cool-to-warm correlation heatmap using the Spotter map-view gradient
    (cool blue-teal for negative/low, pale neutral for near-zero, warm
    orange-red for positive/high), with correlation values annotated in
    each cell.

    Args:
        corr: Square correlation matrix (e.g. df.corr()).
        stage: Pipeline stage subfolder to save under.
        filename: Output filename, including extension.
        title: Axis title.
        figsize: (width, height) in inches.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    logger.info("Plotting correlation heatmap: %s (%d x %d)", title or filename, *corr.shape)
    fig, ax = _new_figure(figsize)

    im = ax.imshow(corr.values, cmap=HEATMAP_CMAP, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.values[i, j]
            cell_color = HEATMAP_CMAP((value + 1) / 2)  # imshow's vmin=-1/vmax=1 mapping
            cell_hex = "#{:02X}{:02X}{:02X}".format(
                int(cell_color[0] * 255), int(cell_color[1] * 255), int(cell_color[2] * 255)
            )
            text_color = _readable_text_color(cell_hex)
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5, color=text_color)

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color=THEME["text_muted"])
    cbar.outline.set_visible(False)
    plt.setp(cbar.ax.get_yticklabels(), color=THEME["text_muted"])

    fig.tight_layout()
    _savefig(fig, stage, filename)
    return fig


def plot_histogram(
    series: pd.Series,
    stage: str,
    filename: str,
    title: str = "",
    xlabel: str = "",
    bins: int = 50,
    log_series: pd.Series | None = None,
    log_label: str = "log-transformed",
    figsize: tuple[float, float] = (10.5, 4.5),
) -> plt.Figure:
    """
    Histogram of a single numeric column, optionally paired side by side
    with a log-transformed version for a skew comparison (e.g. deciding
    whether log(posted_rate) is worth modeling on instead of the raw
    target).

    Args:
        series: Values to histogram (raw scale).
        stage: Pipeline stage subfolder to save under.
        filename: Output filename, including extension.
        title: Overall figure title.
        xlabel: X-axis label for the raw-scale panel.
        bins: Number of histogram bins.
        log_series: If given, plotted as a second panel alongside
            series, for a direct skew comparison.
        log_label: X-axis label for the log-transformed panel.
        figsize: (width, height) in inches.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    logger.info("Plotting histogram: %s (n=%d)", title or filename, len(series))
    ncols = 2 if log_series is not None else 1
    fig, axes = _new_figure(figsize, ncols=ncols)
    axes = np.atleast_1d(axes)

    axes[0].hist(series.dropna(), bins=bins, color=THEME["primary"], alpha=0.85)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    if log_series is not None:
        axes[1].hist(log_series.dropna(), bins=bins, color=THEME["accent"], alpha=0.85)
        axes[1].set_xlabel(log_label)
        axes[1].grid(axis="y", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
        axes[1].spines[["top", "right"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=THEME["text"])
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig


def plot_scatter(
    x: pd.Series,
    y: pd.Series,
    stage: str,
    filename: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    hue: pd.Series | None = None,
    hue_label: str = "",
    alpha: float = 0.55,
    figsize: tuple[float, float] = (8.0, 5.5),
) -> plt.Figure:
    """
    Scatter plot for a feature-vs-target or feature-vs-feature
    relationship, optionally colored by a categorical grouping (e.g.
    equipment type) using the brand qualitative palette.

    Args:
        x: X-axis values.
        y: Y-axis values, same length/order as x.
        stage: Pipeline stage subfolder to save under.
        filename: Output filename, including extension.
        title: Axis title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        hue: Optional categorical series (same length as x/y) to color
            points by.
        hue_label: Legend title when hue is given.
        alpha: Point transparency. 0.55 by default -- an earlier, lower
            default (0.35) combined with grey-blue palette colors close
            in luminance to the panel background made points hard to
            see even before accounting for overlap; raised alongside
            the palette fix below, plus a thin edge stroke on each
            point for a further contrast boost in dense regions.
        figsize: (width, height) in inches.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    logger.info("Plotting scatter: %s (n=%d)", title or filename, len(x))
    fig, ax = _new_figure(figsize)

    if hue is not None:
        categories = pd.Series(hue).astype("category")
        for i, level in enumerate(categories.cat.categories):
            mask = (categories == level).values
            color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
            ax.scatter(
                np.asarray(x)[mask], np.asarray(y)[mask],
                s=14, alpha=alpha, color=color, label=str(level),
                linewidths=0.3, edgecolors=THEME["background"],
            )
        legend = ax.legend(title=hue_label, loc="upper left", framealpha=0.9)
        if legend.get_title():
            legend.get_title().set_color(THEME["text"])
        for legend_handle in legend.legend_handles:
            legend_handle.set_alpha(1.0)
    else:
        ax.scatter(
            x, y, s=14, alpha=alpha, color=THEME["primary"],
            linewidths=0.3, edgecolors=THEME["background"],
        )

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=THEME["gridline"], alpha=0.4, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig


def plot_grouped_box(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    stage: str,
    filename: str,
    title: str = "",
    ylabel: str = "",
    order: list[str] | None = None,
    figsize: tuple[float, float] = (8.5, 5.0),
) -> plt.Figure:
    """
    Boxplot of a numeric column split by a categorical grouping (e.g.
    posted_rate by equipment type), using the brand qualitative palette
    per group.

    Args:
        df: DataFrame containing group_col and value_col.
        group_col: Categorical column to split by.
        value_col: Numeric column to summarize per group.
        stage: Pipeline stage subfolder to save under.
        filename: Output filename, including extension.
        title: Axis title.
        ylabel: Y-axis label.
        order: Optional explicit category order; defaults to descending
            median of value_col.
        figsize: (width, height) in inches.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    if order is None:
        order = (
            df.groupby(group_col)[value_col].median().sort_values(ascending=False).index.tolist()
        )
    logger.info("Plotting grouped box: %s by %s (%d groups)", value_col, group_col, len(order))

    fig, ax = _new_figure(figsize)
    data = [df.loc[df[group_col] == level, value_col].dropna() for level in order]

    box = ax.boxplot(
        data,
        tick_labels=order,
        patch_artist=True,
        medianprops=dict(color=THEME["background"], linewidth=1.6),
        whiskerprops=dict(color=THEME["text_muted"]),
        capprops=dict(color=THEME["text_muted"]),
        flierprops=dict(
            markerfacecolor=THEME["text_muted"], markeredgecolor=THEME["text_muted"], markersize=4
        ),
    )
    for i, patch in enumerate(box["boxes"]):
        color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    stage: str,
    filename: str,
    title: str = "",
    axis_label: str = "posted_rate ($)",
    figsize: tuple[float, float] = (7.0, 7.0),
) -> plt.Figure:
    """
    Actual vs. predicted scatter with a y = x reference line -- points
    on the line are perfect predictions, points above it are
    over-predictions, below it are under-predictions.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values, same length/order as y_true.
        stage: Pipeline stage subfolder to save under, e.g. "modeling".
        filename: Output filename, including extension.
        title: Axis title.
        axis_label: Shared label for both axes (same units on each).
        figsize: (width, height) in inches. Square by default so the
            y = x reference line reads at a true 45 degrees.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    logger.info("Plotting actual vs. predicted: %s (n=%d)", title or filename, len(y_true))
    fig, ax = _new_figure(figsize)

    y_true_arr, y_pred_arr = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    lo = min(y_true_arr.min(), y_pred_arr.min())
    hi = max(y_true_arr.max(), y_pred_arr.max())

    ax.plot([lo, hi], [lo, hi], color=THEME["text_muted"], linewidth=1.2, linestyle="--", zorder=2)
    ax.scatter(
        y_true_arr, y_pred_arr, s=14, alpha=0.5, color=THEME["primary"],
        linewidths=0.3, edgecolors=THEME["background"], zorder=3,
    )

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"actual {axis_label}")
    ax.set_ylabel(f"predicted {axis_label}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=THEME["gridline"], alpha=0.4, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig


def plot_residuals(
    y_pred: pd.Series,
    residual_values: np.ndarray,
    stage: str,
    filename: str,
    title: str = "",
    xlabel: str = "predicted posted_rate ($)",
    figsize: tuple[float, float] = (9.0, 5.0),
) -> plt.Figure:
    """
    Residuals-vs-predicted scatter (left) paired with a residual
    histogram (right) -- the standard pair for checking whether error
    grows with the predicted value (heteroscedasticity) and whether
    residuals are centered near zero and roughly symmetric.

    Args:
        y_pred: Predicted target values.
        residual_values: Residuals (predicted - actual), same
            length/order as y_pred -- typically from
            evaluate.residuals().
        stage: Pipeline stage subfolder to save under.
        filename: Output filename, including extension.
        title: Overall figure title.
        xlabel: X-axis label for the scatter panel.
        figsize: (width, height) in inches.

    Returns:
        The Matplotlib Figure, already saved to
        figures/<stage>/<filename>.
    """
    logger.info("Plotting residuals: %s (n=%d)", title or filename, len(y_pred))
    fig, (ax_scatter, ax_hist) = _new_figure(figsize, ncols=2)

    ax_scatter.axhline(0, color=THEME["text_muted"], linewidth=1.2, linestyle="--", zorder=2)
    ax_scatter.scatter(
        y_pred, residual_values, s=14, alpha=0.5, color=THEME["primary"],
        linewidths=0.3, edgecolors=THEME["background"], zorder=3,
    )
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel("residual (predicted - actual)")
    ax_scatter.grid(color=THEME["gridline"], alpha=0.4, linewidth=0.8)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    ax_hist.hist(residual_values, bins=50, color=THEME["accent"], alpha=0.85)
    ax_hist.axvline(0, color=THEME["text_muted"], linewidth=1.2, linestyle="--")
    ax_hist.set_xlabel("residual")
    ax_hist.set_ylabel("count")
    ax_hist.grid(axis="y", color=THEME["gridline"], alpha=0.5, linewidth=0.8)
    ax_hist.spines[["top", "right"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=THEME["text"])
    fig.tight_layout()

    _savefig(fig, stage, filename)
    return fig