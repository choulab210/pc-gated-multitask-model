"""
Plot Figure 2 from Saved Curve Data
===================================

This script DOES NOT rerun bootstrap analysis.

It reads:
    results/figures/figure2_curve_data.csv

and generates:
    figures/main/Figure_2_adsorption_performance.png
    figures/main/Figure_2_adsorption_performance.pdf
    figures/main/Figure_2_adsorption_performance.tif

Run:
    python scripts/figures/plot_figure2_adsorption_performance.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CURVE_DATA_FILE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "figure2_curve_data.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "main"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# Figure style
# ======================================================================

DPI = 600

TITLE_SIZE = 16
LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 14
PANEL_SIZE = 18

MAIN_LINEWIDTH = 2.5
REFERENCE_LINEWIDTH = 1.5

CI_ALPHA = 0.20


# ======================================================================
# Values shown in legend
# ======================================================================
#
# These are the final reproducible held-out test metrics.
# They are intentionally specified here rather than recalculated.
#

AUROC = 0.9059
AUPRC = 0.8370


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )
    print(
        "PLOTTING FIGURE 2 FROM SAVED CURVE DATA"
    )
    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Check input
    # ------------------------------------------------------------------

    if not CURVE_DATA_FILE.exists():

        raise FileNotFoundError(
            "Figure 2 curve-data file not found:\n"
            f"{CURVE_DATA_FILE}\n\n"
            "Run the Figure 2 bootstrap/data-generation script first."
        )

    # ------------------------------------------------------------------
    # Load saved curves
    # ------------------------------------------------------------------

    data = pd.read_csv(
        CURVE_DATA_FILE
    )

    required_columns = [
        "FPR",
        "ROC_mean_TPR",
        "ROC_CI95_lower",
        "ROC_CI95_upper",
        "Recall",
        "PR_mean_precision",
        "PR_CI95_lower",
        "PR_CI95_upper",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required column(s) in curve-data file:\n"
            f"{missing_columns}"
        )

    # ------------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------------

    fpr = data[
        "FPR"
    ].to_numpy()

    roc_mean = data[
        "ROC_mean_TPR"
    ].to_numpy()

    roc_lower = data[
        "ROC_CI95_lower"
    ].to_numpy()

    roc_upper = data[
        "ROC_CI95_upper"
    ].to_numpy()

    recall = data[
        "Recall"
    ].to_numpy()

    pr_mean = data[
        "PR_mean_precision"
    ].to_numpy()

    pr_lower = data[
        "PR_CI95_lower"
    ].to_numpy()

    pr_upper = data[
        "PR_CI95_upper"
    ].to_numpy()

    # ------------------------------------------------------------------
    # Mean positive-class prevalence
    #
    # Keep this consistent with the current figure theme.
    # Replace if your saved Figure 2 data file later includes prevalence.
    # ------------------------------------------------------------------

    prevalence = 0.50

    # ==================================================================
    # Figure
    # ==================================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.5,
            5.4,
        ),
    )

    ax_roc = axes[0]
    ax_pr = axes[1]

    # ==================================================================
    # Panel A — ROC
    # ==================================================================

    ax_roc.fill_between(
        fpr,
        roc_lower,
        roc_upper,
        alpha=CI_ALPHA,
        linewidth=0,
        label="95% bootstrap CI",
    )

    ax_roc.plot(
        fpr,
        roc_mean,
        linewidth=MAIN_LINEWIDTH,
        label=(
            f"Two-head model "
            f"(AUROC = {AUROC:.3f})"
        ),
    )

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        label="Random classifier",
    )

    ax_roc.set_xlim(
        0,
        1,
    )

    ax_roc.set_ylim(
        0,
        1.02,
    )

    ax_roc.set_xlabel(
        "False positive rate",
        fontsize=LABEL_SIZE, fontweight="bold"
    )

    ax_roc.set_ylabel(
        "True positive rate",
        fontsize=LABEL_SIZE, fontweight="bold"
    )

   
    ax_roc.tick_params(
        labelsize=TICK_SIZE,
    )

    ax_roc.legend(
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower right",
    )

    ax_roc.text(
        -0.13,
        1.06,
        "A",
        transform=ax_roc.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
    )

    # ==================================================================
    # Panel B — Precision–Recall
    # ==================================================================

    ax_pr.fill_between(
        recall,
        pr_lower,
        pr_upper,
        alpha=CI_ALPHA,
        linewidth=0,
        label="95% bootstrap CI",
    )

    ax_pr.plot(
        recall,
        pr_mean,
        linewidth=MAIN_LINEWIDTH,
        label=(
            f"Two-head model "
            f"(AUPRC = {AUPRC:.3f})"
        ),
    )

    ax_pr.axhline(
        prevalence,
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        label=(
            f"Prevalence = "
            f"{prevalence:.3f}"
        ),
    )

    ax_pr.set_xlim(
        0,
        1,
    )

    ax_pr.set_ylim(
        0,
        1.02,
    )

    ax_pr.set_xlabel(
        "Recall",
        fontsize=LABEL_SIZE, fontweight="bold"
    )

    ax_pr.set_ylabel(
        "Precision",
        fontsize=LABEL_SIZE, fontweight="bold"
    )

  
    ax_pr.tick_params(
        labelsize=TICK_SIZE,
    )

    ax_pr.legend(
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower left",
    )

    ax_pr.text(
        -0.13,
        1.06,
        "B",
        transform=ax_pr.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
    )

    # ==================================================================
    # Shared styling
    # ==================================================================

    for axis in axes:

    # Keep all four borders visible
     for spine in [
        "top",
        "bottom",
        "left",
        "right",
     ]:

        axis.spines[
            spine
        ].set_visible(
            True
        )

        axis.spines[
            spine
        ].set_linewidth(
            1.8
        )

    # Make tick marks slightly thicker
    axis.tick_params(
        width=1.5,
        length=5,
        labelsize=TICK_SIZE,
    )

    fig.tight_layout(
        w_pad=2.5
    )

    # ==================================================================
    # Save
    # ==================================================================

    png_file = (
        FIGURE_DIR
        / "Figure_2_adsorption_performance.png"
    )

    pdf_file = (
        FIGURE_DIR
        / "Figure_2_adsorption_performance.pdf"
    )

    tif_file = (
        FIGURE_DIR
        / "Figure_2_adsorption_performance.tif"
    )

    fig.savefig(
        png_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    fig.savefig(
        tif_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ==================================================================
    # Console
    # ==================================================================

    print()

    print(
        "Figure saved:"
    )

    print(
        png_file
    )

    print(
        pdf_file
    )

    print(
        tif_file
    )


if __name__ == "__main__":
    main()