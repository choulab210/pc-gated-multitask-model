"""
Generate Figure 2
=================

Figure 2. Adsorption classification performance on the held-out test set.

Panels
------
A. Receiver operating characteristic (ROC) curve
B. Precision-recall (PR) curve

The figure uses the saved predictions from the final two-head gated model.

Curve construction
------------------
For each protein, an ROC or PR curve is calculated across held-out NP
samples. Protein-specific curves are interpolated onto a common grid and
then averaged across proteins to obtain the macro-average curve.

Confidence intervals
--------------------
The bootstrap unit is the NP sample. Held-out NPs are resampled with
replacement while retaining all protein predictions belonging to each NP.
For each bootstrap sample, the macro-average ROC and PR curves are
recalculated. Pointwise 2.5th and 97.5th percentiles provide the shaded
95% bootstrap intervals.

The reported AUROC and AUPRC values use the same per-protein averaging
strategy as the final model evaluation pipeline.

Outputs
-------
figures/main/Figure_2_adsorption_performance.png
figures/main/Figure_2_adsorption_performance.pdf
figures/main/Figure_2_adsorption_performance.tif

results/figures/figure2_curve_data.csv

Run
---
python scripts/figures/figure2_adsorption_performance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "main"
)

FIGURE_DATA_DIR = (
    RESULTS_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OBSERVED_FILE = (
    RESULTS_DIR
    / "test_presence_observed.csv"
)

PREDICTED_FILE = (
    RESULTS_DIR
    / "test_presence_predictions.csv"
)


# ======================================================================
# Configuration
# ======================================================================

RANDOM_SEED = 42

# 2,000 is sufficient for smooth figure-level uncertainty intervals
# without making the script excessively slow.
#
# The manuscript's scalar metric CIs can continue to use the separate
# 10,000-resample bootstrap analysis.
N_BOOTSTRAP = 2000

GRID_SIZE = 201

DPI = 600


# ======================================================================
# Helpers
# ======================================================================


def load_data():
    """
    Load held-out observations and predicted adsorption probabilities.
    """

    observed = pd.read_csv(
        OBSERVED_FILE
    )

    predicted = pd.read_csv(
        PREDICTED_FILE
    )

    if list(
        observed.columns
    ) != list(
        predicted.columns
    ):
        raise ValueError(
            "Observed and predicted protein columns do not match."
        )

    if len(observed) != len(predicted):
        raise ValueError(
            "Observed and predicted sample counts do not match."
        )

    # --------------------------------------------------------------
    # OTHER is not an individual protein and should not be included
    # in adsorption performance evaluation.
    # --------------------------------------------------------------

    protein_columns = [
        column
        for column in observed.columns
        if str(column).upper() != "OTHER"
    ]

    observed = observed[
        protein_columns
    ]

    predicted = predicted[
        protein_columns
    ]

    return (
        observed.to_numpy(
            dtype=int
        ),
        predicted.to_numpy(
            dtype=float
        ),
        protein_columns,
    )


# ======================================================================
# Macro ROC
# ======================================================================


def macro_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_grid: np.ndarray,
):
    """
    Calculate macro-average ROC curve across proteins.

    Proteins containing only one observed class are skipped.
    """

    interpolated_tprs = []

    auc_values = []

    for protein_index in range(
        y_true.shape[1]
    ):

        truth = y_true[
            :,
            protein_index
        ]

        score = y_score[
            :,
            protein_index
        ]

        if len(
            np.unique(
                truth
            )
        ) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            truth,
            score,
        )

        auc_value = roc_auc_score(
            truth,
            score,
        )

        interp_tpr = np.interp(
            fpr_grid,
            fpr,
            tpr,
        )

        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interpolated_tprs.append(
            interp_tpr
        )

        auc_values.append(
            auc_value
        )

    if not interpolated_tprs:
        return (
            np.full_like(
                fpr_grid,
                np.nan,
                dtype=float,
            ),
            np.nan,
            0,
        )

    macro_tpr = np.nanmean(
        np.vstack(
            interpolated_tprs
        ),
        axis=0,
    )

    mean_auc = float(
        np.nanmean(
            auc_values
        )
    )

    return (
        macro_tpr,
        mean_auc,
        len(
            auc_values
        ),
    )


# ======================================================================
# Macro PR
# ======================================================================


def macro_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_grid: np.ndarray,
):
    """
    Calculate macro-average precision-recall curve across proteins.

    Proteins containing only one observed class are skipped.
    """

    interpolated_precision = []

    auprc_values = []

    prevalence_values = []

    for protein_index in range(
        y_true.shape[1]
    ):

        truth = y_true[
            :,
            protein_index
        ]

        score = y_score[
            :,
            protein_index
        ]

        if len(
            np.unique(
                truth
            )
        ) < 2:
            continue

        precision, recall, _ = (
            precision_recall_curve(
                truth,
                score,
            )
        )

        auprc = average_precision_score(
            truth,
            score,
        )

        # ----------------------------------------------------------
        # sklearn returns recall in decreasing order.
        # Reverse it before interpolation.
        # ----------------------------------------------------------

        recall_ascending = recall[::-1]

        precision_ascending = (
            precision[::-1]
        )

        interp_precision = np.interp(
            recall_grid,
            recall_ascending,
            precision_ascending,
        )

        interpolated_precision.append(
            interp_precision
        )

        auprc_values.append(
            auprc
        )

        prevalence_values.append(
            np.mean(
                truth
            )
        )

    if not interpolated_precision:
        return (
            np.full_like(
                recall_grid,
                np.nan,
                dtype=float,
            ),
            np.nan,
            np.nan,
            0,
        )

    macro_precision = np.nanmean(
        np.vstack(
            interpolated_precision
        ),
        axis=0,
    )

    mean_auprc = float(
        np.nanmean(
            auprc_values
        )
    )

    mean_prevalence = float(
        np.nanmean(
            prevalence_values
        )
    )

    return (
        macro_precision,
        mean_auprc,
        mean_prevalence,
        len(
            auprc_values
        ),
    )


# ======================================================================
# Bootstrap curves
# ======================================================================


def bootstrap_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_grid: np.ndarray,
    recall_grid: np.ndarray,
):
    """
    NP-level bootstrap of macro-average ROC and PR curves.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n_samples = y_true.shape[0]

    roc_draws = np.full(
        (
            N_BOOTSTRAP,
            len(
                fpr_grid
            ),
        ),
        np.nan,
        dtype=float,
    )

    pr_draws = np.full(
        (
            N_BOOTSTRAP,
            len(
                recall_grid
            ),
        ),
        np.nan,
        dtype=float,
    )

    print()

    print(
        f"Running {N_BOOTSTRAP} "
        "NP-level bootstrap resamples..."
    )

    for bootstrap_index in range(
        N_BOOTSTRAP
    ):

        sampled_indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        bootstrap_truth = y_true[
            sampled_indices
        ]

        bootstrap_score = y_score[
            sampled_indices
        ]

        macro_tpr, _, _ = (
            macro_roc_curve(
                bootstrap_truth,
                bootstrap_score,
                fpr_grid,
            )
        )

        macro_precision, _, _, _ = (
            macro_pr_curve(
                bootstrap_truth,
                bootstrap_score,
                recall_grid,
            )
        )

        roc_draws[
            bootstrap_index
        ] = macro_tpr

        pr_draws[
            bootstrap_index
        ] = macro_precision

        if (
            bootstrap_index + 1
        ) % 250 == 0:

            print(
                f"  Completed "
                f"{bootstrap_index + 1}/"
                f"{N_BOOTSTRAP}"
            )

    return (
        roc_draws,
        pr_draws,
    )


# ======================================================================
# CI helper
# ======================================================================


def pointwise_ci(
    draws: np.ndarray,
):
    """
    Calculate pointwise 95% percentile interval.
    """

    lower = np.nanpercentile(
        draws,
        2.5,
        axis=0,
    )

    upper = np.nanpercentile(
        draws,
        97.5,
        axis=0,
    )

    return (
        lower,
        upper,
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "GENERATING FIGURE 2"
    )

    print(
        "=" * 72
    )

    (
        y_true,
        y_score,
        protein_columns,
    ) = load_data()

    print()

    print(
        "Held-out NP samples:",
        y_true.shape[0],
    )

    print(
        "Evaluated proteins:",
        len(
            protein_columns
        ),
    )

    # ==================================================================
    # Common grids
    # ==================================================================

    fpr_grid = np.linspace(
        0.0,
        1.0,
        GRID_SIZE,
    )

    recall_grid = np.linspace(
        0.0,
        1.0,
        GRID_SIZE,
    )

    # ==================================================================
    # Point curves
    # ==================================================================

    (
        macro_tpr,
        mean_auroc,
        n_valid_roc,
    ) = macro_roc_curve(
        y_true,
        y_score,
        fpr_grid,
    )

    (
        macro_precision,
        mean_auprc,
        mean_prevalence,
        n_valid_pr,
    ) = macro_pr_curve(
        y_true,
        y_score,
        recall_grid,
    )

    print()

    print(
        f"Mean AUROC : {mean_auroc:.4f}"
    )

    print(
        f"Mean AUPRC : {mean_auprc:.4f}"
    )

    print(
        f"Valid ROC proteins: {n_valid_roc}"
    )

    print(
        f"Valid PR proteins : {n_valid_pr}"
    )

    print(
        f"Mean prevalence   : "
        f"{mean_prevalence:.4f}"
    )

    # ==================================================================
    # Bootstrap
    # ==================================================================

    (
        roc_draws,
        pr_draws,
    ) = bootstrap_curves(
        y_true,
        y_score,
        fpr_grid,
        recall_grid,
    )

    (
        roc_lower,
        roc_upper,
    ) = pointwise_ci(
        roc_draws
    )

    (
        pr_lower,
        pr_upper,
    ) = pointwise_ci(
        pr_draws
    )

    # ==================================================================
    # Save curve data
    # ==================================================================

    curve_data = pd.DataFrame(
        {
            "FPR":
                fpr_grid,

            "ROC_mean_TPR":
                macro_tpr,

            "ROC_CI95_lower":
                roc_lower,

            "ROC_CI95_upper":
                roc_upper,

            "Recall":
                recall_grid,

            "PR_mean_precision":
                macro_precision,

            "PR_CI95_lower":
                pr_lower,

            "PR_CI95_upper":
                pr_upper,
        }
    )

    curve_data_file = (
        FIGURE_DATA_DIR
        / "figure2_curve_data.csv"
    )

    curve_data.to_csv(
        curve_data_file,
        index=False,
    )

    # ==================================================================
    # Figure
    # ==================================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.5,
            4.6,
        ),
    )

    ax_roc = axes[0]

    ax_pr = axes[1]

    # ------------------------------------------------------------------
    # Panel A — ROC
    # ------------------------------------------------------------------

    ax_roc.fill_between(
        fpr_grid,
        roc_lower,
        roc_upper,
        alpha=0.20,
        linewidth=0,
        label="95% bootstrap CI",
    )

    ax_roc.plot(
        fpr_grid,
        macro_tpr,
        linewidth=2.2,
        label=(
            f"Two-head model "
            f"(AUROC = {mean_auroc:.3f})"
        ),
    )

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
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
        fontsize=11,
    )

    ax_roc.set_ylabel(
        "True positive rate",
        fontsize=11,
    )

    ax_roc.set_title(
        "Receiver operating characteristic",
        fontsize=12,
    )

    ax_roc.legend(
        frameon=False,
        fontsize=9,
        loc="lower right",
    )

    ax_roc.tick_params(
        labelsize=9,
    )

    ax_roc.text(
        -0.13,
        1.06,
        "A",
        transform=ax_roc.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    # ------------------------------------------------------------------
    # Panel B — PR
    # ------------------------------------------------------------------

    ax_pr.fill_between(
        recall_grid,
        pr_lower,
        pr_upper,
        alpha=0.20,
        linewidth=0,
        label="95% bootstrap CI",
    )

    ax_pr.plot(
        recall_grid,
        macro_precision,
        linewidth=2.2,
        label=(
            f"Two-head model "
            f"(AUPRC = {mean_auprc:.3f})"
        ),
    )

    ax_pr.axhline(
        mean_prevalence,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Prevalence = "
            f"{mean_prevalence:.3f}"
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
        fontsize=11,
    )

    ax_pr.set_ylabel(
        "Precision",
        fontsize=11,
    )

    ax_pr.set_title(
        "Precision–recall",
        fontsize=12,
    )

    ax_pr.legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
    )

    ax_pr.tick_params(
        labelsize=9,
    )

    ax_pr.text(
        -0.13,
        1.06,
        "B",
        transform=ax_pr.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    # ------------------------------------------------------------------
    # Common styling
    # ------------------------------------------------------------------

    for axis in axes:

        axis.spines[
            "top"
        ].set_visible(
            False
        )

        axis.spines[
            "right"
        ].set_visible(
            False
        )

    fig.tight_layout(
        w_pad=2.2
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
        "=" * 72
    )

    print(
        "FIGURE 2 COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Saved:"
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

    print()

    print(
        "Curve data:"
    )

    print(
        curve_data_file
    )


if __name__ == "__main__":
    main()