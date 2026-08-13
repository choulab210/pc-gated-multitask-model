"""
Plot Figure 5: External Validation
==================================

Reproduces the grouped circular-bar external-validation figure using
the category-level predicted and observed percentages exported from
the original external-validation notebook.

Inputs
------
results/external_validation/
    external_validation_predicted_observed_percentages.csv

Expected columns
----------------
NP
Original_ID
Category
Predicted_percent
Observed_percent

Outputs
-------
figures/main/Figure_5_external_validation.png
figures/main/Figure_5_external_validation.pdf
figures/main/Figure_5_external_validation.tif

Important
---------
Predicted and observed percentages in the input CSV are the REAL values.
For visualization only, nonzero values below 2% are displayed at 2%,
matching the original notebook.

Run
---
python scripts/figures/plot_figure5_external_validation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "external_validation"
    / "external_validation_predicted_observed_percentages.csv"
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
# Publication style
# ======================================================================

plt.rcParams["font.family"] = "Arial"

DPI = 600

NP_TITLE_SIZE = 18
LEGEND_SIZE = 10
LEGEND_TITLE_SIZE = 11

MIN_DISPLAY_PCT = 2.0


# ======================================================================
# Category order
# ======================================================================

CATEGORY_ORDER = [
    "Apolipoproteins",
    "Coagulation/Fibrinogen",
    "Complement System",
    "Cytoskeletal",
    "Immunoglobulins",
    "Metabolic Enzymes",
    "Protease/Inhibitors",
    "Transport/Binding",
    "Other/Mixed",
]


# ======================================================================
# Category colors
#
# Keep consistent with the original notebook/Tableau-style palette.
# ======================================================================

TABLEAU10 = [
    "#4E79A7",
    "#59A14F",
    "#B07AA1",
    "#E15759",
    "#76B7B2",
    "#BAB0AC",
    "#F28E2B",
    "#9C755F",
    "#EDC948",
]


# ======================================================================
# Helpers
# ======================================================================


def build_category_colors(
    dataframe: pd.DataFrame,
):
    """
    Assign colors using category abundance/representation ordering.

    For consistency across all 8 NP panels, the same category receives
    the same color everywhere.
    """

    present_categories = [
        category
        for category in CATEGORY_ORDER
        if category
        in set(
            dataframe[
                "Category"
            ]
        )
    ]

    color_map = {
        category:
            TABLEAU10[
                index
                % len(TABLEAU10)
            ]
        for index, category in enumerate(
            present_categories
        )
    }

    return (
        present_categories,
        color_map,
    )


def draw_circular_bars(
    axis,
    np_dataframe: pd.DataFrame,
    categories,
    color_map,
    y_min,
    y_max,
    title,
):
    """
    Draw one NP circular grouped-bar plot.
    """

    n_categories = len(
        categories
    )

    slot = (
        2
        * np.pi
        / n_categories
    )

    bar_width = (
        slot
        * 0.40
    )

    slot_centers = (
        np.arange(
            n_categories
        )
        * slot
    )

    pred_angles = (
        slot_centers
        - bar_width
        / 2
    )

    obs_angles = (
        slot_centers
        + bar_width
        / 2
    )

    # --------------------------------------------------------------
    # Retrieve values in fixed category order
    # --------------------------------------------------------------

    pred_values = []

    obs_values = []

    for category in categories:

        row = np_dataframe[
            np_dataframe[
                "Category"
            ]
            == category
        ]

        if row.empty:

            pred = 0.0
            obs = 0.0

        else:

            pred = float(
                row.iloc[
                    0
                ][
                    "Predicted_percent"
                ]
            )

            obs = float(
                row.iloc[
                    0
                ][
                    "Observed_percent"
                ]
            )

        pred_values.append(
            pred
        )

        obs_values.append(
            obs
        )

    pred_values = np.asarray(
        pred_values,
        dtype=float,
    )

    obs_values = np.asarray(
        obs_values,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Display floor only
    # --------------------------------------------------------------

    pred_plot = np.where(
        (
            pred_values > 0
        )
        & (
            pred_values
            < MIN_DISPLAY_PCT
        ),
        MIN_DISPLAY_PCT,
        pred_values,
    )

    obs_plot = np.where(
        (
            obs_values > 0
        )
        & (
            obs_values
            < MIN_DISPLAY_PCT
        ),
        MIN_DISPLAY_PCT,
        obs_values,
    )

    colors = [
        color_map[
            category
        ]
        for category in categories
    ]

    # --------------------------------------------------------------
    # Polar axis configuration
    # --------------------------------------------------------------

    axis.set_theta_offset(
        np.pi
        / 2
    )

    axis.set_theta_direction(
        -1
    )

    axis.set_ylim(
        y_min,
        y_max,
    )

    # --------------------------------------------------------------
    # Predicted — solid
    # --------------------------------------------------------------

    axis.bar(
        pred_angles,
        pred_plot,
        width=bar_width,
        color=colors,
        edgecolor="grey",
        linewidth=0.35,
        zorder=3,
    )

    # --------------------------------------------------------------
    # Observed — hatched
    # --------------------------------------------------------------

    axis.bar(
        obs_angles,
        obs_plot,
        width=bar_width,
        color=colors,
        edgecolor="grey",
        linewidth=0.35,
        hatch="/////",
        zorder=3,
    )

    # --------------------------------------------------------------
    # Remove polar clutter
    # --------------------------------------------------------------

    axis.set_frame_on(
        False
    )

    axis.xaxis.grid(
        False
    )

    axis.yaxis.grid(
        False
    )

    axis.set_xticks(
        []
    )

    axis.set_yticks(
        []
    )

    axis.set_title(
        title,
        fontsize=NP_TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "PLOTTING FIGURE 5 — EXTERNAL VALIDATION"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Input check
    # ------------------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "External-validation composition file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Copy the CSV exported by the external-validation notebook "
            "to results/external_validation/."
        )

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "NP",
        "Original_ID",
        "Category",
        "Predicted_percent",
        "Observed_percent",
    }

    missing = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            "Input CSV is missing required columns:\n"
            f"{sorted(missing)}"
        )

    # ------------------------------------------------------------------
    # NP ordering
    # ------------------------------------------------------------------

    np_ids = list(
        dataframe[
            "NP"
        ]
        .drop_duplicates()
    )

    print()

    print(
        "External NP samples:",
        len(
            np_ids
        ),
    )

    if len(
        np_ids
    ) != 8:

        print(
            "WARNING: expected 8 external validation NPs."
        )

    # ------------------------------------------------------------------
    # Category colors
    # ------------------------------------------------------------------

    (
        categories,
        category_colors,
    ) = build_category_colors(
        dataframe
    )

    print(
        "Protein categories:",
        len(
            categories
        ),
    )

    # ==================================================================
    # Shared radial scale
    # ==================================================================

    max_value = max(
        dataframe[
            "Predicted_percent"
        ].max(),

        dataframe[
            "Observed_percent"
        ].max(),
    )

    y_max = float(
        np.ceil(
            max_value
            / 10
        )
        * 10
    )

    if y_max <= 0:

        y_max = 10.0

    # Empty center, same logic as notebook
    y_min = (
        -y_max
        * 0.08
    )

    print(
        "Shared radial maximum:",
        y_max,
    )

    print(
        "Minimum displayed nonzero bar:",
        f"{MIN_DISPLAY_PCT:.1f}%",
    )

    # ==================================================================
    # Figure
    # ==================================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(
            20.0,
            10.0,
        ),
        subplot_kw={
            "projection":
                "polar"
        },
    )

    axes = axes.flatten()

    # ------------------------------------------------------------------
    # Draw 8 NP plots
    # ------------------------------------------------------------------

    for index, np_id in enumerate(
        np_ids
    ):

        axis = axes[
            index
        ]

        np_dataframe = dataframe[
            dataframe[
                "NP"
            ]
            == np_id
        ]

        draw_circular_bars(
            axis=axis,
            np_dataframe=np_dataframe,
            categories=categories,
            color_map=category_colors,
            y_min=y_min,
            y_max=y_max,
            title=np_id,
        )

    # Hide unused axes if fewer than 8
    for index in range(
        len(
            np_ids
        ),
        len(
            axes
        ),
    ):

        axes[
            index
        ].set_visible(
            False
        )

    # ==================================================================
    # Shared legends
    # ==================================================================

    category_handles = [
        mpatches.Patch(
            facecolor=(
                category_colors[
                    category
                ]
            ),
            edgecolor="none",
            label=category,
        )
        for category in (
            categories
        )
    ]

    bar_handles = [
        mpatches.Patch(
            facecolor="#777777",
            edgecolor="grey",
            label="Predicted",
        ),

        mpatches.Patch(
            facecolor="#777777",
            edgecolor="grey",
            hatch="/////",
            label="Observed",
        ),
    ]

    # ------------------------------------------------------------------
    # Category legend
    # ------------------------------------------------------------------

    category_legend = fig.legend(
        handles=category_handles,
        title="Protein category",
        loc="lower center",
        bbox_to_anchor=(
            0.42,
            -0.02,
        ),
        ncol=3,
        fontsize=LEGEND_SIZE,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.5,
    )

    category_legend.get_title().set_fontsize(
        LEGEND_TITLE_SIZE
    )

    category_legend.get_title().set_fontweight(
        "bold"
    )

    for text in (
        category_legend.get_texts()
    ):

        text.set_fontweight(
            "bold"
        )

    # ------------------------------------------------------------------
    # Predicted / observed legend
    # ------------------------------------------------------------------

    bar_legend = fig.legend(
        handles=bar_handles,
        title="Bar type",
        loc="lower center",
        bbox_to_anchor=(
            0.81,
            -0.02,
        ),
        ncol=2,
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    bar_legend.get_title().set_fontsize(
        LEGEND_TITLE_SIZE
    )

    bar_legend.get_title().set_fontweight(
        "bold"
    )

    for text in (
        bar_legend.get_texts()
    ):

        text.set_fontweight(
            "bold"
        )

    # ==================================================================
    # Layout
    # ==================================================================

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.97,
        bottom=0.14,
        wspace=0.04,
        hspace=0.10,
    )

    # ==================================================================
    # Save
    # ==================================================================

    png_file = (
        FIGURE_DIR
        / "Figure_5_external_validation.png"
    )

    pdf_file = (
        FIGURE_DIR
        / "Figure_5_external_validation.pdf"
    )

    tif_file = (
        FIGURE_DIR
        / "Figure_5_external_validation.tif"
    )

    fig.savefig(
        png_file,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.20,
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
        pad_inches=0.20,
    )

    fig.savefig(
        tif_file,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.20,
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
        "FIGURE 5 COMPLETE"
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


if __name__ == "__main__":
    main()