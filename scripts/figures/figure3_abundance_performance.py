"""
Generate Figure 3
=================

Figure 3. Protein corona abundance prediction performance.

Panel A
-------
Observed versus predicted relative abundance across held-out NP-protein
pairs. Proteins are colored by abundance tertile defined from the
model-development dataset.

Panel B
-------
Per-protein Pearson correlation on the held-out test set stratified by
the same training-derived abundance tertiles. Box colors represent
abundance tertiles, while individual protein points are colored by
functional category.

The OTHER abundance bin is excluded.

For visualization on logarithmic axes, values below 1e-4 are displayed
at 1e-4. This plotting floor does not affect model-performance metrics.

Outputs
-------
figures/main/Figure_3_abundance_performance.png
figures/main/Figure_3_abundance_performance.pdf
figures/main/Figure_3_abundance_performance.tif

results/figures/figure3_protein_tertiles.csv

Run
---
python scripts/figures/figure3_abundance_performance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from matplotlib.lines import Line2D

from pcmodel.data import prepare_model_data


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
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


FEATURE_FILE = DATA_DIR / "Data_1.csv"
ABUNDANCE_FILE = DATA_DIR / "Data_2.csv"

OBSERVED_FILE = (
    RESULTS_DIR
    / "test_abundance_observed.csv"
)

PREDICTED_FILE = (
    RESULTS_DIR
    / "test_abundance_predictions.csv"
)

METADATA_FILE = (
    DATA_DIR
    / "protein_metadata.csv"
)


# ======================================================================
# Publication style — aligned with Figure 2
# ======================================================================

plt.rcParams["font.family"] = "Arial"

DPI = 600

LABEL_SIZE = 16
TICK_SIZE = 12
LEGEND_SIZE = 11
PANEL_SIZE = 18

FRAME_WIDTH = 1.8

SCATTER_SIZE = 18
SCATTER_ALPHA = 0.55

PROTEIN_POINT_SIZE = 24

PLOT_FLOOR = 1e-4


# ======================================================================
# Abundance tertile colors from original notebook
# ======================================================================

TERTILE_ORDER = [
    "High",
    "Middle",
    "Low",
]

TERTILE_COLORS = {
    "High": "#CC79A7",
    "Middle": "#6E8FB0",
    "Low": "#E69F00",
}


# ======================================================================
# Functional category colors
# ======================================================================

CATEGORY_ORDER = [
    "Transport/Binding",
    "Complement System",
    "Immunoglobulins",
    "Cytoskeletal",
    "Other/Mixed",
    "Coagulation/Fibrinogen",
    "Apolipoproteins",
    "Protease/Inhibitors",
    "Metabolic Enzymes",
]

CATEGORY_COLORS = [
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

CATEGORY_COLOR_MAP = {
    category: CATEGORY_COLORS[index]
    for index, category in enumerate(
        CATEGORY_ORDER
    )
}


# ======================================================================
# Helpers
# ======================================================================


def identify_metadata_columns(
    metadata: pd.DataFrame,
):
    """
    Identify protein and functional-category columns.
    """

    protein_candidates = [
        "Protein",
        "protein",
        "Protein_ID",
        "protein_id",
        "UniProt",
        "uniprot",
        "Accession",
        "accession",
    ]

    category_candidates = [
        "Category",
        "category",
        "Protein_Category",
        "protein_category",
        "Functional_Category",
        "functional_category",
    ]

    protein_column = None
    category_column = None

    for candidate in protein_candidates:

        if candidate in metadata.columns:
            protein_column = candidate
            break

    for candidate in category_candidates:

        if candidate in metadata.columns:
            category_column = candidate
            break

    if protein_column is None:

        raise ValueError(
            "Could not identify protein-ID column.\n"
            f"Available columns: {list(metadata.columns)}"
        )

    if category_column is None:

        raise ValueError(
            "Could not identify functional-category column.\n"
            f"Available columns: {list(metadata.columns)}"
        )

    return (
        protein_column,
        category_column,
    )


def safe_pearson(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Pearson r for one protein.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if (
        np.std(observed) <= 1e-8
        or np.std(predicted) <= 1e-8
    ):
        return np.nan

    return float(
        pearsonr(
            observed,
            predicted,
        ).statistic
    )


def apply_figure2_style(
    axis,
):
    """
    Apply the same frame/tick style used in final Figure 2.
    """

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
            FRAME_WIDTH
        )

        axis.spines[
            spine
        ].set_edgecolor(
            "black"
        )

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        width=1.5,
        length=5,
    )

    axis.tick_params(
        axis="both",
        which="minor",
        width=1.0,
        length=3,
    )

    for label in axis.get_xticklabels():
        label.set_fontweight(
            "bold"
        )

    for label in axis.get_yticklabels():
        label.set_fontweight(
            "bold"
        )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "GENERATING FIGURE 3"
    )

    print(
        "=" * 72
    )

    # ==================================================================
    # Reconstruct training abundance data
    # ==================================================================

    prepared = prepare_model_data(
        FEATURE_FILE,
        ABUNDANCE_FILE,
    )

    Ya_train = np.asarray(
        prepared.Y_abundance_train,
        dtype=float,
    )

    panel = list(
        prepared.panel
    )

    n_panel = len(
        panel
    )

    print()

    print(
        "Model-development samples:",
        Ya_train.shape[0],
    )

    print(
        "Individual proteins:",
        n_panel,
    )

    # ==================================================================
    # Held-out observations/predictions
    # ==================================================================

    observed_df = pd.read_csv(
        OBSERVED_FILE
    )

    predicted_df = pd.read_csv(
        PREDICTED_FILE
    )

    if list(
        observed_df.columns
    ) != list(
        predicted_df.columns
    ):

        raise ValueError(
            "Observed and predicted abundance columns do not match."
        )

    protein_columns = [
        column
        for column in observed_df.columns
        if str(
            column
        ).upper()
        != "OTHER"
    ]

    observed_df = observed_df[
        protein_columns
    ]

    predicted_df = predicted_df[
        protein_columns
    ]

    if len(
        protein_columns
    ) != n_panel:

        raise ValueError(
            "Protein panel mismatch:\n"
            f"Training panel = {n_panel}\n"
            f"Saved test panel = {len(protein_columns)}"
        )

    Ya_test = observed_df.to_numpy(
        dtype=float
    )

    D_test = predicted_df.to_numpy(
        dtype=float
    )

    print(
        "Held-out test samples:",
        Ya_test.shape[0],
    )

    # ==================================================================
    # Training-derived abundance tertiles
    # ==================================================================

    Ya_train_panel = Ya_train[
        :,
        :n_panel,
    ]

    mean_abundance_train = (
        Ya_train_panel.mean(
            axis=0
        )
    )

    order = np.argsort(
        -mean_abundance_train
    )

    tertile_groups = np.array_split(
        order,
        3,
    )

    tertile_members = {
        "High":
            tertile_groups[0],

        "Middle":
            tertile_groups[1],

        "Low":
            tertile_groups[2],
    }

    print()

    print(
        "Training-derived tertile counts:"
    )

    for tertile in TERTILE_ORDER:

        print(
            f"  {tertile:<7}: "
            f"{len(tertile_members[tertile])}"
        )

    # ==================================================================
    # Metadata
    # ==================================================================

    metadata = pd.read_csv(
        METADATA_FILE
    )

    (
        protein_column,
        category_column,
    ) = identify_metadata_columns(
        metadata
    )

    metadata[
        protein_column
    ] = (
        metadata[
            protein_column
        ]
        .astype(str)
    )

    category_map = dict(
        zip(
            metadata[
                protein_column
            ],
            metadata[
                category_column
            ],
        )
    )

    # ==================================================================
    # Per-protein Pearson r
    # ==================================================================

    per_protein_r = np.full(
        n_panel,
        np.nan,
        dtype=float,
    )

    for j in range(
        n_panel
    ):

        per_protein_r[
            j
        ] = safe_pearson(
            Ya_test[
                :,
                j
            ],
            D_test[
                :,
                j
            ],
        )

    # ==================================================================
    # Protein summary
    # ==================================================================

    tertile_lookup = {}

    for tertile, indices in (
        tertile_members.items()
    ):

        for index in indices:

            tertile_lookup[
                int(index)
            ] = tertile

    protein_rows = []

    for j, protein in enumerate(
        protein_columns
    ):

        category = category_map.get(
            str(protein),
            "Other/Mixed",
        )

        protein_rows.append(
            {
                "Protein":
                    protein,

                "Training_mean_abundance":
                    mean_abundance_train[
                        j
                    ],

                "Abundance_tertile":
                    tertile_lookup[
                        j
                    ],

                "Pearson_r":
                    per_protein_r[
                        j
                    ],

                "Category":
                    category,
            }
        )

    protein_df = pd.DataFrame(
        protein_rows
    )

    # ==================================================================
    # Panel A data
    # ==================================================================

    obs_by_tertile = {}

    pred_by_tertile = {}

    for tertile, indices in (
        tertile_members.items()
    ):

        obs_by_tertile[
            tertile
        ] = Ya_test[
            :,
            indices
        ].flatten()

        pred_by_tertile[
            tertile
        ] = D_test[
            :,
            indices
        ].flatten()

    all_observed = np.concatenate(
        [
            obs_by_tertile[
                tertile
            ]
            for tertile in TERTILE_ORDER
        ]
    )

    all_predicted = np.concatenate(
        [
            pred_by_tertile[
                tertile
            ]
            for tertile in TERTILE_ORDER
        ]
    )

    print()

    print(
        f"N NP-protein pairs: "
        f"{len(all_observed):,}"
    )

    print(
        f"Observed zero fraction: "
        f"{np.mean(all_observed == 0):.1%}"
    )

    print(
        f"Plotting floor: "
        f"{PLOT_FLOOR:.0e}"
    )

    def floor_for_plot(
        values,
    ):

        return np.maximum(
            values,
            PLOT_FLOOR,
        )

    # ==================================================================
    # Figure
    # ==================================================================

    fig, (
        ax_a,
        ax_b,
    ) = plt.subplots(
        1,
        2,
        figsize=(
            14.0,
            5.8,
        ),
    )

    # ==================================================================
    # Panel A
    # ==================================================================

    plot_order = [
        "Low",
        "Middle",
        "High",
    ]

    for tertile in plot_order:

        ax_a.scatter(
            floor_for_plot(
                obs_by_tertile[
                    tertile
                ]
            ),

            floor_for_plot(
                pred_by_tertile[
                    tertile
                ]
            ),

            s=SCATTER_SIZE,

            alpha=SCATTER_ALPHA,

            color=TERTILE_COLORS[
                tertile
            ],

            edgecolor="none",

            label=tertile,

            zorder=2,
        )

    # Identity line
    ax_a.plot(
        [
            PLOT_FLOOR,
            1.0,
        ],
        [
            PLOT_FLOOR,
            1.0,
        ],
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
        zorder=3,
    )

    ax_a.set_xscale(
        "log"
    )

    ax_a.set_yscale(
        "log"
    )

    ax_a.set_xlim(
        PLOT_FLOOR,
        1.0,
    )

    ax_a.set_ylim(
        PLOT_FLOOR,
        1.0,
    )

    ax_a.set_aspect(
        "equal",
        adjustable="box",
    )

    ax_a.set_xlabel(
        "Observed relative abundance",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        labelpad=12,
    )

    ax_a.set_ylabel(
        "Predicted relative abundance",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        labelpad=12,
    )

    # No internal title — matches final Figure 2

    tertile_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=(
                TERTILE_COLORS[
                    tertile
                ]
            ),
            markeredgecolor="none",
            markersize=8,
            label=tertile,
        )
        for tertile in TERTILE_ORDER
    ]

    legend_a = ax_a.legend(
        handles=tertile_handles,
        loc="upper left",
        fontsize=LEGEND_SIZE,
        frameon=True,
        framealpha=0.97,
    )

    legend_a.get_frame().set_edgecolor(
        "black"
    )

    legend_a.get_frame().set_linewidth(
        1.0
    )

    for text in (
        legend_a.get_texts()
    ):

        text.set_fontweight(
            "bold"
        )

    ax_a.text(
        -0.14,
        1.04,
        "A",
        transform=ax_a.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
    )

    apply_figure2_style(
        ax_a
    )

    # ==================================================================
    # Panel B
    # ==================================================================

    positions = [
        1,
        2,
        3,
    ]

    rng = np.random.default_rng(
        42
    )

    group_values = []

    group_indices = []

    for tertile in TERTILE_ORDER:

        indices = tertile_members[
            tertile
        ]

        values = per_protein_r[
            indices
        ]

        valid = ~np.isnan(
            values
        )

        group_values.append(
            values[
                valid
            ]
        )

        group_indices.append(
            indices[
                valid
            ]
        )

    # Colored boxplots
    boxplot = ax_b.boxplot(
        group_values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": "black",
            "linewidth": 2.2,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.4,
        },
        capprops={
            "color": "black",
            "linewidth": 1.4,
        },
        boxprops={
            "linewidth": 1.4,
            "edgecolor": "black",
        },
    )

    for patch, tertile in zip(
        boxplot[
            "boxes"
        ],
        TERTILE_ORDER,
    ):

        patch.set_facecolor(
            TERTILE_COLORS[
                tertile
            ]
        )

        patch.set_alpha(
            0.55
        )

    # Protein-category points
    for position, values, indices in zip(
        positions,
        group_values,
        group_indices,
    ):

        jitter = rng.uniform(
            -0.18,
            0.18,
            size=len(
                values
            ),
        )

        colors = []

        for index in indices:

            protein = protein_columns[
                index
            ]

            category = category_map.get(
                str(protein),
                "Other/Mixed",
            )

            colors.append(
                CATEGORY_COLOR_MAP.get(
                    category,
                    "#BAB0AC",
                )
            )

        ax_b.scatter(
            np.full(
                len(values),
                position,
            )
            + jitter,

            values,

            s=PROTEIN_POINT_SIZE,

            alpha=0.78,

            c=colors,

            edgecolor="white",

            linewidth=0.35,

            zorder=3,
        )

    # Median labels
    for position, values in zip(
        positions,
        group_values,
    ):

        if len(
            values
        ) == 0:
            continue

        median_r = np.median(
            values
        )

        ax_b.text(
            position,
            1.03,
            f"Median r = {median_r:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            transform=(
                ax_b.get_xaxis_transform()
            ),
        )

    ax_b.axhline(
        0,
        linestyle=":",
        linewidth=1.2,
        color="#909090",
        zorder=1,
    )

    tick_labels = [
        (
            f"{tertile}\n"
            f"(n={len(tertile_members[tertile])})"
        )
        for tertile in TERTILE_ORDER
    ]

    ax_b.set_xticks(
        positions
    )

    ax_b.set_xticklabels(
        tick_labels
    )

    ax_b.set_xlabel(
        "Abundance tertile",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        labelpad=12,
    )

    ax_b.set_ylabel(
        r"Per-protein Pearson $r$",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        labelpad=12,
    )

    ax_b.set_xlim(
        0.4,
        3.6,
    )

    # No internal title — matches final Figure 2

    present_categories = [
        category
        for category in CATEGORY_ORDER
        if category
        in set(
            protein_df[
                "Category"
            ]
        )
    ]

    category_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=(
                CATEGORY_COLOR_MAP[
                    category
                ]
            ),
            markeredgecolor="white",
            markersize=8,
            label=category,
        )
        for category in (
            present_categories
        )
    ]

    legend_b = ax_b.legend(
        handles=category_handles,
        title="Protein function",
        loc="center left",
        bbox_to_anchor=(
            1.02,
            0.5,
        ),
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    legend_b.get_title().set_fontweight(
        "bold"
    )

    ax_b.text(
        -0.14,
        1.04,
        "B",
        transform=ax_b.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
    )

    apply_figure2_style(
        ax_b
    )

    # ==================================================================
    # Layout
    # ==================================================================

    fig.subplots_adjust(
        left=0.08,
        right=0.81,
        bottom=0.17,
        top=0.94,
        wspace=0.34,
    )

    # ==================================================================
    # Save source data
    # ==================================================================

    protein_data_file = (
        FIGURE_DATA_DIR
        / "figure3_protein_tertiles.csv"
    )

    protein_df.to_csv(
        protein_data_file,
        index=False,
    )

    # ==================================================================
    # Save figure
    # ==================================================================

    png_file = (
        FIGURE_DIR
        / "Figure_3_abundance_performance.png"
    )

    pdf_file = (
        FIGURE_DIR
        / "Figure_3_abundance_performance.pdf"
    )

    tif_file = (
        FIGURE_DIR
        / "Figure_3_abundance_performance.tif"
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
    # Summary
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "FIGURE 3 COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Median Pearson r:"
    )

    for tertile, values in zip(
        TERTILE_ORDER,
        group_values,
    ):

        print(
            f"  {tertile:<7}: "
            f"{np.median(values):.4f}"
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