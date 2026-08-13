"""
Plot Figure 4 from Saved SHAP Data
==================================

This script DOES NOT recompute SHAP.

It reads the outputs from:
    prepare_figure4_shap_data.py

and produces the manuscript Figure 4:

A. Top-30 protein × original-feature SHAP heatmap
B. Continuous/ordered feature SHAP summary
C. Top-10 categorical one-hot SHAP summary

Run:
    python scripts/figures/plot_figure4_shap_interpretation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

SHAP_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "figure4"
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

METADATA_FILE = (
    DATA_DIR
    / "protein_metadata.csv"
)


# ======================================================================
# Style — same manuscript family as Figures 2 and 3
# ======================================================================

plt.rcParams[
    "font.family"
] = "Arial"

DPI = 600

LABEL_SIZE = 15
TICK_SIZE = 11
PANEL_SIZE = 18
LEGEND_SIZE = 9

FRAME_WIDTH = 1.6

TOP_N_PROTEINS = 30
TOP_N_CATEGORICAL = 10

MAX_POINTS = 2500

RANDOM_SEED = 42


# ======================================================================
# Feature definitions from original notebook
# ======================================================================

CONTINUOUS_ORDERED_FEATURES = [
    "size_nm",
    "zp_mv",
    "incub_time",
    "washing_steps",
]

EXPERIMENTAL_FEATURES = {
    "incub_time",
    "washing_steps",
    "agitation",
    "hd_solvent",
    "zp_solvent",
}


# ======================================================================
# Helpers
# ======================================================================


def feature_type(
    feature_name,
):
    """
    Parent feature classification for background bars.
    """

    parent = feature_name

    # encoded categorical variable -> parent variable
    candidate_parents = [
        "mod_type",
        "mod_charge",
        "np_type",
        "np_subtype",
        "zp_charge",
        "zp_solvent",
        "hd_solvent",
        "agitation",
    ]

    for candidate in candidate_parents:

        if feature_name.startswith(
            candidate + "_"
        ):

            parent = candidate
            break

    return (
        "experimental"
        if parent in EXPERIMENTAL_FEATURES
        else "np"
    )


def make_color_map():
    """
    Same blue → orange/red logic used in the notebook.
    """

    colors = [
        "#1f5fa8",
        "#7fb3de",
        "#f4a460",
        "#c43c2e",
    ]

    return LinearSegmentedColormap.from_list(
        "shap_feature_value",
        colors,
    )


def stylize_axis(
    axis,
):

    for spine in axis.spines.values():

        spine.set_visible(
            True
        )

        spine.set_linewidth(
            FRAME_WIDTH
        )

        spine.set_edgecolor(
            "black"
        )

    axis.tick_params(
        width=1.4,
        length=5,
        labelsize=TICK_SIZE,
    )

    for label in (
        axis.get_xticklabels()
        + axis.get_yticklabels()
    ):

        label.set_fontweight(
            "bold"
        )


def get_protein_name_map():
    """
    Recover human-readable protein names from protein_metadata.csv.
    """

    if not METADATA_FILE.exists():
        return {}

    df = pd.read_csv(
        METADATA_FILE
    )

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

    name_candidates = [
        "Protein_name",
        "protein_name",
        "Protein Name",
        "Name",
        "name",
        "Recommended_name",
        "recommended_name",
    ]

    protein_col = next(
        (
            column
            for column in protein_candidates
            if column in df.columns
        ),
        None,
    )

    name_col = next(
        (
            column
            for column in name_candidates
            if column in df.columns
        ),
        None,
    )

    if (
        protein_col is None
        or name_col is None
    ):
        return {}

    return dict(
        zip(
            df[
                protein_col
            ].astype(str),
            df[
                name_col
            ].astype(str),
        )
    )


def subsample_rows(
    shap_matrix,
    feature_matrix,
):
    """
    Limit scatter plotting density while keeping deterministic sampling.
    """

    n_rows = shap_matrix.shape[0]

    if n_rows <= MAX_POINTS:

        return (
            shap_matrix,
            feature_matrix,
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = rng.choice(
        n_rows,
        size=MAX_POINTS,
        replace=False,
    )

    return (
        shap_matrix[
            indices
        ],
        feature_matrix[
            indices
        ],
    )


# ======================================================================
# SHAP summary-panel function
# ======================================================================


def draw_shap_summary(
    axis,
    shap_matrix,
    feature_values,
    feature_names,
):
    """
    Reproduce notebook-style:
      translucent importance bars
      SHAP-value scatter
      top secondary axis = mean |SHAP|
    """

    mean_abs = np.mean(
        np.abs(
            shap_matrix
        ),
        axis=0,
    )

    order = np.argsort(
        mean_abs
    )[
        ::-1
    ]

    shap_matrix = shap_matrix[
        :,
        order
    ]

    feature_values = feature_values[
        :,
        order
    ]

    feature_names = [
        feature_names[
            index
        ]
        for index in order
    ]

    mean_abs = mean_abs[
        order
    ]

    percent = (
        mean_abs
        / mean_abs.sum()
        * 100
    )

    scaled_values = MinMaxScaler().fit_transform(
        feature_values
    )

    cmap = make_color_map()

    n_features = len(
        feature_names
    )

    y_positions = np.arange(
        n_features
    )

    # --------------------------------------------------------------
    # Background bars on a second x-axis
    # --------------------------------------------------------------

    bar_axis = axis.twiny()

    bar_axis.set_zorder(
        0
    )

    axis.set_zorder(
        2
    )

    axis.patch.set_alpha(
        0
    )

    bar_axis.patch.set_alpha(
        0
    )

    bar_max = (
        mean_abs.max()
        * 1.18
    )

    for row_index, (
        feature,
        importance,
        percentage,
    ) in enumerate(
        zip(
            feature_names,
            mean_abs,
            percent,
        )
    ):

        bar_color = (
            "#9e9ac8"
            if feature_type(
                feature
            )
            == "experimental"
            else "#a1d99b"
        )

        bar_axis.barh(
            row_index,
            importance,
            height=0.70,
            color=bar_color,
            alpha=0.40,
            edgecolor="none",
        )

        bar_axis.text(
            bar_max * 0.015,
            row_index,
            f"{importance:.3f} ({percentage:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    bar_axis.set_xlim(
        0,
        bar_max,
    )

    bar_axis.set_xlabel(
        "Mean |SHAP|",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    bar_axis.tick_params(
        axis="x",
        labelsize=9,
        width=1.2,
    )

    for label in (
        bar_axis.get_xticklabels()
    ):

        label.set_fontweight(
            "bold"
        )

    bar_axis.set_yticks(
        []
    )

    # --------------------------------------------------------------
    # Scatter
    # --------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    shap_limit = (
        np.max(
            np.abs(
                shap_matrix
            )
        )
        * 1.08
    )

    for feature_index in range(
        n_features
    ):

        values = shap_matrix[
            :,
            feature_index
        ]

        colors = scaled_values[
            :,
            feature_index
        ]

        jitter = rng.normal(
            0,
            0.09,
            size=len(
                values
            ),
        )

        axis.scatter(
            values,
            feature_index
            + jitter,
            c=colors,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=8,
            alpha=0.60,
            linewidth=0,
            zorder=4,
        )

    axis.axvline(
        0,
        linestyle="--",
        linewidth=1.1,
        color="gray",
        alpha=0.8,
    )

    axis.set_xlim(
        -shap_limit,
        shap_limit,
    )

    axis.set_ylim(
        n_features - 0.5,
        -0.5,
    )

    axis.set_yticks(
        y_positions
    )

    axis.set_yticklabels(
        feature_names,
        fontsize=10,
        fontweight="bold",
    )

    axis.set_xlabel(
        "SHAP value",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        labelpad=10,
    )

    stylize_axis(
        axis
    )

    # --------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------

    legend_handles = [
        Patch(
            facecolor="#9e9ac8",
            edgecolor="none",
            label="Experimental Conditions",
        ),
        Patch(
            facecolor="#a1d99b",
            edgecolor="none",
            label="NP Physicochemical Properties",
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7.5,
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
    )

    return cmap


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "PLOTTING FIGURE 4"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------

    with open(
        SHAP_DIR
        / "figure4_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    arrays = np.load(
        SHAP_DIR
        / "figure4_shap_arrays.npz"
    )

    shap_encoded = arrays[
        "shap_encoded"
    ]

    X_encoded = arrays[
        "X_test_encoded"
    ]

    shap_aggregated = arrays[
        "shap_aggregated"
    ]

    X_aggregated = arrays[
        "X_test_aggregated"
    ]

    encoded_columns = metadata[
        "encoded_columns"
    ]

    parent_features = metadata[
        "parent_features"
    ]

    panel = metadata[
        "protein_panel"
    ]

    feature_groups = metadata[
        "feature_groups"
    ]

    # ==================================================================
    # Panel A — heatmap
    # ==================================================================

    heatmap_df = pd.read_csv(
        SHAP_DIR
        / "heatmap_mean_abs_shap.csv",
        index_col=0,
    )

    protein_importance = (
        heatmap_df
        .mean(
            axis=1
        )
        .sort_values(
            ascending=False
        )
    )

    top_proteins = list(
        protein_importance.index[
            :TOP_N_PROTEINS
        ]
    )

    top_features = list(
        heatmap_df.columns
    )

    heatmap_subset = heatmap_df.loc[
        top_proteins,
        top_features,
    ]

    # Scale relative to the maximum cell exactly as notebook figure.
    maximum = heatmap_subset.to_numpy().max()

    heatmap_percent = (
        heatmap_subset
        / maximum
        * 100
    )

    protein_names = (
        get_protein_name_map()
    )

    heatmap_labels = [
        protein_names.get(
            str(
                protein
            ),
            str(
                protein
            ),
        )
        for protein in top_proteins
    ]

    # ==================================================================
    # Panel B — continuous/ordered
    # ==================================================================

    parent_lookup = {
        feature: index
        for index, feature in enumerate(
            parent_features
        )
    }

    continuous_features = [
        feature
        for feature in (
            CONTINUOUS_ORDERED_FEATURES
        )
        if feature in parent_lookup
    ]

    continuous_indices = [
        parent_lookup[
            feature
        ]
        for feature in (
            continuous_features
        )
    ]

    # --------------------------------------------------------------
    # Concatenate all proteins, matching notebook logic.
    #
    # aggregated_shap:
    # sample × parent feature × protein
    # --------------------------------------------------------------

    continuous_shap_parts = []

    continuous_X_parts = []

    for protein_index in range(
        len(
            panel
        )
    ):

        continuous_shap_parts.append(
            shap_aggregated[
                :,
                continuous_indices,
                protein_index
            ]
        )

        continuous_X_parts.append(
            X_aggregated[
                :,
                continuous_indices
            ]
        )

    continuous_shap = np.concatenate(
        continuous_shap_parts,
        axis=0,
    )

    continuous_X = np.concatenate(
        continuous_X_parts,
        axis=0,
    )

    (
        continuous_shap,
        continuous_X,
    ) = subsample_rows(
        continuous_shap,
        continuous_X,
    )

    # ==================================================================
    # Panel C — top categorical one-hot features
    # ==================================================================

    categorical_parent_features = [
        "mod_type",
        "mod_charge",
        "np_type",
        "np_subtype",
        "zp_charge",
        "zp_solvent",
        "hd_solvent",
        "agitation",
    ]

    categorical_encoded_columns = []

    for parent in (
        categorical_parent_features
    ):

        categorical_encoded_columns.extend(
            feature_groups.get(
                parent,
                []
            )
        )

    encoded_lookup = {
        column: index
        for index, column in enumerate(
            encoded_columns
        )
    }

    categorical_indices = [
        encoded_lookup[
            column
        ]
        for column in (
            categorical_encoded_columns
        )
    ]

    # --------------------------------------------------------------
    # Rank encoded categorical variables by global mean |SHAP|.
    # --------------------------------------------------------------

    categorical_global_importance = []

    for encoded_index in (
        categorical_indices
    ):

        importance = float(
            np.mean(
                np.abs(
                    shap_encoded[
                        :,
                        encoded_index,
                        :,
                    ]
                )
            )
        )

        categorical_global_importance.append(
            importance
        )

    categorical_order = np.argsort(
        categorical_global_importance
    )[
        ::-1
    ][
        :TOP_N_CATEGORICAL
    ]

    top_categorical_columns = [
        categorical_encoded_columns[
            index
        ]
        for index in (
            categorical_order
        )
    ]

    top_categorical_indices = [
        categorical_indices[
            index
        ]
        for index in (
            categorical_order
        )
    ]

    categorical_shap_parts = []

    categorical_X_parts = []

    for protein_index in range(
        len(
            panel
        )
    ):

        categorical_shap_parts.append(
            shap_encoded[
                :,
                top_categorical_indices,
                protein_index,
            ]
        )

        categorical_X_parts.append(
            X_encoded[
                :,
                top_categorical_indices
            ]
        )

    categorical_shap = np.concatenate(
        categorical_shap_parts,
        axis=0,
    )

    categorical_X = np.concatenate(
        categorical_X_parts,
        axis=0,
    )

    (
        categorical_shap,
        categorical_X,
    ) = subsample_rows(
        categorical_shap,
        categorical_X,
    )

    # ==================================================================
    # Figure layout
    # ==================================================================

    fig = plt.figure(
        figsize=(
            17.5,
            9.0,
        )
    )

    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[
            1.05,
            1.35,
        ],
        height_ratios=[
            1,
            1,
        ],
        wspace=0.42,
        hspace=0.34,
    )

    ax_heatmap = fig.add_subplot(
        grid[
            :,
            0
        ]
    )

    ax_continuous = fig.add_subplot(
        grid[
            0,
            1
        ]
    )

    ax_categorical = fig.add_subplot(
        grid[
            1,
            1
        ]
    )

    # ==================================================================
    # A — Heatmap
    # ==================================================================

    image = ax_heatmap.imshow(
        heatmap_percent.to_numpy(),
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
    )

    ax_heatmap.set_xticks(
        np.arange(
            len(
                top_features
            )
        )
    )

    ax_heatmap.set_xticklabels(
        top_features,
        rotation=45,
        ha="right",
        fontsize=10,
        fontweight="bold",
    )

    ax_heatmap.set_yticks(
        np.arange(
            len(
                heatmap_labels
            )
        )
    )

    ax_heatmap.set_yticklabels(
        heatmap_labels,
        fontsize=9.5,
    )

    # Cell grid
    ax_heatmap.set_xticks(
        np.arange(
            -0.5,
            len(
                top_features
            ),
            1,
        ),
        minor=True,
    )

    ax_heatmap.set_yticks(
        np.arange(
            -0.5,
            len(
                heatmap_labels
            ),
            1,
        ),
        minor=True,
    )

    ax_heatmap.grid(
        which="minor",
        linewidth=0.35,
        color="gray",
    )

    ax_heatmap.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax_heatmap,
        fraction=0.045,
        pad=0.05,
    )

    

    colorbar.set_ticks(
        [
            0,
            20,
            40,
            60,
            80,
            100,
        ]
    )

    colorbar.set_ticklabels(
        [
            "0%",
            "20%",
            "40%",
            "60%",
            "80%",
            "100%",
        ]
    )

    for label in (
        colorbar.ax.get_yticklabels()
    ):

        label.set_fontweight(
            "bold"
        )

    stylize_axis(
        ax_heatmap
    )

    ax_heatmap.text(
        -0.14,
        1.03,
        "A",
        transform=ax_heatmap.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
    )

    # ==================================================================
    # B — continuous
    # ==================================================================

    cmap = draw_shap_summary(
        ax_continuous,
        continuous_shap,
        continuous_X,
        continuous_features,
    )

    ax_continuous.text(
        -0.14,
        1.08,
        "B",
        transform=ax_continuous.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
    )

    # ==================================================================
    # C — categorical
    # ==================================================================

    draw_shap_summary(
        ax_categorical,
        categorical_shap,
        categorical_X,
        top_categorical_columns,
    )

    ax_categorical.text(
        -0.14,
        1.08,
        "C",
        transform=ax_categorical.transAxes,
        fontsize=PANEL_SIZE,
        fontweight="bold",
        va="top",
    )

    # ==================================================================
    # Shared colorbar for panels B/C feature values
    # ==================================================================

    scalar_map = cm.ScalarMappable(
        norm=Normalize(
            0,
            1,
        ),
        cmap=cmap,
    )

    scalar_map.set_array(
        []
    )

    # ==================================================================
    # Save
    # ==================================================================

    png_file = (
        FIGURE_DIR
        / "Figure_4_SHAP_interpretation.png"
    )

    pdf_file = (
        FIGURE_DIR
        / "Figure_4_SHAP_interpretation.pdf"
    )

    tif_file = (
        FIGURE_DIR
        / "Figure_4_SHAP_interpretation.tif"
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

    print()

    print(
        "=" * 72
    )

    print(
        "FIGURE 4 COMPLETE"
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