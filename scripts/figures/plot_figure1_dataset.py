"""
Figure 1
========

Protein-panel selection and functional representation of the
protein-corona dataset.

Panel A
-------
Binned protein abundance contribution and exact cumulative abundance
coverage in the model-development dataset.

Panel B
-------
Functional-category distribution of the selected 174 proteins.

Run from project root:
    python scripts/figures/plot_figure1_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcmodel.data import (
    COVERAGE_TARGET,
    MIN_NPS_PER_PROTEIN,
    prepare_model_data,
)

from pcmodel.metadata import (
    CATEGORY_ORDER,
    load_protein_metadata,
    metadata_to_mappings,
    validate_metadata_for_panel,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

FIGURE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "main"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_FILE = (
    DATA_DIR
    / "Data_1.csv"
)

ABUNDANCE_FILE = (
    DATA_DIR
    / "Data_2.csv"
)

METADATA_FILE = (
    DATA_DIR
    / "protein_metadata.csv"
)


# ======================================================================
# Plot settings
# ======================================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.2,
    }
)


# ======================================================================
# User-adjustable Figure 1 settings
# ======================================================================

N_BINS = 20


# ======================================================================
# Load reproducible model data
# ======================================================================

print("=" * 72)
print("PREPARING FIGURE 1 DATA")
print("=" * 72)

prepared = prepare_model_data(
    FEATURE_FILE,
    ABUNDANCE_FILE,
)

panel = list(
    prepared.panel
)

panel_result = (
    prepared.panel_result
)

print()
print(
    "Development samples:",
    len(
        prepared.train_ids
    ),
)

print(
    "Held-out samples:",
    len(
        prepared.test_ids
    ),
)

print(
    "Selected individual proteins:",
    len(panel),
)

print(
    "Abundance outputs including OTHER:",
    len(
        prepared.abundance_columns
    ),
)


# ======================================================================
# Panel A data
# ======================================================================

stats = (
    panel_result.stats
    .copy()
)

# Keep proteins passing the development-set frequency filter.
eligible = (
    stats.loc[
        stats["n_nps_detected"]
        >= MIN_NPS_PER_PROTEIN
    ]
    .copy()
)

eligible = (
    eligible
    .sort_values(
        "sum_abundance",
        ascending=False,
    )
    .reset_index()
    .rename(
        columns={
            "index": "protein_id"
        }
    )
)

eligible_total = (
    eligible[
        "sum_abundance"
    ]
    .sum()
)

eligible[
    "relative_abundance"
] = (
    eligible[
        "sum_abundance"
    ]
    / eligible_total
)

eligible[
    "cumulative_abundance"
] = (
    eligible[
        "relative_abundance"
    ]
    .cumsum()
)

eligible[
    "rank"
] = np.arange(
    1,
    len(eligible) + 1,
)

selected_n = len(
    panel
)

selected_cumulative = (
    eligible.iloc[
        selected_n - 1
    ][
        "cumulative_abundance"
    ]
)


# ======================================================================
# Create binned data for Panel A bars
# ======================================================================

# Use equal-width rank bins only for visualization.
eligible[
    "bin"
] = pd.cut(
    eligible[
        "rank"
    ],
    bins=N_BINS,
    labels=False,
    include_lowest=True,
)

binned = (
    eligible
    .groupby(
        "bin",
        as_index=False,
        observed=True,
    )
    .agg(
        rank_min=(
            "rank",
            "min",
        ),
        rank_max=(
            "rank",
            "max",
        ),
        relative_abundance=(
            "relative_abundance",
            "sum",
        ),
    )
)

binned[
    "rank_mid"
] = (
    binned[
        "rank_min"
    ]
    + binned[
        "rank_max"
    ]
) / 2.0

binned[
    "width"
] = (
    binned[
        "rank_max"
    ]
    - binned[
        "rank_min"
    ]
    + 1
)

binned[
    "relative_percent"
] = (
    binned[
        "relative_abundance"
    ]
    * 100.0
)


print()
print(
    "Eligible proteins:",
    len(
        eligible
    ),
)

print(
    "Selected panel:",
    selected_n,
)

print(
    "Number of visualization bins:",
    len(
        binned
    ),
)

print(
    "Cumulative abundance at selected cutoff:",
    selected_cumulative,
)

print(
    "Eligible fraction of total abundance:",
    panel_result.eligible_abundance_fraction_of_all,
)


# ======================================================================
# Panel B data
# ======================================================================

metadata = (
    load_protein_metadata(
        METADATA_FILE
    )
)

validate_metadata_for_panel(
    metadata,
    panel,
    require_all=True,
)

(
    id_to_name,
    id_to_category,
) = metadata_to_mappings(
    metadata
)

category_counts = (
    pd.Series(
        [
            id_to_category[
                protein
            ]
            for protein
            in panel
        ]
    )
    .value_counts()
)

category_rows = []

for category in CATEGORY_ORDER:

    count = int(
        category_counts.get(
            category,
            0,
        )
    )

    category_rows.append(
        {
            "Category":
                category,

            "Count":
                count,

            "Percent":
                100.0
                * count
                / len(panel),
        }
    )

category_df = pd.DataFrame(
    category_rows
)

category_df = (
    category_df
    .sort_values(
        "Count",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)

print()
print("Functional categories:")
print(
    category_df.to_string(
        index=False
    )
)

print()
print(
    "Category count total:",
    category_df[
        "Count"
    ].sum(),
)


# ======================================================================
# Create figure
# ======================================================================

fig = plt.figure(
    figsize=(13.8, 5.6)
)

grid = fig.add_gridspec(
    1,
    2,
    width_ratios=[
        1.18,
        1.0,
    ],
    wspace=0.42,
)


# ======================================================================
# Panel A
# ======================================================================

ax_a = fig.add_subplot(
    grid[0, 0]
)

x = (
    eligible[
        "rank"
    ]
    .to_numpy()
)

cumulative_percent = (
    eligible[
        "cumulative_abundance"
    ]
    .to_numpy()
    * 100.0
)


# ----------------------------------------------------------------------
# Binned abundance contribution bars
# ----------------------------------------------------------------------

ax_a.bar(
    binned[
        "rank_mid"
    ],
    binned[
        "relative_percent"
    ],
    width=(
        binned[
            "width"
        ]
        * 0.88
    ),
    alpha=0.85,
    edgecolor="none",
)


# ----------------------------------------------------------------------
# Primary axis
# ----------------------------------------------------------------------

ax_a.set_xlabel(
    "Proteins ranked by total abundance"
)

ax_a.set_ylabel(
    "Abundance contribution per rank bin (%)"
)

ax_a.set_xlim(
    0,
    len(
        eligible
    )
    + 1,
)

ax_a.set_ylim(
    bottom=0
)


# ----------------------------------------------------------------------
# Secondary axis for exact cumulative abundance
# ----------------------------------------------------------------------

ax_a2 = (
    ax_a.twinx()
)

ax_a2.plot(
    x,
    cumulative_percent,
    linewidth=2.2,
)

ax_a2.set_ylabel(
    "Cumulative abundance (%)"
)

ax_a2.set_ylim(
    0,
    102,
)


# ----------------------------------------------------------------------
# 99% cutoff
# ----------------------------------------------------------------------

ax_a2.axhline(
    COVERAGE_TARGET
    * 100,
    linestyle="--",
    linewidth=1.3,
)

ax_a.axvline(
    selected_n,
    linestyle="--",
    linewidth=1.3,
)


# ----------------------------------------------------------------------
# Annotation
# ----------------------------------------------------------------------

ax_a2.annotate(
    f"99% coverage\nn = {selected_n}",
    xy=(
        selected_n,
        selected_cumulative
        * 100,
    ),
    xytext=(
        max(
            selected_n - 70,
            20,
        ),
        77,
    ),
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.0,
    },
    fontsize=10,
    ha="left",
)


# ----------------------------------------------------------------------
# Panel label
# ----------------------------------------------------------------------

ax_a.text(
    -0.12,
    1.05,
    "A",
    transform=ax_a.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)


# ----------------------------------------------------------------------
# Four-sided frame
# ----------------------------------------------------------------------

for spine in (
    ax_a.spines.values()
):
    spine.set_visible(
        True
    )

for spine in (
    ax_a2.spines.values()
):
    spine.set_visible(
        True
    )


# ======================================================================
# Panel B
# ======================================================================

ax_b = fig.add_subplot(
    grid[0, 1]
)

y = np.arange(
    len(
        category_df
    )
)

bars = ax_b.barh(
    y,
    category_df[
        "Count"
    ],
)

ax_b.set_yticks(
    y
)

ax_b.set_yticklabels(
    category_df[
        "Category"
    ]
)

ax_b.set_xlabel(
    "Number of proteins"
)

ax_b.set_xlim(
    0,
    category_df[
        "Count"
    ].max()
    * 1.42,
)


# ----------------------------------------------------------------------
# Count and percentage labels
# ----------------------------------------------------------------------

for (
    bar,
    count,
    percent,
) in zip(
    bars,
    category_df[
        "Count"
    ],
    category_df[
        "Percent"
    ],
):

    ax_b.text(
        bar.get_width()
        + 0.5,

        bar.get_y()
        + bar.get_height()
        / 2,

        f"{count} ({percent:.1f}%)",

        va="center",
        ha="left",
        fontsize=9.5,
    )


# ----------------------------------------------------------------------
# Panel label
# ----------------------------------------------------------------------

ax_b.text(
    -0.15,
    1.05,
    "B",
    transform=ax_b.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)


# ----------------------------------------------------------------------
# Four-sided frame
# ----------------------------------------------------------------------

for spine in (
    ax_b.spines.values()
):
    spine.set_visible(
        True
    )


# ======================================================================
# Layout
# ======================================================================

fig.subplots_adjust(
    left=0.09,
    right=0.96,
    top=0.94,
    bottom=0.15,
)


# ======================================================================
# Save
# ======================================================================

output_base = (
    FIGURE_DIR
    / "Figure_1_dataset"
)

fig.savefig(
    output_base.with_suffix(
        ".png"
    ),
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    output_base.with_suffix(
        ".pdf"
    ),
    bbox_inches="tight",
)

fig.savefig(
    output_base.with_suffix(
        ".tif"
    ),
    dpi=600,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ======================================================================
# Done
# ======================================================================

print()
print("=" * 72)
print("FIGURE 1 COMPLETE")
print("=" * 72)

print()
print("Saved:")

print(
    output_base.with_suffix(
        ".png"
    )
)

print(
    output_base.with_suffix(
        ".pdf"
    )
)

print(
    output_base.with_suffix(
        ".tif"
    )
)