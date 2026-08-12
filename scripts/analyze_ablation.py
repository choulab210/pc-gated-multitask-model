"""
Analyze Architecture Ablation Results
=====================================

This script analyzes the completed ablation study.

Architectures
-------------
A. Abundance-only
B. Two-head without gating
C. Two-head with gating

Paired comparisons
------------------
B - A
    Effect of adding the adsorption task / multitask learning.

C - B
    Incremental effect of adsorption-guided gating.

C - A
    Total effect of the full gated architecture relative to
    abundance-only prediction.

For each comparison and metric, this script reports:

- Mean paired difference
- Standard deviation of paired differences
- Median paired difference
- 95% bootstrap confidence interval for the mean paired difference
- Wilcoxon signed-rank p-value

Metrics
-------
- Median_r
- 1-TVD
- Cosine

Run from project root:

    python scripts/analyze_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABLATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "ablation"
)

INPUT_FILE = (
    ABLATION_DIR
    / "seed_results.csv"
)

OUTPUT_FILE = (
    ABLATION_DIR
    / "paired_comparisons.csv"
)

MANUSCRIPT_TABLE_FILE = (
    ABLATION_DIR
    / "manuscript_ablation_table.csv"
)

SUMMARY_JSON_FILE = (
    ABLATION_DIR
    / "ablation_analysis_summary.json"
)


# ======================================================================
# Configuration
# ======================================================================

RANDOM_SEED = 42

N_BOOTSTRAP = 10000

CI_LEVEL = 0.95


METRICS = [
    "Median_r",
    "1-TVD",
    "Cosine",
]


VARIANT_A = "abundance_only"

VARIANT_B = "two_head_no_gate"

VARIANT_C = "two_head_gated"


COMPARISONS = [
    (
        "B - A",
        VARIANT_B,
        VARIANT_A,
        "Multitask learning effect",
    ),

    (
        "C - B",
        VARIANT_C,
        VARIANT_B,
        "Incremental gating effect",
    ),

    (
        "C - A",
        VARIANT_C,
        VARIANT_A,
        "Full architecture effect",
    ),
]


# ======================================================================
# Bootstrap confidence interval
# ======================================================================


def bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    seed: int = RANDOM_SEED,
):
    """
    Bootstrap confidence interval for the mean paired difference.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    if len(differences) == 0:

        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    n = len(
        differences
    )

    for i in range(
        n_bootstrap
    ):

        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_means[
            i
        ] = np.mean(
            sample
        )

    alpha = (
        1.0
        - ci_level
    )

    lower_percentile = (
        100.0
        * alpha
        / 2.0
    )

    upper_percentile = (
        100.0
        * (
            1.0
            - alpha / 2.0
        )
    )

    lower = float(
        np.percentile(
            bootstrap_means,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            upper_percentile,
        )
    )

    return (
        lower,
        upper,
    )


# ======================================================================
# Wilcoxon test
# ======================================================================


def paired_wilcoxon(
    differences: np.ndarray,
) -> float:
    """
    Two-sided Wilcoxon signed-rank test on paired differences.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    if (
        len(differences)
        < 2
    ):

        return np.nan

    # If every difference is exactly zero, scipy may raise.
    if np.allclose(
        differences,
        0.0,
    ):

        return 1.0

    try:

        result = wilcoxon(
            differences,
            alternative="two-sided",
        )

        return float(
            result.pvalue
        )

    except ValueError:

        return np.nan


# ======================================================================
# Prepare paired seed table
# ======================================================================


def build_variant_table(
    data: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    """
    Extract seed-level metrics for one architecture.
    """

    subset = (
        data[
            data[
                "Variant"
            ]
            == variant
        ]
        .copy()
    )

    required_columns = (
        ["Seed"]
        + METRICS
    )

    missing = [
        column
        for column in required_columns
        if column
        not in subset.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns for variant {variant}: "
            f"{missing}"
        )

    subset = (
        subset[
            required_columns
        ]
        .sort_values(
            "Seed"
        )
        .reset_index(
            drop=True
        )
    )

    return subset


# ======================================================================
# Analyze one comparison
# ======================================================================


def analyze_comparison(
    data: pd.DataFrame,
    comparison_name: str,
    better_variant: str,
    reference_variant: str,
    interpretation: str,
) -> list[dict]:
    """
    Analyze paired differences for one architecture comparison.
    """

    better = build_variant_table(
        data,
        better_variant,
    )

    reference = build_variant_table(
        data,
        reference_variant,
    )

    merged = better.merge(
        reference,
        on="Seed",
        suffixes=(
            "_better",
            "_reference",
        ),
        how="inner",
    )

    if len(
        merged
    ) == 0:

        raise ValueError(
            f"No shared seeds found for "
            f"{better_variant} and "
            f"{reference_variant}."
        )

    rows = []

    for metric in METRICS:

        differences = (
            merged[
                f"{metric}_better"
            ].to_numpy(
                dtype=float
            )
            -
            merged[
                f"{metric}_reference"
            ].to_numpy(
                dtype=float
            )
        )

        mean_difference = float(
            np.mean(
                differences
            )
        )

        median_difference = float(
            np.median(
                differences
            )
        )

        if (
            len(
                differences
            )
            > 1
        ):

            sd_difference = float(
                np.std(
                    differences,
                    ddof=1,
                )
            )

        else:

            sd_difference = 0.0

        (
            ci_lower,
            ci_upper,
        ) = bootstrap_mean_ci(
            differences
        )

        p_value = paired_wilcoxon(
            differences
        )

        positive_count = int(
            np.sum(
                differences
                > 0
            )
        )

        negative_count = int(
            np.sum(
                differences
                < 0
            )
        )

        zero_count = int(
            np.sum(
                differences
                == 0
            )
        )

        rows.append(
            {
                "Comparison":
                    comparison_name,

                "Interpretation":
                    interpretation,

                "Metric":
                    metric,

                "N_pairs":
                    int(
                        len(
                            differences
                        )
                    ),

                "Mean_difference":
                    mean_difference,

                "SD_difference":
                    sd_difference,

                "Median_difference":
                    median_difference,

                "CI95_lower":
                    ci_lower,

                "CI95_upper":
                    ci_upper,

                "Wilcoxon_p":
                    p_value,

                "Positive_pairs":
                    positive_count,

                "Negative_pairs":
                    negative_count,

                "Zero_pairs":
                    zero_count,
            }
        )

    return rows


# ======================================================================
# Manuscript summary table
# ======================================================================


def build_manuscript_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create mean ± SD table for the three architectures.
    """

    model_order = [
        VARIANT_A,
        VARIANT_B,
        VARIANT_C,
    ]

    model_labels = {
        VARIANT_A:
            "Abundance-only",

        VARIANT_B:
            "Two-head without gating",

        VARIANT_C:
            "Two-head with gating",
    }

    rows = []

    for variant in model_order:

        subset = data[
            data[
                "Variant"
            ]
            == variant
        ]

        row = {
            "Model":
                model_labels[
                    variant
                ],

            "N_seeds":
                int(
                    len(
                        subset
                    )
                ),
        }

        for metric in METRICS:

            values = (
                subset[
                    metric
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            row[
                f"{metric}_sd"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            row[
                f"{metric}_formatted"
            ] = (
                f"{row[f'{metric}_mean']:.3f} "
                f"± "
                f"{row[f'{metric}_sd']:.3f}"
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "ABLATION STATISTICAL ANALYSIS"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Ablation seed results were not found:\n"
            f"{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    print()

    print(
        "Rows loaded:",
        len(
            data
        ),
    )

    print(
        "Seeds:",
        sorted(
            data[
                "Seed"
            ]
            .unique()
            .tolist()
        ),
    )

    # ==================================================================
    # Paired comparisons
    # ==================================================================

    comparison_rows = []

    for (
        comparison_name,
        better_variant,
        reference_variant,
        interpretation,
    ) in COMPARISONS:

        comparison_rows.extend(
            analyze_comparison(
                data,
                comparison_name,
                better_variant,
                reference_variant,
                interpretation,
            )
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ==================================================================
    # Manuscript table
    # ==================================================================

    manuscript_table = (
        build_manuscript_table(
            data
        )
    )

    manuscript_table.to_csv(
        MANUSCRIPT_TABLE_FILE,
        index=False,
    )

    # ==================================================================
    # Console: architecture performance
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "ARCHITECTURE PERFORMANCE"
    )

    print(
        "=" * 72
    )

    display_columns = [
        "Model",
        "Median_r_formatted",
        "1-TVD_formatted",
        "Cosine_formatted",
    ]

    print(
        manuscript_table[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ==================================================================
    # Console: paired differences
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "PAIRED ARCHITECTURE DIFFERENCES"
    )

    print(
        "=" * 72
    )

    for comparison_name in [
        "B - A",
        "C - B",
        "C - A",
    ]:

        subset = comparison_df[
            comparison_df[
                "Comparison"
            ]
            == comparison_name
        ]

        print()

        print(
            comparison_name
        )

        for _, row in (
            subset.iterrows()
        ):

            print(
                f"  {row['Metric']:8s}: "
                f"Δ={row['Mean_difference']:+.4f}, "
                f"95% CI "
                f"[{row['CI95_lower']:+.4f}, "
                f"{row['CI95_upper']:+.4f}], "
                f"p={row['Wilcoxon_p']:.4f}, "
                f"positive seeds="
                f"{int(row['Positive_pairs'])}/"
                f"{int(row['N_pairs'])}"
            )

    # ==================================================================
    # JSON summary
    # ==================================================================

    summary = {
        "n_bootstrap":
            N_BOOTSTRAP,

        "ci_level":
            CI_LEVEL,

        "seeds":
            sorted(
                data[
                    "Seed"
                ]
                .unique()
                .astype(int)
                .tolist()
            ),

        "comparisons":
            comparison_df.to_dict(
                orient="records"
            ),
    }

    with open(
        SUMMARY_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ==================================================================
    # Finish
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "ABLATION ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Files saved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        MANUSCRIPT_TABLE_FILE
    )

    print(
        SUMMARY_JSON_FILE
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()