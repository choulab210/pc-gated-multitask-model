"""
Analyze Generalization Gap
==========================

Compare conventional held-out test performance against stricter
feature-grouped cross-validation performance.

The goal is to quantify how much each model's performance decreases
when identical raw feature signatures are prevented from appearing
across training and validation folds.

Inputs
------
results/benchmarks/
    adsorption_benchmarks.csv
    abundance_benchmarks.csv

results/grouped_benchmarks/
    adsorption_summary.csv
    abundance_summary.csv

results/grouped_validation/
    summary.csv

Outputs
-------
results/generalization/
    adsorption_generalization_gap.csv
    abundance_generalization_gap.csv

Metrics
-------
Adsorption:
    F1
    AUROC
    AUPRC
    MCC

Abundance:
    Median_r
    1-TVD
    Cosine

For each model:

    Absolute change = Grouped score - Random held-out score

    Relative retention (%) =
        Grouped score / Random held-out score * 100

    Relative drop (%) =
        (Random held-out score - Grouped score)
        / Random held-out score * 100

Run:
    python scripts/analyze_generalization_gap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

BENCHMARK_DIR = (
    RESULTS_DIR
    / "benchmarks"
)

GROUPED_BENCHMARK_DIR = (
    RESULTS_DIR
    / "grouped_benchmarks"
)

GROUPED_TWOHEAD_DIR = (
    RESULTS_DIR
    / "grouped_validation"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "generalization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# Input files
# ======================================================================

RANDOM_ADSORPTION_FILE = (
    BENCHMARK_DIR
    / "adsorption_benchmarks.csv"
)

RANDOM_ABUNDANCE_FILE = (
    BENCHMARK_DIR
    / "abundance_benchmarks.csv"
)

GROUPED_ADSORPTION_FILE = (
    GROUPED_BENCHMARK_DIR
    / "adsorption_summary.csv"
)

GROUPED_ABUNDANCE_FILE = (
    GROUPED_BENCHMARK_DIR
    / "abundance_summary.csv"
)

GROUPED_TWOHEAD_FILE = (
    GROUPED_TWOHEAD_DIR
    / "summary.csv"
)


# ======================================================================
# Metrics
# ======================================================================

ADSORPTION_METRICS = [
    "F1",
    "AUROC",
    "AUPRC",
    "MCC",
]

ABUNDANCE_METRICS = [
    "Median_r",
    "1-TVD",
    "Cosine",
]


# ======================================================================
# Utilities
# ======================================================================


def check_files() -> None:

    files = [
        RANDOM_ADSORPTION_FILE,
        RANDOM_ABUNDANCE_FILE,
        GROUPED_ADSORPTION_FILE,
        GROUPED_ABUNDANCE_FILE,
        GROUPED_TWOHEAD_FILE,
    ]

    missing = [
        path
        for path in files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required result file(s):\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def compute_change(
    random_score: float,
    grouped_score: float,
):
    """
    Calculate absolute change, retention, and relative drop.
    """

    absolute_change = (
        grouped_score
        - random_score
    )

    if (
        not np.isfinite(
            random_score
        )
        or random_score == 0
    ):

        retention = np.nan

        relative_drop = np.nan

    else:

        retention = (
            grouped_score
            / random_score
            * 100.0
        )

        relative_drop = (
            (
                random_score
                - grouped_score
            )
            / random_score
            * 100.0
        )

    return (
        absolute_change,
        retention,
        relative_drop,
    )


# ======================================================================
# Load grouped two-head results
# ======================================================================


def load_grouped_twohead() -> dict:
    """
    Convert grouped-validation summary into metric -> value dictionary.
    """

    df = pd.read_csv(
        GROUPED_TWOHEAD_FILE
    )

    required = {
        "Metric",
        "Mean",
    }

    if not required.issubset(
        df.columns
    ):

        raise ValueError(
            "Grouped two-head summary must contain "
            "'Metric' and 'Mean' columns."
        )

    return dict(
        zip(
            df[
                "Metric"
            ],
            df[
                "Mean"
            ],
        )
    )


# ======================================================================
# Adsorption
# ======================================================================


def analyze_adsorption(
    grouped_twohead: dict,
) -> pd.DataFrame:

    random_df = pd.read_csv(
        RANDOM_ADSORPTION_FILE
    )

    grouped_df = pd.read_csv(
        GROUPED_ADSORPTION_FILE
    )

    rows = []

    # ------------------------------------------------------------------
    # Two-head model
    # ------------------------------------------------------------------

    twohead_random = (
        random_df[
            random_df[
                "Model"
            ]
            == "Two-head Model"
        ]
    )

    if len(
        twohead_random
    ) != 1:

        raise ValueError(
            "Could not uniquely identify Two-head Model "
            "in adsorption_benchmarks.csv"
        )

    for metric in ADSORPTION_METRICS:

        random_score = float(
            twohead_random.iloc[
                0
            ][
                metric
            ]
        )

        grouped_score = float(
            grouped_twohead[
                metric
            ]
        )

        (
            absolute_change,
            retention,
            relative_drop,
        ) = compute_change(
            random_score,
            grouped_score,
        )

        rows.append(
            {
                "Model":
                    "Two-head Model",

                "Metric":
                    metric,

                "Random_score":
                    random_score,

                "Grouped_score":
                    grouped_score,

                "Absolute_change":
                    absolute_change,

                "Retention_pct":
                    retention,

                "Relative_drop_pct":
                    relative_drop,
            }
        )

    # ------------------------------------------------------------------
    # Conventional models
    # ------------------------------------------------------------------

    for _, random_row in (
        random_df.iterrows()
    ):

        model = (
            random_row[
                "Model"
            ]
        )

        if (
            model
            == "Two-head Model"
        ):

            continue

        grouped_row = (
            grouped_df[
                grouped_df[
                    "Model"
                ]
                == model
            ]
        )

        if len(
            grouped_row
        ) != 1:

            raise ValueError(
                f"Could not uniquely match grouped adsorption "
                f"result for {model}"
            )

        grouped_row = (
            grouped_row.iloc[
                0
            ]
        )

        for metric in ADSORPTION_METRICS:

            random_score = float(
                random_row[
                    metric
                ]
            )

            grouped_score = float(
                grouped_row[
                    f"{metric}_mean"
                ]
            )

            (
                absolute_change,
                retention,
                relative_drop,
            ) = compute_change(
                random_score,
                grouped_score,
            )

            rows.append(
                {
                    "Model":
                        model,

                    "Metric":
                        metric,

                    "Random_score":
                        random_score,

                    "Grouped_score":
                        grouped_score,

                    "Absolute_change":
                        absolute_change,

                    "Retention_pct":
                        retention,

                    "Relative_drop_pct":
                        relative_drop,
                }
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Abundance
# ======================================================================


def analyze_abundance(
    grouped_twohead: dict,
) -> pd.DataFrame:

    random_df = pd.read_csv(
        RANDOM_ABUNDANCE_FILE
    )

    grouped_df = pd.read_csv(
        GROUPED_ABUNDANCE_FILE
    )

    rows = []

    # ------------------------------------------------------------------
    # Two-head model
    # ------------------------------------------------------------------

    twohead_random = (
        random_df[
            random_df[
                "Model"
            ]
            == "Two-head Model"
        ]
    )

    if len(
        twohead_random
    ) != 1:

        raise ValueError(
            "Could not uniquely identify Two-head Model "
            "in abundance_benchmarks.csv"
        )

    for metric in ABUNDANCE_METRICS:

        random_score = float(
            twohead_random.iloc[
                0
            ][
                metric
            ]
        )

        grouped_score = float(
            grouped_twohead[
                metric
            ]
        )

        (
            absolute_change,
            retention,
            relative_drop,
        ) = compute_change(
            random_score,
            grouped_score,
        )

        rows.append(
            {
                "Model":
                    "Two-head Model",

                "Metric":
                    metric,

                "Random_score":
                    random_score,

                "Grouped_score":
                    grouped_score,

                "Absolute_change":
                    absolute_change,

                "Retention_pct":
                    retention,

                "Relative_drop_pct":
                    relative_drop,
            }
        )

    # ------------------------------------------------------------------
    # Conventional models
    # ------------------------------------------------------------------

    for _, random_row in (
        random_df.iterrows()
    ):

        model = (
            random_row[
                "Model"
            ]
        )

        if (
            model
            == "Two-head Model"
        ):

            continue

        grouped_row = (
            grouped_df[
                grouped_df[
                    "Model"
                ]
                == model
            ]
        )

        if len(
            grouped_row
        ) != 1:

            raise ValueError(
                f"Could not uniquely match grouped abundance "
                f"result for {model}"
            )

        grouped_row = (
            grouped_row.iloc[
                0
            ]
        )

        for metric in ABUNDANCE_METRICS:

            random_score = float(
                random_row[
                    metric
                ]
            )

            grouped_score = float(
                grouped_row[
                    f"{metric}_mean"
                ]
            )

            (
                absolute_change,
                retention,
                relative_drop,
            ) = compute_change(
                random_score,
                grouped_score,
            )

            rows.append(
                {
                    "Model":
                        model,

                    "Metric":
                        metric,

                    "Random_score":
                        random_score,

                    "Grouped_score":
                        grouped_score,

                    "Absolute_change":
                        absolute_change,

                    "Retention_pct":
                        retention,

                    "Relative_drop_pct":
                        relative_drop,
                }
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Wide manuscript-friendly tables
# ======================================================================


def make_wide_table(
    long_df: pd.DataFrame,
) -> pd.DataFrame:

    pieces = []

    for model in (
        long_df[
            "Model"
        ]
        .drop_duplicates()
    ):

        subset = (
            long_df[
                long_df[
                    "Model"
                ]
                == model
            ]
        )

        row = {
            "Model":
                model,
        }

        for _, metric_row in (
            subset.iterrows()
        ):

            metric = (
                metric_row[
                    "Metric"
                ]
            )

            row[
                f"{metric}_Random"
            ] = (
                metric_row[
                    "Random_score"
                ]
            )

            row[
                f"{metric}_Grouped"
            ] = (
                metric_row[
                    "Grouped_score"
                ]
            )

            row[
                f"{metric}_Drop_pct"
            ] = (
                metric_row[
                    "Relative_drop_pct"
                ]
            )

        pieces.append(
            row
        )

    return pd.DataFrame(
        pieces
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 78
    )

    print(
        "GENERALIZATION GAP ANALYSIS"
    )

    print(
        "=" * 78
    )

    check_files()

    grouped_twohead = (
        load_grouped_twohead()
    )

    adsorption = (
        analyze_adsorption(
            grouped_twohead
        )
    )

    abundance = (
        analyze_abundance(
            grouped_twohead
        )
    )

    adsorption_wide = (
        make_wide_table(
            adsorption
        )
    )

    abundance_wide = (
        make_wide_table(
            abundance
        )
    )

    # ==================================================================
    # Save
    # ==================================================================

    adsorption.to_csv(
        OUTPUT_DIR
        / "adsorption_generalization_gap.csv",
        index=False,
    )

    abundance.to_csv(
        OUTPUT_DIR
        / "abundance_generalization_gap.csv",
        index=False,
    )

    adsorption_wide.to_csv(
        OUTPUT_DIR
        / "adsorption_generalization_table.csv",
        index=False,
    )

    abundance_wide.to_csv(
        OUTPUT_DIR
        / "abundance_generalization_table.csv",
        index=False,
    )

    # ==================================================================
    # Console summary
    # ==================================================================

    print()

    print(
        "=" * 78
    )

    print(
        "ADSORPTION — RELATIVE PERFORMANCE DROP"
    )

    print(
        "=" * 78
    )

    adsorption_display = (
        adsorption.pivot(
            index="Model",
            columns="Metric",
            values="Relative_drop_pct",
        )
        .reset_index()
    )

    print(
        adsorption_display.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}%",
        )
    )

    print()

    print(
        "=" * 78
    )

    print(
        "ABUNDANCE — RELATIVE PERFORMANCE DROP"
    )

    print(
        "=" * 78
    )

    abundance_display = (
        abundance.pivot(
            index="Model",
            columns="Metric",
            values="Relative_drop_pct",
        )
        .reset_index()
    )

    print(
        abundance_display.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}%",
        )
    )

    print()

    print(
        "=" * 78
    )

    print(
        "GENERALIZATION ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()