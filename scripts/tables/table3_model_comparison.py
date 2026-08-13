"""
Generate Table 3
================

Comparison of neural-network architectures and conventional
machine-learning models for protein adsorption and abundance prediction.

Expected inputs
---------------
results/benchmarks/adsorption_benchmarks.csv
results/benchmarks/abundance_benchmarks.csv

results/neural_baselines/adsorption_metrics.csv
results/neural_baselines/abundance_metrics.csv

Outputs
-------
tables/Table_3_model_comparison.csv
tables/Table_3_model_comparison.xlsx

Run:
    python scripts/tables/table3_model_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

BENCHMARK_DIR = (
    RESULTS_DIR
    / "benchmarks"
)

NEURAL_DIR = (
    RESULTS_DIR
    / "neural_baselines"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ADSORPTION_BENCHMARK_FILE = (
    BENCHMARK_DIR
    / "adsorption_benchmarks.csv"
)

ABUNDANCE_BENCHMARK_FILE = (
    BENCHMARK_DIR
    / "abundance_benchmarks.csv"
)

ADSORPTION_NEURAL_FILE = (
    NEURAL_DIR
    / "adsorption_metrics.csv"
)

ABUNDANCE_NEURAL_FILE = (
    NEURAL_DIR
    / "abundance_metrics.csv"
)


# ======================================================================
# Desired manuscript model order
# ======================================================================

MODEL_ORDER = [
    "Two-head with gating",
    "Two-head without gating",
    "Single-task NN",
    "Random Forest",
    "XGBoost",
]


# ======================================================================
# Helpers
# ======================================================================


def get_metric(
    dataframe: pd.DataFrame,
    model: str,
    metric: str,
):
    """
    Return one model metric or NaN if unavailable.
    """

    rows = dataframe[
        dataframe[
            "Model"
        ]
        == model
    ]

    if rows.empty:

        return np.nan

    if metric not in rows.columns:

        return np.nan

    return float(
        rows.iloc[
            0
        ][
            metric
        ]
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "GENERATING TABLE 3"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Required benchmark files
    # ------------------------------------------------------------------

    if not ADSORPTION_BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            ADSORPTION_BENCHMARK_FILE
        )

    if not ABUNDANCE_BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            ABUNDANCE_BENCHMARK_FILE
        )

    benchmark_adsorption = pd.read_csv(
        ADSORPTION_BENCHMARK_FILE
    )

    benchmark_abundance = pd.read_csv(
        ABUNDANCE_BENCHMARK_FILE
    )

    # ------------------------------------------------------------------
    # Neural baseline files
    #
    # These contain held-out results for:
    #   Two-head with gating
    #   Two-head without gating
    #   Single-task NN
    # ------------------------------------------------------------------

    if ADSORPTION_NEURAL_FILE.exists():

        neural_adsorption = pd.read_csv(
            ADSORPTION_NEURAL_FILE
        )

    else:

        print(
            "WARNING: adsorption neural-baseline file not found."
        )

        neural_adsorption = pd.DataFrame()

    if ABUNDANCE_NEURAL_FILE.exists():

        neural_abundance = pd.read_csv(
            ABUNDANCE_NEURAL_FILE
        )

    else:

        print(
            "WARNING: abundance neural-baseline file not found."
        )

        neural_abundance = pd.DataFrame()

    # ------------------------------------------------------------------
    # Rename existing final two-head benchmark model
    # ------------------------------------------------------------------

    benchmark_adsorption[
        "Model"
    ] = (
        benchmark_adsorption[
            "Model"
        ]
        .replace(
            {
                "Two-head Model":
                    "Two-head with gating"
            }
        )
    )

    benchmark_abundance[
        "Model"
    ] = (
        benchmark_abundance[
            "Model"
        ]
        .replace(
            {
                "Two-head Model":
                    "Two-head with gating"
            }
        )
    )

    # ------------------------------------------------------------------
    # Combine neural + conventional results
    #
    # Neural results take priority if duplicated.
    # ------------------------------------------------------------------

    adsorption_all = pd.concat(
        [
            neural_adsorption,
            benchmark_adsorption,
        ],
        ignore_index=True,
    )

    abundance_all = pd.concat(
        [
            neural_abundance,
            benchmark_abundance,
        ],
        ignore_index=True,
    )

    adsorption_all = (
        adsorption_all
        .drop_duplicates(
            subset="Model",
            keep="first",
        )
    )

    abundance_all = (
        abundance_all
        .drop_duplicates(
            subset="Model",
            keep="first",
        )
    )

    # ==================================================================
    # Table rows
    # ==================================================================

    rows = []

    # ------------------------------------------------------------------
    # Adsorption section
    # ------------------------------------------------------------------

    adsorption_metrics = [
        ("Acc.", "Acc"),
        ("F1", "F1"),
        ("Prec.", "Precision"),
        ("Recall", "Recall"),
        ("AUROC", "AUROC"),
        ("AUPRC", "AUPRC"),
        ("MCC", "MCC"),
    ]

    for display_metric, source_metric in adsorption_metrics:

        row = {
            "Task / Metric":
                display_metric,
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = get_metric(
                adsorption_all,
                model,
                source_metric,
            )

        rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Spacer / label handled separately in output
    # ------------------------------------------------------------------

    abundance_metrics = [
        ("Median r", "Median_r"),
        ("1-TVD", "1-TVD"),
        ("Cosine", "Cosine"),
    ]

    for display_metric, source_metric in abundance_metrics:

        row = {
            "Task / Metric":
                display_metric,
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = get_metric(
                abundance_all,
                model,
                source_metric,
            )

        rows.append(
            row
        )

    table = pd.DataFrame(
        rows
    )

    # ==================================================================
    # Round
    # ==================================================================

    for model in MODEL_ORDER:

        table[
            model
        ] = (
            pd.to_numeric(
                table[
                    model
                ],
                errors="coerce",
            )
            .round(
                3
            )
        )

    # ==================================================================
    # Save machine-readable table
    # ==================================================================

    csv_file = (
        TABLE_DIR
        / "Table_3_model_comparison.csv"
    )

    xlsx_file = (
        TABLE_DIR
        / "Table_3_model_comparison.xlsx"
    )

    table.to_csv(
        csv_file,
        index=False,
    )

    # ==================================================================
    # Excel manuscript-style layout
    # ==================================================================

    with pd.ExcelWriter(
        xlsx_file,
        engine="openpyxl",
    ) as writer:

        # --------------------------------------------------------------
        # Adsorption block
        # --------------------------------------------------------------

        adsorption_block = table.iloc[
            0:7
        ].copy()

        adsorption_block.to_excel(
            writer,
            sheet_name="Table 3",
            index=False,
            startrow=1,
        )

        worksheet = writer.book[
            "Table 3"
        ]

        worksheet[
            "A1"
        ] = (
            "Adsorption Classification"
        )

        # --------------------------------------------------------------
        # Abundance block
        # --------------------------------------------------------------

        abundance_block = table.iloc[
            7:
        ].copy()

        abundance_start = 11

        worksheet.cell(
            row=abundance_start,
            column=1,
            value="Abundance Prediction",
        )

        abundance_block.to_excel(
            writer,
            sheet_name="Table 3",
            index=False,
            startrow=abundance_start,
        )

        # --------------------------------------------------------------
        # Column widths
        # --------------------------------------------------------------

        worksheet.column_dimensions[
            "A"
        ].width = 20

        worksheet.column_dimensions[
            "B"
        ].width = 23

        worksheet.column_dimensions[
            "C"
        ].width = 26

        worksheet.column_dimensions[
            "D"
        ].width = 18

        worksheet.column_dimensions[
            "E"
        ].width = 16

        worksheet.column_dimensions[
            "F"
        ].width = 16

    # ==================================================================
    # Console
    # ==================================================================

    print()

    print(
        table.to_string(
            index=False,
            na_rep="—",
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        csv_file
    )

    print(
        xlsx_file
    )

    # ------------------------------------------------------------------
    # Missing-result warning
    # ------------------------------------------------------------------

    if (
        table[
            MODEL_ORDER
        ]
        .isna()
        .any()
        .any()
    ):

        print()

        print(
            "NOTE:"
        )

        print(
            "Some cells are empty because the corresponding "
            "held-out neural baseline has not yet been generated."
        )


if __name__ == "__main__":
    main()