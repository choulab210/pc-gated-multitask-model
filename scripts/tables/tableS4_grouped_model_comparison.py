"""
Generate Supplementary Table S4
===============================

Table S4.
Comparison of Model Performance under Conventional Held-Out
and Feature-Grouped Validation.

Models
------
- Two-head gated
- Two-head ungated
- Single-task NN
- Random Forest
- XGBoost

The table contains three blocks for each prediction task:

1. Held-out test performance
2. Feature-grouped cross-validation performance
3. Relative performance drop (%)

Current grouped neural-baseline results may not yet exist.
If grouped results for the ungated two-head or single-task NN are
missing, the corresponding cells are left blank / shown as "—".

Once grouped neural-baseline files are generated, rerunning this script
will populate those cells automatically.

Expected inputs
---------------
Held-out:
results/benchmarks/adsorption_benchmarks.csv
results/benchmarks/abundance_benchmarks.csv
results/neural_baselines/adsorption_metrics.csv
results/neural_baselines/abundance_metrics.csv

Grouped:
results/grouped_validation/summary.csv
results/grouped_benchmarks/adsorption_summary.csv
results/grouped_benchmarks/abundance_summary.csv

Optional future grouped neural results:
results/grouped_neural_baselines/adsorption_summary.csv
results/grouped_neural_baselines/abundance_summary.csv

Outputs
-------
tables/Table_S4_grouped_model_comparison.csv
tables/Table_S4_grouped_model_comparison.xlsx

Run
---
python scripts/tables/tableS4_grouped_model_comparison.py
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
TABLE_DIR = PROJECT_ROOT / "tables"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Held-out benchmark results
HELDOUT_ADSORPTION_FILE = (
    RESULTS_DIR
    / "benchmarks"
    / "adsorption_benchmarks.csv"
)

HELDOUT_ABUNDANCE_FILE = (
    RESULTS_DIR
    / "benchmarks"
    / "abundance_benchmarks.csv"
)

HELDOUT_NEURAL_ADSORPTION_FILE = (
    RESULTS_DIR
    / "neural_baselines"
    / "adsorption_metrics.csv"
)

HELDOUT_NEURAL_ABUNDANCE_FILE = (
    RESULTS_DIR
    / "neural_baselines"
    / "abundance_metrics.csv"
)


# Grouped gated two-head result
GROUPED_GATED_FILE = (
    RESULTS_DIR
    / "grouped_validation"
    / "summary.csv"
)


# Grouped conventional benchmarks
GROUPED_ADSORPTION_FILE = (
    RESULTS_DIR
    / "grouped_benchmarks"
    / "adsorption_summary.csv"
)

GROUPED_ABUNDANCE_FILE = (
    RESULTS_DIR
    / "grouped_benchmarks"
    / "abundance_summary.csv"
)


# Optional future grouped neural baselines
GROUPED_NEURAL_ADSORPTION_FILE = (
    RESULTS_DIR
    / "grouped_neural_baselines"
    / "adsorption_summary.csv"
)

GROUPED_NEURAL_ABUNDANCE_FILE = (
    RESULTS_DIR
    / "grouped_neural_baselines"
    / "abundance_summary.csv"
)


# ======================================================================
# Model order
# ======================================================================

MODEL_ORDER = [
    "Two-head gated",
    "Two-head ungated",
    "Single-task NN",
    "Random Forest",
    "XGBoost",
]


# ======================================================================
# Metric definitions
# ======================================================================

ADSORPTION_METRICS = [
    ("Acc.", "Acc"),
    ("F1", "F1"),
    ("Prec.", "Precision"),
    ("Recall", "Recall"),
    ("AUROC", "AUROC"),
    ("AUPRC", "AUPRC"),
    ("MCC", "MCC"),
]

ABUNDANCE_METRICS = [
    ("Median r", "Median_r"),
    ("1-TVD", "1-TVD"),
    ("Cosine", "Cosine"),
]


# ======================================================================
# Utilities
# ======================================================================


def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Read CSV if present; otherwise return an empty DataFrame.
    """

    if not path.exists():

        print(
            f"WARNING: file not found:\n"
            f"  {path}"
        )

        return pd.DataFrame()

    return pd.read_csv(
        path
    )


def get_wide_metric(
    dataframe: pd.DataFrame,
    model: str,
    metric: str,
):
    """
    Retrieve a metric from a standard wide model-results table.

    Expected form:
        Model | F1 | AUROC | ...
    """

    if dataframe.empty:

        return np.nan

    if (
        "Model"
        not in dataframe.columns
    ):

        return np.nan

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

    value = rows.iloc[
        0
    ][
        metric
    ]

    return float(
        value
    )


def get_grouped_metric(
    dataframe: pd.DataFrame,
    model: str,
    metric: str,
):
    """
    Retrieve mean grouped-validation performance.

    Supports columns such as:
        F1_mean
        AUROC_mean
        Median_r_mean
    """

    if dataframe.empty:

        return np.nan

    if (
        "Model"
        not in dataframe.columns
    ):

        return np.nan

    rows = dataframe[
        dataframe[
            "Model"
        ]
        == model
    ]

    if rows.empty:

        return np.nan

    possible_columns = [
        f"{metric}_mean",
        metric,
    ]

    for column in possible_columns:

        if column in rows.columns:

            return float(
                rows.iloc[
                    0
                ][
                    column
                ]
            )

    return np.nan


def get_gated_grouped_metric(
    grouped_gated: pd.DataFrame,
    metric: str,
):
    """
    Retrieve gated two-head grouped-validation result.

    Current file format:
        Metric | Mean | SD
    """

    if grouped_gated.empty:

        return np.nan

    if not {
        "Metric",
        "Mean",
    }.issubset(
        grouped_gated.columns
    ):

        return np.nan

    rows = grouped_gated[
        grouped_gated[
            "Metric"
        ]
        == metric
    ]

    if rows.empty:

        return np.nan

    return float(
        rows.iloc[
            0
        ][
            "Mean"
        ]
    )


def relative_drop(
    heldout,
    grouped,
):
    """
    Relative performance decrease:

    (held-out - grouped) / held-out * 100
    """

    if (
        pd.isna(
            heldout
        )
        or pd.isna(
            grouped
        )
        or heldout == 0
    ):

        return np.nan

    return float(
        (
            heldout
            - grouped
        )
        / heldout
        * 100.0
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 78
    )

    print(
        "GENERATING SUPPLEMENTARY TABLE S4"
    )

    print(
        "=" * 78
    )

    # ==================================================================
    # Load held-out results
    # ==================================================================

    heldout_adsorption = safe_read_csv(
        HELDOUT_ADSORPTION_FILE
    )

    heldout_abundance = safe_read_csv(
        HELDOUT_ABUNDANCE_FILE
    )

    neural_adsorption = safe_read_csv(
        HELDOUT_NEURAL_ADSORPTION_FILE
    )

    neural_abundance = safe_read_csv(
        HELDOUT_NEURAL_ABUNDANCE_FILE
    )

    # ------------------------------------------------------------------
    # Rename final gated model
    # ------------------------------------------------------------------

    if not heldout_adsorption.empty:

        heldout_adsorption[
            "Model"
        ] = (
            heldout_adsorption[
                "Model"
            ]
            .replace(
                {
                    "Two-head Model":
                        "Two-head gated"
                }
            )
        )

    if not heldout_abundance.empty:

        heldout_abundance[
            "Model"
        ] = (
            heldout_abundance[
                "Model"
            ]
            .replace(
                {
                    "Two-head Model":
                        "Two-head gated"
                }
            )
        )

    # ------------------------------------------------------------------
    # Rename neural models to manuscript terminology
    # ------------------------------------------------------------------

    for dataframe in [
        neural_adsorption,
        neural_abundance,
    ]:

        if (
            not dataframe.empty
            and "Model"
            in dataframe.columns
        ):

            dataframe[
                "Model"
            ] = (
                dataframe[
                    "Model"
                ]
                .replace(
                    {
                        "Two-head without gating":
                            "Two-head ungated"
                    }
                )
            )

    heldout_adsorption_all = pd.concat(
        [
            neural_adsorption,
            heldout_adsorption,
        ],
        ignore_index=True,
    )

    heldout_abundance_all = pd.concat(
        [
            neural_abundance,
            heldout_abundance,
        ],
        ignore_index=True,
    )

    heldout_adsorption_all = (
        heldout_adsorption_all
        .drop_duplicates(
            subset="Model",
            keep="first",
        )
    )

    heldout_abundance_all = (
        heldout_abundance_all
        .drop_duplicates(
            subset="Model",
            keep="first",
        )
    )

    # ==================================================================
    # Load grouped results
    # ==================================================================

    grouped_gated = safe_read_csv(
        GROUPED_GATED_FILE
    )

    grouped_adsorption = safe_read_csv(
        GROUPED_ADSORPTION_FILE
    )

    grouped_abundance = safe_read_csv(
        GROUPED_ABUNDANCE_FILE
    )

    grouped_neural_adsorption = safe_read_csv(
        GROUPED_NEURAL_ADSORPTION_FILE
    )

    grouped_neural_abundance = safe_read_csv(
        GROUPED_NEURAL_ABUNDANCE_FILE
    )

    # ------------------------------------------------------------------
    # Standardize grouped neural model naming
    # ------------------------------------------------------------------

    for dataframe in [
        grouped_neural_adsorption,
        grouped_neural_abundance,
    ]:

        if (
            not dataframe.empty
            and "Model"
            in dataframe.columns
        ):

            dataframe[
                "Model"
            ] = (
                dataframe[
                    "Model"
                ]
                .replace(
                    {
                        "Two-head without gating":
                            "Two-head ungated"
                    }
                )
            )

    # ==================================================================
    # Build lookup dictionaries
    # ==================================================================

    heldout_lookup = {
        "adsorption": {},
        "abundance": {},
    }

    grouped_lookup = {
        "adsorption": {},
        "abundance": {},
    }

    # ------------------------------------------------------------------
    # Held-out adsorption
    # ------------------------------------------------------------------

    for model in MODEL_ORDER:

        heldout_lookup[
            "adsorption"
        ][
            model
        ] = {}

        for _, metric in ADSORPTION_METRICS:

            heldout_lookup[
                "adsorption"
            ][
                model
            ][
                metric
            ] = get_wide_metric(
                heldout_adsorption_all,
                model,
                metric,
            )

    # ------------------------------------------------------------------
    # Held-out abundance
    # ------------------------------------------------------------------

    for model in MODEL_ORDER:

        heldout_lookup[
            "abundance"
        ][
            model
        ] = {}

        for _, metric in ABUNDANCE_METRICS:

            heldout_lookup[
                "abundance"
            ][
                model
            ][
                metric
            ] = get_wide_metric(
                heldout_abundance_all,
                model,
                metric,
            )

    # ==================================================================
    # Grouped gated two-head
    # ==================================================================

    grouped_lookup[
        "adsorption"
    ][
        "Two-head gated"
    ] = {}

    for _, metric in ADSORPTION_METRICS:

        grouped_lookup[
            "adsorption"
        ][
            "Two-head gated"
        ][
            metric
        ] = get_gated_grouped_metric(
            grouped_gated,
            metric,
        )

    grouped_lookup[
        "abundance"
    ][
        "Two-head gated"
    ] = {}

    for _, metric in ABUNDANCE_METRICS:

        grouped_lookup[
            "abundance"
        ][
            "Two-head gated"
        ][
            metric
        ] = get_gated_grouped_metric(
            grouped_gated,
            metric,
        )

    # ==================================================================
    # Grouped conventional models
    # ==================================================================

    for model in [
        "Random Forest",
        "XGBoost",
    ]:

        grouped_lookup[
            "adsorption"
        ][
            model
        ] = {}

        for _, metric in ADSORPTION_METRICS:

            grouped_lookup[
                "adsorption"
            ][
                model
            ][
                metric
            ] = get_grouped_metric(
                grouped_adsorption,
                model,
                metric,
            )

        grouped_lookup[
            "abundance"
        ][
            model
        ] = {}

        for _, metric in ABUNDANCE_METRICS:

            grouped_lookup[
                "abundance"
            ][
                model
            ][
                metric
            ] = get_grouped_metric(
                grouped_abundance,
                model,
                metric,
            )

    # ==================================================================
    # Optional grouped neural models
    # ==================================================================

    for model in [
        "Two-head ungated",
        "Single-task NN",
    ]:

        grouped_lookup[
            "adsorption"
        ][
            model
        ] = {}

        for _, metric in ADSORPTION_METRICS:

            grouped_lookup[
                "adsorption"
            ][
                model
            ][
                metric
            ] = get_grouped_metric(
                grouped_neural_adsorption,
                model,
                metric,
            )

        grouped_lookup[
            "abundance"
        ][
            model
        ] = {}

        for _, metric in ABUNDANCE_METRICS:

            grouped_lookup[
                "abundance"
            ][
                model
            ][
                metric
            ] = get_grouped_metric(
                grouped_neural_abundance,
                model,
                metric,
            )

    # ==================================================================
    # Build Table S4
    # ==================================================================

    rows = []

    # ------------------------------------------------------------------
    # Adsorption — held-out
    # ------------------------------------------------------------------

    rows.append(
        {
            "Validation / Metric":
                "Adsorption — Held-out"
        }
    )

    for display_metric, metric in (
        ADSORPTION_METRICS
    ):

        row = {
            "Validation / Metric":
                display_metric
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = heldout_lookup[
                "adsorption"
            ][
                model
            ][
                metric
            ]

        rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Adsorption — grouped
    # ------------------------------------------------------------------

    rows.append(
        {
            "Validation / Metric":
                "Adsorption — Feature-grouped"
        }
    )

    for display_metric, metric in (
        ADSORPTION_METRICS
    ):

        row = {
            "Validation / Metric":
                display_metric
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = grouped_lookup[
                "adsorption"
            ][
                model
            ][
                metric
            ]

        rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Adsorption — relative drop
    # ------------------------------------------------------------------

    rows.append(
        {
            "Validation / Metric":
                "Adsorption — Relative drop (%)"
        }
    )

    for display_metric, metric in (
        ADSORPTION_METRICS
    ):

        row = {
            "Validation / Metric":
                display_metric
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = relative_drop(
                heldout_lookup[
                    "adsorption"
                ][
                    model
                ][
                    metric
                ],

                grouped_lookup[
                    "adsorption"
                ][
                    model
                ][
                    metric
                ],
            )

        rows.append(
            row
        )

    # ==================================================================
    # Abundance — held-out
    # ==================================================================

    rows.append(
        {
            "Validation / Metric":
                "Abundance — Held-out"
        }
    )

    for display_metric, metric in (
        ABUNDANCE_METRICS
    ):

        row = {
            "Validation / Metric":
                display_metric
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = heldout_lookup[
                "abundance"
            ][
                model
            ][
                metric
            ]

        rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Abundance — grouped
    # ------------------------------------------------------------------

    rows.append(
        {
            "Validation / Metric":
                "Abundance — Feature-grouped"
        }
    )

    for display_metric, metric in (
        ABUNDANCE_METRICS
    ):

        row = {
            "Validation / Metric":
                display_metric
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = grouped_lookup[
                "abundance"
            ][
                model
            ][
                metric
            ]

        rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Abundance — relative drop
    # ------------------------------------------------------------------

    rows.append(
        {
            "Validation / Metric":
                "Abundance — Relative drop (%)"
        }
    )

    for display_metric, metric in (
        ABUNDANCE_METRICS
    ):

        row = {
            "Validation / Metric":
                display_metric
        }

        for model in MODEL_ORDER:

            row[
                model
            ] = relative_drop(
                heldout_lookup[
                    "abundance"
                ][
                    model
                ][
                    metric
                ],

                grouped_lookup[
                    "abundance"
                ][
                    model
                ][
                    metric
                ],
            )

        rows.append(
            row
        )

    table = pd.DataFrame(
        rows
    )

    # ==================================================================
    # Round numerical values
    # ==================================================================

    for model in MODEL_ORDER:

        table[
            model
        ] = pd.to_numeric(
            table[
                model
            ],
            errors="coerce",
        )

        table[
            model
        ] = table[
            model
        ].round(
            3
        )

    # ==================================================================
    # Save CSV
    # ==================================================================

    csv_file = (
        TABLE_DIR
        / "Table_S4_grouped_model_comparison.csv"
    )

    table.to_csv(
        csv_file,
        index=False,
    )

    # ==================================================================
    # Save Excel
    # ==================================================================

    xlsx_file = (
        TABLE_DIR
        / "Table_S4_grouped_model_comparison.xlsx"
    )

    with pd.ExcelWriter(
        xlsx_file,
        engine="openpyxl",
    ) as writer:

        table.to_excel(
            writer,
            sheet_name="Table S4",
            index=False,
        )

        worksheet = writer.book[
            "Table S4"
        ]

        worksheet.column_dimensions[
            "A"
        ].width = 32

        worksheet.column_dimensions[
            "B"
        ].width = 20

        worksheet.column_dimensions[
            "C"
        ].width = 22

        worksheet.column_dimensions[
            "D"
        ].width = 20

        worksheet.column_dimensions[
            "E"
        ].width = 18

        worksheet.column_dimensions[
            "F"
        ].width = 16

    # ==================================================================
    # Console display
    # ==================================================================

    display_table = table.copy()

    for model in MODEL_ORDER:

        display_table[
            model
        ] = display_table[
            model
        ].apply(
            lambda value:
                "—"
                if pd.isna(
                    value
                )
                else f"{value:.3f}"
        )

    print()

    print(
        display_table.to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 78
    )

    print(
        "TABLE S4 COMPLETE"
    )

    print(
        "=" * 78
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
    # Explicit missing grouped-neural warning
    # ------------------------------------------------------------------

    if (
        not GROUPED_NEURAL_ADSORPTION_FILE.exists()
        or not GROUPED_NEURAL_ABUNDANCE_FILE.exists()
    ):

        print()

        print(
            "NOTE:"
        )

        print(
            "Grouped results for the ungated two-head and/or "
            "single-task NN are not yet available."
        )

        print(
            "These cells are shown as '—'."
        )

        print(
            "After grouped neural-baseline analysis is completed, "
            "rerun this script to populate them automatically."
        )


if __name__ == "__main__":
    main()