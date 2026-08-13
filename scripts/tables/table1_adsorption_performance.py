"""
Generate Table 1
================

Table 1.
Adsorption Classification Performance on the Held-Out Test Set
by Protein Functional Category.

The script uses the saved held-out observations/predictions from the
final two-head model and protein functional-category metadata.

Metrics
-------
Accuracy
F1
Precision
Recall
AUROC
AUPRC
MCC

Category-level metrics follow the same strategy as the model evaluation:
metrics are calculated per protein and then summarized within each
functional category.

Outputs
-------
tables/Table_1_adsorption_performance.csv
tables/Table_1_adsorption_performance.xlsx

Run from project root:

    python scripts/tables/table1_adsorption_performance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR = PROJECT_ROOT / "data"

TABLE_DIR = PROJECT_ROOT / "tables"

TABLE_DIR.mkdir(
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

METADATA_FILE = (
    DATA_DIR
    / "protein_metadata.csv"
)


# ======================================================================
# Configuration
# ======================================================================

THRESHOLD = 0.5


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
# Helpers
# ======================================================================


def safe_auroc(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> float:
    """
    Calculate AUROC if both classes are represented.
    """

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        return np.nan

    return float(
        roc_auc_score(
            y_true,
            y_probability,
        )
    )


def safe_auprc(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> float:
    """
    Calculate AUPRC if both classes are represented.
    """

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        return np.nan

    return float(
        average_precision_score(
            y_true,
            y_probability,
        )
    )


def protein_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> dict:
    """
    Calculate adsorption metrics for one protein.
    """

    y_predicted = (
        y_probability
        >= THRESHOLD
    ).astype(int)

    return {
        "Accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_predicted,
                )
            ),

        "F1":
            float(
                f1_score(
                    y_true,
                    y_predicted,
                    zero_division=0,
                )
            ),

        "Precision":
            float(
                precision_score(
                    y_true,
                    y_predicted,
                    zero_division=0,
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    y_predicted,
                    zero_division=0,
                )
            ),

        "AUROC":
            safe_auroc(
                y_true,
                y_probability,
            ),

        "AUPRC":
            safe_auprc(
                y_true,
                y_probability,
            ),

        "MCC":
            float(
                matthews_corrcoef(
                    y_true,
                    y_predicted,
                )
            ),
    }


# ======================================================================
# Metadata
# ======================================================================


def identify_metadata_columns(
    metadata: pd.DataFrame,
):
    """
    Identify protein-ID and functional-category columns.

    This supports several likely column naming conventions.
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
            "Could not identify the protein-ID column in "
            "protein_metadata.csv.\n"
            f"Available columns: {list(metadata.columns)}"
        )

    if category_column is None:

        raise ValueError(
            "Could not identify the functional-category column in "
            "protein_metadata.csv.\n"
            f"Available columns: {list(metadata.columns)}"
        )

    return (
        protein_column,
        category_column,
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "GENERATING TABLE 1"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Check files
    # ------------------------------------------------------------------

    for path in [
        OBSERVED_FILE,
        PREDICTED_FILE,
        METADATA_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------

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

    if len(
        observed
    ) != len(
        predicted
    ):

        raise ValueError(
            "Observed and predicted sample counts do not match."
        )

    # ------------------------------------------------------------------
    # Remove OTHER if present
    # ------------------------------------------------------------------

    protein_columns = [
        column
        for column in observed.columns
        if str(
            column
        ).upper() != "OTHER"
    ]

    observed = observed[
        protein_columns
    ]

    predicted = predicted[
        protein_columns
    ]

    print()

    print(
        "Held-out samples:",
        len(
            observed
        ),
    )

    print(
        "Individual proteins:",
        len(
            protein_columns
        ),
    )

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Calculate per-protein metrics
    # ------------------------------------------------------------------

    protein_rows = []

    for protein in (
        protein_columns
    ):

        y_true = (
            observed[
                protein
            ]
            .to_numpy(
                dtype=int
            )
        )

        y_probability = (
            predicted[
                protein
            ]
            .to_numpy(
                dtype=float
            )
        )

        metrics = protein_metrics(
            y_true,
            y_probability,
        )

        category = category_map.get(
            str(
                protein
            ),
            "Other/Mixed",
        )

        protein_rows.append(
            {
                "Protein":
                    protein,

                "Category":
                    category,

                **metrics,
            }
        )

    protein_df = pd.DataFrame(
        protein_rows
    )

    # ------------------------------------------------------------------
    # Overall metrics
    #
    # Important:
    # Use the same aggregation convention as category metrics:
    # arithmetic mean of per-protein classification metrics.
    #
    # AUROC/AUPRC also use the mean across valid per-protein metrics.
    # ------------------------------------------------------------------

    metric_columns = [
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "AUROC",
        "AUPRC",
        "MCC",
    ]

    overall_row = {
        "Protein Category":
            "Overall",

        "N":
            len(
                protein_df
            ),
    }

    for metric in metric_columns:

        overall_row[
            metric
        ] = float(
            np.nanmean(
                protein_df[
                    metric
                ]
            )
        )

    # ------------------------------------------------------------------
    # Category metrics
    # ------------------------------------------------------------------

    category_rows = []

    for category in CATEGORY_ORDER:

        subset = protein_df[
            protein_df[
                "Category"
            ]
            == category
        ]

        if subset.empty:

            print(
                f"WARNING: no proteins found for category: {category}"
            )

            continue

        row = {
            "Protein Category":
                category,

            "N":
                len(
                    subset
                ),
        }

        for metric in metric_columns:

            row[
                metric
            ] = float(
                np.nanmean(
                    subset[
                        metric
                    ]
                )
            )

        category_rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Final Table 1
    # ------------------------------------------------------------------

    table = pd.DataFrame(
        [
            overall_row,
            *category_rows,
        ]
    )

    # ------------------------------------------------------------------
    # Round manuscript values
    # ------------------------------------------------------------------

    display_table = table.copy()

    for metric in metric_columns:

        display_table[
            metric
        ] = (
            display_table[
                metric
            ]
            .round(
                3
            )
        )

    # ==================================================================
    # Save
    # ==================================================================

    csv_file = (
        TABLE_DIR
        / "Table_1_adsorption_performance.csv"
    )

    xlsx_file = (
        TABLE_DIR
        / "Table_1_adsorption_performance.xlsx"
    )

    per_protein_file = (
        TABLE_DIR
        / "Table_1_per_protein_metrics.csv"
    )

    display_table.to_csv(
        csv_file,
        index=False,
    )

    display_table.to_excel(
        xlsx_file,
        index=False,
    )

    protein_df.to_csv(
        per_protein_file,
        index=False,
    )

    # ==================================================================
    # Console
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "TABLE 1"
    )

    print(
        "=" * 72
    )

    print(
        display_table.to_string(
            index=False
        )
    )

    print()

    print(
        "Files saved:"
    )

    print(
        csv_file
    )

    print(
        xlsx_file
    )

    print()

    print(
        "Per-protein metrics saved:"
    )

    print(
        per_protein_file
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()