"""
Generate Table 2
================

Table 2.
Protein Corona Abundance Prediction Performance on the Held-Out
Test Set by Protein Functional Category.

Metrics
-------
Median r
1-TVD
Cosine similarity

Definitions
-----------
Median r:
    Median per-protein Pearson correlation between observed and predicted
    relative abundance across held-out NP samples.

1-TVD:
    For each NP, observed and predicted abundance profiles are restricted
    to proteins in the relevant category, renormalized within that category,
    and compared using one minus total variation distance. The category-level
    value is the mean across held-out NPs.

Cosine:
    For each NP, observed and predicted abundance vectors are restricted to
    proteins in the relevant category. Cosine similarity is then calculated,
    and the category-level value is the mean across held-out NPs.

The OTHER abundance bin is excluded because it does not represent an
individual protein.

Outputs
-------
tables/Table_2_abundance_performance.csv
tables/Table_2_abundance_performance.xlsx
tables/Table_2_per_protein_correlations.csv

Run from project root:

    python scripts/tables/table2_abundance_performance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


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


EPS = 1e-12


# ======================================================================
# Metadata helpers
# ======================================================================


def identify_metadata_columns(
    metadata: pd.DataFrame,
):
    """
    Identify protein ID and protein functional-category columns.
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
            "Could not identify the category column in "
            "protein_metadata.csv.\n"
            f"Available columns: {list(metadata.columns)}"
        )

    return (
        protein_column,
        category_column,
    )


# ======================================================================
# Metric helpers
# ======================================================================


def safe_pearson(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Pearson correlation for one protein.

    Returns NaN if correlation cannot be calculated.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
    )

    observed = observed[
        valid
    ]

    predicted = predicted[
        valid
    ]

    if len(
        observed
    ) < 2:

        return np.nan

    if np.std(
        observed
    ) <= EPS:

        return np.nan

    if np.std(
        predicted
    ) <= EPS:

        return np.nan

    return float(
        pearsonr(
            observed,
            predicted,
        ).statistic
    )


def normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Normalize abundance vectors within each NP.
    """

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    matrix = np.clip(
        matrix,
        0.0,
        None,
    )

    row_sum = matrix.sum(
        axis=1,
        keepdims=True,
    )

    result = np.zeros_like(
        matrix,
        dtype=float,
    )

    valid_rows = (
        row_sum[
            :,
            0
        ]
        > EPS
    )

    result[
        valid_rows
    ] = (
        matrix[
            valid_rows
        ]
        / row_sum[
            valid_rows
        ]
    )

    return result


def mean_one_minus_tvd(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate mean 1-TVD across NP samples.

    Profiles are renormalized within the selected protein subset.
    """

    observed_norm = normalize_rows(
        observed
    )

    predicted_norm = normalize_rows(
        predicted
    )

    observed_sum = observed.sum(
        axis=1
    )

    predicted_sum = predicted.sum(
        axis=1
    )

    valid = (
        observed_sum > EPS
    ) & (
        predicted_sum > EPS
    )

    if not np.any(
        valid
    ):

        return np.nan

    tvd = (
        0.5
        * np.abs(
            observed_norm[
                valid
            ]
            - predicted_norm[
                valid
            ]
        ).sum(
            axis=1
        )
    )

    return float(
        np.mean(
            1.0 - tvd
        )
    )


def mean_cosine_similarity(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate mean cosine similarity across NP samples.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    numerator = np.sum(
        observed
        * predicted,
        axis=1,
    )

    observed_norm = np.linalg.norm(
        observed,
        axis=1,
    )

    predicted_norm = np.linalg.norm(
        predicted,
        axis=1,
    )

    denominator = (
        observed_norm
        * predicted_norm
    )

    valid = (
        denominator
        > EPS
    )

    if not np.any(
        valid
    ):

        return np.nan

    cosine = (
        numerator[
            valid
        ]
        / denominator[
            valid
        ]
    )

    return float(
        np.mean(
            cosine
        )
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "GENERATING TABLE 2"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # File checks
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
    # Load data
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
            "Observed and predicted abundance columns do not match."
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
    # Exclude OTHER
    # ------------------------------------------------------------------

    protein_columns = [
        column
        for column in observed.columns
        if str(
            column
        ).upper()
        != "OTHER"
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
    # Metadata
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
    # Per-protein Pearson r
    # ------------------------------------------------------------------

    correlation_rows = []

    for protein in protein_columns:

        correlation = safe_pearson(
            observed[
                protein
            ].to_numpy(
                dtype=float
            ),

            predicted[
                protein
            ].to_numpy(
                dtype=float
            ),
        )

        category = category_map.get(
            str(
                protein
            ),
            "Other/Mixed",
        )

        correlation_rows.append(
            {
                "Protein":
                    protein,

                "Category":
                    category,

                "Pearson_r":
                    correlation,
            }
        )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    # ==================================================================
    # Overall metrics
    # ==================================================================

    overall_median_r = float(
        np.nanmedian(
            correlation_df[
                "Pearson_r"
            ]
        )
    )

    overall_one_minus_tvd = (
        mean_one_minus_tvd(
            observed.to_numpy(
                dtype=float
            ),
            predicted.to_numpy(
                dtype=float
            ),
        )
    )

    overall_cosine = (
        mean_cosine_similarity(
            observed.to_numpy(
                dtype=float
            ),
            predicted.to_numpy(
                dtype=float
            ),
        )
    )

    overall_row = {
        "Protein Category":
            "Overall",

        "N":
            len(
                protein_columns
            ),

        "Median_r":
            overall_median_r,

        "1-TVD":
            overall_one_minus_tvd,

        "Cosine":
            overall_cosine,
    }

    # ==================================================================
    # Category-level metrics
    # ==================================================================

    category_rows = []

    for category in CATEGORY_ORDER:

        category_proteins = [
            protein
            for protein in protein_columns
            if category_map.get(
                str(
                    protein
                ),
                "Other/Mixed",
            )
            == category
        ]

        if not category_proteins:

            print(
                f"WARNING: no proteins found for category: {category}"
            )

            continue

        # --------------------------------------------------------------
        # Median per-protein Pearson r
        # --------------------------------------------------------------

        category_correlations = (
            correlation_df[
                correlation_df[
                    "Category"
                ]
                == category
            ][
                "Pearson_r"
            ]
            .to_numpy(
                dtype=float
            )
        )

        median_r = float(
            np.nanmedian(
                category_correlations
            )
        )

        # --------------------------------------------------------------
        # Category abundance matrices
        # --------------------------------------------------------------

        observed_category = (
            observed[
                category_proteins
            ]
            .to_numpy(
                dtype=float
            )
        )

        predicted_category = (
            predicted[
                category_proteins
            ]
            .to_numpy(
                dtype=float
            )
        )

        one_minus_tvd = (
            mean_one_minus_tvd(
                observed_category,
                predicted_category,
            )
        )

        cosine = (
            mean_cosine_similarity(
                observed_category,
                predicted_category,
            )
        )

        category_rows.append(
            {
                "Protein Category":
                    category,

                "N":
                    len(
                        category_proteins
                    ),

                "Median_r":
                    median_r,

                "1-TVD":
                    one_minus_tvd,

                "Cosine":
                    cosine,
            }
        )

    # ==================================================================
    # Final table
    # ==================================================================

    table = pd.DataFrame(
        [
            overall_row,
            *category_rows,
        ]
    )

    display_table = table.copy()

    for metric in [
        "Median_r",
        "1-TVD",
        "Cosine",
    ]:

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
        / "Table_2_abundance_performance.csv"
    )

    xlsx_file = (
        TABLE_DIR
        / "Table_2_abundance_performance.xlsx"
    )

    correlation_file = (
        TABLE_DIR
        / "Table_2_per_protein_correlations.csv"
    )

    display_table.to_csv(
        csv_file,
        index=False,
    )

    display_table.to_excel(
        xlsx_file,
        index=False,
    )

    correlation_df.to_csv(
        correlation_file,
        index=False,
    )

    # ==================================================================
    # Console output
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "TABLE 2"
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
        "Per-protein correlations saved:"
    )

    print(
        correlation_file
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()