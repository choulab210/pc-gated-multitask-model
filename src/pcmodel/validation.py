"""
External Validation Utilities
=============================

Reusable functions for evaluating the two-head protein corona model
on independent nanoparticle-protein-corona datasets.

Responsibilities
----------------
- Match model output proteins to external-validation columns.
- Identify detected overlapping proteins.
- Identify missing and all-zero proteins.
- Build normalized abundance and binary adsorption targets.
- Renormalize predictions over the external overlap panel.
- Calculate overall external-validation F1 and cosine similarity.

Feature preprocessing is handled by data.py.
Model inference is handled by training.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score


EPS = 1e-12
DEFAULT_THRESHOLD = 0.5


# ======================================================================
# Data container
# ======================================================================


@dataclass
class ExternalTargets:
    """
    External-validation targets aligned to model output order.
    """

    Y_raw: np.ndarray

    Y_presence: np.ndarray

    Y_abundance: np.ndarray

    overlap_indices: List[int]

    overlap_proteins: List[str]

    missing_proteins: List[str]

    all_zero_proteins: List[str]

    overlap_table: pd.DataFrame


# ======================================================================
# Build external targets
# ======================================================================


def build_external_targets(
    validation_df: pd.DataFrame,
    output_columns: Sequence[str],
    *,
    id_to_name: Optional[
        Mapping[str, str]
    ] = None,
    other_name: str = "OTHER",
) -> ExternalTargets:
    """
    Align an external-validation dataset with model output proteins.

    Parameters
    ----------
    validation_df
        External-validation dataframe containing protein abundance
        columns.

    output_columns
        Model output order, normally:

            174 individual UniProt IDs + OTHER

    id_to_name
        Optional mapping:

            UniProt accession -> protein name used in validation_df

        When no mapping is supplied, the UniProt accession itself is
        used as the expected validation column name.

    other_name
        Name of the residual abundance output.

    Returns
    -------
    ExternalTargets
        Aligned observed targets and protein-overlap information.
    """

    if id_to_name is None:

        id_to_name = {}

    output_columns = list(
        output_columns
    )

    y_raw_columns: Dict[
        str,
        np.ndarray,
    ] = {}

    overlap_indices: List[int] = []

    overlap_proteins: List[str] = []

    missing_proteins: List[str] = []

    all_zero_proteins: List[str] = []

    overlap_records = []

    # --------------------------------------------------------------
    # Match every model output with validation data
    # --------------------------------------------------------------

    for output_index, protein_id in enumerate(
        output_columns
    ):

        if protein_id == other_name:

            validation_column = other_name

        else:

            validation_column = id_to_name.get(
                protein_id,
                protein_id,
            )

        # ----------------------------------------------------------
        # Protein exists in validation dataframe
        # ----------------------------------------------------------

        if validation_column in validation_df.columns:

            values = pd.to_numeric(
                validation_df[
                    validation_column
                ],
                errors="coerce",
            )

            # Missing and negative values are treated as zero,
            # matching the original external-validation notebook.
            values = (
                values
                .fillna(0.0)
                .clip(lower=0.0)
            )

            values_array = (
                values.to_numpy(
                    dtype=np.float64
                )
            )

            y_raw_columns[
                protein_id
            ] = values_array

            if protein_id != other_name:

                total_abundance = float(
                    values_array.sum()
                )

                if total_abundance > 0.0:

                    overlap_indices.append(
                        output_index
                    )

                    overlap_proteins.append(
                        protein_id
                    )

                    status = (
                        "overlap_detected"
                    )

                else:

                    all_zero_proteins.append(
                        protein_id
                    )

                    status = (
                        "all_zero_in_validation"
                    )

                overlap_records.append(
                    {
                        "UniProt_ID": protein_id,
                        "Protein_name": validation_column,
                        "Status": status,
                        "Total_abundance_in_validation":
                            total_abundance,
                    }
                )

        # ----------------------------------------------------------
        # Protein absent from validation dataframe
        # ----------------------------------------------------------

        else:

            y_raw_columns[
                protein_id
            ] = np.zeros(
                len(validation_df),
                dtype=np.float64,
            )

            if protein_id != other_name:

                missing_proteins.append(
                    protein_id
                )

                overlap_records.append(
                    {
                        "UniProt_ID": protein_id,
                        "Protein_name": validation_column,
                        "Status":
                            "missing_from_validation",
                        "Total_abundance_in_validation":
                            0.0,
                    }
                )

    # --------------------------------------------------------------
    # Assemble target matrix in exact model-output order
    # --------------------------------------------------------------

    raw_df = pd.DataFrame(
        y_raw_columns,
        index=validation_df.index,
    )

    raw_df = raw_df.reindex(
        columns=output_columns,
        fill_value=0.0,
    )

    Y_raw = raw_df.to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------------
    # Check external samples
    # --------------------------------------------------------------

    row_sum = Y_raw.sum(
        axis=1,
        keepdims=True,
    )

    bad_rows = np.where(
        row_sum.flatten() <= 0
    )[0]

    if len(bad_rows) > 0:

        raise ValueError(
            "External-validation samples with zero total "
            "protein abundance were detected at rows: "
            f"{bad_rows.tolist()}"
        )

    # --------------------------------------------------------------
    # Abundance target
    # --------------------------------------------------------------

    Y_abundance = (
        Y_raw
        / row_sum
    )

    # --------------------------------------------------------------
    # Presence target
    # --------------------------------------------------------------

    Y_presence = (
        Y_raw > 0.0
    ).astype(int)

    if len(overlap_indices) == 0:

        raise ValueError(
            "No detected overlapping proteins were found "
            "between the model panel and validation dataset."
        )

    overlap_table = pd.DataFrame(
        overlap_records
    )

    return ExternalTargets(
        Y_raw=Y_raw,
        Y_presence=Y_presence,
        Y_abundance=Y_abundance,
        overlap_indices=overlap_indices,
        overlap_proteins=overlap_proteins,
        missing_proteins=missing_proteins,
        all_zero_proteins=all_zero_proteins,
        overlap_table=overlap_table,
    )


# ======================================================================
# Renormalization
# ======================================================================


def renormalize_over_columns(
    values: np.ndarray,
    indices: Sequence[int],
    *,
    eps: float = EPS,
) -> np.ndarray:
    """
    Restrict abundance distributions to selected proteins and
    renormalize each NP to sum to one.

    This is required for external validation because only the proteins
    represented and detected in the independent dataset are evaluated.
    """

    indices = list(
        indices
    )

    subset = np.asarray(
        values,
        dtype=np.float64,
    )[:, indices]

    row_sum = subset.sum(
        axis=1,
        keepdims=True,
    )

    safe_sum = np.where(
        row_sum > eps,
        row_sum,
        1.0,
    )

    return (
        subset
        / safe_sum
    )


# ======================================================================
# External validation metrics
# ======================================================================


def external_presence_f1(
    observed_presence: np.ndarray,
    predicted_probability: np.ndarray,
    indices: Sequence[int],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """
    Calculate macro F1 across externally validated proteins.
    """

    indices = list(
        indices
    )

    observed = (
        observed_presence[
            :,
            indices
        ]
    )

    predicted = (
        predicted_probability[
            :,
            indices
        ]
        >= threshold
    ).astype(int)

    return float(
        f1_score(
            observed,
            predicted,
            average="macro",
            zero_division=0,
        )
    )


def external_cosine_similarity(
    observed_abundance: np.ndarray,
    predicted_abundance: np.ndarray,
    indices: Sequence[int],
    *,
    eps: float = EPS,
) -> float:
    """
    Calculate mean per-NP cosine similarity on overlapping proteins.

    Both observed and predicted abundance distributions are first
    restricted to the overlap panel and renormalized.
    """

    observed = renormalize_over_columns(
        observed_abundance,
        indices,
        eps=eps,
    )

    predicted = renormalize_over_columns(
        predicted_abundance,
        indices,
        eps=eps,
    )

    numerator = (
        observed
        * predicted
    ).sum(
        axis=1
    )

    denominator = np.clip(
        np.linalg.norm(
            observed,
            axis=1,
        )
        * np.linalg.norm(
            predicted,
            axis=1,
        ),
        eps,
        None,
    )

    cosine = (
        numerator
        / denominator
    )

    return float(
        np.mean(
            cosine
        )
    )


def external_validation_metrics(
    targets: ExternalTargets,
    predicted_presence_probability: np.ndarray,
    predicted_abundance: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculate the main external-validation metrics.

    Returns
    -------
    Dictionary containing:
        N
        F1
        Cosine
    """

    indices = (
        targets.overlap_indices
    )

    f1 = external_presence_f1(
        targets.Y_presence,
        predicted_presence_probability,
        indices,
        threshold=threshold,
    )

    cosine = external_cosine_similarity(
        targets.Y_abundance,
        predicted_abundance,
        indices,
    )

    return {
        "N": int(
            len(indices)
        ),
        "F1": f1,
        "Cosine": cosine,
    }


# ======================================================================
# Functional-category aggregation
# ======================================================================


def group_overlap_indices_by_category(
    overlap_proteins: Sequence[str],
    id_to_category: Mapping[
        str,
        str,
    ],
    category_order: Sequence[str],
) -> Dict[str, List[int]]:
    """
    Group local overlap-panel indices by functional category.

    Note
    ----
    These are local indices within the overlap panel, not indices in
    the complete model-output array.
    """

    result = {
        category: []
        for category
        in category_order
    }

    for local_index, protein_id in enumerate(
        overlap_proteins
    ):

        category = id_to_category.get(
            protein_id,
            "Other/Mixed",
        )

        if category not in result:

            category = (
                "Other/Mixed"
            )

        result[
            category
        ].append(
            local_index
        )

    return result


def aggregate_abundance_by_category(
    abundance: np.ndarray,
    protein_ids: Sequence[str],
    id_to_category: Mapping[
        str,
        str,
    ],
    category_order: Sequence[str],
) -> pd.DataFrame:
    """
    Sum protein abundance within functional categories.

    Parameters
    ----------
    abundance
        Shape:
            n_samples x n_proteins

    protein_ids
        Protein order corresponding to abundance columns.

    Returns
    -------
    DataFrame
        Rows are NP samples and columns are functional categories.
    """

    abundance = np.asarray(
        abundance,
        dtype=float,
    )

    if (
        abundance.shape[1]
        != len(protein_ids)
    ):

        raise ValueError(
            "Number of protein IDs does not match "
            "the abundance matrix width."
        )

    output = np.zeros(
        (
            abundance.shape[0],
            len(category_order),
        ),
        dtype=float,
    )

    category_index = {
        category: i
        for i, category
        in enumerate(
            category_order
        )
    }

    for protein_index, protein_id in enumerate(
        protein_ids
    ):

        category = id_to_category.get(
            protein_id,
            "Other/Mixed",
        )

        if category not in category_index:

            category = (
                "Other/Mixed"
            )

        output[
            :,
            category_index[
                category
            ],
        ] += abundance[
            :,
            protein_index
        ]

    return pd.DataFrame(
        output,
        columns=list(
            category_order
        ),
    )