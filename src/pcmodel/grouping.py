"""
Grouped Validation Utilities
============================

Utilities for constructing leakage-resistant groups from nanoparticle
feature profiles.

The current dataset does not contain a study identifier, so grouped
validation is based on identical encoded/raw feature signatures.

Samples with identical model input features receive the same group ID.
These groups can then be passed to GroupKFold so that identical profiles
never appear in both training and validation folds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ======================================================================
# Group construction
# ======================================================================


def build_feature_signature_groups(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Assign group IDs based on identical feature profiles.

    Parameters
    ----------
    feature_df
        DataFrame containing NP_ID and model input variables.

    feature_columns
        Columns used to define the feature signature.

    Returns
    -------
    DataFrame
        Columns:
            NP_ID
            group_id
            group_size
    """

    feature_columns = list(
        feature_columns
    )

    required = (
        ["NP_ID"]
        + feature_columns
    )

    missing = [
        column
        for column in required
        if column not in feature_df.columns
    ]

    if missing:

        raise ValueError(
            "Feature dataframe is missing required columns: "
            f"{missing}"
        )

    working = (
        feature_df[
            required
        ]
        .copy()
    )

    # --------------------------------------------------------------
    # Make missing values deterministic before hashing.
    # --------------------------------------------------------------

    for column in feature_columns:

        if pd.api.types.is_numeric_dtype(
            working[
                column
            ]
        ):

            working[
                column
            ] = (
                pd.to_numeric(
                    working[
                        column
                    ],
                    errors="coerce",
                )
                .fillna(
                    -999999.0
                )
            )

        else:

            working[
                column
            ] = (
                working[
                    column
                ]
                .fillna(
                    "<MISSING>"
                )
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------------
    # Hash the feature signature.
    # --------------------------------------------------------------

    signature_hash = (
        pd.util.hash_pandas_object(
            working[
                feature_columns
            ],
            index=False,
        )
        .astype(str)
    )

    # --------------------------------------------------------------
    # Convert hashes to compact group numbers.
    # --------------------------------------------------------------

    group_codes, _ = pd.factorize(
        signature_hash,
        sort=True,
    )

    working[
        "group_id"
    ] = group_codes.astype(int)

    group_sizes = (
        working[
            "group_id"
        ]
        .value_counts()
        .to_dict()
    )

    working[
        "group_size"
    ] = (
        working[
            "group_id"
        ]
        .map(
            group_sizes
        )
        .astype(int)
    )

    return working[
        [
            "NP_ID",
            "group_id",
            "group_size",
        ]
    ]


# ======================================================================
# Group audit
# ======================================================================


def summarize_groups(
    group_table: pd.DataFrame,
) -> dict:
    """
    Summarize the grouped feature profiles.
    """

    if (
        "group_id"
        not in group_table.columns
        or "group_size"
        not in group_table.columns
    ):

        raise ValueError(
            "group_table must contain group_id and group_size."
        )

    group_sizes = (
        group_table[
            [
                "group_id",
                "group_size",
            ]
        ]
        .drop_duplicates(
            "group_id"
        )
    )

    repeated_groups = (
        group_sizes[
            group_sizes[
                "group_size"
            ]
            > 1
        ]
    )

    repeated_samples = int(
        repeated_groups[
            "group_size"
        ].sum()
    )

    return {
        "n_samples":
            int(
                len(
                    group_table
                )
            ),

        "n_unique_groups":
            int(
                group_table[
                    "group_id"
                ]
                .nunique()
            ),

        "n_repeated_groups":
            int(
                len(
                    repeated_groups
                )
            ),

        "n_samples_in_repeated_groups":
            repeated_samples,

        "fraction_samples_in_repeated_groups":
            float(
                repeated_samples
                / len(
                    group_table
                )
            )
            if len(
                group_table
            ) > 0
            else np.nan,

        "largest_group":
            int(
                group_table[
                    "group_size"
                ].max()
            ),
    }


# ======================================================================
# Leakage audit
# ======================================================================


def count_shared_groups(
    train_group_ids,
    test_group_ids,
) -> int:
    """
    Count feature-signature groups shared between two data partitions.
    """

    train_set = set(
        np.asarray(
            train_group_ids
        ).tolist()
    )

    test_set = set(
        np.asarray(
            test_group_ids
        ).tolist()
    )

    return len(
        train_set
        & test_set
    )