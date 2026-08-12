"""
Grouped Cross-Validation Benchmarks
===================================

Compare conventional machine-learning models against the two-head model
using feature-signature grouped cross-validation.

Models
------
Adsorption:
    - Random Forest
    - Logistic Regression
    - XGBoost

Abundance:
    - Random Forest
    - Ridge Regression
    - XGBoost

Grouping
--------
Samples with identical raw input-feature profiles are assigned to the
same group. GroupKFold ensures that an identical feature signature never
appears in both training and validation folds.

This script uses the same 317 model-development samples used in
run_grouped_validation.py.

Run:
    python scripts/run_grouped_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from pcmodel.benchmarks import (
    ADSORPTION_BENCHMARKS,
    ABUNDANCE_BENCHMARKS,
    BenchmarkConfig,
    fit_adsorption_models,
    fit_abundance_models,
    predict_adsorption_probabilities,
    predict_abundance_distribution,
)

from pcmodel.data import (
    FEATURE_COLS,
    prepare_model_data,
)

from pcmodel.grouping import (
    build_feature_signature_groups,
)

from pcmodel.metrics import (
    adsorption_metrics,
    abundance_metrics,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "grouped_benchmarks"
)

RESULTS_DIR.mkdir(
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


# ======================================================================
# Configuration
# ======================================================================

N_SPLITS = 5

RANDOM_SEED = 42


# ======================================================================
# Helpers
# ======================================================================


def get_train_ids(
    prepared,
    n_expected: int,
):
    """
    Retrieve model-development NP IDs from PreparedData.
    """

    possible_attributes = [
        "train_ids",
        "train_np_ids",
        "np_ids_train",
    ]

    for attribute in possible_attributes:

        if hasattr(
            prepared,
            attribute,
        ):

            candidate = getattr(
                prepared,
                attribute,
            )

            if (
                candidate is not None
                and len(candidate) == n_expected
            ):

                return [
                    str(value)
                    for value in candidate
                ]

    raise RuntimeError(
        "PreparedData does not expose model-development NP IDs."
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "GROUPED BENCHMARK VALIDATION"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Prepare model-development data
    # ------------------------------------------------------------------

    prepared = prepare_model_data(
        FEATURE_FILE,
        ABUNDANCE_FILE,
    )

    X = np.asarray(
        prepared.X_train,
        dtype=np.float32,
    )

    Y_presence = np.asarray(
        prepared.Y_presence_train,
        dtype=float,
    )

    Y_abundance = np.asarray(
        prepared.Y_abundance_train,
        dtype=float,
    )

    n_samples = len(
        X
    )

    n_proteins = len(
        prepared.panel
    )

    protein_indices = list(
        range(
            n_proteins
        )
    )

    train_ids = get_train_ids(
        prepared,
        n_samples,
    )

    print()

    print(
        "Development samples:",
        n_samples,
    )

    print(
        "Encoded features:",
        X.shape[1],
    )

    print(
        "Individual proteins:",
        n_proteins,
    )

    # ------------------------------------------------------------------
    # Build identical-feature groups
    # ------------------------------------------------------------------

    raw_features = pd.read_csv(
        FEATURE_FILE
    )

    raw_features[
        "NP_ID"
    ] = (
        raw_features[
            "NP_ID"
        ]
        .astype(str)
    )

    development_features = (
        raw_features[
            raw_features[
                "NP_ID"
            ].isin(
                train_ids
            )
        ]
        .copy()
    )

    development_features = (
        development_features
        .set_index(
            "NP_ID"
        )
        .loc[
            train_ids
        ]
        .reset_index()
    )

    group_table = (
        build_feature_signature_groups(
            development_features,
            FEATURE_COLS,
        )
    )

    groups = (
        group_table[
            "group_id"
        ]
        .to_numpy()
    )

    print(
        "Unique feature groups:",
        len(
            np.unique(
                groups
            )
        ),
    )

    # ------------------------------------------------------------------
    # Benchmark config
    # ------------------------------------------------------------------

    config = BenchmarkConfig(
        random_state=RANDOM_SEED,
    )

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    adsorption_fold_rows = []

    abundance_fold_rows = []

    fold_assignment_rows = []

    # ==================================================================
    # Cross-validation loop
    # ==================================================================

    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        group_kfold.split(
            X,
            groups=groups,
        ),
        start=1,
    ):

        print()

        print(
            "=" * 72
        )

        print(
            f"FOLD {fold_number}/{N_SPLITS}"
        )

        print(
            "=" * 72
        )

        train_groups = set(
            groups[
                train_index
            ]
        )

        validation_groups = set(
            groups[
                validation_index
            ]
        )

        shared_groups = (
            train_groups
            & validation_groups
        )

        if shared_groups:

            raise RuntimeError(
                "Grouped split leakage detected."
            )

        print(
            "Train samples:",
            len(
                train_index
            ),
        )

        print(
            "Validation samples:",
            len(
                validation_index
            ),
        )

        print(
            "Shared groups:",
            len(
                shared_groups
            ),
        )

        X_train = X[
            train_index
        ]

        X_validation = X[
            validation_index
        ]

        Yp_train = Y_presence[
            train_index
        ]

        Yp_validation = Y_presence[
            validation_index
        ]

        Ya_train = Y_abundance[
            train_index
        ]

        Ya_validation = Y_abundance[
            validation_index
        ]

        # ==============================================================
        # Adsorption models
        # ==============================================================

        for (
            model_key,
            model_label,
        ) in ADSORPTION_BENCHMARKS.items():

            print()

            print(
                f"Adsorption: {model_label}"
            )

            models = (
                fit_adsorption_models(
                    X_train,
                    Yp_train,
                    model_key,
                    config,
                )
            )

            probability = (
                predict_adsorption_probabilities(
                    models,
                    X_validation,
                )
            )

            metrics = (
                adsorption_metrics(
                    Yp_validation,
                    probability,
                    indices=protein_indices,
                )
            )

            adsorption_fold_rows.append(
                {
                    "Fold":
                        fold_number,

                    "Model":
                        model_label,

                    "N_train":
                        int(
                            len(
                                train_index
                            )
                        ),

                    "N_validation":
                        int(
                            len(
                                validation_index
                            )
                        ),

                    "Acc":
                        metrics[
                            "Acc"
                        ],

                    "F1":
                        metrics[
                            "F1"
                        ],

                    "Precision":
                        metrics[
                            "Precision"
                        ],

                    "Recall":
                        metrics[
                            "Recall"
                        ],

                    "AUROC":
                        metrics[
                            "AUROC"
                        ],

                    "AUPRC":
                        metrics[
                            "AUPRC"
                        ],

                    "MCC":
                        metrics[
                            "MCC"
                        ],
                }
            )

            print(
                f"  F1    : "
                f"{metrics['F1']:.4f}"
            )

            print(
                f"  AUROC : "
                f"{metrics['AUROC']:.4f}"
            )

            print(
                f"  AUPRC : "
                f"{metrics['AUPRC']:.4f}"
            )

        # ==============================================================
        # Abundance models
        # ==============================================================

        for (
            model_key,
            model_label,
        ) in ABUNDANCE_BENCHMARKS.items():

            print()

            print(
                f"Abundance: {model_label}"
            )

            models = (
                fit_abundance_models(
                    X_train,
                    Ya_train,
                    model_key,
                    config,
                )
            )

            prediction = (
                predict_abundance_distribution(
                    models,
                    X_validation,
                )
            )

            metrics = (
                abundance_metrics(
                    Ya_validation,
                    prediction,
                    indices=protein_indices,
                )
            )

            abundance_fold_rows.append(
                {
                    "Fold":
                        fold_number,

                    "Model":
                        model_label,

                    "N_train":
                        int(
                            len(
                                train_index
                            )
                        ),

                    "N_validation":
                        int(
                            len(
                                validation_index
                            )
                        ),

                    "Median_r":
                        metrics[
                            "Median_r"
                        ],

                    "1-TVD":
                        metrics[
                            "1-TVD"
                        ],

                    "Cosine":
                        metrics[
                            "Cosine"
                        ],
                }
            )

            print(
                f"  Median_r : "
                f"{metrics['Median_r']:.4f}"
            )

            print(
                f"  1-TVD    : "
                f"{metrics['1-TVD']:.4f}"
            )

            print(
                f"  Cosine   : "
                f"{metrics['Cosine']:.4f}"
            )

        # ==============================================================
        # Save fold assignments
        # ==============================================================

        for global_index in validation_index:

            fold_assignment_rows.append(
                {
                    "Fold":
                        fold_number,

                    "NP_ID":
                        train_ids[
                            global_index
                        ],

                    "Group_ID":
                        int(
                            groups[
                                global_index
                            ]
                        ),
                }
            )

        # Incremental saves
        pd.DataFrame(
            adsorption_fold_rows
        ).to_csv(
            RESULTS_DIR
            / "adsorption_fold_results.csv",
            index=False,
        )

        pd.DataFrame(
            abundance_fold_rows
        ).to_csv(
            RESULTS_DIR
            / "abundance_fold_results.csv",
            index=False,
        )

    # ==================================================================
    # Summaries
    # ==================================================================

    adsorption_df = pd.DataFrame(
        adsorption_fold_rows
    )

    abundance_df = pd.DataFrame(
        abundance_fold_rows
    )

    # ------------------------------------------------------------------
    # Adsorption summary
    # ------------------------------------------------------------------

    adsorption_metrics_list = [
        "Acc",
        "F1",
        "Precision",
        "Recall",
        "AUROC",
        "AUPRC",
        "MCC",
    ]

    adsorption_summary_rows = []

    for model_label in (
        ADSORPTION_BENCHMARKS.values()
    ):

        model_rows = (
            adsorption_df[
                adsorption_df[
                    "Model"
                ]
                == model_label
            ]
        )

        row = {
            "Model":
                model_label,
        }

        for metric in (
            adsorption_metrics_list
        ):

            values = (
                model_rows[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.nanmean(
                    values
                )
            )

            row[
                f"{metric}_sd"
            ] = float(
                np.nanstd(
                    values,
                    ddof=1,
                )
            )

        adsorption_summary_rows.append(
            row
        )

    adsorption_summary = pd.DataFrame(
        adsorption_summary_rows
    )

    # ------------------------------------------------------------------
    # Abundance summary
    # ------------------------------------------------------------------

    abundance_metrics_list = [
        "Median_r",
        "1-TVD",
        "Cosine",
    ]

    abundance_summary_rows = []

    for model_label in (
        ABUNDANCE_BENCHMARKS.values()
    ):

        model_rows = (
            abundance_df[
                abundance_df[
                    "Model"
                ]
                == model_label
            ]
        )

        row = {
            "Model":
                model_label,
        }

        for metric in (
            abundance_metrics_list
        ):

            values = (
                model_rows[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.nanmean(
                    values
                )
            )

            row[
                f"{metric}_sd"
            ] = float(
                np.nanstd(
                    values,
                    ddof=1,
                )
            )

        abundance_summary_rows.append(
            row
        )

    abundance_summary = pd.DataFrame(
        abundance_summary_rows
    )

    # ==================================================================
    # Save
    # ==================================================================

    group_table.to_csv(
        RESULTS_DIR
        / "feature_signature_groups.csv",
        index=False,
    )

    pd.DataFrame(
        fold_assignment_rows
    ).to_csv(
        RESULTS_DIR
        / "fold_assignments.csv",
        index=False,
    )

    adsorption_df.to_csv(
        RESULTS_DIR
        / "adsorption_fold_results.csv",
        index=False,
    )

    abundance_df.to_csv(
        RESULTS_DIR
        / "abundance_fold_results.csv",
        index=False,
    )

    adsorption_summary.to_csv(
        RESULTS_DIR
        / "adsorption_summary.csv",
        index=False,
    )

    abundance_summary.to_csv(
        RESULTS_DIR
        / "abundance_summary.csv",
        index=False,
    )

    with open(
        RESULTS_DIR
        / "grouped_benchmark_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "n_splits":
                    N_SPLITS,

                "random_seed":
                    RANDOM_SEED,

                "grouping":
                    "identical raw feature signature",

                "development_samples":
                    int(
                        n_samples
                    ),

                "n_unique_groups":
                    int(
                        len(
                            np.unique(
                                groups
                            )
                        )
                    ),
            },
            file,
            indent=4,
        )

    # ==================================================================
    # Console summary
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "GROUPED ADSORPTION BENCHMARK SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        adsorption_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "GROUPED ABUNDANCE BENCHMARK SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        abundance_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()