"""
Run Applicability Domain Analysis
=================================

This script:

1. Reconstructs the final model-development/test split.
2. Fits the k-NN applicability domain using development samples only.
3. Assigns held-out test samples as In-AD or Out-of-AD.
4. Loads the saved final two-head model.
5. Generates held-out predictions.
6. Calculates adsorption and abundance performance for:
       - All test samples
       - In-AD test samples
       - Out-of-AD test samples
7. Saves AD assignments and performance summaries.

Run from project root:

    python scripts/run_applicability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcmodel.applicability import (
    fit_applicability_domain,
)

from pcmodel.data import (
    prepare_model_data,
)

from pcmodel.metrics import (
    adsorption_metrics,
    abundance_metrics,
)

from pcmodel.models import TwoHead

from pcmodel.training import (
    get_device,
    predict_probabilities,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "applicability"
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

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "results"
    / "twohead_model_checkpoint.pt"
)


# ======================================================================
# AD configuration
# ======================================================================

N_NEIGHBORS = 5

THRESHOLD_PERCENTILE = 85.0


# ======================================================================
# Model loading
# ======================================================================


def load_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> TwoHead:
    """
    Reconstruct the saved TwoHead model.
    """

    config = checkpoint[
        "training_config"
    ]

    model = TwoHead(
        in_dim=int(
            checkpoint[
                "input_dim"
            ]
        ),

        hidden=list(
            config[
                "hidden"
            ]
        ),

        k=int(
            checkpoint[
                "n_outputs"
            ]
        ),

        dropout=float(
            config[
                "dropout"
            ]
        ),

        alpha_gate=float(
            config[
                "alpha_gate"
            ]
        ),

        temp_init=float(
            config[
                "temp_init"
            ]
        ),

        stopgrad_gate=True,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ======================================================================
# Sample IDs
# ======================================================================


def get_test_sample_ids(
    prepared,
) -> list[str]:
    """
    Recover test NP identifiers when available.

    Falls back to Test_001, Test_002, ... if the PreparedData object
    does not expose test IDs directly.
    """

    possible_attributes = [
        "test_ids",
        "test_np_ids",
        "np_ids_test",
    ]

    for attribute in possible_attributes:

        if hasattr(
            prepared,
            attribute,
        ):

            values = getattr(
                prepared,
                attribute,
            )

            if values is not None:

                values = list(
                    values
                )

                if (
                    len(values)
                    == len(
                        prepared.X_test
                    )
                ):

                    return [
                        str(value)
                        for value
                        in values
                    ]

    return [
        f"Test_{i + 1:03d}"
        for i in range(
            len(
                prepared.X_test
            )
        )
    ]


# ======================================================================
# Performance helper
# ======================================================================


def evaluate_subset(
    label: str,
    sample_mask: np.ndarray,
    y_presence: np.ndarray,
    presence_probability: np.ndarray,
    y_abundance: np.ndarray,
    abundance_prediction: np.ndarray,
    protein_indices: list[int],
) -> dict:
    """
    Evaluate one subset of test samples.
    """

    sample_mask = np.asarray(
        sample_mask,
        dtype=bool,
    )

    n_samples = int(
        sample_mask.sum()
    )

    if n_samples == 0:

        return {
            "Group": label,
            "N_samples": 0,
            "Acc": np.nan,
            "F1": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "MCC": np.nan,
            "Median_r": np.nan,
            "1-TVD": np.nan,
            "Cosine": np.nan,
        }

    # ------------------------------------------------------------------
    # Adsorption
    # ------------------------------------------------------------------

    adsorption = adsorption_metrics(
        y_presence[
            sample_mask
        ],
        presence_probability[
            sample_mask
        ],
        indices=protein_indices,
    )

    # ------------------------------------------------------------------
    # Abundance
    # ------------------------------------------------------------------

    abundance = abundance_metrics(
        y_abundance[
            sample_mask
        ],
        abundance_prediction[
            sample_mask
        ],
        indices=protein_indices,
    )

    return {
        "Group":
            label,

        "N_samples":
            n_samples,

        "Acc":
            adsorption[
                "Acc"
            ],

        "F1":
            adsorption[
                "F1"
            ],

        "Precision":
            adsorption[
                "Precision"
            ],

        "Recall":
            adsorption[
                "Recall"
            ],

        "AUROC":
            adsorption[
                "AUROC"
            ],

        "AUPRC":
            adsorption[
                "AUPRC"
            ],

        "MCC":
            adsorption[
                "MCC"
            ],

        "Median_r":
            abundance[
                "Median_r"
            ],

        "1-TVD":
            abundance[
                "1-TVD"
            ],

        "Cosine":
            abundance[
                "Cosine"
            ],
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "APPLICABILITY DOMAIN ANALYSIS"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Load/preprocess data
    # ------------------------------------------------------------------

    print()

    print(
        "Preparing model data..."
    )

    prepared = prepare_model_data(
        FEATURE_FILE,
        ABUNDANCE_FILE,
    )

    print(
        "Development samples:",
        prepared.X_train.shape[0],
    )

    print(
        "Held-out samples:",
        prepared.X_test.shape[0],
    )

    print(
        "Encoded features:",
        prepared.X_train.shape[1],
    )

    # ------------------------------------------------------------------
    # Fit AD
    # ------------------------------------------------------------------

    print()

    print(
        "Fitting applicability domain..."
    )

    ad_model = fit_applicability_domain(
        prepared.X_train,
        n_neighbors=N_NEIGHBORS,
        threshold_percentile=(
            THRESHOLD_PERCENTILE
        ),
    )

    ad_result = ad_model.evaluate(
        prepared.X_test
    )

    in_ad_mask = (
        ad_result.in_domain
    )

    out_ad_mask = (
        ~ad_result.in_domain
    )

    print()

    print(
        "=" * 72
    )

    print(
        "AD SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"k neighbors       : "
        f"{N_NEIGHBORS}"
    )

    print(
        f"Threshold percentile: "
        f"{THRESHOLD_PERCENTILE:.1f}"
    )

    print(
        f"AD threshold      : "
        f"{ad_result.threshold:.6f}"
    )

    print(
        f"In-AD samples     : "
        f"{int(in_ad_mask.sum())}"
    )

    print(
        f"Out-of-AD samples : "
        f"{int(out_ad_mask.sum())}"
    )

    # ------------------------------------------------------------------
    # Load saved model
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    # Safety checks

    if (
        prepared.X_test.shape[1]
        != int(
            checkpoint[
                "input_dim"
            ]
        )
    ):

        raise ValueError(
            "Encoded feature dimension does not match "
            "the saved model checkpoint."
        )

    if (
        list(
            prepared.panel
        )
        != list(
            checkpoint[
                "panel"
            ]
        )
    ):

        raise ValueError(
            "Protein panel does not match "
            "the saved model checkpoint."
        )

    device = get_device()

    print()

    print(
        "Device:",
        device,
    )

    model = load_model_from_checkpoint(
        checkpoint,
        device,
    )

    # ------------------------------------------------------------------
    # Predict held-out test set
    # ------------------------------------------------------------------

    print()

    print(
        "Generating held-out predictions..."
    )

    (
        presence_probability,
        abundance_prediction,
    ) = predict_probabilities(
        model,
        prepared.X_test,
        device=device,
    )

    # ------------------------------------------------------------------
    # Individual protein columns
    # ------------------------------------------------------------------
    #
    # OTHER is the final output and is excluded from protein-level
    # performance evaluation.
    # ------------------------------------------------------------------

    protein_indices = list(
        range(
            len(
                prepared.panel
            )
        )
    )

    # ------------------------------------------------------------------
    # Evaluate All / In-AD / Out-of-AD
    # ------------------------------------------------------------------

    all_mask = np.ones(
        len(
            prepared.X_test
        ),
        dtype=bool,
    )

    performance_rows = [
        evaluate_subset(
            "All",
            all_mask,
            prepared.Y_presence_test,
            presence_probability,
            prepared.Y_abundance_test,
            abundance_prediction,
            protein_indices,
        ),

        evaluate_subset(
            "In-AD",
            in_ad_mask,
            prepared.Y_presence_test,
            presence_probability,
            prepared.Y_abundance_test,
            abundance_prediction,
            protein_indices,
        ),

        evaluate_subset(
            "Out-of-AD",
            out_ad_mask,
            prepared.Y_presence_test,
            presence_probability,
            prepared.Y_abundance_test,
            abundance_prediction,
            protein_indices,
        ),
    ]

    performance_df = pd.DataFrame(
        performance_rows
    )

    print()

    print(
        "=" * 72
    )

    print(
        "PERFORMANCE BY APPLICABILITY DOMAIN"
    )

    print(
        "=" * 72
    )

    print(
        performance_df.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Save sample-level AD assignments
    # ------------------------------------------------------------------

    test_ids = get_test_sample_ids(
        prepared
    )

    assignments_df = (
        ad_result.to_dataframe(
            sample_ids=test_ids
        )
    )

    assignments_df.to_csv(
        RESULTS_DIR
        / "test_ad_assignments.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save performance summary
    # ------------------------------------------------------------------

    performance_df.to_csv(
        RESULTS_DIR
        / "performance_by_ad.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save training distance distribution
    # ------------------------------------------------------------------

    training_distance_df = pd.DataFrame(
        {
            "Training_sample":
                [
                    f"Train_{i + 1:03d}"
                    for i in range(
                        len(
                            ad_model.training_distances_
                        )
                    )
                ],

            "Mean_kNN_distance":
                ad_model.training_distances_,
        }
    )

    training_distance_df.to_csv(
        RESULTS_DIR
        / "training_ad_distances.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save AD summary
    # ------------------------------------------------------------------

    ad_summary = (
        ad_model.summary()
    )

    ad_summary.update(
        {
            "n_test_samples":
                int(
                    len(
                        prepared.X_test
                    )
                ),

            "n_in_ad":
                int(
                    in_ad_mask.sum()
                ),

            "n_out_of_ad":
                int(
                    out_ad_mask.sum()
                ),
        }
    )

    with open(
        RESULTS_DIR
        / "ad_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            ad_summary,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "APPLICABILITY DOMAIN ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
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