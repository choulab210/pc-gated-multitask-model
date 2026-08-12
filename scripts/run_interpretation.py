"""
Run Abundance-Head Interpretation
=================================

This script evaluates how abundance-prediction performance varies
across proteins with different abundance levels.

Workflow
--------
1. Reconstruct the model-development/test data.
2. Load the saved final model.
3. Predict the held-out test set.
4. Exclude the OTHER residual bin.
5. Rank the 174 individual proteins by mean training abundance.
6. Divide proteins into High / Middle / Low abundance tertiles.
7. Calculate per-protein Pearson correlation.
8. Add protein names and functional categories.
9. Save protein-level and tertile-level interpretation results.
10. Save observed-vs-predicted data for later figure generation.

Run from project root:

    python scripts/run_interpretation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcmodel.data import (
    prepare_model_data,
)

from pcmodel.interpretation import (
    abundance_interpretation,
)

from pcmodel.metadata import (
    load_protein_metadata,
    metadata_to_mappings,
    validate_metadata_for_panel,
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
    / "interpretation"
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

METADATA_FILE = (
    DATA_DIR
    / "protein_metadata.csv"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "results"
    / "twohead_model_checkpoint.pt"
)


# ======================================================================
# File checks
# ======================================================================


def check_required_files() -> None:
    """
    Confirm that all required files exist.
    """

    required = [
        FEATURE_FILE,
        ABUNDANCE_FILE,
        METADATA_FILE,
        CHECKPOINT_FILE,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        text = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "Required file(s) missing:\n"
            + text
        )


# ======================================================================
# Load model
# ======================================================================


def load_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> TwoHead:
    """
    Reconstruct the final two-head model.
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
# Observed-vs-predicted long table
# ======================================================================


def build_observed_predicted_table(
    observed: np.ndarray,
    predicted: np.ndarray,
    protein_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a long-format table for later observed-vs-predicted plots.

    Each row represents one:

        NP sample x protein

    combination.
    """

    n_samples, n_proteins = (
        observed.shape
    )

    if (
        predicted.shape
        != observed.shape
    ):

        raise ValueError(
            "Observed and predicted abundance "
            "matrices must have identical shape."
        )

    if (
        len(protein_table)
        != n_proteins
    ):

        raise ValueError(
            "Protein table does not match "
            "abundance matrix width."
        )

    # --------------------------------------------------------------
    # Ensure protein order matches the abundance matrix order.
    # --------------------------------------------------------------

    protein_lookup = (
        protein_table
        .set_index(
            "protein"
        )
    )

    protein_order = list(
        protein_table[
            "protein"
        ]
    )

    rows = []

    for protein_index, protein_id in enumerate(
        protein_order
    ):

        metadata_row = (
            protein_lookup.loc[
                protein_id
            ]
        )

        protein_name = (
            metadata_row[
                "protein_name"
            ]
            if (
                "protein_name"
                in metadata_row.index
            )
            else protein_id
        )

        category = (
            metadata_row[
                "category"
            ]
            if (
                "category"
                in metadata_row.index
            )
            else "Other/Mixed"
        )

        tertile = (
            metadata_row[
                "abundance_tertile"
            ]
        )

        pearson_r = (
            metadata_row[
                "pearson_r"
            ]
        )

        for sample_index in range(
            n_samples
        ):

            rows.append(
                {
                    "Test_sample":
                        sample_index + 1,

                    "protein":
                        protein_id,

                    "protein_name":
                        protein_name,

                    "category":
                        category,

                    "abundance_tertile":
                        tertile,

                    "pearson_r":
                        pearson_r,

                    "Observed":
                        float(
                            observed[
                                sample_index,
                                protein_index,
                            ]
                        ),

                    "Predicted":
                        float(
                            predicted[
                                sample_index,
                                protein_index,
                            ]
                        ),
                }
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
        "ABUNDANCE-HEAD INTERPRETATION"
    )

    print(
        "=" * 72
    )

    check_required_files()

    # ------------------------------------------------------------------
    # Reconstruct data
    # ------------------------------------------------------------------

    print()

    print(
        "Preparing model data..."
    )

    prepared = (
        prepare_model_data(
            FEATURE_FILE,
            ABUNDANCE_FILE,
        )
    )

    panel = list(
        prepared.panel
    )

    n_proteins = len(
        panel
    )

    print(
        "Selected individual proteins:",
        n_proteins,
    )

    print(
        "Development samples:",
        prepared.X_train.shape[0],
    )

    print(
        "Held-out samples:",
        prepared.X_test.shape[0],
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    # Safety checks

    if (
        list(
            checkpoint[
                "panel"
            ]
        )
        != panel
    ):

        raise ValueError(
            "Saved checkpoint protein panel "
            "does not match reconstructed data."
        )

    if (
        int(
            checkpoint[
                "input_dim"
            ]
        )
        != prepared.X_test.shape[1]
    ):

        raise ValueError(
            "Input feature dimension mismatch."
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    device = get_device()

    print()

    print(
        "Device:",
        device,
    )

    model = (
        load_model_from_checkpoint(
            checkpoint,
            device,
        )
    )

    # ------------------------------------------------------------------
    # Predict held-out test set
    # ------------------------------------------------------------------

    print()

    print(
        "Generating test predictions..."
    )

    (
        _,
        abundance_prediction_full,
    ) = predict_probabilities(
        model,
        prepared.X_test,
        device=device,
    )

    # ------------------------------------------------------------------
    # Remove OTHER
    # ------------------------------------------------------------------
    #
    # Current architecture:
    #
    #     174 individual proteins + OTHER
    #
    # Interpretation is performed only on the 174 individual proteins.
    # ------------------------------------------------------------------

    training_abundance = (
        prepared.Y_abundance_train[
            :,
            :n_proteins
        ]
    )

    observed_abundance = (
        prepared.Y_abundance_test[
            :,
            :n_proteins
        ]
    )

    predicted_abundance = (
        abundance_prediction_full[
            :,
            :n_proteins
        ]
    )

    print()

    print(
        "Training abundance matrix:",
        training_abundance.shape,
    )

    print(
        "Observed test abundance:",
        observed_abundance.shape,
    )

    print(
        "Predicted test abundance:",
        predicted_abundance.shape,
    )

    # ------------------------------------------------------------------
    # Protein metadata
    # ------------------------------------------------------------------

    metadata = (
        load_protein_metadata(
            METADATA_FILE
        )
    )

    validate_metadata_for_panel(
        metadata,
        panel,
        require_all=True,
    )

    (
        id_to_name,
        id_to_category,
    ) = metadata_to_mappings(
        metadata
    )

    # ------------------------------------------------------------------
    # Run abundance interpretation
    # ------------------------------------------------------------------

    interpretation = (
        abundance_interpretation(
            observed_abundance,
            predicted_abundance,
            training_abundance,
            panel,
            id_to_name=id_to_name,
            id_to_category=(
                id_to_category
            ),
        )
    )

    protein_table = (
        interpretation.protein_table
    )

    tertile_summary = (
        interpretation.tertile_summary
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # abundance_interpretation() sorts the protein table by abundance
    # rank. For observed/predicted long-format data we need model-panel
    # order.
    # ------------------------------------------------------------------

    protein_table_model_order = (
        protein_table
        .set_index(
            "protein"
        )
        .loc[
            panel
        ]
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Display tertile results
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "ABUNDANCE TERTILE SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        tertile_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Top proteins by training abundance
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "TOP 10 PROTEINS BY TRAINING ABUNDANCE"
    )

    print(
        "=" * 72
    )

    display_columns = [
        "protein",
        "protein_name",
        "category",
        "mean_training_abundance",
        "pearson_r",
        "abundance_tertile",
    ]

    print(
        protein_table[
            display_columns
        ]
        .head(
            10
        )
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ==================================================================
    # Save protein-level interpretation
    # ==================================================================

    protein_table.to_csv(
        RESULTS_DIR
        / "protein_abundance_interpretation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save tertile summary
    # ------------------------------------------------------------------

    tertile_summary.to_csv(
        RESULTS_DIR
        / "abundance_tertile_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Long-format observed vs predicted table
    # ------------------------------------------------------------------

    observed_predicted_table = (
        build_observed_predicted_table(
            observed_abundance,
            predicted_abundance,
            protein_table_model_order,
        )
    )

    observed_predicted_table.to_csv(
        RESULTS_DIR
        / "observed_vs_predicted_abundance.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save compact protein table in original model order
    # ------------------------------------------------------------------

    protein_table_model_order.to_csv(
        RESULTS_DIR
        / "protein_abundance_model_order.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "ABUNDANCE INTERPRETATION COMPLETE"
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