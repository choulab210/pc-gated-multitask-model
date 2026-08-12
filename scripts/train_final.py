"""
Train Final Two-Head Protein Corona Model
=========================================

This script:

1. Loads and preprocesses Data_1.csv and Data_2.csv.
2. Trains the final two-head neural network.
3. Evaluates the held-out test set.
4. Saves model weights, training history, predictions, and metrics.

Run from the project root:

    python scripts/train_final.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcmodel.data import prepare_model_data

from pcmodel.metrics import (
    adsorption_metrics,
    abundance_metrics,
)

from pcmodel.training import (
    TrainingConfig,
    get_device,
    predict_probabilities,
    train_with_early_stopping,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FEATURE_FILE = DATA_DIR / "Data_1.csv"
ABUNDANCE_FILE = DATA_DIR / "Data_2.csv"


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print("=" * 70)
    print("Protein Corona Two-Head Model")
    print("Final Baseline Training")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = get_device()

    print()
    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------

    print()
    print("Preparing data...")

    data = prepare_model_data(
        FEATURE_FILE,
        ABUNDANCE_FILE,
    )

    print()
    print("Data summary")
    print("-" * 40)

    print(
        "Training samples:",
        data.X_train.shape[0],
    )

    print(
        "Test samples:",
        data.X_test.shape[0],
    )

    print(
        "Encoded input features:",
        data.X_train.shape[1],
    )

    print(
        "Selected individual proteins:",
        len(data.panel),
    )

    print(
        "Presence outputs:",
        data.Y_presence_train.shape[1],
    )

    print(
        "Abundance outputs:",
        data.Y_abundance_train.shape[1],
    )

    # ------------------------------------------------------------------
    # Final training configuration
    # ------------------------------------------------------------------

    config = TrainingConfig(
        hidden=(
            320,
            320,
            320,
        ),
        dropout=0.2597530378485485,
        learning_rate=0.0019913470844860467,
        weight_decay=0.0,
        weight_presence=1.0,
        weight_abundance=1.9528320541332875,
        alpha_gate=1.4356053039813828,
        temp_init=0.7623618527415651,
        batch_size=64,
        max_epochs=100,
        patience=10,
        dev_ratio=0.15,
    )

    print()
    print("Training model...")
    print("-" * 40)

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------

    result = train_with_early_stopping(
        data.X_train,
        data.Y_presence_train,
        data.Y_abundance_train,
        config=config,
        seed=42,
        device=device,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Predict held-out test set
    # ------------------------------------------------------------------

    print()
    print("Generating held-out test predictions...")

    (
        presence_probability,
        abundance_prediction,
    ) = predict_probabilities(
        result.model,
        data.X_test,
        device=device,
    )

    print(
        "Presence prediction shape:",
        presence_probability.shape,
    )

    print(
        "Abundance prediction shape:",
        abundance_prediction.shape,
    )

    # ------------------------------------------------------------------
    # Evaluate individual proteins
    # ------------------------------------------------------------------
    #
    # The current compatibility architecture includes OTHER as the final
    # output. Manuscript protein-level performance should evaluate only
    # the individual proteins.
    # ------------------------------------------------------------------

    protein_indices = list(
        range(
            len(data.panel)
        )
    )

    adsorption_results = adsorption_metrics(
        data.Y_presence_test,
        presence_probability,
        indices=protein_indices,
    )

    abundance_results = abundance_metrics(
        data.Y_abundance_test,
        abundance_prediction,
        indices=protein_indices,
    )

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("HELD-OUT TEST PERFORMANCE")
    print("=" * 70)

    print()
    print("Adsorption Head")
    print("-" * 40)

    for key, value in adsorption_results.items():

        if isinstance(
            value,
            (float, np.floating),
        ):

            print(
                f"{key:12s}: {value:.4f}"
            )

        else:

            print(
                f"{key:12s}: {value}"
            )

    print()
    print("Abundance Head")
    print("-" * 40)

    for key, value in abundance_results.items():

        if isinstance(
            value,
            (float, np.floating),
        ):

            print(
                f"{key:12s}: {value:.4f}"
            )

        else:

            print(
                f"{key:12s}: {value}"
            )

    # ==================================================================
    # Save training history
    # ==================================================================

    history_df = pd.DataFrame(
        result.history
    )

    history_path = (
        RESULTS_DIR
        / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # ==================================================================
    # Save predictions
    # ==================================================================

    presence_columns = list(
        data.presence_columns
    )

    abundance_columns = list(
        data.abundance_columns
    )

    presence_df = pd.DataFrame(
        presence_probability,
        columns=presence_columns,
    )

    abundance_df = pd.DataFrame(
        abundance_prediction,
        columns=abundance_columns,
    )

    presence_path = (
        RESULTS_DIR
        / "test_presence_predictions.csv"
    )

    abundance_path = (
        RESULTS_DIR
        / "test_abundance_predictions.csv"
    )

    presence_df.to_csv(
        presence_path,
        index=False,
    )

    abundance_df.to_csv(
        abundance_path,
        index=False,
    )

    # ==================================================================
    # Save observed test targets
    # ==================================================================

    observed_presence_df = pd.DataFrame(
        data.Y_presence_test,
        columns=presence_columns,
    )

    observed_abundance_df = pd.DataFrame(
        data.Y_abundance_test,
        columns=abundance_columns,
    )

    observed_presence_path = (
        RESULTS_DIR
        / "test_presence_observed.csv"
    )

    observed_abundance_path = (
        RESULTS_DIR
        / "test_abundance_observed.csv"
    )

    observed_presence_df.to_csv(
        observed_presence_path,
        index=False,
    )

    observed_abundance_df.to_csv(
        observed_abundance_path,
        index=False,
    )

    # ==================================================================
    # Save metrics
    # ==================================================================

    metrics = {
        "best_epoch": int(
            result.best_epoch
        ),
        "best_development_score": float(
            result.best_score
        ),
        "adsorption": {
            key: (
                int(value)
                if key == "N"
                else float(value)
            )
            for key, value
            in adsorption_results.items()
        },
        "abundance": {
            key: (
                int(value)
                if key == "N"
                else float(value)
            )
            for key, value
            in abundance_results.items()
        },
    }

    metrics_path = (
        RESULTS_DIR
        / "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ==================================================================
    # Save selected protein panel
    # ==================================================================

    protein_panel_path = (
        RESULTS_DIR
        / "protein_panel.csv"
    )

    pd.DataFrame(
        {
            "protein": list(
                data.panel
            )
        }
    ).to_csv(
        protein_panel_path,
        index=False,
    )

    # ==================================================================
    # Save model checkpoint
    # ==================================================================

    checkpoint_path = (
        RESULTS_DIR
        / "twohead_model_checkpoint.pt"
    )

    # Store tensors on CPU so the checkpoint can later be loaded
    # on either CPU or GPU.
    model_state = {
        key: value.detach().cpu()
        for key, value
        in result.model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": model_state,
        "training_config": asdict(
            result.config
        ),
        "input_dim": int(
            data.X_train.shape[1]
        ),
        "n_outputs": int(
            data.Y_presence_train.shape[1]
        ),
        "panel": list(
            data.panel
        ),
        "presence_cols": presence_columns,
        "abundance_cols": abundance_columns,
        "best_epoch": int(
            result.best_epoch
        ),
        "best_score": float(
            result.best_score
        ),
        "seed": 42,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    # ==================================================================
    # Finish
    # ==================================================================

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        checkpoint_path
    )

    print(
        metrics_path
    )

    print(
        history_path
    )

    print(
        presence_path
    )

    print(
        abundance_path
    )

    print()
    print(
        "Final baseline training completed successfully."
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()