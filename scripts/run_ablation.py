"""
Run Architecture Ablation Study
===============================

Compare three architectures:

A. Abundance-only
B. Two-head without gating
C. Two-head with adsorption-guided gating

Each architecture is trained across multiple random seeds.

Default seeds:
    42, 52, 62, 72, 82

Outputs
-------
results/ablation/
    architecture_table.csv
    seed_results.csv
    summary.csv

Run from project root:

    python scripts/run_ablation.py
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from pcmodel.ablation import (
    AblationModelConfig,
    AblationVariant,
    VARIANT_LABELS,
    ablation_architecture_table,
    build_ablation_model,
    has_presence_head,
)

from pcmodel.data import (
    prepare_model_data,
)

from pcmodel.metrics import (
    abundance_distribution_metrics,
    abundance_metrics,
    adsorption_hpo_metrics,
    adsorption_metrics,
    composite_score,
)

from pcmodel.training import (
    TrainingConfig,
    compute_pos_weight,
    get_device,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ablation"
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
# Ablation configuration
# ======================================================================

SEEDS = [
    42,
    52,
    62,
    72,
    82,
]


VARIANTS = [
    AblationVariant.ABUNDANCE_ONLY,
    AblationVariant.TWO_HEAD_NO_GATE,
    AblationVariant.TWO_HEAD_GATED,
]


# ======================================================================
# Reproducibility
# ======================================================================


def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ======================================================================
# DataLoader
# ======================================================================


def make_loader(
    X: np.ndarray,
    Y_presence: np.ndarray,
    Y_abundance: np.ndarray,
    *,
    batch_size: int,
) -> DataLoader:

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),

        torch.tensor(
            Y_presence,
            dtype=torch.float32,
        ),

        torch.tensor(
            Y_abundance,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


# ======================================================================
# Prediction
# ======================================================================


@torch.no_grad()
def predict_ablation_model(
    model,
    variant: AblationVariant,
    X: np.ndarray,
    device: torch.device,
):
    """
    Return:

        presence_probability or None
        abundance_distribution
    """

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    if (
        variant
        == AblationVariant.ABUNDANCE_ONLY
    ):

        abundance_logprob = model(
            X_tensor
        )

        abundance_prediction = (
            torch.exp(
                abundance_logprob
            )
            .cpu()
            .numpy()
        )

        return (
            None,
            abundance_prediction,
        )

    (
        presence_logits,
        abundance_logprob,
    ) = model(
        X_tensor
    )

    presence_probability = (
        torch.sigmoid(
            presence_logits
        )
        .cpu()
        .numpy()
    )

    abundance_prediction = (
        torch.exp(
            abundance_logprob
        )
        .cpu()
        .numpy()
    )

    return (
        presence_probability,
        abundance_prediction,
    )


# ======================================================================
# Train one model
# ======================================================================


def train_one_variant(
    variant: AblationVariant,
    seed: int,
    X: np.ndarray,
    Y_presence: np.ndarray,
    Y_abundance: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
):
    """
    Train one architecture for one seed.
    """

    set_seed(
        seed
    )

    # ------------------------------------------------------------------
    # Internal train/dev split
    # ------------------------------------------------------------------

    indices = np.arange(
        len(X)
    )

    train_idx, dev_idx = (
        train_test_split(
            indices,
            test_size=config.dev_ratio,
            random_state=seed,
        )
    )

    X_train = X[
        train_idx
    ]

    X_dev = X[
        dev_idx
    ]

    Yp_train = Y_presence[
        train_idx
    ]

    Yp_dev = Y_presence[
        dev_idx
    ]

    Ya_train = Y_abundance[
        train_idx
    ]

    Ya_dev = Y_abundance[
        dev_idx
    ]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model_config = AblationModelConfig(
        in_dim=X.shape[1],
        hidden=list(
            config.hidden
        ),
        n_outputs=(
            Y_abundance.shape[1]
        ),
        dropout=config.dropout,
        alpha_gate=config.alpha_gate,
        temp_init=config.temp_init,
        stopgrad_gate=True,
    )

    model = build_ablation_model(
        variant,
        model_config,
    ).to(
        device
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    abundance_loss_fn = nn.KLDivLoss(
        reduction="batchmean"
    )

    if has_presence_head(
        variant
    ):

        pos_weight = compute_pos_weight(
            Yp_train,
            device,
        )

        presence_loss_fn = (
            nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction="mean",
            )
        )

    else:

        presence_loss_fn = None

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    train_loader = make_loader(
        X_train,
        Yp_train,
        Ya_train,
        batch_size=config.batch_size,
    )

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    best_score = -np.inf

    best_state = None

    best_epoch = 0

    no_improvement = 0

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    for epoch in range(
        1,
        config.max_epochs + 1,
    ):

        model.train()

        for (
            X_batch,
            Yp_batch,
            Ya_batch,
        ) in train_loader:

            X_batch = X_batch.to(
                device
            )

            Yp_batch = Yp_batch.to(
                device
            )

            Ya_batch = Ya_batch.to(
                device
            )

            optimizer.zero_grad()

            # ----------------------------------------------------------
            # A. Abundance only
            # ----------------------------------------------------------

            if (
                variant
                == AblationVariant.ABUNDANCE_ONLY
            ):

                abundance_logprob = model(
                    X_batch
                )

                loss_abundance = (
                    abundance_loss_fn(
                        abundance_logprob,
                        Ya_batch,
                    )
                )

                total_loss = (
                    config.weight_abundance
                    * loss_abundance
                )

            # ----------------------------------------------------------
            # B/C. Two-head models
            # ----------------------------------------------------------

            else:

                (
                    presence_logits,
                    abundance_logprob,
                ) = model(
                    X_batch
                )

                loss_presence = (
                    presence_loss_fn(
                        presence_logits,
                        Yp_batch,
                    )
                )

                loss_abundance = (
                    abundance_loss_fn(
                        abundance_logprob,
                        Ya_batch,
                    )
                )

                total_loss = (
                    config.weight_presence
                    * loss_presence
                    +
                    config.weight_abundance
                    * loss_abundance
                )

            total_loss.backward()

            optimizer.step()

        # ==============================================================
        # Development evaluation
        # ==============================================================

        (
            dev_presence_probability,
            dev_abundance_prediction,
        ) = predict_ablation_model(
            model,
            variant,
            X_dev,
            device,
        )

        abundance_dev = (
            abundance_distribution_metrics(
                Ya_dev,
                dev_abundance_prediction,
            )
        )

        # --------------------------------------------------------------
        # A. Abundance-only early stopping
        #
        # Use abundance performance only.
        # --------------------------------------------------------------

        if (
            variant
            == AblationVariant.ABUNDANCE_ONLY
        ):

            score = float(
                abundance_dev[
                    "one_minus_tvd"
                ]
            )

        # --------------------------------------------------------------
        # B/C. Same composite criterion as final model
        # --------------------------------------------------------------

        else:

            adsorption_dev = (
                adsorption_hpo_metrics(
                    Yp_dev,
                    dev_presence_probability,
                )
            )

            score = composite_score(
                adsorption_dev,
                abundance_dev,
            )

        # --------------------------------------------------------------
        # Early stopping update
        # --------------------------------------------------------------

        if (
            score
            > best_score
            + 1e-6
        ):

            best_score = score

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            no_improvement = 0

        else:

            no_improvement += 1

        if (
            no_improvement
            >= config.patience
        ):

            break

    # ------------------------------------------------------------------
    # Restore best model
    # ------------------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No valid ablation checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        best_score,
    )


# ======================================================================
# Evaluate one seed
# ======================================================================


def evaluate_seed(
    variant: AblationVariant,
    seed: int,
    prepared,
    config: TrainingConfig,
    device: torch.device,
):
    """
    Train one architecture/seed and evaluate held-out test performance.
    """

    (
        model,
        best_epoch,
        best_dev_score,
    ) = train_one_variant(
        variant,
        seed,
        prepared.X_train,
        prepared.Y_presence_train,
        prepared.Y_abundance_train,
        config,
        device,
    )

    (
        presence_probability,
        abundance_prediction,
    ) = predict_ablation_model(
        model,
        variant,
        prepared.X_test,
        device,
    )

    # ------------------------------------------------------------------
    # Evaluate 174 individual proteins only
    # ------------------------------------------------------------------

    protein_indices = list(
        range(
            len(
                prepared.panel
            )
        )
    )

    abundance_result = (
        abundance_metrics(
            prepared.Y_abundance_test,
            abundance_prediction,
            indices=protein_indices,
        )
    )

    result = {
        "Variant":
            variant.value,

        "Model":
            VARIANT_LABELS[
                variant
            ],

        "Seed":
            seed,

        "Best_epoch":
            best_epoch,

        "Best_dev_score":
            best_dev_score,

        "Median_r":
            abundance_result[
                "Median_r"
            ],

        "1-TVD":
            abundance_result[
                "1-TVD"
            ],

        "Cosine":
            abundance_result[
                "Cosine"
            ],
    }

    # ------------------------------------------------------------------
    # Adsorption metrics for B/C only
    # ------------------------------------------------------------------

    if (
        presence_probability
        is not None
    ):

        adsorption_result = (
            adsorption_metrics(
                prepared.Y_presence_test,
                presence_probability,
                indices=protein_indices,
            )
        )

        result.update(
            {
                "Adsorption_F1":
                    adsorption_result[
                        "F1"
                    ],

                "Adsorption_AUROC":
                    adsorption_result[
                        "AUROC"
                    ],

                "Adsorption_AUPRC":
                    adsorption_result[
                        "AUPRC"
                    ],
            }
        )

    else:

        result.update(
            {
                "Adsorption_F1":
                    np.nan,

                "Adsorption_AUROC":
                    np.nan,

                "Adsorption_AUPRC":
                    np.nan,
            }
        )

    return result


# ======================================================================
# Aggregate results
# ======================================================================


def summarize_results(
    seed_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean and SD across seeds.
    """

    metric_columns = [
        "Median_r",
        "1-TVD",
        "Cosine",
        "Adsorption_F1",
        "Adsorption_AUROC",
        "Adsorption_AUPRC",
    ]

    rows = []

    for variant in VARIANTS:

        subset = seed_results[
            seed_results[
                "Variant"
            ]
            == variant.value
        ]

        row = {
            "Variant":
                variant.value,

            "Model":
                VARIANT_LABELS[
                    variant
                ],

            "N_seeds":
                len(
                    subset
                ),
        }

        for metric in metric_columns:

            values = (
                subset[
                    metric
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            if (
                len(values)
                == 0
            ):

                row[
                    f"{metric}_mean"
                ] = np.nan

                row[
                    f"{metric}_sd"
                ] = np.nan

            else:

                row[
                    f"{metric}_mean"
                ] = float(
                    np.mean(
                        values
                    )
                )

                row[
                    f"{metric}_sd"
                ] = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                    if len(
                        values
                    ) > 1
                    else 0.0
                )

        rows.append(
            row
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
        "ARCHITECTURE ABLATION STUDY"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Architecture table
    # ------------------------------------------------------------------

    architecture_df = (
        ablation_architecture_table()
    )

    architecture_df.to_csv(
        RESULTS_DIR
        / "architecture_table.csv",
        index=False,
    )

    print()

    print(
        architecture_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------

    print()

    print(
        "Preparing data..."
    )

    prepared = (
        prepare_model_data(
            FEATURE_FILE,
            ABUNDANCE_FILE,
        )
    )

    print(
        "Development samples:",
        prepared.X_train.shape[0],
    )

    print(
        "Test samples:",
        prepared.X_test.shape[0],
    )

    print(
        "Input features:",
        prepared.X_train.shape[1],
    )

    print(
        "Individual proteins:",
        len(
            prepared.panel
        ),
    )

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    config = TrainingConfig(
        hidden=(
            320,
            320,
            320,
        ),

        dropout=(
            0.2597530378485485
        ),

        learning_rate=(
            0.0019913470844860467
        ),

        weight_decay=0.0,

        weight_presence=1.0,

        weight_abundance=(
            1.9528320541332875
        ),

        alpha_gate=(
            1.4356053039813828
        ),

        temp_init=(
            0.7623618527415651
        ),

        batch_size=64,

        max_epochs=100,

        patience=10,

        dev_ratio=0.15,
    )

    device = get_device()

    print()

    print(
        "Device:",
        device,
    )

    print(
        "Total training runs:",
        len(
            VARIANTS
        )
        * len(
            SEEDS
        ),
    )

    # ==================================================================
    # Run study
    # ==================================================================

    results = []

    total_runs = (
        len(
            VARIANTS
        )
        * len(
            SEEDS
        )
    )

    run_number = 0

    for variant in VARIANTS:

        print()

        print(
            "=" * 72
        )

        print(
            VARIANT_LABELS[
                variant
            ]
        )

        print(
            "=" * 72
        )

        for seed in SEEDS:

            run_number += 1

            print()

            print(
                f"[{run_number}/{total_runs}] "
                f"Variant={variant.value} "
                f"Seed={seed}"
            )

            result = evaluate_seed(
                variant,
                seed,
                prepared,
                config,
                device,
            )

            results.append(
                result
            )

            print(
                f"  Median_r = "
                f"{result['Median_r']:.4f}"
            )

            print(
                f"  1-TVD    = "
                f"{result['1-TVD']:.4f}"
            )

            print(
                f"  Cosine   = "
                f"{result['Cosine']:.4f}"
            )

            print(
                f"  Best epoch = "
                f"{result['Best_epoch']}"
            )

            # Save incrementally in case a later run fails.
            pd.DataFrame(
                results
            ).to_csv(
                RESULTS_DIR
                / "seed_results.csv",
                index=False,
            )

    # ==================================================================
    # Final seed table
    # ==================================================================

    seed_results = pd.DataFrame(
        results
    )

    seed_results.to_csv(
        RESULTS_DIR
        / "seed_results.csv",
        index=False,
    )

    # ==================================================================
    # Aggregate
    # ==================================================================

    summary = summarize_results(
        seed_results
    )

    summary.to_csv(
        RESULTS_DIR
        / "summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save JSON configuration
    # ------------------------------------------------------------------

    with open(
        RESULTS_DIR
        / "ablation_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "seeds":
                    SEEDS,

                "variants":
                    [
                        variant.value
                        for variant
                        in VARIANTS
                    ],

                "n_runs":
                    total_runs,

                "max_epochs":
                    config.max_epochs,

                "patience":
                    config.patience,

                "dev_ratio":
                    config.dev_ratio,

                "batch_size":
                    config.batch_size,

                "learning_rate":
                    config.learning_rate,

                "weight_presence":
                    config.weight_presence,

                "weight_abundance":
                    config.weight_abundance,
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
        "ABLATION SUMMARY"
    )

    print(
        "=" * 72
    )

    display_columns = [
        "Model",
        "N_seeds",
        "Median_r_mean",
        "Median_r_sd",
        "1-TVD_mean",
        "1-TVD_sd",
        "Cosine_mean",
        "Cosine_sd",
    ]

    print(
        summary[
            display_columns
        ].to_string(
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
        "ABLATION COMPLETE"
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