"""
Run Grouped Cross-Validation
============================

Evaluate the two-head gated protein-corona model using grouped
cross-validation.

Samples with identical NP feature profiles are assigned to the same
group. GroupKFold ensures that a feature-signature group never appears
in both training and validation folds.

This analysis evaluates generalization under a stricter split than the
original random NP-level held-out test split.

Run from project root:

    python scripts/run_grouped_validation.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from pcmodel.ablation import (
    AblationModelConfig,
    AblationVariant,
    build_ablation_model,
)

from pcmodel.data import (
    FEATURE_COLS,
    prepare_model_data,
)

from pcmodel.grouping import (
    build_feature_signature_groups,
    summarize_groups,
)

from pcmodel.metrics import (
    abundance_metrics,
    adsorption_metrics,
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

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "grouped_validation"
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
# Loader
# ======================================================================


def make_loader(
    X,
    Y_presence,
    Y_abundance,
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
# Train one grouped fold
# ======================================================================


def train_fold(
    X_train,
    Yp_train,
    Ya_train,
    config: TrainingConfig,
    device: torch.device,
    seed: int,
):
    """
    Train the full gated two-head architecture for one grouped fold.

    This grouped-validation implementation uses the same final
    hyperparameters and a fixed maximum training budget.
    """

    set_seed(
        seed
    )

    model_config = (
        AblationModelConfig(
            in_dim=X_train.shape[1],

            hidden=list(
                config.hidden
            ),

            n_outputs=(
                Yp_train.shape[1]
            ),

            dropout=config.dropout,

            alpha_gate=(
                config.alpha_gate
            ),

            temp_init=(
                config.temp_init
            ),

            stopgrad_gate=True,
        )
    )

    model = (
        build_ablation_model(
            AblationVariant.TWO_HEAD_GATED,
            model_config,
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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

    abundance_loss_fn = (
        nn.KLDivLoss(
            reduction="batchmean"
        )
    )

    loader = make_loader(
        X_train,
        Yp_train,
        Ya_train,
        batch_size=config.batch_size,
    )

    # --------------------------------------------------------------
    # Use a fixed training budget for grouped CV.
    #
    # This avoids using validation-fold information for early stopping.
    # --------------------------------------------------------------

    n_epochs = min(
        config.max_epochs,
        60,
    )

    for _ in range(
        n_epochs
    ):

        model.train()

        for (
            X_batch,
            Yp_batch,
            Ya_batch,
        ) in loader:

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

    return model


# ======================================================================
# Prediction
# ======================================================================


@torch.no_grad()
def predict_model(
    model,
    X,
    device,
):
    """
    Generate presence probabilities and abundance predictions.
    """

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
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
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "GROUPED VALIDATION"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Reconstruct original model-development dataset
    # ------------------------------------------------------------------

    prepared = prepare_model_data(
        FEATURE_FILE,
        ABUNDANCE_FILE,
    )

    # ------------------------------------------------------------------
    # Important:
    #
    # Grouped validation should use the 317 development samples,
    # not the original 80 held-out test samples.
    # ------------------------------------------------------------------

    X = np.asarray(
        prepared.X_train,
        dtype=np.float32,
    )

    Y_presence = np.asarray(
        prepared.Y_presence_train,
        dtype=np.float32,
    )

    Y_abundance = np.asarray(
        prepared.Y_abundance_train,
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Recover development NP IDs.
    #
    # The PreparedData object should retain train IDs. If not, we
    # reconstruct them from the original feature table by using the
    # deterministic train split information.
    # ------------------------------------------------------------------

    possible_id_attributes = [
        "train_ids",
        "train_np_ids",
        "np_ids_train",
    ]

    train_ids = None

    for attribute in (
        possible_id_attributes
    ):

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
                and len(
                    candidate
                ) == len(
                    X
                )
            ):

                train_ids = [
                    str(value)
                    for value
                    in candidate
                ]

                break

    if train_ids is None:

        raise RuntimeError(
            "PreparedData does not expose model-development NP IDs. "
            "Add train IDs to PreparedData before running grouped "
            "validation."
        )

    # ------------------------------------------------------------------
    # Raw feature table
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

    development_feature_df = (
        raw_features[
            raw_features[
                "NP_ID"
            ].isin(
                train_ids
            )
        ]
        .copy()
    )

    # Preserve exact development order.
    development_feature_df = (
        development_feature_df
        .set_index(
            "NP_ID"
        )
        .loc[
            train_ids
        ]
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Construct identical-feature groups
    # ------------------------------------------------------------------

    group_table = (
        build_feature_signature_groups(
            development_feature_df,
            FEATURE_COLS,
        )
    )

    group_summary = (
        summarize_groups(
            group_table
        )
    )

    groups = (
        group_table[
            "group_id"
        ]
        .to_numpy()
    )

    print()

    print(
        "Development samples:",
        len(
            X
        ),
    )

    print(
        "Unique feature groups:",
        group_summary[
            "n_unique_groups"
        ],
    )

    print(
        "Repeated groups:",
        group_summary[
            "n_repeated_groups"
        ],
    )

    print(
        "Samples in repeated groups:",
        group_summary[
            "n_samples_in_repeated_groups"
        ],
    )

    print(
        "Fraction in repeated groups:",
        f"{group_summary['fraction_samples_in_repeated_groups']:.3f}",
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

    # ------------------------------------------------------------------
    # GroupKFold
    # ------------------------------------------------------------------

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    protein_indices = list(
        range(
            len(
                prepared.panel
            )
        )
    )

    fold_rows = []

    prediction_rows = []

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

        fold_train_groups = set(
            groups[
                train_index
            ]
        )

        fold_validation_groups = set(
            groups[
                validation_index
            ]
        )

        shared_groups = (
            fold_train_groups
            & fold_validation_groups
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

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------

        model = train_fold(
            X[
                train_index
            ],

            Y_presence[
                train_index
            ],

            Y_abundance[
                train_index
            ],

            config,

            device,

            seed=(
                RANDOM_SEED
                + fold_number
            ),
        )

        # --------------------------------------------------------------
        # Predict validation fold
        # --------------------------------------------------------------

        (
            presence_probability,
            abundance_prediction,
        ) = predict_model(
            model,
            X[
                validation_index
            ],
            device,
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        adsorption = (
            adsorption_metrics(
                Y_presence[
                    validation_index
                ],
                presence_probability,
                indices=protein_indices,
            )
        )

        abundance = (
            abundance_metrics(
                Y_abundance[
                    validation_index
                ],
                abundance_prediction,
                indices=protein_indices,
            )
        )

        fold_result = {
            "Fold":
                fold_number,

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

            "N_train_groups":
                int(
                    len(
                        fold_train_groups
                    )
                ),

            "N_validation_groups":
                int(
                    len(
                        fold_validation_groups
                    )
                ),

            "Shared_groups":
                0,

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

        fold_rows.append(
            fold_result
        )

        print()

        print(
            f"F1       : "
            f"{fold_result['F1']:.4f}"
        )

        print(
            f"AUROC    : "
            f"{fold_result['AUROC']:.4f}"
        )

        print(
            f"Median_r : "
            f"{fold_result['Median_r']:.4f}"
        )

        print(
            f"1-TVD    : "
            f"{fold_result['1-TVD']:.4f}"
        )

        print(
            f"Cosine   : "
            f"{fold_result['Cosine']:.4f}"
        )

        # --------------------------------------------------------------
        # Save fold assignments
        # --------------------------------------------------------------

        for local_index, global_index in enumerate(
            validation_index
        ):

            prediction_rows.append(
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

        # Incremental save
        pd.DataFrame(
            fold_rows
        ).to_csv(
            RESULTS_DIR
            / "fold_results.csv",
            index=False,
        )

    # ==================================================================
    # Aggregate summary
    # ==================================================================

    fold_results = pd.DataFrame(
        fold_rows
    )

    metric_columns = [
        "Acc",
        "F1",
        "Precision",
        "Recall",
        "AUROC",
        "AUPRC",
        "MCC",
        "Median_r",
        "1-TVD",
        "Cosine",
    ]

    summary_rows = []

    for metric in metric_columns:

        values = (
            fold_results[
                metric
            ]
            .to_numpy(
                dtype=float
            )
        )

        summary_rows.append(
            {
                "Metric":
                    metric,

                "Mean":
                    float(
                        np.nanmean(
                            values
                        )
                    ),

                "SD":
                    float(
                        np.nanstd(
                            values,
                            ddof=1,
                        )
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ==================================================================
    # Save
    # ==================================================================

    group_table.to_csv(
        RESULTS_DIR
        / "feature_signature_groups.csv",
        index=False,
    )

    fold_results.to_csv(
        RESULTS_DIR
        / "fold_results.csv",
        index=False,
    )

    pd.DataFrame(
        prediction_rows
    ).to_csv(
        RESULTS_DIR
        / "fold_assignments.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "summary.csv",
        index=False,
    )

    with open(
        RESULTS_DIR
        / "group_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            group_summary,
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
        "GROUPED VALIDATION SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        summary_df.to_string(
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