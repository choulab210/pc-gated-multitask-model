"""
Run Grouped Neural Baselines
============================

Evaluate neural-network baseline architectures using the same
feature-signature grouped cross-validation strategy used for the
gated two-head model.

Models
------
Adsorption:
    - Two-head ungated
    - Single-task NN

Abundance:
    - Two-head ungated
    - Single-task NN

The grouped folds are defined from identical raw input-feature profiles,
matching the grouped-validation analysis used elsewhere in the project.

Outputs
-------
results/grouped_neural_baselines/
    adsorption_fold_results.csv
    abundance_fold_results.csv
    adsorption_summary.csv
    abundance_summary.csv

Run
---
python scripts/run_grouped_neural_baselines.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

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

from pcmodel.training import (
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
    / "grouped_neural_baselines"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_FILE = DATA_DIR / "Data_1.csv"
ABUNDANCE_FILE = DATA_DIR / "Data_2.csv"


# ======================================================================
# Configuration
# ======================================================================

N_SPLITS = 5
SEED = 42

HIDDEN = [
    320,
    320,
    320,
]

DROPOUT = 0.2597530378485485
LEARNING_RATE = 0.0019913470844860467
WEIGHT_DECAY = 0.0

BATCH_SIZE = 64
MAX_EPOCHS = 60

PRESENCE_WEIGHT = 1.0
ABUNDANCE_WEIGHT = 1.9528320541332875


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
# Encoder
# ======================================================================


class Encoder(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden,
        dropout: float,
    ):

        super().__init__()

        layers = []

        current_dim = in_dim

        for hidden_dim in hidden:

            layers.append(
                nn.Linear(
                    current_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.Dropout(
                    dropout
                )
            )

            current_dim = hidden_dim

        self.network = nn.Sequential(
            *layers
        )

        self.out_dim = current_dim

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ======================================================================
# Models
# ======================================================================


class TwoHeadUngated(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden,
        n_outputs: int,
        dropout: float,
    ):

        super().__init__()

        self.encoder = Encoder(
            in_dim,
            hidden,
            dropout,
        )

        self.presence_head = nn.Linear(
            self.encoder.out_dim,
            n_outputs,
        )

        self.abundance_head = nn.Linear(
            self.encoder.out_dim,
            n_outputs,
        )

    def forward(
        self,
        x,
    ):

        representation = self.encoder(
            x
        )

        presence_logits = (
            self.presence_head(
                representation
            )
        )

        abundance_logits = (
            self.abundance_head(
                representation
            )
        )

        abundance_logprob = (
            torch.log_softmax(
                abundance_logits,
                dim=1,
            )
        )

        return (
            presence_logits,
            abundance_logprob,
        )


class AdsorptionOnlyNN(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden,
        n_outputs: int,
        dropout: float,
    ):

        super().__init__()

        self.encoder = Encoder(
            in_dim,
            hidden,
            dropout,
        )

        self.head = nn.Linear(
            self.encoder.out_dim,
            n_outputs,
        )

    def forward(
        self,
        x,
    ):

        representation = self.encoder(
            x
        )

        return self.head(
            representation
        )


class AbundanceOnlyNN(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden,
        n_outputs: int,
        dropout: float,
    ):

        super().__init__()

        self.encoder = Encoder(
            in_dim,
            hidden,
            dropout,
        )

        self.head = nn.Linear(
            self.encoder.out_dim,
            n_outputs,
        )

    def forward(
        self,
        x,
    ):

        representation = self.encoder(
            x
        )

        logits = self.head(
            representation
        )

        return torch.log_softmax(
            logits,
            dim=1,
        )


# ======================================================================
# Helpers
# ======================================================================


def get_train_ids(
    prepared,
    n_expected: int,
):

    candidates = [
        "train_ids",
        "train_np_ids",
        "np_ids_train",
    ]

    for attribute in candidates:

        if hasattr(
            prepared,
            attribute,
        ):

            values = getattr(
                prepared,
                attribute,
            )

            if (
                values is not None
                and len(
                    values
                ) == n_expected
            ):

                return [
                    str(
                        value
                    )
                    for value in values
                ]

    raise RuntimeError(
        "PreparedData does not expose model-development NP IDs."
    )


def make_presence_loader(
    X,
    Y,
):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            Y,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


def make_abundance_loader(
    X,
    Y,
):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            Y,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


def make_twohead_loader(
    X,
    Yp,
    Ya,
):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            Yp,
            dtype=torch.float32,
        ),
        torch.tensor(
            Ya,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


# ======================================================================
# Training
# ======================================================================


def train_ungated(
    X_train,
    Yp_train,
    Ya_train,
    device,
    seed,
):

    set_seed(
        seed
    )

    model = TwoHeadUngated(
        in_dim=X_train.shape[1],
        hidden=HIDDEN,
        n_outputs=Yp_train.shape[1],
        dropout=DROPOUT,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    pos_weight = compute_pos_weight(
        Yp_train,
        device,
    )

    presence_loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    abundance_loss_fn = (
        nn.KLDivLoss(
            reduction="batchmean"
        )
    )

    loader = make_twohead_loader(
        X_train,
        Yp_train,
        Ya_train,
    )

    for _ in range(
        MAX_EPOCHS
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
                PRESENCE_WEIGHT
                * loss_presence
                +
                ABUNDANCE_WEIGHT
                * loss_abundance
            )

            total_loss.backward()

            optimizer.step()

    return model


def train_adsorption_only(
    X_train,
    Y_train,
    device,
    seed,
):

    set_seed(
        seed
    )

    model = AdsorptionOnlyNN(
        in_dim=X_train.shape[1],
        hidden=HIDDEN,
        n_outputs=Y_train.shape[1],
        dropout=DROPOUT,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    pos_weight = compute_pos_weight(
        Y_train,
        device,
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    loader = make_presence_loader(
        X_train,
        Y_train,
    )

    for _ in range(
        MAX_EPOCHS
    ):

        model.train()

        for (
            X_batch,
            Y_batch,
        ) in loader:

            X_batch = X_batch.to(
                device
            )

            Y_batch = Y_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = loss_fn(
                logits,
                Y_batch,
            )

            loss.backward()

            optimizer.step()

    return model


def train_abundance_only(
    X_train,
    Y_train,
    device,
    seed,
):

    set_seed(
        seed
    )

    model = AbundanceOnlyNN(
        in_dim=X_train.shape[1],
        hidden=HIDDEN,
        n_outputs=Y_train.shape[1],
        dropout=DROPOUT,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.KLDivLoss(
        reduction="batchmean"
    )

    loader = make_abundance_loader(
        X_train,
        Y_train,
    )

    for _ in range(
        MAX_EPOCHS
    ):

        model.train()

        for (
            X_batch,
            Y_batch,
        ) in loader:

            X_batch = X_batch.to(
                device
            )

            Y_batch = Y_batch.to(
                device
            )

            optimizer.zero_grad()

            logprob = model(
                X_batch
            )

            loss = loss_fn(
                logprob,
                Y_batch,
            )

            loss.backward()

            optimizer.step()

    return model


# ======================================================================
# Prediction
# ======================================================================


@torch.no_grad()
def predict_ungated(
    model,
    X,
    device,
):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    (
        presence_logits,
        abundance_logprob,
    ) = model(
        tensor
    )

    return (
        torch.sigmoid(
            presence_logits
        )
        .cpu()
        .numpy(),

        torch.exp(
            abundance_logprob
        )
        .cpu()
        .numpy(),
    )


@torch.no_grad()
def predict_presence(
    model,
    X,
    device,
):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    return (
        torch.sigmoid(
            model(
                tensor
            )
        )
        .cpu()
        .numpy()
    )


@torch.no_grad()
def predict_abundance(
    model,
    X,
    device,
):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    return (
        torch.exp(
            model(
                tensor
            )
        )
        .cpu()
        .numpy()
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 78
    )

    print(
        "GROUPED NEURAL BASELINE VALIDATION"
    )

    print(
        "=" * 78
    )

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

    print()

    print(
        "Development samples:",
        n_samples,
    )

    print(
        "Unique feature groups:",
        len(
            np.unique(
                groups
            )
        ),
    )

    device = get_device()

    print(
        "Device:",
        device,
    )

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    adsorption_rows = []

    abundance_rows = []

    # ==================================================================
    # CV
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
            "=" * 78
        )

        print(
            f"FOLD {fold_number}/{N_SPLITS}"
        )

        print(
            "=" * 78
        )

        X_train = X[
            train_index
        ]

        X_val = X[
            validation_index
        ]

        Yp_train = Y_presence[
            train_index
        ]

        Yp_val = Y_presence[
            validation_index
        ]

        Ya_train = Y_abundance[
            train_index
        ]

        Ya_val = Y_abundance[
            validation_index
        ]

        fold_seed = (
            SEED
            + fold_number
        )

        # --------------------------------------------------------------
        # Two-head ungated
        # --------------------------------------------------------------

        print(
            "Training Two-head ungated..."
        )

        ungated = train_ungated(
            X_train,
            Yp_train,
            Ya_train,
            device,
            fold_seed,
        )

        (
            ungated_presence,
            ungated_abundance,
        ) = predict_ungated(
            ungated,
            X_val,
            device,
        )

        ungated_ads = adsorption_metrics(
            Yp_val,
            ungated_presence,
            indices=protein_indices,
        )

        ungated_abun = abundance_metrics(
            Ya_val,
            ungated_abundance,
            indices=protein_indices,
        )

        # --------------------------------------------------------------
        # Adsorption-only NN
        # --------------------------------------------------------------

        print(
            "Training Single-task adsorption NN..."
        )

        adsorption_only = (
            train_adsorption_only(
                X_train,
                Yp_train,
                device,
                fold_seed,
            )
        )

        adsorption_prediction = (
            predict_presence(
                adsorption_only,
                X_val,
                device,
            )
        )

        single_ads = adsorption_metrics(
            Yp_val,
            adsorption_prediction,
            indices=protein_indices,
        )

        # --------------------------------------------------------------
        # Abundance-only NN
        # --------------------------------------------------------------

        print(
            "Training Single-task abundance NN..."
        )

        abundance_only = (
            train_abundance_only(
                X_train,
                Ya_train,
                device,
                fold_seed,
            )
        )

        abundance_prediction = (
            predict_abundance(
                abundance_only,
                X_val,
                device,
            )
        )

        single_abun = abundance_metrics(
            Ya_val,
            abundance_prediction,
            indices=protein_indices,
        )

        # --------------------------------------------------------------
        # Store adsorption
        # --------------------------------------------------------------

        for model_name, metrics in [
            (
                "Two-head ungated",
                ungated_ads,
            ),
            (
                "Single-task NN",
                single_ads,
            ),
        ]:

            adsorption_rows.append(
                {
                    "Fold":
                        fold_number,

                    "Model":
                        model_name,

                    "Acc":
                        metrics["Acc"],

                    "F1":
                        metrics["F1"],

                    "Precision":
                        metrics["Precision"],

                    "Recall":
                        metrics["Recall"],

                    "AUROC":
                        metrics["AUROC"],

                    "AUPRC":
                        metrics["AUPRC"],

                    "MCC":
                        metrics["MCC"],
                }
            )

        # --------------------------------------------------------------
        # Store abundance
        # --------------------------------------------------------------

        for model_name, metrics in [
            (
                "Two-head ungated",
                ungated_abun,
            ),
            (
                "Single-task NN",
                single_abun,
            ),
        ]:

            abundance_rows.append(
                {
                    "Fold":
                        fold_number,

                    "Model":
                        model_name,

                    "Median_r":
                        metrics["Median_r"],

                    "1-TVD":
                        metrics["1-TVD"],

                    "Cosine":
                        metrics["Cosine"],
                }
            )

        pd.DataFrame(
            adsorption_rows
        ).to_csv(
            RESULTS_DIR
            / "adsorption_fold_results.csv",
            index=False,
        )

        pd.DataFrame(
            abundance_rows
        ).to_csv(
            RESULTS_DIR
            / "abundance_fold_results.csv",
            index=False,
        )

    # ==================================================================
    # Summaries
    # ==================================================================

    adsorption_df = pd.DataFrame(
        adsorption_rows
    )

    abundance_df = pd.DataFrame(
        abundance_rows
    )

    adsorption_summary = (
        adsorption_df
        .groupby(
            "Model",
            as_index=False,
        )
        .agg(
            Acc_mean=("Acc", "mean"),
            Acc_sd=("Acc", "std"),
            F1_mean=("F1", "mean"),
            F1_sd=("F1", "std"),
            Precision_mean=("Precision", "mean"),
            Precision_sd=("Precision", "std"),
            Recall_mean=("Recall", "mean"),
            Recall_sd=("Recall", "std"),
            AUROC_mean=("AUROC", "mean"),
            AUROC_sd=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_sd=("AUPRC", "std"),
            MCC_mean=("MCC", "mean"),
            MCC_sd=("MCC", "std"),
        )
    )

    abundance_summary = (
        abundance_df
        .groupby(
            "Model",
            as_index=False,
        )
        .agg(
            Median_r_mean=("Median_r", "mean"),
            Median_r_sd=("Median_r", "std"),
            **{
                "1-TVD_mean": ("1-TVD", "mean"),
                "1-TVD_sd": ("1-TVD", "std"),
            },
            Cosine_mean=("Cosine", "mean"),
            Cosine_sd=("Cosine", "std"),
        )
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

    # ==================================================================
    # Print
    # ==================================================================

    print()

    print(
        "=" * 78
    )

    print(
        "GROUPED ADSORPTION NEURAL BASELINES"
    )

    print(
        "=" * 78
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
        "=" * 78
    )

    print(
        "GROUPED ABUNDANCE NEURAL BASELINES"
    )

    print(
        "=" * 78
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
        "Saved to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()