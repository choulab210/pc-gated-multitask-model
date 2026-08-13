"""
Run Neural Baselines on the Held-Out Test Set
=============================================

This script evaluates neural-network baseline architectures using the
same model-development/test split and preprocessing as the final model.

Models
------
Adsorption:
    - Two-head without gating
    - Single-task adsorption NN

Abundance:
    - Two-head without gating
    - Single-task abundance NN

The final gated two-head model is already stored in the benchmark files,
so this script focuses on the missing neural baselines needed for Table 3.

Outputs
-------
results/neural_baselines/
    adsorption_metrics.csv
    abundance_metrics.csv

Run:
    python scripts/run_neural_baselines.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from pcmodel.data import prepare_model_data
from pcmodel.metrics import adsorption_metrics, abundance_metrics
from pcmodel.training import compute_pos_weight, get_device


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "neural_baselines"
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

MAX_EPOCHS = 100

PRESENCE_WEIGHT = 1.0

ABUNDANCE_WEIGHT = 1.9528320541332875


# ======================================================================
# Reproducibility
# ======================================================================


def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ======================================================================
# Shared encoder
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

    def forward(self, x):

        return self.network(x)


# ======================================================================
# Two-head without gating
# ======================================================================


class TwoHeadNoGate(nn.Module):

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

    def forward(self, x):

        representation = self.encoder(x)

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


# ======================================================================
# Single-task adsorption
# ======================================================================


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

    def forward(self, x):

        representation = self.encoder(x)

        return self.head(
            representation
        )


# ======================================================================
# Single-task abundance
# ======================================================================


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

    def forward(self, x):

        representation = self.encoder(x)

        logits = self.head(
            representation
        )

        return torch.log_softmax(
            logits,
            dim=1,
        )


# ======================================================================
# Data loader
# ======================================================================


def make_loader(
    X,
    Y_presence,
    Y_abundance,
):

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
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


# ======================================================================
# Train two-head without gating
# ======================================================================


def train_twohead_no_gate(
    X_train,
    Yp_train,
    Ya_train,
    device,
):

    set_seed(SEED)

    model = TwoHeadNoGate(
        in_dim=X_train.shape[1],
        hidden=HIDDEN,
        n_outputs=Yp_train.shape[1],
        dropout=DROPOUT,
    ).to(device)

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

    loader = make_loader(
        X_train,
        Yp_train,
        Ya_train,
    )

    for _ in range(MAX_EPOCHS):

        model.train()

        for (
            X_batch,
            Yp_batch,
            Ya_batch,
        ) in loader:

            X_batch = X_batch.to(device)
            Yp_batch = Yp_batch.to(device)
            Ya_batch = Ya_batch.to(device)

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


# ======================================================================
# Train adsorption-only
# ======================================================================


def train_adsorption_only(
    X_train,
    Yp_train,
    device,
):

    set_seed(SEED)

    model = AdsorptionOnlyNN(
        in_dim=X_train.shape[1],
        hidden=HIDDEN,
        n_outputs=Yp_train.shape[1],
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    pos_weight = compute_pos_weight(
        Yp_train,
        device,
    )

    loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            Yp_train,
            dtype=torch.float32,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for _ in range(MAX_EPOCHS):

        model.train()

        for (
            X_batch,
            Y_batch,
        ) in loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

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


# ======================================================================
# Train abundance-only
# ======================================================================


def train_abundance_only(
    X_train,
    Ya_train,
    device,
):

    set_seed(SEED)

    model = AbundanceOnlyNN(
        in_dim=X_train.shape[1],
        hidden=HIDDEN,
        n_outputs=Ya_train.shape[1],
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.KLDivLoss(
        reduction="batchmean"
    )

    dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            Ya_train,
            dtype=torch.float32,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for _ in range(MAX_EPOCHS):

        model.train()

        for (
            X_batch,
            Y_batch,
        ) in loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

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
def predict_twohead(
    model,
    X,
    device,
):

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


@torch.no_grad()
def predict_adsorption(
    model,
    X,
    device,
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    logits = model(
        X_tensor
    )

    return (
        torch.sigmoid(
            logits
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

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    logprob = model(
        X_tensor
    )

    return (
        torch.exp(
            logprob
        )
        .cpu()
        .numpy()
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "NEURAL BASELINE EVALUATION"
    )

    print(
        "=" * 72
    )

    prepared = prepare_model_data(
        FEATURE_FILE,
        ABUNDANCE_FILE,
    )

    X_train = np.asarray(
        prepared.X_train,
        dtype=np.float32,
    )

    X_test = np.asarray(
        prepared.X_test,
        dtype=np.float32,
    )

    Yp_train = np.asarray(
        prepared.Y_presence_train,
        dtype=float,
    )

    Yp_test = np.asarray(
        prepared.Y_presence_test,
        dtype=float,
    )

    Ya_train = np.asarray(
        prepared.Y_abundance_train,
        dtype=float,
    )

    Ya_test = np.asarray(
        prepared.Y_abundance_test,
        dtype=float,
    )

    n_proteins = len(
        prepared.panel
    )

    protein_indices = list(
        range(
            n_proteins
        )
    )

    device = get_device()

    print()

    print(
        "Device:",
        device,
    )

    print(
        "Development samples:",
        len(
            X_train
        ),
    )

    print(
        "Held-out samples:",
        len(
            X_test
        ),
    )

    # ==================================================================
    # Two-head without gating
    # ==================================================================

    print()

    print(
        "Training two-head without gating..."
    )

    twohead = train_twohead_no_gate(
        X_train,
        Yp_train,
        Ya_train,
        device,
    )

    (
        twohead_presence,
        twohead_abundance,
    ) = predict_twohead(
        twohead,
        X_test,
        device,
    )

    twohead_adsorption_metrics = (
        adsorption_metrics(
            Yp_test,
            twohead_presence,
            indices=protein_indices,
        )
    )

    twohead_abundance_metrics = (
        abundance_metrics(
            Ya_test,
            twohead_abundance,
            indices=protein_indices,
        )
    )

    # ==================================================================
    # Single-task adsorption
    # ==================================================================

    print(
        "Training adsorption-only NN..."
    )

    adsorption_model = (
        train_adsorption_only(
            X_train,
            Yp_train,
            device,
        )
    )

    adsorption_prediction = (
        predict_adsorption(
            adsorption_model,
            X_test,
            device,
        )
    )

    adsorption_only_metrics = (
        adsorption_metrics(
            Yp_test,
            adsorption_prediction,
            indices=protein_indices,
        )
    )

    # ==================================================================
    # Single-task abundance
    # ==================================================================

    print(
        "Training abundance-only NN..."
    )

    abundance_model = (
        train_abundance_only(
            X_train,
            Ya_train,
            device,
        )
    )

    abundance_prediction = (
        predict_abundance(
            abundance_model,
            X_test,
            device,
        )
    )

    abundance_only_metrics = (
        abundance_metrics(
            Ya_test,
            abundance_prediction,
            indices=protein_indices,
        )
    )

    # ==================================================================
    # Save adsorption
    # ==================================================================

    adsorption_df = pd.DataFrame(
        [
            {
                "Model":
                    "Two-head without gating",

                "Acc":
                    twohead_adsorption_metrics[
                        "Acc"
                    ],

                "F1":
                    twohead_adsorption_metrics[
                        "F1"
                    ],

                "Precision":
                    twohead_adsorption_metrics[
                        "Precision"
                    ],

                "Recall":
                    twohead_adsorption_metrics[
                        "Recall"
                    ],

                "AUROC":
                    twohead_adsorption_metrics[
                        "AUROC"
                    ],

                "AUPRC":
                    twohead_adsorption_metrics[
                        "AUPRC"
                    ],

                "MCC":
                    twohead_adsorption_metrics[
                        "MCC"
                    ],
            },

            {
                "Model":
                    "Single-task NN",

                "Acc":
                    adsorption_only_metrics[
                        "Acc"
                    ],

                "F1":
                    adsorption_only_metrics[
                        "F1"
                    ],

                "Precision":
                    adsorption_only_metrics[
                        "Precision"
                    ],

                "Recall":
                    adsorption_only_metrics[
                        "Recall"
                    ],

                "AUROC":
                    adsorption_only_metrics[
                        "AUROC"
                    ],

                "AUPRC":
                    adsorption_only_metrics[
                        "AUPRC"
                    ],

                "MCC":
                    adsorption_only_metrics[
                        "MCC"
                    ],
            },
        ]
    )

    adsorption_df.to_csv(
        RESULTS_DIR
        / "adsorption_metrics.csv",
        index=False,
    )

    # ==================================================================
    # Save abundance
    # ==================================================================

    abundance_df = pd.DataFrame(
        [
            {
                "Model":
                    "Two-head without gating",

                "Median_r":
                    twohead_abundance_metrics[
                        "Median_r"
                    ],

                "1-TVD":
                    twohead_abundance_metrics[
                        "1-TVD"
                    ],

                "Cosine":
                    twohead_abundance_metrics[
                        "Cosine"
                    ],
            },

            {
                "Model":
                    "Single-task NN",

                "Median_r":
                    abundance_only_metrics[
                        "Median_r"
                    ],

                "1-TVD":
                    abundance_only_metrics[
                        "1-TVD"
                    ],

                "Cosine":
                    abundance_only_metrics[
                        "Cosine"
                    ],
            },
        ]
    )

    abundance_df.to_csv(
        RESULTS_DIR
        / "abundance_metrics.csv",
        index=False,
    )

    # ==================================================================
    # Console
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "ADSORPTION NEURAL BASELINES"
    )

    print(
        "=" * 72
    )

    print(
        adsorption_df.to_string(
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
        "ABUNDANCE NEURAL BASELINES"
    )

    print(
        "=" * 72
    )

    print(
        abundance_df.to_string(
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