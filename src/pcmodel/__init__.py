"""
Protein Corona Model Training
=============================

This module contains reusable PyTorch training utilities for the two-head
protein corona prediction model.

Responsibilities
----------------
- Reproducibility and device selection.
- Protein-specific class-weight calculation.
- BCE + KL multitask loss calculation.
- Adam optimization.
- Model prediction.
- Internal development-set splitting.
- Early stopping using the composite adsorption/abundance score.

This module does NOT:
- Load or preprocess data -> data.py
- Define model architecture -> models.py
- Define performance metrics -> metrics.py
- Run a specific experiment -> scripts/

The current implementation is designed to remain compatible with the
original two-head notebook during the initial refactoring stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import copy
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset

from pcmodel.models import TwoHead

from pcmodel.metrics import (
    adsorption_hpo_metrics,
    abundance_distribution_metrics,
    composite_score,
)


# ======================================================================
# Default settings
# ======================================================================

DEFAULT_SEED = 42

DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_DEV_RATIO = 0.15


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class TrainingConfig:
    """
    Configuration for training one two-head model.

    The defaults below correspond to the best configuration identified
    in the original notebook and are retained as a reproducibility
    reference during refactoring.
    """

    hidden: Tuple[int, ...] = (
        320,
        320,
        320,
    )

    dropout: float = 0.2597530378485485

    learning_rate: float = 0.0019913470844860467

    weight_decay: float = 0.0

    weight_presence: float = 1.0

    weight_abundance: float = 1.9528320541332875

    alpha_gate: float = 1.4356053039813828

    temp_init: float = 0.7623618527415651

    batch_size: int = DEFAULT_BATCH_SIZE

    max_epochs: int = DEFAULT_MAX_EPOCHS

    patience: int = DEFAULT_PATIENCE

    dev_ratio: float = DEFAULT_DEV_RATIO


@dataclass
class TrainingResult:
    """
    Results returned by final model training.
    """

    model: TwoHead

    best_score: float

    best_epoch: int

    history: List[Dict[str, float]]

    train_indices: np.ndarray

    dev_indices: np.ndarray

    config: TrainingConfig


# ======================================================================
# Reproducibility
# ======================================================================


def set_seed(
    seed: int = DEFAULT_SEED,
) -> None:
    """
    Set random seeds for reproducible model training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def get_device() -> torch.device:
    """
    Select GPU when available; otherwise use CPU.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ======================================================================
# Class weights
# ======================================================================


def compute_pos_weight(
    y_presence: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate protein-specific positive-class weights.

    Weight for each protein:

        n_negative / n_positive

    with a minimum value of 1.

    This reproduces the class-weighting strategy used in the
    original notebook.
    """

    y_presence = np.asarray(
        y_presence,
        dtype=np.float32,
    )

    n_positive = (
        y_presence
        .sum(axis=0)
        .clip(min=1)
    )

    n_negative = (
        (1.0 - y_presence)
        .sum(axis=0)
        .clip(min=1)
    )

    weights = (
        n_negative
        / n_positive
    ).clip(
        min=1.0
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


# ======================================================================
# Prediction
# ======================================================================


@torch.no_grad()
def predict_probabilities(
    model: TwoHead,
    X: np.ndarray,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate adsorption probabilities and abundance distributions.

    Returns
    -------
    presence_probability
        Sigmoid-transformed adsorption probabilities.

    abundance_distribution
        Predicted relative abundance distributions.
    """

    if device is None:
        device = get_device()

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

    abundance_distribution = (
        torch.exp(
            abundance_logprob
        )
        .cpu()
        .numpy()
    )

    return (
        presence_probability,
        abundance_distribution,
    )


# ======================================================================
# Trainer
# ======================================================================


class TwoHeadTrainer:
    """
    Train the two-head model using class-weighted BCE and KL divergence.
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        config: TrainingConfig,
        pos_weight: torch.Tensor,
        device: torch.device,
    ) -> None:

        self.device = device

        self.config = config

        # ----------------------------------------------------------
        # Model
        # ----------------------------------------------------------

        self.model = TwoHead(
            in_dim=input_dim,
            hidden=list(
                config.hidden
            ),
            k=n_outputs,
            dropout=config.dropout,
            alpha_gate=config.alpha_gate,
            temp_init=config.temp_init,
            stopgrad_gate=True,
        ).to(
            device
        )

        # ----------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=(
                config.weight_decay
            ),
        )

        # ----------------------------------------------------------
        # Presence loss
        # ----------------------------------------------------------

        self.presence_loss = (
            nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction="mean",
            )
        )

        # ----------------------------------------------------------
        # Abundance loss
        # ----------------------------------------------------------

        self.abundance_loss = (
            nn.KLDivLoss(
                reduction="batchmean"
            )
        )

    def step(
        self,
        X_batch: torch.Tensor,
        Y_presence_batch: torch.Tensor,
        Y_abundance_batch: torch.Tensor,
        *,
        train: bool = True,
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """
        Perform one training or evaluation step.
        """

        if train:

            self.model.train()

            self.optimizer.zero_grad()

        else:

            self.model.eval()

        (
            presence_logits,
            abundance_logprob,
        ) = self.model(
            X_batch
        )

        # ----------------------------------------------------------
        # Presence loss
        # ----------------------------------------------------------

        loss_presence = (
            self.presence_loss(
                presence_logits,
                Y_presence_batch,
            )
        )

        # ----------------------------------------------------------
        # Abundance loss
        # ----------------------------------------------------------

        loss_abundance = (
            self.abundance_loss(
                abundance_logprob,
                Y_abundance_batch,
            )
        )

        # ----------------------------------------------------------
        # Combined multitask loss
        # ----------------------------------------------------------

        total_loss = (
            self.config.weight_presence
            * loss_presence

            +

            self.config.weight_abundance
            * loss_abundance
        )

        if train:

            total_loss.backward()

            self.optimizer.step()

        return (
            float(
                total_loss.item()
            ),
            float(
                loss_presence.item()
            ),
            float(
                loss_abundance.item()
            ),
        )


# ======================================================================
# DataLoader
# ======================================================================


def make_training_loader(
    X: np.ndarray,
    Y_presence: np.ndarray,
    Y_abundance: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.
    """

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    Y_presence_tensor = torch.tensor(
        Y_presence,
        dtype=torch.float32,
    )

    Y_abundance_tensor = torch.tensor(
        Y_abundance,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        Y_presence_tensor,
        Y_abundance_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ======================================================================
# Final model training
# ======================================================================


def train_with_early_stopping(
    X: np.ndarray,
    Y_presence: np.ndarray,
    Y_abundance: np.ndarray,
    *,
    config: Optional[
        TrainingConfig
    ] = None,
    seed: int = DEFAULT_SEED,
    device: Optional[
        torch.device
    ] = None,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train the two-head model using an internal development subset.

    The model-development data are divided into:

        training portion
            Used for gradient-based model fitting.

        development portion
            Used only for early stopping.

    The held-out external test set must NOT be supplied to this function.
    """

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    if config is None:

        config = TrainingConfig()

    if device is None:

        device = get_device()

    set_seed(
        seed
    )

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    Y_presence = np.asarray(
        Y_presence,
        dtype=np.float32,
    )

    Y_abundance = np.asarray(
        Y_abundance,
        dtype=np.float32,
    )

    if not (
        len(X)
        == len(Y_presence)
        == len(Y_abundance)
    ):

        raise ValueError(
            "X, Y_presence, and Y_abundance must contain "
            "the same number of samples."
        )

    if (
        Y_presence.shape[1]
        != Y_abundance.shape[1]
    ):

        raise ValueError(
            "The current compatibility model requires the "
            "presence and abundance heads to have the same "
            "number of outputs."
        )

    # --------------------------------------------------------------
    # Internal train/dev split
    # --------------------------------------------------------------

    all_indices = np.arange(
        len(X)
    )

    (
        train_indices,
        dev_indices,
    ) = train_test_split(
        all_indices,
        test_size=config.dev_ratio,
        random_state=seed,
    )

    X_train = (
        X[
            train_indices
        ]
    )

    X_dev = (
        X[
            dev_indices
        ]
    )

    Yp_train = (
        Y_presence[
            train_indices
        ]
    )

    Yp_dev = (
        Y_presence[
            dev_indices
        ]
    )

    Ya_train = (
        Y_abundance[
            train_indices
        ]
    )

    Ya_dev = (
        Y_abundance[
            dev_indices
        ]
    )

    # --------------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------------

    pos_weight = compute_pos_weight(
        Yp_train,
        device,
    )

    # --------------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------------

    train_loader = (
        make_training_loader(
            X_train,
            Yp_train,
            Ya_train,
            batch_size=(
                config.batch_size
            ),
            shuffle=True,
        )
    )

    # --------------------------------------------------------------
    # Initialize trainer
    # --------------------------------------------------------------

    trainer = TwoHeadTrainer(
        input_dim=X.shape[1],
        n_outputs=Y_presence.shape[1],
        config=config,
        pos_weight=pos_weight,
        device=device,
    )

    # --------------------------------------------------------------
    # Early-stopping state
    # --------------------------------------------------------------

    best_score = -np.inf

    best_epoch = 0

    best_state = None

    no_improvement = 0

    history: List[
        Dict[str, float]
    ] = []

    # --------------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------------

    for epoch in range(
        1,
        config.max_epochs + 1,
    ):

        train_losses = []

        presence_losses = []

        abundance_losses = []

        # ----------------------------------------------------------
        # Training pass
        # ----------------------------------------------------------

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

            (
                total_loss,
                presence_loss,
                abundance_loss,
            ) = trainer.step(
                X_batch,
                Yp_batch,
                Ya_batch,
                train=True,
            )

            train_losses.append(
                total_loss
            )

            presence_losses.append(
                presence_loss
            )

            abundance_losses.append(
                abundance_loss
            )

        # ----------------------------------------------------------
        # Development prediction
        # ----------------------------------------------------------

        (
            dev_presence_prob,
            dev_abundance_pred,
        ) = predict_probabilities(
            trainer.model,
            X_dev,
            device=device,
        )

        # ----------------------------------------------------------
        # Development metrics
        # ----------------------------------------------------------

        adsorption_results = (
            adsorption_hpo_metrics(
                Yp_dev,
                dev_presence_prob,
            )
        )

        abundance_results = (
            abundance_distribution_metrics(
                Ya_dev,
                dev_abundance_pred,
            )
        )

        score = composite_score(
            adsorption_results,
            abundance_results,
        )

        # ----------------------------------------------------------
        # Record history
        # ----------------------------------------------------------

        epoch_record = {
            "epoch": float(
                epoch
            ),

            "train_loss": float(
                np.mean(
                    train_losses
                )
            ),

            "presence_loss": float(
                np.mean(
                    presence_losses
                )
            ),

            "abundance_loss": float(
                np.mean(
                    abundance_losses
                )
            ),

            "dev_composite": float(
                score
            ),

            "dev_auroc": float(
                adsorption_results[
                    "macro_auroc"
                ]
            ),

            "dev_auprc": float(
                adsorption_results[
                    "macro_auprc"
                ]
            ),

            "dev_cosine": float(
                abundance_results[
                    "mean_cosine"
                ]
            ),
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------

        if (
            score
            > best_score
            + 1e-6
        ):

            best_score = score

            best_epoch = epoch

            no_improvement = 0

            best_state = copy.deepcopy(
                trainer.model.state_dict()
            )

        else:

            no_improvement += 1

        if verbose:

            print(
                f"[Epoch {epoch:03d}] "
                f"Loss={epoch_record['train_loss']:.4f} | "
                f"AUROC={epoch_record['dev_auroc']:.3f} | "
                f"AUPRC={epoch_record['dev_auprc']:.3f} | "
                f"Cosine={epoch_record['dev_cosine']:.3f} | "
                f"Composite={score:.3f}"
            )

        if (
            no_improvement
            >= config.patience
        ):

            if verbose:

                print(
                    f"[EARLY STOP] Epoch {epoch}. "
                    f"Best epoch = {best_epoch}."
                )

            break

    # --------------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "Training completed without a valid model checkpoint."
        )

    trainer.model.load_state_dict(
        best_state
    )

    if verbose:

        print(
            f"[TRAINING COMPLETE] "
            f"Best dev composite = {best_score:.4f} "
            f"at epoch {best_epoch}."
        )

    return TrainingResult(
        model=trainer.model,
        best_score=float(
            best_score
        ),
        best_epoch=int(
            best_epoch
        ),
        history=history,
        train_indices=(
            train_indices
        ),
        dev_indices=(
            dev_indices
        ),
        config=config,
    )