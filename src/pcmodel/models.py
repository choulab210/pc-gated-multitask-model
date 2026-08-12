"""
Protein Corona Prediction Models
================================

This module contains the reusable PyTorch neural-network architectures used
for joint prediction of protein adsorption and quantitative protein corona
composition.

The initial implementation was refactored from the original Jupyter notebook
used for development of the two-head protein corona model. During the initial
refactoring stage, the goal is to preserve the behavior of the original model
while separating reusable model definitions from data preprocessing, training,
evaluation, and visualization code.

Main components
---------------
MLP
    Shared multilayer perceptron encoder used to learn representations from
    nanoparticle physicochemical properties and experimental conditions.

TwoHead
    Multitask neural-network model containing:
    1. a protein adsorption/presence head,
    2. a protein abundance/composition head, and
    3. an adsorption-guided gating mechanism that modifies abundance
       predictions according to predicted protein adsorption.

Important
---------
- This file should contain model architecture definitions only.
- Data preprocessing belongs in ``data.py``.
- Training and loss calculations belong in ``training.py``.
- Performance metrics belong in ``metrics.py``.
- Experimental workflows should be implemented in the ``scripts/`` directory.
- During refactoring, changes to model architecture should be made only after
  confirming that the modular implementation reproduces the original notebook
  results.

Future model variants, such as abundance-only and two-head models without
adsorption-guided gating, can also be defined in this module for architecture
ablation studies.
"""


from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Shared multilayer perceptron encoder."""

    def __init__(
        self,
        in_dim: int,
        widths: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        d = in_dim

        for w in widths:
            layers.extend([
                nn.Linear(d, w),
                nn.ReLU(),
            ])

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            d = w

        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoHead(nn.Module):
    """Two-head model for protein adsorption and corona abundance."""

    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        k: int,
        dropout: float,
        alpha_gate: float = 2.0,
        temp_init: float = 1.0,
        stopgrad_gate: bool = True,
    ):
        super().__init__()

        # Shared encoder
        self.enc = MLP(
            in_dim=in_dim,
            widths=hidden,
            dropout=dropout,
        )

        hidden_dim = self.enc.out_dim

        # Protein adsorption head
        self.pres = nn.Linear(hidden_dim, k)

        # Protein abundance head
        self.abun = nn.Linear(hidden_dim, k)

        # Trainable gating strength
        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_gate))
        )

        # Trainable softmax temperature
        self.temp = nn.Parameter(
            torch.tensor(float(temp_init))
        )

        self.stopgrad_gate = stopgrad_gate

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Shared representation
        z = self.enc(x)

        # Two output heads
        pres_logits = self.pres(z)
        abund_logits = self.abun(z)

        # Adsorption-guided gating
        if self.stopgrad_gate:
            gate = torch.sigmoid(
                self.alpha * pres_logits.detach()
            )
        else:
            gate = torch.sigmoid(
                self.alpha * pres_logits
            )

        # Apply gate before abundance normalization
        masked_logits = (
            abund_logits
            + torch.log(gate + 1e-8)
        )

        # Temperature-scaled abundance distribution
        temperature = self.temp.clamp(
            0.5,
            5.0,
        )

        abund_logprob = F.log_softmax(
            masked_logits / temperature,
            dim=1,
        )

        return pres_logits, abund_logprob