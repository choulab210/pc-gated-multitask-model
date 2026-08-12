"""
Ablation Model Architectures
============================

This module defines neural-network architectures used to evaluate the
contribution of multitask learning and adsorption-guided gating.

Three model variants are implemented:

A. AbundanceOnly
   - Shared-style MLP encoder.
   - Abundance head only.
   - No adsorption task.
   - No gating.

B. TwoHeadNoGate
   - Shared MLP encoder.
   - Adsorption head + abundance head.
   - Both tasks are trained jointly.
   - Adsorption predictions do NOT modify abundance predictions.

C. TwoHeadGated
   - The current final model.
   - Shared MLP encoder.
   - Adsorption head + abundance head.
   - Adsorption probabilities guide abundance prediction through the
     gating mechanism.

The purpose of this ablation is to separate:

    A -> B:
        contribution of multitask learning

    B -> C:
        contribution of explicit adsorption-guided gating

All architectures are designed to use the same hidden-layer structure,
dropout, temperature parameterization, optimizer settings, and training
budget wherever applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcmodel.models import (
    MLP,
    TwoHead,
)


# ======================================================================
# Ablation variant names
# ======================================================================


class AblationVariant(str, Enum):
    """
    Available ablation architectures.
    """

    ABUNDANCE_ONLY = "abundance_only"

    TWO_HEAD_NO_GATE = "two_head_no_gate"

    TWO_HEAD_GATED = "two_head_gated"


# ======================================================================
# Human-readable labels
# ======================================================================


VARIANT_LABELS = {
    AblationVariant.ABUNDANCE_ONLY:
        "Abundance-only",

    AblationVariant.TWO_HEAD_NO_GATE:
        "Two-head without gating",

    AblationVariant.TWO_HEAD_GATED:
        "Two-head with gating",
}


# ======================================================================
# Configuration container
# ======================================================================


@dataclass
class AblationModelConfig:
    """
    Architecture settings shared by ablation models.
    """

    in_dim: int

    hidden: List[int]

    n_outputs: int

    dropout: float

    alpha_gate: float = 2.0

    temp_init: float = 1.0

    stopgrad_gate: bool = True


# ======================================================================
# A. Abundance-only model
# ======================================================================


class AbundanceOnly(nn.Module):
    """
    Abundance-only neural network.

    Architecture
    ------------
    Input
        ↓
    MLP encoder
        ↓
    abundance logits
        ↓
    temperature scaling
        ↓
    log-softmax
        ↓
    abundance distribution

    There is no adsorption head and no gating mechanism.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        n_outputs: int,
        dropout: float,
        temp_init: float = 1.0,
    ) -> None:

        super().__init__()

        # ----------------------------------------------------------
        # Encoder
        # ----------------------------------------------------------

        self.enc = MLP(
            in_dim,
            hidden,
            dropout,
        )

        hidden_dim = (
            self.enc.out_dim
        )

        # ----------------------------------------------------------
        # Abundance head
        # ----------------------------------------------------------

        self.abun = nn.Linear(
            hidden_dim,
            n_outputs,
        )

        # ----------------------------------------------------------
        # Trainable temperature
        # ----------------------------------------------------------

        self.temp = nn.Parameter(
            torch.tensor(
                float(
                    temp_init
                )
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return abundance log probabilities.
        """

        z = self.enc(
            x
        )

        abundance_logits = (
            self.abun(
                z
            )
        )

        temperature = (
            self.temp.clamp(
                0.5,
                5.0,
            )
        )

        abundance_logprob = (
            F.log_softmax(
                abundance_logits
                / temperature,
                dim=1,
            )
        )

        return abundance_logprob


# ======================================================================
# B. Two-head model without gating
# ======================================================================


class TwoHeadNoGate(nn.Module):
    """
    Two-head multitask model without adsorption-guided gating.

    Architecture
    ------------
                         ┌─ adsorption head
    Input -> encoder ----|
                         └─ abundance head

    Both tasks share the encoder, but the adsorption output does not
    constrain abundance prediction.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        n_outputs: int,
        dropout: float,
        temp_init: float = 1.0,
    ) -> None:

        super().__init__()

        # ----------------------------------------------------------
        # Shared encoder
        # ----------------------------------------------------------

        self.enc = MLP(
            in_dim,
            hidden,
            dropout,
        )

        hidden_dim = (
            self.enc.out_dim
        )

        # ----------------------------------------------------------
        # Adsorption head
        # ----------------------------------------------------------

        self.pres = nn.Linear(
            hidden_dim,
            n_outputs,
        )

        # ----------------------------------------------------------
        # Abundance head
        # ----------------------------------------------------------

        self.abun = nn.Linear(
            hidden_dim,
            n_outputs,
        )

        # ----------------------------------------------------------
        # Trainable temperature
        # ----------------------------------------------------------

        self.temp = nn.Parameter(
            torch.tensor(
                float(
                    temp_init
                )
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Return:

            presence_logits
            abundance_logprob
        """

        z = self.enc(
            x
        )

        presence_logits = (
            self.pres(
                z
            )
        )

        abundance_logits = (
            self.abun(
                z
            )
        )

        temperature = (
            self.temp.clamp(
                0.5,
                5.0,
            )
        )

        abundance_logprob = (
            F.log_softmax(
                abundance_logits
                / temperature,
                dim=1,
            )
        )

        return (
            presence_logits,
            abundance_logprob,
        )


# ======================================================================
# C. Gated model
# ======================================================================


class TwoHeadGated(TwoHead):
    """
    Alias/subclass for the final adsorption-guided TwoHead model.

    This architecture is intentionally inherited directly from the
    production model in models.py so that the gated ablation condition
    uses exactly the same implementation as the final model.
    """

    pass


# ======================================================================
# Factory
# ======================================================================


def build_ablation_model(
    variant: AblationVariant | str,
    config: AblationModelConfig,
) -> nn.Module:
    """
    Construct one ablation architecture.

    Parameters
    ----------
    variant
        One of:

            abundance_only
            two_head_no_gate
            two_head_gated

    config
        Shared model architecture configuration.

    Returns
    -------
    torch.nn.Module
        Requested model.
    """

    if isinstance(
        variant,
        str,
    ):

        variant = AblationVariant(
            variant
        )

    # ------------------------------------------------------------------
    # A. Abundance only
    # ------------------------------------------------------------------

    if (
        variant
        == AblationVariant.ABUNDANCE_ONLY
    ):

        return AbundanceOnly(
            in_dim=config.in_dim,
            hidden=list(
                config.hidden
            ),
            n_outputs=(
                config.n_outputs
            ),
            dropout=config.dropout,
            temp_init=(
                config.temp_init
            ),
        )

    # ------------------------------------------------------------------
    # B. Two-head without gate
    # ------------------------------------------------------------------

    if (
        variant
        == AblationVariant.TWO_HEAD_NO_GATE
    ):

        return TwoHeadNoGate(
            in_dim=config.in_dim,
            hidden=list(
                config.hidden
            ),
            n_outputs=(
                config.n_outputs
            ),
            dropout=config.dropout,
            temp_init=(
                config.temp_init
            ),
        )

    # ------------------------------------------------------------------
    # C. Final gated two-head model
    # ------------------------------------------------------------------

    if (
        variant
        == AblationVariant.TWO_HEAD_GATED
    ):

        return TwoHeadGated(
            in_dim=config.in_dim,
            hidden=list(
                config.hidden
            ),

            # TwoHead uses "k" instead of n_outputs.
            k=config.n_outputs,

            dropout=config.dropout,

            alpha_gate=(
                config.alpha_gate
            ),

            temp_init=(
                config.temp_init
            ),

            stopgrad_gate=(
                config.stopgrad_gate
            ),
        )

    raise ValueError(
        f"Unsupported ablation variant: {variant}"
    )


# ======================================================================
# Variant properties
# ======================================================================


def has_presence_head(
    variant: AblationVariant | str,
) -> bool:
    """
    Return whether the architecture contains an adsorption head.
    """

    if isinstance(
        variant,
        str,
    ):

        variant = AblationVariant(
            variant
        )

    return (
        variant
        != AblationVariant.ABUNDANCE_ONLY
    )


def uses_gating(
    variant: AblationVariant | str,
) -> bool:
    """
    Return whether the architecture uses adsorption-guided gating.
    """

    if isinstance(
        variant,
        str,
    ):

        variant = AblationVariant(
            variant
        )

    return (
        variant
        == AblationVariant.TWO_HEAD_GATED
    )


# ======================================================================
# Architecture summary
# ======================================================================


def ablation_architecture_table():
    """
    Return a manuscript-friendly description of the three variants.
    """

    import pandas as pd

    return pd.DataFrame(
        [
            {
                "Variant":
                    "A",

                "Model":
                    VARIANT_LABELS[
                        AblationVariant.ABUNDANCE_ONLY
                    ],

                "Shared_encoder":
                    True,

                "Adsorption_head":
                    False,

                "Abundance_head":
                    True,

                "Multitask_training":
                    False,

                "Adsorption_guided_gating":
                    False,
            },

            {
                "Variant":
                    "B",

                "Model":
                    VARIANT_LABELS[
                        AblationVariant.TWO_HEAD_NO_GATE
                    ],

                "Shared_encoder":
                    True,

                "Adsorption_head":
                    True,

                "Abundance_head":
                    True,

                "Multitask_training":
                    True,

                "Adsorption_guided_gating":
                    False,
            },

            {
                "Variant":
                    "C",

                "Model":
                    VARIANT_LABELS[
                        AblationVariant.TWO_HEAD_GATED
                    ],

                "Shared_encoder":
                    True,

                "Adsorption_head":
                    True,

                "Abundance_head":
                    True,

                "Multitask_training":
                    True,

                "Adsorption_guided_gating":
                    True,
            },
        ]
    )