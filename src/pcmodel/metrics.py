"""
Protein Corona Model Evaluation Metrics
=======================================

This module contains reusable performance metrics for evaluating:

1. Protein adsorption/presence prediction.
2. Quantitative protein corona composition prediction.
3. Hyperparameter optimization and early-stopping criteria.

The metric definitions are based on the original two-head model notebook
and are centralized here so that final training, ablation studies,
cross-validation, external validation, and benchmark models all use
the same calculations.

Main conventions
----------------
Adsorption:
    - Binary prediction threshold = 0.5.
    - AUROC and AUPRC are calculated separately for each protein and
      averaged across proteins with both positive and negative observations.
    - F1, precision, and recall use macro averaging across proteins.
    - Accuracy is calculated after flattening all protein/sample labels.
    - MCC is calculated per protein and averaged across valid proteins.

Abundance:
    - Pearson correlation is calculated separately for each protein
      across NP samples.
    - Overall Pearson performance is summarized by the median correlation.
    - Cosine similarity is calculated per NP and averaged across samples.
    - 1-TVD is calculated from normalized abundance distributions.
    - Category-level 1-TVD follows the original notebook and includes
      samples whose true abundance for the category exceeds 1%.

This module calculates metrics only. Plotting belongs in plotting.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from scipy.stats import pearsonr

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================
# Constants
# ======================================================================

DEFAULT_THRESHOLD = 0.5

EPS = 1e-12

# Original notebook threshold used for category-level 1-TVD.
TVD_TRUE_SUM_THRESHOLD = 0.01

# Composite-score weights used in the original notebook.
W_AUROC = 0.5
W_ONE_MINUS_TVD = 0.5


# ======================================================================
# Validation helpers
# ======================================================================


def _validate_same_shape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Verify that observed and predicted arrays have identical shape.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Observed and predicted arrays must have the same shape. "
            f"Received {y_true.shape} and {y_pred.shape}."
        )


def _subset_columns(
    array: np.ndarray,
    indices: Optional[Sequence[int]],
) -> np.ndarray:
    """
    Select output columns when an index subset is supplied.
    """

    if indices is None:
        return array

    indices = list(indices)

    return array[:, indices]


# ======================================================================
# Adsorption metrics
# ======================================================================


def per_protein_auroc_auprc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Calculate AUROC and AUPRC separately for each protein.

    Proteins with only one observed class are assigned NaN because
    AUROC/AUPRC are not meaningfully defined for those proteins.
    """

    _validate_same_shape(
        y_true,
        y_prob,
    )

    n_outputs = y_true.shape[1]

    auroc = np.full(
        n_outputs,
        np.nan,
        dtype=float,
    )

    auprc = np.full(
        n_outputs,
        np.nan,
        dtype=float,
    )

    for j in range(n_outputs):

        yt = y_true[:, j]
        yp = y_prob[:, j]

        if len(np.unique(yt)) < 2:
            continue

        try:

            auroc[j] = roc_auc_score(
                yt,
                yp,
            )

            auprc[j] = average_precision_score(
                yt,
                yp,
            )

        except ValueError:
            continue

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
    }


def per_protein_mcc(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
) -> np.ndarray:
    """
    Calculate MCC separately for each protein.

    To reproduce the original notebook, proteins with only one observed
    class are excluded and assigned NaN.
    """

    _validate_same_shape(
        y_true,
        y_pred_binary,
    )

    n_outputs = y_true.shape[1]

    mcc = np.full(
        n_outputs,
        np.nan,
        dtype=float,
    )

    for j in range(n_outputs):

        yt = y_true[:, j]
        yh = y_pred_binary[:, j]

        if len(np.unique(yt)) < 2:
            continue

        mcc[j] = matthews_corrcoef(
            yt,
            yh,
        )

    return mcc


def adsorption_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Calculate manuscript-level adsorption prediction metrics.

    Parameters
    ----------
    y_true
        Binary observed adsorption/detection labels with shape
        (n_samples, n_proteins).

    y_prob
        Predicted adsorption probabilities with the same shape.

    threshold
        Probability threshold used to generate binary predictions.

    indices
        Optional protein-column indices. This is useful for excluding
        OTHER or evaluating a functional protein category.

    Returns
    -------
    Dictionary containing:
        N, Acc, F1, Precision, Recall, AUROC, AUPRC, MCC
    """

    _validate_same_shape(
        y_true,
        y_prob,
    )

    y_true_sub = _subset_columns(
        np.asarray(y_true),
        indices,
    )

    y_prob_sub = _subset_columns(
        np.asarray(y_prob),
        indices,
    )

    if y_true_sub.shape[1] == 0:

        return {
            "N": 0,
            "Acc": np.nan,
            "F1": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "MCC": np.nan,
        }

    y_pred = (
        y_prob_sub
        >= threshold
    ).astype(int)

    probability_metrics = (
        per_protein_auroc_auprc(
            y_true_sub,
            y_prob_sub,
        )
    )

    mcc_values = per_protein_mcc(
        y_true_sub,
        y_pred,
    )

    return {
        "N": int(
            y_true_sub.shape[1]
        ),

        # Original notebook calculated accuracy after flattening.
        "Acc": float(
            accuracy_score(
                y_true_sub.flatten(),
                y_pred.flatten(),
            )
        ),

        "F1": float(
            f1_score(
                y_true_sub,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "Precision": float(
            precision_score(
                y_true_sub,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "Recall": float(
            recall_score(
                y_true_sub,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "AUROC": float(
            np.nanmean(
                probability_metrics[
                    "AUROC"
                ]
            )
        ),

        "AUPRC": float(
            np.nanmean(
                probability_metrics[
                    "AUPRC"
                ]
            )
        ),

        "MCC": float(
            np.nanmean(
                mcc_values
            )
        ),
    }


# ======================================================================
# Metrics used during hyperparameter optimization
# ======================================================================


def adsorption_hpo_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculate adsorption metrics used by the original training notebook.

    Includes macro/micro F1 because those values were used for diagnostic
    evaluation during model development.
    """

    _validate_same_shape(
        y_true,
        y_prob,
    )

    probability_metrics = (
        per_protein_auroc_auprc(
            y_true,
            y_prob,
        )
    )

    y_pred = (
        y_prob
        >= threshold
    ).astype(int)

    return {
        "macro_auroc": float(
            np.nanmean(
                probability_metrics[
                    "AUROC"
                ]
            )
        ),

        "macro_auprc": float(
            np.nanmean(
                probability_metrics[
                    "AUPRC"
                ]
            )
        ),

        "macro_f1@0.5": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "micro_f1@0.5": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),

        "acc@0.5": float(
            accuracy_score(
                y_true.flatten(),
                y_pred.flatten(),
            )
        ),
    }


# ======================================================================
# Abundance metrics
# ======================================================================


def normalize_distributions(
    values: np.ndarray,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """
    Normalize each row to sum to one.

    Small values are clipped to eps to reproduce the numerical behavior
    of the original notebook during global abundance evaluation.
    """

    values = np.asarray(
        values,
        dtype=float,
    ).copy()

    values = np.clip(
        values,
        eps,
        None,
    )

    row_sum = values.sum(
        axis=1,
        keepdims=True,
    )

    row_sum = np.clip(
        row_sum,
        eps,
        None,
    )

    return (
        values
        / row_sum
    )


def per_sample_cosine(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """
    Calculate cosine similarity for each NP sample.
    """

    _validate_same_shape(
        y_true,
        y_pred,
    )

    numerator = (
        y_true
        * y_pred
    ).sum(
        axis=1
    )

    denominator = np.clip(
        np.linalg.norm(
            y_true,
            axis=1,
        )
        * np.linalg.norm(
            y_pred,
            axis=1,
        ),
        eps,
        None,
    )

    return (
        numerator
        / denominator
    )


def per_sample_one_minus_tvd(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate 1-TVD for each NP sample.

    TVD = 0.5 * sum(|observed - predicted|)
    """

    _validate_same_shape(
        y_true,
        y_pred,
    )

    tvd = (
        0.5
        * np.abs(
            y_true
            - y_pred
        ).sum(
            axis=1
        )
    )

    return (
        1.0
        - tvd
    )


def per_protein_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_std: float = 1e-8,
) -> np.ndarray:
    """
    Calculate Pearson correlation for each protein across NP samples.

    Proteins without sufficient observed or predicted variation are
    assigned NaN.
    """

    _validate_same_shape(
        y_true,
        y_pred,
    )

    n_outputs = (
        y_true.shape[1]
    )

    correlations = np.full(
        n_outputs,
        np.nan,
        dtype=float,
    )

    for j in range(
        n_outputs
    ):

        yt = y_true[:, j]
        yp = y_pred[:, j]

        if (
            yt.std() <= min_std
            or yp.std() <= min_std
        ):
            continue

        correlations[j], _ = pearsonr(
            yt,
            yp,
        )

    return correlations


def abundance_distribution_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate global abundance metrics used during model optimization.

    This reproduces the original notebook's use of normalized
    distributions, mean cosine similarity, and mean L1 distance.
    """

    _validate_same_shape(
        y_true,
        y_pred,
    )

    true_norm = normalize_distributions(
        y_true
    )

    pred_norm = normalize_distributions(
        y_pred
    )

    cosine_values = (
        per_sample_cosine(
            true_norm,
            pred_norm,
        )
    )

    l1_values = np.abs(
        true_norm
        - pred_norm
    ).sum(
        axis=1
    )

    mean_l1 = float(
        np.mean(
            l1_values
        )
    )

    return {
        "mean_cosine": float(
            np.mean(
                cosine_values
            )
        ),

        "mean_L1": mean_l1,

        "one_minus_tvd": float(
            max(
                0.0,
                1.0
                - mean_l1 / 2.0,
            )
        ),
    }


def abundance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    indices: Optional[Sequence[int]] = None,
    pearson_values: Optional[np.ndarray] = None,
    true_sum_threshold: float = TVD_TRUE_SUM_THRESHOLD,
    eps: float = EPS,
) -> Dict[str, float]:
    """
    Calculate manuscript-level abundance metrics.

    This reproduces the category-level logic of the original notebook.

    Pearson:
        Median per-protein Pearson correlation.

    1-TVD:
        For the requested protein subset, true and predicted abundances
        are normalized within that subset. Samples are included when the
        true abundance of the subset exceeds 1%.

    Cosine:
        Per-sample cosine similarity for the selected protein subset,
        averaged across all NP samples.
    """

    _validate_same_shape(
        y_true,
        y_pred,
    )

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if indices is None:

        indices = list(
            range(
                y_true.shape[1]
            )
        )

    else:

        indices = list(
            indices
        )

    if len(indices) == 0:

        return {
            "N": 0,
            "Median_r": np.nan,
            "1-TVD": np.nan,
            "Cosine": np.nan,
        }

    true_subset = (
        y_true[
            :,
            indices
        ]
    )

    pred_subset = (
        y_pred[
            :,
            indices
        ]
    )

    # --------------------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------------------

    if pearson_values is None:

        pearson_values = (
            per_protein_pearson(
                y_true,
                y_pred,
            )
        )

    subset_r = (
        pearson_values[
            indices
        ]
    )

    valid_r = subset_r[
        ~np.isnan(
            subset_r
        )
    ]

    median_r = (
        float(
            np.median(
                valid_r
            )
        )
        if len(valid_r)
        else np.nan
    )

    # --------------------------------------------------------------
    # 1-TVD
    # --------------------------------------------------------------

    true_sum = (
        true_subset.sum(
            axis=1
        )
    )

    pred_sum = (
        pred_subset.sum(
            axis=1
        )
    )

    valid_mask = (
        (true_sum > true_sum_threshold)
        & (pred_sum > eps)
    )

    if valid_mask.any():

        true_valid = (
            true_subset[
                valid_mask
            ]
        )

        pred_valid = (
            pred_subset[
                valid_mask
            ]
        )

        true_norm = (
            true_valid
            / true_sum[
                valid_mask
            ][:, None]
        )

        pred_norm = (
            pred_valid
            / pred_sum[
                valid_mask
            ][:, None]
        )

        one_minus_tvd = (
            per_sample_one_minus_tvd(
                true_norm,
                pred_norm,
            )
        )

        mean_one_minus_tvd = float(
            np.mean(
                one_minus_tvd
            )
        )

    else:

        mean_one_minus_tvd = np.nan

    # --------------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------------

    cosine_values = (
        per_sample_cosine(
            true_subset,
            pred_subset,
            eps=eps,
        )
    )

    mean_cosine = float(
        np.mean(
            cosine_values
        )
    )

    return {
        "N": int(
            len(indices)
        ),

        "Median_r": median_r,

        "1-TVD": mean_one_minus_tvd,

        "Cosine": mean_cosine,
    }


# ======================================================================
# Composite model-selection score
# ======================================================================


def composite_score(
    adsorption_results: Dict[str, float],
    abundance_results: Dict[str, float],
    *,
    w_auroc: float = W_AUROC,
    w_one_minus_tvd: float = W_ONE_MINUS_TVD,
) -> float:
    """
    Calculate the composite score used for hyperparameter optimization
    and early stopping in the original notebook.

    Score =
        0.5 * macro AUROC
        +
        0.5 * (1 - mean L1 / 2)
    """

    auroc = adsorption_results.get(
        "macro_auroc",
        np.nan,
    )

    mean_l1 = abundance_results.get(
        "mean_L1",
        np.nan,
    )

    if np.isnan(
        auroc
    ):
        auroc = 0.0

    if np.isnan(
        mean_l1
    ):
        mean_l1 = 2.0

    one_minus_tvd = max(
        0.0,
        1.0
        - mean_l1 / 2.0,
    )

    return float(
        w_auroc
        * auroc
        +
        w_one_minus_tvd
        * one_minus_tvd
    )