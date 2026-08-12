"""
Conventional Benchmark Models
=============================

Reusable conventional machine-learning benchmarks for comparison with
the two-head protein corona model.

Adsorption benchmarks
---------------------
- Random Forest
- Logistic Regression
- XGBoost

Abundance benchmarks
--------------------
- Random Forest
- Ridge Regression
- XGBoost

Each protein is modeled independently because outputs correspond to
different proteins.

For abundance predictions:
    1. Negative predictions are clipped to zero.
    2. Predictions are normalized within each NP so the abundance
       profile sums to one.

This module defines models and helper functions only.
Experiment execution belongs in scripts/run_benchmarks.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from sklearn.base import clone

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)


# ======================================================================
# Defaults
# ======================================================================

DEFAULT_RANDOM_STATE = 42

EPS = 1e-12


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class BenchmarkConfig:
    """
    Conventional benchmark hyperparameters.
    """

    random_state: int = DEFAULT_RANDOM_STATE

    # Random Forest
    rf_n_estimators: int = 500

    rf_max_depth: Optional[int] = None

    rf_min_samples_leaf: int = 1

    # Logistic Regression
    logistic_c: float = 1.0

    logistic_max_iter: int = 2000

    # Ridge
    ridge_alpha: float = 1.0

    # XGBoost
    xgb_n_estimators: int = 300

    xgb_max_depth: int = 4

    xgb_learning_rate: float = 0.05

    xgb_subsample: float = 0.8

    xgb_colsample_bytree: float = 0.8


# ======================================================================
# Constant classifier fallback
# ======================================================================


class ConstantProbabilityClassifier:
    """
    Fallback for proteins containing only one observed class.
    """

    def __init__(
        self,
        probability: float,
    ) -> None:

        self.probability = float(
            probability
        )

    def predict_proba(
        self,
        X,
    ) -> np.ndarray:

        n = len(
            X
        )

        positive = np.full(
            n,
            self.probability,
            dtype=float,
        )

        negative = (
            1.0
            - positive
        )

        return np.column_stack(
            [
                negative,
                positive,
            ]
        )


# ======================================================================
# Adsorption model factories
# ======================================================================


def make_adsorption_model(
    name: str,
    config: BenchmarkConfig,
):
    """
    Construct one adsorption benchmark estimator.
    """

    name = name.lower()

    if name == "random_forest":

        return RandomForestClassifier(
            n_estimators=(
                config.rf_n_estimators
            ),
            max_depth=(
                config.rf_max_depth
            ),
            min_samples_leaf=(
                config.rf_min_samples_leaf
            ),
            class_weight="balanced",
            random_state=(
                config.random_state
            ),
            n_jobs=-1,
        )

    if name == "logistic_regression":

        return LogisticRegression(
            C=config.logistic_c,
            max_iter=(
                config.logistic_max_iter
            ),
            class_weight="balanced",
            solver="liblinear",
            random_state=(
                config.random_state
            ),
        )

    if name == "xgboost":

        return XGBClassifier(
            n_estimators=(
                config.xgb_n_estimators
            ),
            max_depth=(
                config.xgb_max_depth
            ),
            learning_rate=(
                config.xgb_learning_rate
            ),
            subsample=(
                config.xgb_subsample
            ),
            colsample_bytree=(
                config.xgb_colsample_bytree
            ),
            objective=(
                "binary:logistic"
            ),
            eval_metric="logloss",
            random_state=(
                config.random_state
            ),
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown adsorption model: {name}"
    )


# ======================================================================
# Abundance model factories
# ======================================================================


def make_abundance_model(
    name: str,
    config: BenchmarkConfig,
):
    """
    Construct one abundance benchmark estimator.
    """

    name = name.lower()

    if name == "random_forest":

        return RandomForestRegressor(
            n_estimators=(
                config.rf_n_estimators
            ),
            max_depth=(
                config.rf_max_depth
            ),
            min_samples_leaf=(
                config.rf_min_samples_leaf
            ),
            random_state=(
                config.random_state
            ),
            n_jobs=-1,
        )

    if name == "ridge":

        return Ridge(
            alpha=(
                config.ridge_alpha
            )
        )

    if name == "xgboost":

        return XGBRegressor(
            n_estimators=(
                config.xgb_n_estimators
            ),
            max_depth=(
                config.xgb_max_depth
            ),
            learning_rate=(
                config.xgb_learning_rate
            ),
            subsample=(
                config.xgb_subsample
            ),
            colsample_bytree=(
                config.xgb_colsample_bytree
            ),
            objective="reg:squarederror",
            random_state=(
                config.random_state
            ),
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown abundance model: {name}"
    )


# ======================================================================
# Fit adsorption models
# ======================================================================


def fit_adsorption_models(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    model_name: str,
    config: BenchmarkConfig,
):
    """
    Train one binary classifier for each protein.
    """

    models = []

    for protein_index in range(
        Y_train.shape[1]
    ):

        y = (
            Y_train[
                :,
                protein_index
            ]
            .astype(int)
        )

        unique = np.unique(
            y
        )

        # ----------------------------------------------------------
        # Single-class protein
        # ----------------------------------------------------------

        if (
            len(unique)
            < 2
        ):

            model = (
                ConstantProbabilityClassifier(
                    float(
                        unique[0]
                    )
                )
            )

        else:

            estimator = (
                make_adsorption_model(
                    model_name,
                    config,
                )
            )

            model = clone(
                estimator
            )

            model.fit(
                X_train,
                y,
            )

        models.append(
            model
        )

    return models


# ======================================================================
# Predict adsorption probabilities
# ======================================================================


def predict_adsorption_probabilities(
    models,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate protein-specific adsorption probabilities.
    """

    predictions = []

    for model in models:

        probability = (
            model.predict_proba(
                X
            )[
                :,
                1
            ]
        )

        predictions.append(
            probability
        )

    return np.column_stack(
        predictions
    )


# ======================================================================
# Fit abundance models
# ======================================================================


def fit_abundance_models(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    model_name: str,
    config: BenchmarkConfig,
):
    """
    Train one regression model for each abundance output.
    """

    models = []

    for protein_index in range(
        Y_train.shape[1]
    ):

        estimator = (
            make_abundance_model(
                model_name,
                config,
            )
        )

        model = clone(
            estimator
        )

        model.fit(
            X_train,
            Y_train[
                :,
                protein_index
            ],
        )

        models.append(
            model
        )

    return models


# ======================================================================
# Predict abundance
# ======================================================================


def predict_abundance_distribution(
    models,
    X: np.ndarray,
    *,
    eps: float = EPS,
) -> np.ndarray:
    """
    Generate normalized abundance distributions.
    """

    predictions = []

    for model in models:

        prediction = (
            model.predict(
                X
            )
        )

        predictions.append(
            prediction
        )

    matrix = np.column_stack(
        predictions
    ).astype(
        float
    )

    # --------------------------------------------------------------
    # Relative abundance cannot be negative.
    # --------------------------------------------------------------

    matrix = np.clip(
        matrix,
        0.0,
        None,
    )

    # --------------------------------------------------------------
    # Normalize each NP profile.
    # --------------------------------------------------------------

    row_sum = matrix.sum(
        axis=1,
        keepdims=True,
    )

    zero_rows = (
        row_sum.flatten()
        <= eps
    )

    if np.any(
        zero_rows
    ):

        # Uniform fallback for pathological all-zero predictions.
        matrix[
            zero_rows,
            :
        ] = 1.0

        row_sum = matrix.sum(
            axis=1,
            keepdims=True,
        )

    return (
        matrix
        / row_sum
    )


# ======================================================================
# Available models
# ======================================================================


ADSORPTION_BENCHMARKS = {
    "random_forest":
        "Random Forest",

    "logistic_regression":
        "Logistic Regression",

    "xgboost":
        "XGBoost",
}


ABUNDANCE_BENCHMARKS = {
    "random_forest":
        "Random Forest",

    "ridge":
        "Ridge Regression",

    "xgboost":
        "XGBoost",
}