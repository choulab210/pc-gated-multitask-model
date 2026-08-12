"""
Applicability Domain Utilities
==============================

This module implements the k-nearest-neighbor applicability-domain
analysis used for the protein corona prediction framework.

The applicability domain (AD) is defined from the encoded
model-development feature space.

Procedure
---------
1. For each reference/training sample, calculate the mean Euclidean
   distance to its k nearest OTHER reference samples.
2. Define the AD threshold as a selected percentile of those
   training-reference distances.
3. For a new sample, calculate its mean distance to the k nearest
   reference samples.
4. Samples with distance <= threshold are classified as In-AD.
5. Samples with distance > threshold are classified as Out-of-AD.

Default settings reproduce the original analysis:

    k = 5
    threshold percentile = 85

This module contains domain calculations only.

Model evaluation belongs in metrics.py.
Experiment execution belongs in scripts/run_applicability.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


# ======================================================================
# Defaults
# ======================================================================

DEFAULT_N_NEIGHBORS = 5

DEFAULT_THRESHOLD_PERCENTILE = 85.0

DEFAULT_METRIC = "euclidean"


# ======================================================================
# Result container
# ======================================================================


@dataclass
class ApplicabilityResult:
    """
    Applicability-domain results for a set of query samples.
    """

    distances: np.ndarray

    in_domain: np.ndarray

    threshold: float

    n_neighbors: int

    threshold_percentile: float

    def to_dataframe(
        self,
        sample_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> pd.DataFrame:
        """
        Convert AD results to a dataframe.
        """

        n_samples = len(
            self.distances
        )

        if sample_ids is None:

            sample_ids = [
                f"Sample_{i + 1}"
                for i in range(
                    n_samples
                )
            ]

        sample_ids = list(
            sample_ids
        )

        if (
            len(sample_ids)
            != n_samples
        ):

            raise ValueError(
                "Number of sample IDs does not match "
                "the number of AD results."
            )

        labels = np.where(
            self.in_domain,
            "In-AD",
            "Out-of-AD",
        )

        return pd.DataFrame(
            {
                "Sample_ID":
                    sample_ids,

                "AD_distance":
                    self.distances,

                "AD_threshold":
                    self.threshold,

                "In_AD":
                    self.in_domain,

                "AD_label":
                    labels,
            }
        )


# ======================================================================
# Input validation
# ======================================================================


def _as_numeric_matrix(
    X,
    *,
    name: str,
) -> np.ndarray:
    """
    Convert feature input into a finite 2D float matrix.
    """

    if isinstance(
        X,
        pd.DataFrame,
    ):

        X = X.copy()

        # ----------------------------------------------------------
        # Remove NP_ID when present.
        # ----------------------------------------------------------

        if (
            "NP_ID"
            in X.columns
        ):

            X = X.drop(
                columns=[
                    "NP_ID"
                ]
            )

        X = X.to_numpy(
            dtype=np.float64
        )

    else:

        X = np.asarray(
            X,
            dtype=np.float64,
        )

    if X.ndim != 2:

        raise ValueError(
            f"{name} must be a 2D feature matrix. "
            f"Received shape {X.shape}."
        )

    if (
        X.shape[0] == 0
        or X.shape[1] == 0
    ):

        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return X


# ======================================================================
# Applicability-domain model
# ======================================================================


class ApplicabilityDomain:
    """
    k-nearest-neighbor applicability-domain model.

    Parameters
    ----------
    n_neighbors
        Number of nearest reference samples used to calculate
        the mean distance.

    threshold_percentile
        Percentile of the training-reference distance distribution
        used as the AD threshold.

    metric
        Distance metric used by sklearn NearestNeighbors.
    """

    def __init__(
        self,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        threshold_percentile: float = (
            DEFAULT_THRESHOLD_PERCENTILE
        ),
        metric: str = DEFAULT_METRIC,
    ) -> None:

        if n_neighbors < 1:

            raise ValueError(
                "n_neighbors must be >= 1."
            )

        if not (
            0.0
            < threshold_percentile
            <= 100.0
        ):

            raise ValueError(
                "threshold_percentile must be "
                "between 0 and 100."
            )

        self.n_neighbors = int(
            n_neighbors
        )

        self.threshold_percentile = float(
            threshold_percentile
        )

        self.metric = metric

        self.reference_X_: Optional[
            np.ndarray
        ] = None

        self.training_distances_: Optional[
            np.ndarray
        ] = None

        self.threshold_: Optional[
            float
        ] = None

        self.n_features_in_: Optional[
            int
        ] = None

    # ==================================================================
    # Fit
    # ==================================================================

    def fit(
        self,
        X_reference,
    ) -> "ApplicabilityDomain":
        """
        Fit the applicability domain using reference/training samples.

        For each reference sample, its own self-distance is excluded.
        The mean distance to the next k nearest reference samples is
        calculated.

        The AD threshold is then defined from the requested percentile
        of this training-reference distance distribution.
        """

        X_reference = _as_numeric_matrix(
            X_reference,
            name="X_reference",
        )

        n_samples = (
            X_reference.shape[0]
        )

        if (
            n_samples
            <= self.n_neighbors
        ):

            raise ValueError(
                "Reference dataset must contain more samples "
                "than n_neighbors. "
                f"Received {n_samples} samples and "
                f"k={self.n_neighbors}."
            )

        self.reference_X_ = (
            X_reference.copy()
        )

        self.n_features_in_ = int(
            X_reference.shape[1]
        )

        # ----------------------------------------------------------
        # k + 1 because every training sample is its own nearest
        # neighbor at distance zero.
        # ----------------------------------------------------------

        neighbor_model = (
            NearestNeighbors(
                n_neighbors=(
                    self.n_neighbors
                    + 1
                ),
                metric=self.metric,
            )
        )

        neighbor_model.fit(
            X_reference
        )

        distances, _ = (
            neighbor_model.kneighbors(
                X_reference
            )
        )

        # ----------------------------------------------------------
        # Distances are returned in ascending order.
        #
        # Drop exactly one nearest entry corresponding to the sample
        # itself. This remains valid even when duplicate samples exist,
        # because only one self-neighbor should be removed.
        # ----------------------------------------------------------

        neighbor_distances = (
            distances[
                :,
                1:
                self.n_neighbors
                + 1
            ]
        )

        training_distances = (
            neighbor_distances.mean(
                axis=1
            )
        )

        threshold = float(
            np.percentile(
                training_distances,
                self.threshold_percentile,
            )
        )

        self.training_distances_ = (
            training_distances
        )

        self.threshold_ = (
            threshold
        )

        return self

    # ==================================================================
    # Internal fitted check
    # ==================================================================

    def _check_fitted(
        self,
    ) -> None:
        """
        Confirm that fit() has already been called.
        """

        if (
            self.reference_X_
            is None
            or self.threshold_
            is None
            or self.training_distances_
            is None
        ):

            raise RuntimeError(
                "ApplicabilityDomain has not been fitted. "
                "Call fit(X_reference) first."
            )

    # ==================================================================
    # Query distances
    # ==================================================================

    def distances(
        self,
        X_query,
    ) -> np.ndarray:
        """
        Calculate mean distance from each query sample to its
        k nearest reference samples.
        """

        self._check_fitted()

        X_query = _as_numeric_matrix(
            X_query,
            name="X_query",
        )

        if (
            X_query.shape[1]
            != self.n_features_in_
        ):

            raise ValueError(
                "Feature dimension mismatch. "
                f"Reference data contain "
                f"{self.n_features_in_} features, "
                f"but query data contain "
                f"{X_query.shape[1]}."
            )

        neighbor_model = (
            NearestNeighbors(
                n_neighbors=(
                    self.n_neighbors
                ),
                metric=self.metric,
            )
        )

        neighbor_model.fit(
            self.reference_X_
        )

        distances, _ = (
            neighbor_model.kneighbors(
                X_query
            )
        )

        mean_distances = (
            distances.mean(
                axis=1
            )
        )

        return mean_distances

    # ==================================================================
    # Predict labels
    # ==================================================================

    def predict(
        self,
        X_query,
    ) -> np.ndarray:
        """
        Return boolean In-AD labels.

        True  = In-AD
        False = Out-of-AD
        """

        query_distances = (
            self.distances(
                X_query
            )
        )

        return (
            query_distances
            <= self.threshold_
        )

    # ==================================================================
    # Full evaluation
    # ==================================================================

    def evaluate(
        self,
        X_query,
    ) -> ApplicabilityResult:
        """
        Calculate query distances and AD labels.
        """

        query_distances = (
            self.distances(
                X_query
            )
        )

        in_domain = (
            query_distances
            <= self.threshold_
        )

        return ApplicabilityResult(
            distances=query_distances,
            in_domain=in_domain,
            threshold=float(
                self.threshold_
            ),
            n_neighbors=self.n_neighbors,
            threshold_percentile=(
                self.threshold_percentile
            ),
        )

    # ==================================================================
    # Summary
    # ==================================================================

    def summary(
        self,
    ) -> Dict[str, float]:
        """
        Return the fitted AD configuration and reference statistics.
        """

        self._check_fitted()

        return {
            "n_reference_samples":
                int(
                    len(
                        self.reference_X_
                    )
                ),

            "n_features":
                int(
                    self.n_features_in_
                ),

            "n_neighbors":
                int(
                    self.n_neighbors
                ),

            "threshold_percentile":
                float(
                    self.threshold_percentile
                ),

            "threshold":
                float(
                    self.threshold_
                ),

            "training_distance_mean":
                float(
                    np.mean(
                        self.training_distances_
                    )
                ),

            "training_distance_median":
                float(
                    np.median(
                        self.training_distances_
                    )
                ),

            "training_distance_min":
                float(
                    np.min(
                        self.training_distances_
                    )
                ),

            "training_distance_max":
                float(
                    np.max(
                        self.training_distances_
                    )
                ),
        }


# ======================================================================
# Convenience function
# ======================================================================


def fit_applicability_domain(
    X_reference,
    *,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    threshold_percentile: float = (
        DEFAULT_THRESHOLD_PERCENTILE
    ),
    metric: str = DEFAULT_METRIC,
) -> ApplicabilityDomain:
    """
    Convenience function that creates and fits an AD model.
    """

    model = ApplicabilityDomain(
        n_neighbors=n_neighbors,
        threshold_percentile=(
            threshold_percentile
        ),
        metric=metric,
    )

    model.fit(
        X_reference
    )

    return model