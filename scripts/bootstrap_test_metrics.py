"""
Bootstrap Confidence Intervals for Held-Out Test Metrics
========================================================

This script calculates sample-level bootstrap confidence intervals for
the primary held-out test-set performance metrics.

The bootstrap unit is the nanoparticle (NP) sample.

For each bootstrap iteration:
    1. Resample the held-out NPs with replacement.
    2. Keep all protein outputs together for each sampled NP.
    3. Recalculate adsorption and abundance performance metrics.

Metrics
-------
Adsorption:
    - F1
    - AUROC
    - AUPRC
    - MCC

Abundance:
    - Median Pearson r
    - 1-TVD
    - Cosine similarity

Default:
    10,000 bootstrap resamples
    95% percentile confidence interval
    random seed = 42

Run from project root:

    python scripts/bootstrap_test_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcmodel.metrics import (
    adsorption_metrics,
    abundance_metrics,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "bootstrap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OBSERVED_PRESENCE_FILE = (
    RESULTS_DIR
    / "test_presence_observed.csv"
)

PREDICTED_PRESENCE_FILE = (
    RESULTS_DIR
    / "test_presence_predictions.csv"
)

OBSERVED_ABUNDANCE_FILE = (
    RESULTS_DIR
    / "test_abundance_observed.csv"
)

PREDICTED_ABUNDANCE_FILE = (
    RESULTS_DIR
    / "test_abundance_predictions.csv"
)


# ======================================================================
# Configuration
# ======================================================================

RANDOM_SEED = 42

N_BOOTSTRAP = 10000

CI_LEVEL = 0.95


# ======================================================================
# Metrics to report
# ======================================================================

ADSORPTION_METRICS = [
    "F1",
    "AUROC",
    "AUPRC",
    "MCC",
]

ABUNDANCE_METRICS = [
    "Median_r",
    "1-TVD",
    "Cosine",
]


# ======================================================================
# File checks
# ======================================================================


def check_required_files() -> None:

    required = [
        OBSERVED_PRESENCE_FILE,
        PREDICTED_PRESENCE_FILE,
        OBSERVED_ABUNDANCE_FILE,
        PREDICTED_ABUNDANCE_FILE,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        text = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "Required test-result file(s) missing:\n"
            + text
        )


# ======================================================================
# Load arrays
# ======================================================================


def load_test_arrays():
    """
    Load saved held-out observations and predictions.
    """

    observed_presence_df = pd.read_csv(
        OBSERVED_PRESENCE_FILE
    )

    predicted_presence_df = pd.read_csv(
        PREDICTED_PRESENCE_FILE
    )

    observed_abundance_df = pd.read_csv(
        OBSERVED_ABUNDANCE_FILE
    )

    predicted_abundance_df = pd.read_csv(
        PREDICTED_ABUNDANCE_FILE
    )

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    if (
        list(
            observed_presence_df.columns
        )
        != list(
            predicted_presence_df.columns
        )
    ):

        raise ValueError(
            "Observed and predicted presence columns do not match."
        )

    if (
        list(
            observed_abundance_df.columns
        )
        != list(
            predicted_abundance_df.columns
        )
    ):

        raise ValueError(
            "Observed and predicted abundance columns do not match."
        )

    if not (
        len(
            observed_presence_df
        )
        == len(
            predicted_presence_df
        )
        == len(
            observed_abundance_df
        )
        == len(
            predicted_abundance_df
        )
    ):

        raise ValueError(
            "Held-out sample counts do not match across saved files."
        )

    return (
        observed_presence_df.to_numpy(
            dtype=float
        ),

        predicted_presence_df.to_numpy(
            dtype=float
        ),

        observed_abundance_df.to_numpy(
            dtype=float
        ),

        predicted_abundance_df.to_numpy(
            dtype=float
        ),

        list(
            observed_presence_df.columns
        ),
    )


# ======================================================================
# Percentile CI
# ======================================================================


def percentile_ci(
    values: np.ndarray,
    *,
    ci_level: float = CI_LEVEL,
):
    """
    Calculate percentile bootstrap CI.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if (
        len(values)
        == 0
    ):

        return (
            np.nan,
            np.nan,
        )

    alpha = (
        1.0
        - ci_level
    )

    lower = float(
        np.percentile(
            values,
            100.0
            * alpha
            / 2.0,
        )
    )

    upper = float(
        np.percentile(
            values,
            100.0
            * (
                1.0
                - alpha / 2.0
            ),
        )
    )

    return (
        lower,
        upper,
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "BOOTSTRAP CONFIDENCE INTERVALS"
    )

    print(
        "=" * 72
    )

    check_required_files()

    (
        y_presence,
        presence_probability,
        y_abundance,
        abundance_prediction,
        output_columns,
    ) = load_test_arrays()

    n_samples = (
        y_presence.shape[0]
    )

    n_outputs = (
        y_presence.shape[1]
    )

    # ------------------------------------------------------------------
    # Current compatibility architecture:
    #
    # 174 individual proteins + OTHER
    #
    # Exclude OTHER from manuscript protein-level evaluation.
    # ------------------------------------------------------------------

    if (
        output_columns[
            -1
        ]
        == "OTHER"
    ):

        protein_indices = list(
            range(
                n_outputs - 1
            )
        )

    else:

        # Fallback:
        # assume all columns are individual proteins.
        protein_indices = list(
            range(
                n_outputs
            )
        )

    print()

    print(
        "Held-out NP samples:",
        n_samples,
    )

    print(
        "Individual proteins evaluated:",
        len(
            protein_indices
        ),
    )

    print(
        "Bootstrap resamples:",
        N_BOOTSTRAP,
    )

    # ==================================================================
    # Original point estimates
    # ==================================================================

    adsorption_point = (
        adsorption_metrics(
            y_presence,
            presence_probability,
            indices=protein_indices,
        )
    )

    abundance_point = (
        abundance_metrics(
            y_abundance,
            abundance_prediction,
            indices=protein_indices,
        )
    )

    # ==================================================================
    # Bootstrap
    # ==================================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_results = {
        metric: np.full(
            N_BOOTSTRAP,
            np.nan,
            dtype=float,
        )
        for metric in (
            ADSORPTION_METRICS
            + ABUNDANCE_METRICS
        )
    }

    print()

    print(
        "Running bootstrap..."
    )

    for bootstrap_index in range(
        N_BOOTSTRAP
    ):

        # --------------------------------------------------------------
        # Resample NP rows with replacement.
        # --------------------------------------------------------------

        sampled_indices = (
            rng.integers(
                low=0,
                high=n_samples,
                size=n_samples,
            )
        )

        yp_obs = (
            y_presence[
                sampled_indices
            ]
        )

        yp_pred = (
            presence_probability[
                sampled_indices
            ]
        )

        ya_obs = (
            y_abundance[
                sampled_indices
            ]
        )

        ya_pred = (
            abundance_prediction[
                sampled_indices
            ]
        )

        # --------------------------------------------------------------
        # Adsorption
        # --------------------------------------------------------------

        adsorption = (
            adsorption_metrics(
                yp_obs,
                yp_pred,
                indices=protein_indices,
            )
        )

        # --------------------------------------------------------------
        # Abundance
        # --------------------------------------------------------------

        abundance = (
            abundance_metrics(
                ya_obs,
                ya_pred,
                indices=protein_indices,
            )
        )

        # --------------------------------------------------------------
        # Store
        # --------------------------------------------------------------

        for metric in (
            ADSORPTION_METRICS
        ):

            bootstrap_results[
                metric
            ][
                bootstrap_index
            ] = adsorption[
                metric
            ]

        for metric in (
            ABUNDANCE_METRICS
        ):

            bootstrap_results[
                metric
            ][
                bootstrap_index
            ] = abundance[
                metric
            ]

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------

        if (
            bootstrap_index + 1
        ) % 1000 == 0:

            print(
                f"  Completed "
                f"{bootstrap_index + 1}/"
                f"{N_BOOTSTRAP}"
            )

    # ==================================================================
    # Summary table
    # ==================================================================

    rows = []

    for metric in ADSORPTION_METRICS:

        values = (
            bootstrap_results[
                metric
            ]
        )

        (
            ci_lower,
            ci_upper,
        ) = percentile_ci(
            values
        )

        rows.append(
            {
                "Head":
                    "Adsorption",

                "Metric":
                    metric,

                "Point_estimate":
                    float(
                        adsorption_point[
                            metric
                        ]
                    ),

                "Bootstrap_mean":
                    float(
                        np.nanmean(
                            values
                        )
                    ),

                "Bootstrap_SD":
                    float(
                        np.nanstd(
                            values,
                            ddof=1,
                        )
                    ),

                "CI95_lower":
                    ci_lower,

                "CI95_upper":
                    ci_upper,
            }
        )

    for metric in ABUNDANCE_METRICS:

        values = (
            bootstrap_results[
                metric
            ]
        )

        (
            ci_lower,
            ci_upper,
        ) = percentile_ci(
            values
        )

        rows.append(
            {
                "Head":
                    "Abundance",

                "Metric":
                    metric,

                "Point_estimate":
                    float(
                        abundance_point[
                            metric
                        ]
                    ),

                "Bootstrap_mean":
                    float(
                        np.nanmean(
                            values
                        )
                    ),

                "Bootstrap_SD":
                    float(
                        np.nanstd(
                            values,
                            ddof=1,
                        )
                    ),

                "CI95_lower":
                    ci_lower,

                "CI95_upper":
                    ci_upper,
            }
        )

    summary_df = pd.DataFrame(
        rows
    )

    # ==================================================================
    # Save summary
    # ==================================================================

    summary_df.to_csv(
        OUTPUT_DIR
        / "bootstrap_ci_summary.csv",
        index=False,
    )

    # ==================================================================
    # Save all bootstrap draws
    # ==================================================================

    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )

    bootstrap_df.insert(
        0,
        "Bootstrap_iteration",
        np.arange(
            1,
            N_BOOTSTRAP + 1,
        ),
    )

    bootstrap_df.to_csv(
        OUTPUT_DIR
        / "bootstrap_metric_draws.csv",
        index=False,
    )

    # ==================================================================
    # JSON manifest
    # ==================================================================

    manifest = {
        "random_seed":
            RANDOM_SEED,

        "n_bootstrap":
            N_BOOTSTRAP,

        "ci_level":
            CI_LEVEL,

        "bootstrap_unit":
            "NP sample",

        "n_test_samples":
            int(
                n_samples
            ),

        "n_individual_proteins":
            int(
                len(
                    protein_indices
                )
            ),

        "ci_method":
            "percentile bootstrap",
    }

    with open(
        OUTPUT_DIR
        / "bootstrap_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    # ==================================================================
    # Console output
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "BOOTSTRAP CI SUMMARY"
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
        "=" * 72
    )

    print(
        "BOOTSTRAP COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()