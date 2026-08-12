"""
Run Conventional Model Benchmarks
=================================

Compare the final two-head protein corona model against conventional
machine-learning algorithms using the exact same development/test split
and preprocessing.

Adsorption
----------
- Two-head model
- Random Forest
- Logistic Regression
- XGBoost

Abundance
---------
- Two-head model
- Random Forest
- Ridge Regression
- XGBoost

Outputs
-------
results/benchmarks/
    adsorption_benchmarks.csv
    abundance_benchmarks.csv
    benchmark_config.json
    predictions/

Run:

    python scripts/run_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcmodel.benchmarks import (
    ADSORPTION_BENCHMARKS,
    ABUNDANCE_BENCHMARKS,
    BenchmarkConfig,
    fit_adsorption_models,
    fit_abundance_models,
    predict_adsorption_probabilities,
    predict_abundance_distribution,
)

from pcmodel.data import (
    prepare_model_data,
)

from pcmodel.metrics import (
    adsorption_metrics,
    abundance_metrics,
)

from pcmodel.models import TwoHead

from pcmodel.training import (
    get_device,
    predict_probabilities,
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
    / "benchmarks"
)

PREDICTION_DIR = (
    RESULTS_DIR
    / "predictions"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
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

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "results"
    / "twohead_model_checkpoint.pt"
)


# ======================================================================
# Load final two-head model
# ======================================================================


def load_twohead_model(
    checkpoint: dict,
    device: torch.device,
) -> TwoHead:
    """
    Reconstruct saved final model.
    """

    config = checkpoint[
        "training_config"
    ]

    model = TwoHead(
        in_dim=int(
            checkpoint[
                "input_dim"
            ]
        ),

        hidden=list(
            config[
                "hidden"
            ]
        ),

        k=int(
            checkpoint[
                "n_outputs"
            ]
        ),

        dropout=float(
            config[
                "dropout"
            ]
        ),

        alpha_gate=float(
            config[
                "alpha_gate"
            ]
        ),

        temp_init=float(
            config[
                "temp_init"
            ]
        ),

        stopgrad_gate=True,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ======================================================================
# Save prediction matrix
# ======================================================================


def save_predictions(
    filename: str,
    prediction: np.ndarray,
    columns,
) -> None:

    pd.DataFrame(
        prediction,
        columns=list(
            columns
        ),
    ).to_csv(
        PREDICTION_DIR
        / filename,
        index=False,
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "CONVENTIONAL MODEL BENCHMARKS"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Prepare identical data split/preprocessing
    # ------------------------------------------------------------------

    print()

    print(
        "Preparing model data..."
    )

    prepared = (
        prepare_model_data(
            FEATURE_FILE,
            ABUNDANCE_FILE,
        )
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

    print(
        "Encoded features:",
        X_train.shape[1],
    )

    print(
        "Individual proteins:",
        n_proteins,
    )

    # ------------------------------------------------------------------
    # Benchmark configuration
    # ------------------------------------------------------------------

    config = BenchmarkConfig(
        random_state=42,
    )

    # ==================================================================
    # Two-head reference
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "FINAL TWO-HEAD MODEL"
    )

    print(
        "=" * 72
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    device = get_device()

    model = load_twohead_model(
        checkpoint,
        device,
    )

    (
        twohead_presence,
        twohead_abundance,
    ) = predict_probabilities(
        model,
        X_test,
        device=device,
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
    # Adsorption benchmarks
    # ==================================================================

    adsorption_rows = [
        {
            "Model":
                "Two-head Model",

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
        }
    ]

    save_predictions(
        "twohead_presence.csv",
        twohead_presence,
        prepared.presence_columns,
    )

    # ------------------------------------------------------------------
    # Conventional classifiers
    # ------------------------------------------------------------------

    for model_key, model_label in (
        ADSORPTION_BENCHMARKS.items()
    ):

        print()

        print(
            "=" * 72
        )

        print(
            f"ADSORPTION — {model_label}"
        )

        print(
            "=" * 72
        )

        models = fit_adsorption_models(
            X_train,
            Yp_train,
            model_key,
            config,
        )

        probability = (
            predict_adsorption_probabilities(
                models,
                X_test,
            )
        )

        metrics = adsorption_metrics(
            Yp_test,
            probability,
            indices=protein_indices,
        )

        adsorption_rows.append(
            {
                "Model":
                    model_label,

                "Acc":
                    metrics[
                        "Acc"
                    ],

                "F1":
                    metrics[
                        "F1"
                    ],

                "Precision":
                    metrics[
                        "Precision"
                    ],

                "Recall":
                    metrics[
                        "Recall"
                    ],

                "AUROC":
                    metrics[
                        "AUROC"
                    ],

                "AUPRC":
                    metrics[
                        "AUPRC"
                    ],

                "MCC":
                    metrics[
                        "MCC"
                    ],
            }
        )

        save_predictions(
            (
                f"{model_key}"
                "_presence.csv"
            ),
            probability,
            prepared.presence_columns,
        )

        print(
            f"F1     : "
            f"{metrics['F1']:.4f}"
        )

        print(
            f"AUROC  : "
            f"{metrics['AUROC']:.4f}"
        )

        print(
            f"AUPRC  : "
            f"{metrics['AUPRC']:.4f}"
        )

    # ==================================================================
    # Abundance benchmarks
    # ==================================================================

    abundance_rows = [
        {
            "Model":
                "Two-head Model",

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
        }
    ]

    save_predictions(
        "twohead_abundance.csv",
        twohead_abundance,
        prepared.abundance_columns,
    )

    # ------------------------------------------------------------------
    # Conventional regressors
    # ------------------------------------------------------------------

    for model_key, model_label in (
        ABUNDANCE_BENCHMARKS.items()
    ):

        print()

        print(
            "=" * 72
        )

        print(
            f"ABUNDANCE — {model_label}"
        )

        print(
            "=" * 72
        )

        models = fit_abundance_models(
            X_train,
            Ya_train,
            model_key,
            config,
        )

        abundance_prediction = (
            predict_abundance_distribution(
                models,
                X_test,
            )
        )

        metrics = abundance_metrics(
            Ya_test,
            abundance_prediction,
            indices=protein_indices,
        )

        abundance_rows.append(
            {
                "Model":
                    model_label,

                "Median_r":
                    metrics[
                        "Median_r"
                    ],

                "1-TVD":
                    metrics[
                        "1-TVD"
                    ],

                "Cosine":
                    metrics[
                        "Cosine"
                    ],
            }
        )

        save_predictions(
            (
                f"{model_key}"
                "_abundance.csv"
            ),
            abundance_prediction,
            prepared.abundance_columns,
        )

        print(
            f"Median_r : "
            f"{metrics['Median_r']:.4f}"
        )

        print(
            f"1-TVD    : "
            f"{metrics['1-TVD']:.4f}"
        )

        print(
            f"Cosine   : "
            f"{metrics['Cosine']:.4f}"
        )

    # ==================================================================
    # Save benchmark tables
    # ==================================================================

    adsorption_df = pd.DataFrame(
        adsorption_rows
    )

    abundance_df = pd.DataFrame(
        abundance_rows
    )

    adsorption_df.to_csv(
        RESULTS_DIR
        / "adsorption_benchmarks.csv",
        index=False,
    )

    abundance_df.to_csv(
        RESULTS_DIR
        / "abundance_benchmarks.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save configuration
    # ------------------------------------------------------------------

    with open(
        RESULTS_DIR
        / "benchmark_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "random_state":
                    config.random_state,

                "rf_n_estimators":
                    config.rf_n_estimators,

                "rf_max_depth":
                    config.rf_max_depth,

                "logistic_C":
                    config.logistic_c,

                "ridge_alpha":
                    config.ridge_alpha,

                "xgb_n_estimators":
                    config.xgb_n_estimators,

                "xgb_max_depth":
                    config.xgb_max_depth,

                "xgb_learning_rate":
                    config.xgb_learning_rate,

                "xgb_subsample":
                    config.xgb_subsample,

                "xgb_colsample_bytree":
                    config.xgb_colsample_bytree,
            },
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
        "ADSORPTION BENCHMARK SUMMARY"
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
        "ABUNDANCE BENCHMARK SUMMARY"
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
        "=" * 72
    )

    print(
        "BENCHMARK ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
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