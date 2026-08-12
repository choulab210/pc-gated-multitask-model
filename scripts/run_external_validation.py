"""
Run External Validation
=======================

Evaluate the saved two-head protein corona model on the independent
external-validation dataset (Data_val.csv).

Workflow
--------
1. Reconstruct the training preprocessing.
2. Load the saved model checkpoint.
3. Load protein metadata.
4. Preprocess external NP features using the training-fitted preprocessor.
5. Match external proteins to the model protein panel.
6. Run inference without retraining.
7. Calculate overall, category-level, and per-NP performance.
8. Save validation results.

Run from the project root:

    python scripts/run_external_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pcmodel.data import (
    FEATURE_COLS,
    prepare_model_data,
)

from pcmodel.metadata import (
    CATEGORY_ORDER,
    load_protein_metadata,
    metadata_to_mappings,
    validate_metadata_for_panel,
)

from pcmodel.models import TwoHead

from pcmodel.training import (
    get_device,
    predict_probabilities,
)

from pcmodel.validation import (
    build_external_targets,
    external_cosine_similarity,
    external_presence_f1,
    external_validation_metrics,
    renormalize_over_columns,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "external_validation"
)

RESULTS_DIR.mkdir(
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

VALIDATION_FILE = (
    DATA_DIR
    / "Data_val.csv"
)

METADATA_FILE = (
    DATA_DIR
    / "protein_metadata.csv"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "results"
    / "twohead_model_checkpoint.pt"
)


# ======================================================================
# File checks
# ======================================================================


def check_required_files() -> None:
    """
    Confirm that all files required for external validation exist.
    """

    required_files = [
        FEATURE_FILE,
        ABUNDANCE_FILE,
        VALIDATION_FILE,
        METADATA_FILE,
        CHECKPOINT_FILE,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:

        missing_text = "\n".join(
            str(path)
            for path in missing_files
        )

        raise FileNotFoundError(
            "Required file(s) are missing:\n"
            + missing_text
        )


# ======================================================================
# Model loading
# ======================================================================


def load_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> TwoHead:
    """
    Reconstruct the TwoHead model from the saved checkpoint.
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
# External feature preprocessing
# ======================================================================


def prepare_external_feature_matrix(
    validation_df: pd.DataFrame,
    prepared,
    checkpoint: dict,
) -> np.ndarray:
    """
    Transform external features using the preprocessing fitted on
    the original model-development data.

    NP_ID is retained while passing through the preprocessor because
    data.py uses it for sample tracking. It is removed before neural
    network inference.
    """

    # ------------------------------------------------------------------
    # Check required feature columns
    # ------------------------------------------------------------------

    required_columns = (
        ["NP_ID"]
        + list(
            FEATURE_COLS
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in validation_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Data_val.csv is missing required feature column(s): "
            f"{missing_columns}"
        )

    # ------------------------------------------------------------------
    # Keep NP_ID + input features
    # ------------------------------------------------------------------

    external_feature_df = (
        validation_df[
            required_columns
        ].copy()
    )

    # ------------------------------------------------------------------
    # Apply training-fitted preprocessing
    # ------------------------------------------------------------------

    transformed = (
        prepared.preprocessor.transform(
            external_feature_df
        )
    )

    # ------------------------------------------------------------------
    # data.py preserves NP_ID in the transformed dataframe.
    # Remove it before passing the matrix into PyTorch.
    # ------------------------------------------------------------------

    if isinstance(
        transformed,
        pd.DataFrame,
    ):

        if (
            "NP_ID"
            in transformed.columns
        ):

            transformed = (
                transformed.drop(
                    columns=[
                        "NP_ID"
                    ]
                )
            )

        X_external = (
            transformed.to_numpy(
                dtype=np.float32
            )
        )

    else:

        X_external = np.asarray(
            transformed,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------

    expected_input_dim = int(
        checkpoint[
            "input_dim"
        ]
    )

    if (
        X_external.shape[1]
        != expected_input_dim
    ):

        raise ValueError(
            "External feature dimension mismatch. "
            f"Model expects {expected_input_dim}, "
            f"but external preprocessing produced "
            f"{X_external.shape[1]}."
        )

    return X_external


# ======================================================================
# Category-level performance
# ======================================================================


def build_category_summary(
    targets,
    presence_probability: np.ndarray,
    abundance_prediction: np.ndarray,
    id_to_category: dict,
) -> pd.DataFrame:
    """
    Calculate overall and category-level external-validation metrics.
    """

    rows = []

    # ------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------

    overall = (
        external_validation_metrics(
            targets,
            presence_probability,
            abundance_prediction,
        )
    )

    rows.append(
        {
            "Category": "Overall",
            "N": overall[
                "N"
            ],
            "F1": overall[
                "F1"
            ],
            "Cosine": overall[
                "Cosine"
            ],
        }
    )

    # ------------------------------------------------------------------
    # Create category -> full model-output index mapping
    # ------------------------------------------------------------------

    category_indices = {
        category: []
        for category
        in CATEGORY_ORDER
    }

    for (
        model_index,
        protein_id,
    ) in zip(
        targets.overlap_indices,
        targets.overlap_proteins,
    ):

        category = (
            id_to_category.get(
                protein_id,
                "Other/Mixed",
            )
        )

        if (
            category
            not in CATEGORY_ORDER
        ):

            category = (
                "Other/Mixed"
            )

        category_indices[
            category
        ].append(
            model_index
        )

    # ------------------------------------------------------------------
    # Category metrics
    # ------------------------------------------------------------------

    for category in CATEGORY_ORDER:

        indices = (
            category_indices[
                category
            ]
        )

        if not indices:
            continue

        f1 = (
            external_presence_f1(
                targets.Y_presence,
                presence_probability,
                indices,
            )
        )

        cosine = (
            external_cosine_similarity(
                targets.Y_abundance,
                abundance_prediction,
                indices,
            )
        )

        rows.append(
            {
                "Category":
                    category,

                "N":
                    len(
                        indices
                    ),

                "F1":
                    f1,

                "Cosine":
                    cosine,
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Per-NP performance
# ======================================================================


def build_per_np_summary(
    validation_df: pd.DataFrame,
    targets,
    presence_probability: np.ndarray,
    abundance_prediction: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate F1 and cosine similarity separately for each external NP.
    """

    indices = list(
        targets.overlap_indices
    )

    # ------------------------------------------------------------------
    # Presence
    # ------------------------------------------------------------------

    observed_presence = (
        targets.Y_presence[
            :,
            indices
        ]
    )

    predicted_presence = (
        presence_probability[
            :,
            indices
        ]
        >= 0.5
    ).astype(int)

    # ------------------------------------------------------------------
    # Abundance
    # ------------------------------------------------------------------

    observed_abundance = (
        renormalize_over_columns(
            targets.Y_abundance,
            indices,
        )
    )

    predicted_abundance = (
        renormalize_over_columns(
            abundance_prediction,
            indices,
        )
    )

    rows = []

    for sample_index in range(
        len(validation_df)
    ):

        # --------------------------------------------------------------
        # NP identifier
        # --------------------------------------------------------------

        np_id = str(
            validation_df[
                "NP_ID"
            ].iloc[
                sample_index
            ]
        )

        # --------------------------------------------------------------
        # F1
        # --------------------------------------------------------------

        y_true = (
            observed_presence[
                sample_index
            ]
        )

        y_pred = (
            predicted_presence[
                sample_index
            ]
        )

        tp = int(
            np.sum(
                (y_true == 1)
                & (y_pred == 1)
            )
        )

        fp = int(
            np.sum(
                (y_true == 0)
                & (y_pred == 1)
            )
        )

        fn = int(
            np.sum(
                (y_true == 1)
                & (y_pred == 0)
            )
        )

        f1_denominator = (
            2 * tp
            + fp
            + fn
        )

        if (
            f1_denominator
            > 0
        ):

            f1_value = (
                2 * tp
                / f1_denominator
            )

        else:

            f1_value = 0.0

        # --------------------------------------------------------------
        # Cosine similarity
        # --------------------------------------------------------------

        observed_vector = (
            observed_abundance[
                sample_index
            ]
        )

        predicted_vector = (
            predicted_abundance[
                sample_index
            ]
        )

        cosine_denominator = (
            np.linalg.norm(
                observed_vector
            )
            * np.linalg.norm(
                predicted_vector
            )
        )

        if (
            cosine_denominator
            > 0
        ):

            cosine_value = float(
                np.dot(
                    observed_vector,
                    predicted_vector,
                )
                / cosine_denominator
            )

        else:

            cosine_value = np.nan

        rows.append(
            {
                "NP_ID":
                    np_id,

                "F1":
                    float(
                        f1_value
                    ),

                "Cosine":
                    float(
                        cosine_value
                    ),

                "Observed_present":
                    int(
                        y_true.sum()
                    ),

                "Predicted_present":
                    int(
                        y_pred.sum()
                    ),

                "TP":
                    tp,

                "FP":
                    fp,

                "FN":
                    fn,
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Category-level abundance composition
# ======================================================================


def build_category_composition(
    validation_df: pd.DataFrame,
    targets,
    abundance_prediction: np.ndarray,
    id_to_category: dict,
) -> pd.DataFrame:
    """
    Aggregate observed and predicted protein abundance into functional
    categories for each external NP.
    """

    indices = list(
        targets.overlap_indices
    )

    protein_ids = list(
        targets.overlap_proteins
    )

    # ------------------------------------------------------------------
    # Restrict to external overlap panel
    # ------------------------------------------------------------------

    observed = (
        renormalize_over_columns(
            targets.Y_abundance,
            indices,
        )
    )

    predicted = (
        renormalize_over_columns(
            abundance_prediction,
            indices,
        )
    )

    rows = []

    for sample_index in range(
        len(validation_df)
    ):

        np_id = str(
            validation_df[
                "NP_ID"
            ].iloc[
                sample_index
            ]
        )

        observed_category = {
            category: 0.0
            for category
            in CATEGORY_ORDER
        }

        predicted_category = {
            category: 0.0
            for category
            in CATEGORY_ORDER
        }

        for (
            local_index,
            protein_id,
        ) in enumerate(
            protein_ids
        ):

            category = (
                id_to_category.get(
                    protein_id,
                    "Other/Mixed",
                )
            )

            if (
                category
                not in CATEGORY_ORDER
            ):

                category = (
                    "Other/Mixed"
                )

            observed_category[
                category
            ] += float(
                observed[
                    sample_index,
                    local_index,
                ]
            )

            predicted_category[
                category
            ] += float(
                predicted[
                    sample_index,
                    local_index,
                ]
            )

        for category in CATEGORY_ORDER:

            rows.append(
                {
                    "NP_ID":
                        np_id,

                    "Category":
                        category,

                    "Observed":
                        observed_category[
                            category
                        ],

                    "Predicted":
                        predicted_category[
                            category
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Save full prediction matrices
# ======================================================================


def save_prediction_matrices(
    validation_df: pd.DataFrame,
    output_columns: list[str],
    presence_probability: np.ndarray,
    abundance_prediction: np.ndarray,
) -> None:
    """
    Save complete model prediction matrices.
    """

    presence_df = pd.DataFrame(
        presence_probability,
        columns=output_columns,
    )

    abundance_df = pd.DataFrame(
        abundance_prediction,
        columns=output_columns,
    )

    np_ids = (
        validation_df[
            "NP_ID"
        ].astype(
            str
        ).values
    )

    presence_df.insert(
        0,
        "NP_ID",
        np_ids,
    )

    abundance_df.insert(
        0,
        "NP_ID",
        np_ids,
    )

    presence_df.to_csv(
        RESULTS_DIR
        / "presence_predictions.csv",
        index=False,
    )

    abundance_df.to_csv(
        RESULTS_DIR
        / "abundance_predictions.csv",
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
        "EXTERNAL VALIDATION"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Required files
    # ------------------------------------------------------------------

    check_required_files()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = get_device()

    print()

    print(
        "Device:",
        device,
    )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    panel = list(
        checkpoint[
            "panel"
        ]
    )

    output_columns = list(
        checkpoint[
            "abundance_cols"
        ]
    )

    print()

    print(
        "Checkpoint panel:",
        len(panel),
        "proteins",
    )

    print(
        "Model outputs:",
        len(
            output_columns
        ),
    )

    # ------------------------------------------------------------------
    # Reconstruct preprocessing
    # ------------------------------------------------------------------

    print()

    print(
        "Reconstructing frozen preprocessing..."
    )

    prepared = (
        prepare_model_data(
            FEATURE_FILE,
            ABUNDANCE_FILE,
        )
    )

    # ------------------------------------------------------------------
    # Verify preprocessing
    # ------------------------------------------------------------------

    if (
        prepared.X_train.shape[1]
        != int(
            checkpoint[
                "input_dim"
            ]
        )
    ):

        raise ValueError(
            "Reconstructed preprocessing does not match "
            "the model checkpoint."
        )

    if (
        list(
            prepared.panel
        )
        != panel
    ):

        raise ValueError(
            "Reconstructed protein panel does not match "
            "the saved model checkpoint."
        )

    print(
        "Preprocessing verified."
    )

    # ------------------------------------------------------------------
    # Load external data
    # ------------------------------------------------------------------

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    if (
        "NP_ID"
        not in validation_df.columns
    ):

        raise ValueError(
            "Data_val.csv must contain an NP_ID column."
        )

    print()

    print(
        "External NP samples:",
        len(
            validation_df
        ),
    )

    # ------------------------------------------------------------------
    # External feature preprocessing
    # ------------------------------------------------------------------

    X_external = (
        prepare_external_feature_matrix(
            validation_df,
            prepared,
            checkpoint,
        )
    )

    print(
        "External encoded feature matrix:",
        X_external.shape,
    )

    # ------------------------------------------------------------------
    # Protein metadata
    # ------------------------------------------------------------------

    metadata = (
        load_protein_metadata(
            METADATA_FILE
        )
    )

    metadata_check = (
        validate_metadata_for_panel(
            metadata,
            panel,
            require_all=True,
        )
    )

    print()

    print(
        "Protein metadata matched:",
        metadata_check[
            "matched"
        ],
        "/",
        metadata_check[
            "panel_size"
        ],
    )

    (
        id_to_name,
        id_to_category,
    ) = metadata_to_mappings(
        metadata
    )

    # ------------------------------------------------------------------
    # Build observed external targets
    # ------------------------------------------------------------------

    targets = (
        build_external_targets(
            validation_df,
            output_columns,
            id_to_name=id_to_name,
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "PROTEIN OVERLAP"
    )

    print(
        "=" * 72
    )

    print(
        "Model proteins:",
        len(
            panel
        ),
    )

    print(
        "Detected overlap:",
        len(
            targets.overlap_indices
        ),
    )

    print(
        "All-zero in validation:",
        len(
            targets.all_zero_proteins
        ),
    )

    print(
        "Missing from validation:",
        len(
            targets.missing_proteins
        ),
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    model = (
        load_model_from_checkpoint(
            checkpoint,
            device,
        )
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    print()

    print(
        "Running model inference..."
    )

    (
        presence_probability,
        abundance_prediction,
    ) = predict_probabilities(
        model,
        X_external,
        device=device,
    )

    print(
        "Presence predictions:",
        presence_probability.shape,
    )

    print(
        "Abundance predictions:",
        abundance_prediction.shape,
    )

    # ------------------------------------------------------------------
    # Overall external validation
    # ------------------------------------------------------------------

    overall_results = (
        external_validation_metrics(
            targets,
            presence_probability,
            abundance_prediction,
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "EXTERNAL VALIDATION PERFORMANCE"
    )

    print(
        "=" * 72
    )

    print(
        f"N proteins : "
        f"{overall_results['N']}"
    )

    print(
        f"F1         : "
        f"{overall_results['F1']:.4f}"
    )

    print(
        f"Cosine     : "
        f"{overall_results['Cosine']:.4f}"
    )

    # ------------------------------------------------------------------
    # Category performance
    # ------------------------------------------------------------------

    category_summary = (
        build_category_summary(
            targets,
            presence_probability,
            abundance_prediction,
            id_to_category,
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "CATEGORY PERFORMANCE"
    )

    print(
        "=" * 72
    )

    print(
        category_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Per-NP performance
    # ------------------------------------------------------------------

    per_np_summary = (
        build_per_np_summary(
            validation_df,
            targets,
            presence_probability,
            abundance_prediction,
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "PER-NP PERFORMANCE"
    )

    print(
        "=" * 72
    )

    print(
        per_np_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Category composition
    # ------------------------------------------------------------------

    category_composition = (
        build_category_composition(
            validation_df,
            targets,
            abundance_prediction,
            id_to_category,
        )
    )

    # ==================================================================
    # Save results
    # ==================================================================

    targets.overlap_table.to_csv(
        RESULTS_DIR
        / "protein_overlap.csv",
        index=False,
    )

    category_summary.to_csv(
        RESULTS_DIR
        / "category_performance.csv",
        index=False,
    )

    per_np_summary.to_csv(
        RESULTS_DIR
        / "per_np_performance.csv",
        index=False,
    )

    category_composition.to_csv(
        RESULTS_DIR
        / "category_composition.csv",
        index=False,
    )

    save_prediction_matrices(
        validation_df,
        output_columns,
        presence_probability,
        abundance_prediction,
    )

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------

    summary = {
        "n_external_samples":
            int(
                len(
                    validation_df
                )
            ),

        "model_panel_size":
            int(
                len(
                    panel
                )
            ),

        "model_output_size":
            int(
                len(
                    output_columns
                )
            ),

        "overlap_proteins":
            int(
                len(
                    targets.overlap_indices
                )
            ),

        "all_zero_proteins":
            int(
                len(
                    targets.all_zero_proteins
                )
            ),

        "missing_proteins":
            int(
                len(
                    targets.missing_proteins
                )
            ),

        "F1":
            float(
                overall_results[
                    "F1"
                ]
            ),

        "Cosine":
            float(
                overall_results[
                    "Cosine"
                ]
            ),
    }

    with open(
        RESULTS_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "EXTERNAL VALIDATION COMPLETE"
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