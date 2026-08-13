"""
Prepare Figure 4 SHAP Data
==========================

Compute SHAP values for the ADSORPTION HEAD of the final gated
two-head protein-corona model.

This follows the original notebook logic:

1. Load the final gated model.
2. Use adsorption probabilities as the SHAP prediction target.
3. Use 100 model-development samples as the SHAP background.
4. Explain all held-out test samples.
5. Aggregate one-hot encoded SHAP values back to the 12 original
   NP/experimental features.
6. Save both aggregated and encoded SHAP data so Figure 4 can be
   replotted without rerunning SHAP.

Run:
    python scripts/figures/prepare_figure4_shap_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from pcmodel.data import (
    FEATURE_COLS,
    prepare_model_data,
)

from pcmodel.models import TwoHead
from pcmodel.training import get_device


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = (
    RESULTS_DIR
    / "figures"
    / "figure4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_FILE = DATA_DIR / "Data_1.csv"
ABUNDANCE_FILE = DATA_DIR / "Data_2.csv"

CHECKPOINT_FILE = (
    RESULTS_DIR
    / "twohead_model_checkpoint.pt"
)


# ======================================================================
# Configuration
# ======================================================================

RANDOM_SEED = 42

SHAP_BACKGROUND_SAMPLES = 100


# ======================================================================
# Model loading
# ======================================================================


def load_final_gated_model(
    checkpoint: dict,
    device,
):
    """
    Reconstruct final gated TwoHead model from checkpoint.
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
# Encoded feature names
# ======================================================================


def get_encoded_feature_names(
    prepared,
    n_features: int,
):
    """
    Recover encoded feature names from PreparedData/preprocessor.

    Several fallbacks are included because preprocessing implementations
    may expose names differently.
    """

    # --------------------------------------------------------------
    # DataFrame columns
    # --------------------------------------------------------------

    if isinstance(
        prepared.X_train,
        pd.DataFrame,
    ):

        columns = list(
            prepared.X_train.columns
        )

        if len(columns) == n_features:
            return columns

    # --------------------------------------------------------------
    # Common preprocessor attributes
    # --------------------------------------------------------------

    preprocessor = getattr(
        prepared,
        "preprocessor",
        None,
    )

    if preprocessor is not None:

        candidate_attributes = [
            "feature_names",
            "feature_names_",
            "output_columns",
            "encoded_columns",
        ]

        for attribute in candidate_attributes:

            if hasattr(
                preprocessor,
                attribute,
            ):

                values = list(
                    getattr(
                        preprocessor,
                        attribute,
                    )
                )

                if len(values) == n_features:
                    return values

        if hasattr(
            preprocessor,
            "get_feature_names_out",
        ):

            try:

                values = list(
                    preprocessor.get_feature_names_out()
                )

                if len(values) == n_features:
                    return values

            except Exception:
                pass

    # --------------------------------------------------------------
    # Last-resort generic names
    # --------------------------------------------------------------

    raise RuntimeError(
        "Could not recover encoded feature names from the current "
        "PreparedData/preprocessor object. Do not use generic feature "
        "names because Figure 4 requires mapping encoded features back "
        "to the original 12 variables."
    )


# ======================================================================
# Parent-feature mapping
# ======================================================================


def build_feature_groups(
    encoded_columns,
):
    """
    Aggregate encoded columns back to original model-input features.

    This follows the notebook strategy:
        exact match for continuous/ordinal features
        prefix match for one-hot categorical features
    """

    encoded_columns = list(
        encoded_columns
    )

    groups = {}

    for parent in FEATURE_COLS:

        exact = [
            column
            for column in encoded_columns
            if column == parent
        ]

        prefixed = [
            column
            for column in encoded_columns
            if column.startswith(
                parent + "_"
            )
        ]

        matched = exact + prefixed

        # Avoid duplicates
        matched = list(
            dict.fromkeys(
                matched
            )
        )

        if matched:

            groups[
                parent
            ] = matched

    return groups


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "PREPARING FIGURE 4 SHAP DATA"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

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

    n_features = X_train.shape[1]

    encoded_columns = (
        get_encoded_feature_names(
            prepared,
            n_features,
        )
    )

    panel = list(
        prepared.panel
    )

    print()

    print(
        "Development samples:",
        X_train.shape[0],
    )

    print(
        "Held-out test samples:",
        X_test.shape[0],
    )

    print(
        "Encoded features:",
        n_features,
    )

    print(
        "Individual proteins:",
        len(panel),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    device = get_device()

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = load_final_gated_model(
        checkpoint,
        device,
    )

    # ------------------------------------------------------------------
    # Adsorption-head prediction wrapper
    # ------------------------------------------------------------------

    def predict_presence(
        X_numpy,
    ):

        model.eval()

        with torch.no_grad():

            tensor = torch.tensor(
                X_numpy,
                dtype=torch.float32,
                device=device,
            )

            presence_logits, _ = model(
                tensor
            )

            probabilities = torch.sigmoid(
                presence_logits
            )

        return (
            probabilities
            .cpu()
            .numpy()
        )

    test_output = predict_presence(
        X_test[:3]
    )

    print(
        "Prediction wrapper output:",
        test_output.shape,
    )

    # ------------------------------------------------------------------
    # Background samples
    # ------------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    background_indices = rng.choice(
        len(
            X_train
        ),
        size=min(
            SHAP_BACKGROUND_SAMPLES,
            len(
                X_train
            ),
        ),
        replace=False,
    )

    background = X_train[
        background_indices
    ]

    # ==================================================================
    # SHAP
    # ==================================================================

    print()

    print(
        "Computing SHAP values..."
    )

    explainer = shap.Explainer(
        predict_presence,
        background,
        seed=RANDOM_SEED,
    )

    shap_object = explainer(
        X_test
    )

    shap_array = np.asarray(
        shap_object.values
    )

    print(
        "Raw SHAP shape:",
        shap_array.shape,
    )

    # Expected:
    # samples × encoded features × outputs

    if shap_array.ndim != 3:

        raise RuntimeError(
            "Unexpected SHAP shape. Expected "
            "(samples, features, outputs), got "
            f"{shap_array.shape}"
        )

    # ------------------------------------------------------------------
    # Restrict to 174 individual proteins.
    #
    # Compatibility architecture may contain OTHER as an extra output.
    # ------------------------------------------------------------------

    n_panel = len(
        panel
    )

    shap_array = shap_array[
        :,
        :,
        :n_panel,
    ]

    # ==================================================================
    # Aggregate encoded SHAP -> original 12 features
    # ==================================================================

    feature_groups = build_feature_groups(
        encoded_columns
    )

    parent_features = list(
        feature_groups.keys()
    )

    print()

    print(
        "Recovered original features:"
    )

    for feature in parent_features:

        print(
            f"  {feature:<18} -> "
            f"{len(feature_groups[feature])} encoded column(s)"
        )

    encoded_lookup = {
        column: index
        for index, column in enumerate(
            encoded_columns
        )
    }

    # --------------------------------------------------------------
    # Shape:
    # samples × parent features × proteins
    # --------------------------------------------------------------

    aggregated_shap = np.zeros(
        (
            X_test.shape[0],
            len(
                parent_features
            ),
            n_panel,
        ),
        dtype=np.float32,
    )

    aggregated_X = np.zeros(
        (
            X_test.shape[0],
            len(
                parent_features
            ),
        ),
        dtype=np.float32,
    )

    for feature_index, feature in enumerate(
        parent_features
    ):

        encoded_group = feature_groups[
            feature
        ]

        indices = [
            encoded_lookup[
                column
            ]
            for column in encoded_group
        ]

        # ----------------------------------------------------------
        # Signed SHAP aggregation, as in original notebook.
        # ----------------------------------------------------------

        aggregated_shap[
            :,
            feature_index,
            :,
        ] = shap_array[
            :,
            indices,
            :,
        ].sum(
            axis=1
        )

        # ----------------------------------------------------------
        # Feature value used for SHAP scatter coloring.
        # Exact features use their actual encoded value.
        # One-hot groups use the active-category index.
        # ----------------------------------------------------------

        if len(
            indices
        ) == 1:

            aggregated_X[
                :,
                feature_index
            ] = X_test[
                :,
                indices[0]
            ]

        else:

            aggregated_X[
                :,
                feature_index
            ] = np.argmax(
                X_test[
                    :,
                    indices
                ],
                axis=1,
            )

    # ==================================================================
    # Heatmap data
    # ==================================================================

    # proteins × original features

    heatmap_matrix = np.mean(
        np.abs(
            aggregated_shap
        ),
        axis=0,
    ).T

    heatmap_df = pd.DataFrame(
        heatmap_matrix,
        index=panel,
        columns=parent_features,
    )

    global_feature_importance = (
        heatmap_df
        .mean(
            axis=0
        )
        .sort_values(
            ascending=False
        )
    )

    sorted_features = list(
        global_feature_importance.index
    )

    heatmap_df = heatmap_df[
        sorted_features
    ]

    # ==================================================================
    # Save
    # ==================================================================

    np.savez_compressed(
        OUTPUT_DIR
        / "figure4_shap_arrays.npz",

        shap_encoded=shap_array,

        X_test_encoded=X_test,

        shap_aggregated=aggregated_shap,

        X_test_aggregated=aggregated_X,
    )

    pd.DataFrame(
        X_test,
        columns=encoded_columns,
    ).to_csv(
        OUTPUT_DIR
        / "X_test_encoded.csv",
        index=False,
    )

    pd.DataFrame(
        aggregated_X,
        columns=parent_features,
    ).to_csv(
        OUTPUT_DIR
        / "X_test_aggregated.csv",
        index=False,
    )

    heatmap_df.to_csv(
        OUTPUT_DIR
        / "heatmap_mean_abs_shap.csv"
    )

    global_feature_importance.rename(
        "Mean_abs_SHAP"
    ).to_csv(
        OUTPUT_DIR
        / "global_feature_importance.csv"
    )

    metadata = {
        "random_seed":
            RANDOM_SEED,

        "background_samples":
            int(
                len(
                    background
                )
            ),

        "n_test":
            int(
                len(
                    X_test
                )
            ),

        "n_proteins":
            int(
                n_panel
            ),

        "encoded_columns":
            encoded_columns,

        "parent_features":
            parent_features,

        "sorted_features":
            sorted_features,

        "protein_panel":
            panel,

        "feature_groups":
            feature_groups,
    }

    with open(
        OUTPUT_DIR
        / "figure4_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()

    print(
        "=" * 72
    )

    print(
        "FIGURE 4 SHAP DATA COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()