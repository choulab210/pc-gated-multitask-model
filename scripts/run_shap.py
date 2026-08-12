"""
Run SHAP Analysis
=================

Calculate SHAP values for the adsorption/presence head of the final
two-head protein corona model.

Workflow
--------
1. Reconstruct the final development/test preprocessing.
2. Load the saved final model checkpoint.
3. Recover the 55 encoded feature names.
4. Sample background observations from the development set.
5. Calculate SHAP values for adsorption probabilities.
6. Aggregate one-hot encoded categorical features back to the
   original model features.
7. Calculate mean absolute SHAP importance.
8. Save encoded and original-feature SHAP results.
9. Generate tables for later publication-quality plotting.

The SHAP analysis is performed on the adsorption head only.

Default manuscript-style run:
    python scripts/run_shap.py

Quick pipeline test:
    python scripts/run_shap.py --quick

The quick run is only for debugging and should NOT be used for
manuscript results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
import torch

from pcmodel.data import (
    FEATURE_COLS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    NOMINAL_COLS,
    prepare_model_data,
)

from pcmodel.interpretation import (
    global_shap_feature_importance,
)

from pcmodel.metadata import (
    load_protein_metadata,
    metadata_to_mappings,
    validate_metadata_for_panel,
)

from pcmodel.models import TwoHead

from pcmodel.training import (
    get_device,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

FEATURE_FILE = (
    DATA_DIR
    / "Data_1.csv"
)

ABUNDANCE_FILE = (
    DATA_DIR
    / "Data_2.csv"
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
# Defaults
# ======================================================================

RANDOM_SEED = 42

DEFAULT_BACKGROUND_SAMPLES = 100

TOP_N_PROTEINS = 30

EXPERIMENTAL_FEATURES = {
    "incub_time",
    "washing_steps",
    "agitation",
    "hd_solvent",
    "zp_solvent",
}


# ======================================================================
# Command-line arguments
# ======================================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate SHAP values for the adsorption "
            "head of the protein corona model."
        )
    )

    parser.add_argument(
        "--background",
        type=int,
        default=DEFAULT_BACKGROUND_SAMPLES,
        help=(
            "Number of development samples used as "
            "the SHAP background. Default = 100."
        ),
    )

    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help=(
            "Maximum number of held-out samples to explain. "
            "0 means use all test samples."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Quick debugging run using 20 background samples "
            "and 8 explained samples."
        ),
    )

    return parser.parse_args()


# ======================================================================
# File checks
# ======================================================================


def check_required_files() -> None:

    required = [
        FEATURE_FILE,
        ABUNDANCE_FILE,
        METADATA_FILE,
        CHECKPOINT_FILE,
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
            "Required file(s) missing:\n"
            + text
        )


# ======================================================================
# Load model
# ======================================================================


def load_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> TwoHead:

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
# Recover encoded feature names
# ======================================================================


def get_encoded_feature_names(
    prepared,
) -> List[str]:
    """
    Recover the names of the 55 encoded model features.

    data.py preserves NP_ID during preprocessing, so NP_ID is removed
    before extracting neural-network feature names.
    """

    raw_features = pd.read_csv(
        FEATURE_FILE
    )

    required_columns = (
        ["NP_ID"]
        + list(
            FEATURE_COLS
        )
    )

    transformed = (
        prepared.preprocessor.transform(
            raw_features[
                required_columns
            ].copy()
        )
    )

    if not isinstance(
        transformed,
        pd.DataFrame,
    ):

        raise TypeError(
            "Expected FeaturePreprocessor.transform() "
            "to return a pandas DataFrame."
        )

    if (
        "NP_ID"
        in transformed.columns
    ):

        transformed = transformed.drop(
            columns=[
                "NP_ID"
            ]
        )

    feature_names = list(
        transformed.columns
    )

    if (
        len(feature_names)
        != prepared.X_train.shape[1]
    ):

        raise ValueError(
            "Encoded feature-name count does not match "
            "the neural-network feature dimension. "
            f"Names={len(feature_names)}, "
            f"matrix={prepared.X_train.shape[1]}"
        )

    return feature_names


# ======================================================================
# Feature grouping
# ======================================================================


def build_feature_groups(
    encoded_feature_names: List[str],
) -> Dict[
    str,
    List[str],
]:
    """
    Group encoded features back to their original model variables.

    Continuous/ordinal features map to one encoded column.

    Nominal features may map to multiple one-hot encoded columns.
    """

    groups: Dict[
        str,
        List[str],
    ] = {}

    encoded_feature_names = list(
        encoded_feature_names
    )

    # ------------------------------------------------------------------
    # Numerical + ordinal
    # ------------------------------------------------------------------

    for feature in (
        list(
            NUMERIC_COLS
        )
        + list(
            ORDINAL_COLS
        )
    ):

        if (
            feature
            in encoded_feature_names
        ):

            groups[
                feature
            ] = [
                feature
            ]

    # ------------------------------------------------------------------
    # Nominal / one-hot features
    # ------------------------------------------------------------------

    for parent in NOMINAL_COLS:

        matched = [
            column
            for column
            in encoded_feature_names
            if (
                column.startswith(
                    parent + "_"
                )
                or column.startswith(
                    parent + "__"
                )
                or column.startswith(
                    parent + "="
                )
            )
        ]

        if matched:

            groups[
                parent
            ] = matched

    # ------------------------------------------------------------------
    # Verify that every encoded feature belongs to exactly one group
    # ------------------------------------------------------------------

    grouped_columns = []

    for children in groups.values():

        grouped_columns.extend(
            children
        )

    missing_encoded = [
        column
        for column
        in encoded_feature_names
        if column
        not in grouped_columns
    ]

    if missing_encoded:

        raise ValueError(
            "Some encoded features could not be mapped "
            "to their original feature groups:\n"
            f"{missing_encoded}"
        )

    return groups


# ======================================================================
# SHAP shape handling
# ======================================================================


def ensure_shap_shape(
    shap_values,
    *,
    n_samples: int,
    n_features: int,
    n_outputs: int,
) -> np.ndarray:
    """
    Standardize SHAP output to:

        samples x encoded_features x outputs
    """

    values = np.asarray(
        shap_values,
        dtype=np.float64,
    )

    expected = (
        n_samples,
        n_features,
        n_outputs,
    )

    if (
        values.shape
        == expected
    ):

        return values

    # ------------------------------------------------------------------
    # Some older SHAP interfaces may return:
    #
    # outputs x samples x features
    # ------------------------------------------------------------------

    alternative = (
        n_outputs,
        n_samples,
        n_features,
    )

    if (
        values.shape
        == alternative
    ):

        return np.transpose(
            values,
            (
                1,
                2,
                0,
            ),
        )

    raise ValueError(
        "Unexpected SHAP output shape. "
        f"Received {values.shape}; "
        f"expected {expected}."
    )


# ======================================================================
# Aggregate encoded SHAP to original features
# ======================================================================


def aggregate_shap_to_original_features(
    shap_encoded: np.ndarray,
    encoded_feature_names: List[str],
    feature_groups: Dict[
        str,
        List[str],
    ],
):
    """
    Sum SHAP values across one-hot columns belonging to the same
    original feature.

    Returns
    -------
    shap_original
        samples x original_features x outputs

    original_feature_names
        Names corresponding to axis 1.
    """

    encoded_lookup = {
        name: index
        for index, name
        in enumerate(
            encoded_feature_names
        )
    }

    original_feature_names = list(
        feature_groups.keys()
    )

    aggregated = []

    for feature in original_feature_names:

        child_columns = (
            feature_groups[
                feature
            ]
        )

        child_indices = [
            encoded_lookup[
                column
            ]
            for column
            in child_columns
        ]

        # --------------------------------------------------------------
        # Additive SHAP aggregation
        # --------------------------------------------------------------

        feature_shap = (
            shap_encoded[
                :,
                child_indices,
                :,
            ]
            .sum(
                axis=1
            )
        )

        aggregated.append(
            feature_shap
        )

    # ------------------------------------------------------------------
    # List of:
    #     samples x outputs
    #
    # Stack into:
    #     samples x original_features x outputs
    # ------------------------------------------------------------------

    shap_original = np.stack(
        aggregated,
        axis=1,
    )

    return (
        shap_original,
        original_feature_names,
    )


# ======================================================================
# Build heatmap tables
# ======================================================================


def build_heatmap_table(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_names: List[str],
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP for every:

        output protein x feature

    combination.
    """

    mean_abs = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    # mean_abs:
    #     feature x output
    #
    # transpose:
    #     output x feature

    return pd.DataFrame(
        mean_abs.T,
        index=output_names,
        columns=feature_names,
    )


# ======================================================================
# Relative heatmap
# ======================================================================


def relative_to_global_maximum(
    heatmap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert mean absolute SHAP to:

        Relative mean |SHAP| (% of maximum)

    These values do NOT sum to 100%.
    """

    maximum = float(
        np.nanmax(
            heatmap.to_numpy(
                dtype=float
            )
        )
    )

    if (
        not np.isfinite(
            maximum
        )
        or maximum <= 0
    ):

        return heatmap * 0.0

    return (
        heatmap
        / maximum
        * 100.0
    )


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    args = parse_args()

    check_required_files()

    # ------------------------------------------------------------------
    # Quick mode
    # ------------------------------------------------------------------

    if args.quick:

        background_samples = 20

        max_test_samples = 8

        results_dir = (
            PROJECT_ROOT
            / "results"
            / "shap_quick"
        )

        print(
            "QUICK MODE enabled."
        )

        print(
            "This run is for debugging only."
        )

    else:

        background_samples = (
            args.background
        )

        max_test_samples = (
            args.max_test_samples
        )

        results_dir = (
            PROJECT_ROOT
            / "results"
            / "shap"
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "SHAP ANALYSIS — ADSORPTION HEAD"
    )

    print(
        "=" * 72
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    print()

    print(
        "[1/7] Preparing model data..."
    )

    prepared = (
        prepare_model_data(
            FEATURE_FILE,
            ABUNDANCE_FILE,
        )
    )

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

    output_names = list(
        checkpoint[
            "presence_cols"
        ]
    )

    n_outputs = len(
        output_names
    )

    if (
        list(
            prepared.panel
        )
        != panel
    ):

        raise ValueError(
            "Reconstructed protein panel does not "
            "match the saved checkpoint."
        )

    print(
        "Development samples:",
        prepared.X_train.shape[0],
    )

    print(
        "Held-out samples:",
        prepared.X_test.shape[0],
    )

    print(
        "Encoded features:",
        prepared.X_train.shape[1],
    )

    print(
        "Model outputs:",
        n_outputs,
    )

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    print()

    print(
        "[2/7] Recovering encoded feature names..."
    )

    encoded_feature_names = (
        get_encoded_feature_names(
            prepared
        )
    )

    feature_groups = (
        build_feature_groups(
            encoded_feature_names
        )
    )

    print(
        "Encoded features recovered:",
        len(
            encoded_feature_names
        ),
    )

    print(
        "Original feature groups:",
        len(
            feature_groups
        ),
    )

    print()

    for parent, children in (
        feature_groups.items()
    ):

        print(
            f"  {parent:16s}: "
            f"{len(children)} encoded column(s)"
        )

    # ------------------------------------------------------------------
    # Load protein metadata
    # ------------------------------------------------------------------

    metadata = (
        load_protein_metadata(
            METADATA_FILE
        )
    )

    validate_metadata_for_panel(
        metadata,
        panel,
        require_all=True,
    )

    (
        id_to_name,
        id_to_category,
    ) = metadata_to_mappings(
        metadata
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print()

    print(
        "[3/7] Loading final model..."
    )

    device = get_device()

    print(
        "Device:",
        device,
    )

    model = (
        load_model_from_checkpoint(
            checkpoint,
            device,
        )
    )

    # ==================================================================
    # Presence prediction wrapper
    # ==================================================================

    def predict_presence(
        X_numpy,
    ) -> np.ndarray:
        """
        Return adsorption probabilities for SHAP.
        """

        X_numpy = np.asarray(
            X_numpy,
            dtype=np.float32,
        )

        X_tensor = torch.tensor(
            X_numpy,
            dtype=torch.float32,
            device=device,
        )

        model.eval()

        with torch.no_grad():

            (
                presence_logits,
                _,
            ) = model(
                X_tensor
            )

            probabilities = (
                torch.sigmoid(
                    presence_logits
                )
                .cpu()
                .numpy()
            )

        return probabilities

    # ------------------------------------------------------------------
    # Wrapper sanity test
    # ------------------------------------------------------------------

    test_output = predict_presence(
        prepared.X_test[
            :3
        ]
    )

    expected_shape = (
        3,
        n_outputs,
    )

    if (
        test_output.shape
        != expected_shape
    ):

        raise ValueError(
            "Presence prediction wrapper returned "
            f"{test_output.shape}; "
            f"expected {expected_shape}."
        )

    print(
        "Prediction wrapper verified:",
        test_output.shape,
    )

    # ==================================================================
    # Select SHAP background
    # ==================================================================

    print()

    print(
        "[4/7] Selecting SHAP background..."
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n_background = min(
        int(
            background_samples
        ),
        len(
            prepared.X_train
        ),
    )

    background_indices = (
        rng.choice(
            len(
                prepared.X_train
            ),
            size=n_background,
            replace=False,
        )
    )

    background = (
        prepared.X_train[
            background_indices
        ]
    )

    # ------------------------------------------------------------------
    # Select explained samples
    # ------------------------------------------------------------------

    X_explain = (
        prepared.X_test
    )

    test_indices = np.arange(
        len(
            X_explain
        )
    )

    if (
        max_test_samples
        and max_test_samples > 0
        and max_test_samples
        < len(
            X_explain
        )
    ):

        selected = (
            rng.choice(
                len(
                    X_explain
                ),
                size=int(
                    max_test_samples
                ),
                replace=False,
            )
        )

        selected = np.sort(
            selected
        )

        X_explain = (
            X_explain[
                selected
            ]
        )

        test_indices = (
            test_indices[
                selected
            ]
        )

    print(
        "Background samples:",
        len(
            background
        ),
    )

    print(
        "Explained test samples:",
        len(
            X_explain
        ),
    )

    # ==================================================================
    # SHAP
    # ==================================================================

    print()

    print(
        "[5/7] Computing SHAP values..."
    )

    print(
        "This may take substantially longer "
        "than the previous analyses."
    )

    explainer = shap.Explainer(
        predict_presence,
        background,
        seed=RANDOM_SEED,
    )

    shap_object = explainer(
        X_explain
    )

    shap_encoded = (
        ensure_shap_shape(
            shap_object.values,
            n_samples=len(
                X_explain
            ),
            n_features=(
                prepared.X_train.shape[1]
            ),
            n_outputs=n_outputs,
        )
    )

    print(
        "Encoded SHAP shape:",
        shap_encoded.shape,
    )

    # ==================================================================
    # Aggregate one-hot features
    # ==================================================================

    print()

    print(
        "[6/7] Aggregating one-hot SHAP "
        "to original features..."
    )

    (
        shap_original,
        original_feature_names,
    ) = (
        aggregate_shap_to_original_features(
            shap_encoded,
            encoded_feature_names,
            feature_groups,
        )
    )

    print(
        "Aggregated SHAP shape:",
        shap_original.shape,
    )

    print(
        "Original features:",
        original_feature_names,
    )

    # ==================================================================
    # Heatmap tables
    # ==================================================================

    heatmap_all = (
        build_heatmap_table(
            shap_original,
            original_feature_names,
            output_names,
        )
    )

    # ------------------------------------------------------------------
    # Exclude OTHER for biological protein interpretation
    # ------------------------------------------------------------------

    individual_outputs = [
        protein
        for protein in panel
        if protein != "OTHER"
    ]

    heatmap_proteins = (
        heatmap_all.loc[
            individual_outputs
        ].copy()
    )

    # ------------------------------------------------------------------
    # Sort original features by global importance
    # ------------------------------------------------------------------

    global_original_importance = (
        heatmap_proteins
        .mean(
            axis=0
        )
        .sort_values(
            ascending=False
        )
    )

    sorted_original_features = (
        global_original_importance
        .index
        .tolist()
    )

    heatmap_all = (
        heatmap_all[
            sorted_original_features
        ]
    )

    heatmap_proteins = (
        heatmap_proteins[
            sorted_original_features
        ]
    )

    # ------------------------------------------------------------------
    # Relative heatmap
    # ------------------------------------------------------------------

    relative_heatmap = (
        relative_to_global_maximum(
            heatmap_proteins
        )
    )

    # ------------------------------------------------------------------
    # Top 30 proteins
    # ------------------------------------------------------------------

    protein_importance = (
        heatmap_proteins
        .mean(
            axis=1
        )
        .sort_values(
            ascending=False
        )
    )

    top_protein_ids = (
        protein_importance
        .head(
            TOP_N_PROTEINS
        )
        .index
        .tolist()
    )

    top30_heatmap = (
        relative_heatmap.loc[
            top_protein_ids
        ]
    )

    # ==================================================================
    # Encoded-feature global importance
    # ==================================================================

    # Exclude OTHER output.
    panel_output_indices = [
        output_names.index(
            protein
        )
        for protein
        in panel
    ]

    shap_encoded_panel = (
        shap_encoded[
            :,
            :,
            panel_output_indices,
        ]
    )

    encoded_importance = (
        global_shap_feature_importance(
            shap_encoded_panel,
            encoded_feature_names,
            relative=True,
        )
    )

    # ==================================================================
    # Original-feature importance table
    # ==================================================================

    original_importance_df = pd.DataFrame(
        {
            "feature":
                global_original_importance.index,

            "mean_abs_shap":
                global_original_importance.values,
        }
    )

    maximum_original = float(
        original_importance_df[
            "mean_abs_shap"
        ].max()
    )

    original_importance_df[
        "relative_mean_abs_shap"
    ] = (
        original_importance_df[
            "mean_abs_shap"
        ]
        / maximum_original
        * 100.0
    )

    original_importance_df[
        "feature_group"
    ] = (
        original_importance_df[
            "feature"
        ]
        .apply(
            lambda feature:
                (
                    "Experimental Conditions"
                    if feature
                    in EXPERIMENTAL_FEATURES
                    else
                    "NP Physicochemical Properties"
                )
        )
    )

    # ==================================================================
    # Top-protein annotation table
    # ==================================================================

    top30_annotation = []

    for rank, protein_id in enumerate(
        top_protein_ids,
        start=1,
    ):

        top30_annotation.append(
            {
                "rank":
                    rank,

                "accession":
                    protein_id,

                "protein_name":
                    id_to_name.get(
                        protein_id,
                        "<UNKNOWN>",
                    ),

                "category":
                    id_to_category.get(
                        protein_id,
                        "Other/Mixed",
                    ),

                "mean_abs_shap":
                    float(
                        protein_importance[
                            protein_id
                        ]
                    ),
            }
        )

    top30_annotation_df = (
        pd.DataFrame(
            top30_annotation
        )
    )

    # ==================================================================
    # Save results
    # ==================================================================

    print()

    print(
        "[7/7] Saving SHAP outputs..."
    )

    # ------------------------------------------------------------------
    # Raw encoded SHAP values
    # ------------------------------------------------------------------

    np.savez_compressed(
        results_dir
        / "shap_values_encoded.npz",

        shap_values=(
            shap_encoded.astype(
                np.float32
            )
        ),

        explained_test_indices=(
            test_indices.astype(
                np.int32
            )
        ),

        background_indices=(
            background_indices.astype(
                np.int32
            )
        ),
    )

    # ------------------------------------------------------------------
    # Original-feature SHAP values
    # ------------------------------------------------------------------

    np.savez_compressed(
        results_dir
        / "shap_values_original_features.npz",

        shap_values=(
            shap_original.astype(
                np.float32
            )
        ),

        explained_test_indices=(
            test_indices.astype(
                np.int32
            )
        ),
    )

    # ------------------------------------------------------------------
    # Encoded test feature values
    # ------------------------------------------------------------------

    pd.DataFrame(
        X_explain,
        columns=encoded_feature_names,
    ).to_csv(
        results_dir
        / "explained_test_features_encoded.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    pd.DataFrame(
        {
            "encoded_feature":
                encoded_feature_names
        }
    ).to_csv(
        results_dir
        / "encoded_feature_names.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    with open(
        results_dir
        / "feature_groups.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feature_groups,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------

    heatmap_all.to_csv(
        results_dir
        / "shap_heatmap_all_outputs.csv"
    )

    heatmap_proteins.to_csv(
        results_dir
        / "shap_heatmap_individual_proteins.csv"
    )

    relative_heatmap.to_csv(
        results_dir
        / "shap_heatmap_relative_pct_max.csv"
    )

    top30_heatmap.to_csv(
        results_dir
        / "shap_heatmap_top30_relative_pct_max.csv"
    )

    # ------------------------------------------------------------------
    # Importance tables
    # ------------------------------------------------------------------

    original_importance_df.to_csv(
        results_dir
        / "global_original_feature_importance.csv",
        index=False,
    )

    encoded_importance.to_csv(
        results_dir
        / "global_encoded_feature_importance.csv",
        index=False,
    )

    top30_annotation_df.to_csv(
        results_dir
        / "top30_proteins.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = {
        "random_seed":
            RANDOM_SEED,

        "quick_mode":
            bool(
                args.quick
            ),

        "background_samples":
            int(
                len(
                    background
                )
            ),

        "explained_test_samples":
            int(
                len(
                    X_explain
                )
            ),

        "development_samples":
            int(
                len(
                    prepared.X_train
                )
            ),

        "total_test_samples":
            int(
                len(
                    prepared.X_test
                )
            ),

        "encoded_features":
            int(
                len(
                    encoded_feature_names
                )
            ),

        "original_features":
            int(
                len(
                    original_feature_names
                )
            ),

        "model_outputs":
            int(
                n_outputs
            ),

        "individual_proteins":
            int(
                len(
                    panel
                )
            ),

        "shap_output":
            "adsorption probability",

        "relative_shap_definition":
            (
                "mean absolute SHAP divided by "
                "global maximum, multiplied by 100"
            ),

        "relative_shap_axis_label":
            (
                "Relative mean |SHAP| (% of maximum)"
            ),
    }

    with open(
        results_dir
        / "shap_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    # ==================================================================
    # Console summary
    # ==================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "TOP ORIGINAL FEATURES"
    )

    print(
        "=" * 72
    )

    print(
        original_importance_df[
            [
                "feature",
                "mean_abs_shap",
                "relative_mean_abs_shap",
                "feature_group",
            ]
        ]
        .head(
            12
        )
        .to_string(
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
        "TOP 10 PROTEINS BY SHAP IMPORTANCE"
    )

    print(
        "=" * 72
    )

    print(
        top30_annotation_df
        .head(
            10
        )
        .to_string(
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
        "SHAP ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        results_dir
    )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()