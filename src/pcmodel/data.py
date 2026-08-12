"""
Protein Corona Data Processing
==============================

This module contains reusable functions for loading, splitting, preprocessing,
and preparing the nanoparticle-protein corona datasets used by the prediction
models.

Main responsibilities
---------------------
1. Load nanoparticle feature and protein abundance datasets.
2. Create reproducible model-development and held-out test splits.
3. Select the representative protein panel using development data only.
4. Construct protein adsorption and relative-abundance targets.
5. Fit feature preprocessing using development data only.
6. Apply the same preprocessing to held-out and external-validation samples.

Design principles
-----------------
- Protein-panel selection is performed using model-development data only.
- Feature preprocessing is fitted on model-development data only to avoid
  information leakage.
- Data processing is separated from model architecture and training code.
- The current default reproduces the original notebook behavior by including
  the OTHER bin in both the adsorption and abundance targets.

Model definitions belong in ``models.py``.
Training procedures belong in ``training.py``.
Performance calculations belong in ``metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ======================================================================
# Type aliases
# ======================================================================

PathLike = Union[str, Path]


# ======================================================================
# Reproducibility and split settings
# ======================================================================

SEED = 42
TEST_SIZE = 0.20


# ======================================================================
# Feature definitions
# ======================================================================

# Continuous variables
NUMERIC_COLS = [
    "size_nm",
    "zp_mv",
]

# Ordered categorical variables
ORDINAL_COLS = [
    "incub_time",
    "washing_steps",
]

# Nominal categorical variables
NOMINAL_COLS = [
    "mod_type",
    "mod_charge",
    "np_type",
    "np_subtype",
    "zp_charge",
    "zp_solvent",
    "hd_solvent",
    "agitation",
]

FEATURE_COLS = (
    NUMERIC_COLS
    + ORDINAL_COLS
    + NOMINAL_COLS
)

# Defined ordinal order
ORDINAL_ORDER: Dict[str, List[str]] = {
    "incub_time": [
        "<30",
        "30~60",
        ">60",
    ],
    "washing_steps": [
        "1~3",
        "4~6",
        "Not Reported",
    ],
}


# ======================================================================
# Known category harmonization
# ======================================================================

# Data_val contains "Surfactant", while the development dataset uses
# "Surfactants". These should represent the same category.
CATEGORY_ALIASES: Dict[str, Dict[str, str]] = {
    "mod_type": {
        "Surfactant": "Surfactants",
    },
}


# ======================================================================
# Protein-panel settings
# ======================================================================

MIN_DETECT_THRESHOLD = 0.0
MIN_NPS_PER_PROTEIN = 10
COVERAGE_TARGET = 0.99


# ======================================================================
# Data containers
# ======================================================================


@dataclass
class ProteinPanelResult:
    """
    Store results from protein-panel selection.
    """

    panel: List[str]

    # Protein-level selection statistics
    stats: pd.DataFrame

    # Cumulative abundance among eligible proteins
    cumulative_abundance: pd.Series

    # Fraction of total abundance retained after frequency filtering
    eligible_abundance_fraction_of_all: float


@dataclass
class PreparedData:
    """
    Container for fully prepared model-development and test data.
    """

    X_train: np.ndarray
    X_test: np.ndarray

    Y_presence_train: np.ndarray
    Y_presence_test: np.ndarray

    Y_abundance_train: np.ndarray
    Y_abundance_test: np.ndarray

    train_ids: List[str]
    test_ids: List[str]

    feature_names: List[str]

    panel: List[str]
    presence_columns: List[str]
    abundance_columns: List[str]

    preprocessor: "FeaturePreprocessor"
    panel_result: ProteinPanelResult

    feature_train_df: pd.DataFrame
    feature_test_df: pd.DataFrame

    abundance_train_df: pd.DataFrame
    abundance_test_df: pd.DataFrame


# ======================================================================
# Validation helpers
# ======================================================================


def _require_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    dataframe_name: str,
) -> None:
    """
    Verify that all required columns are present.
    """

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def _check_unique_np_ids(
    df: pd.DataFrame,
    dataframe_name: str,
) -> None:
    """
    Verify that NP_ID uniquely identifies samples.
    """

    duplicated = df.loc[
        df["NP_ID"].duplicated(),
        "NP_ID",
    ].tolist()

    if duplicated:
        raise ValueError(
            f"{dataframe_name} contains duplicated NP_ID values. "
            f"Examples: {duplicated[:10]}"
        )


# ======================================================================
# Category harmonization
# ======================================================================


def canonicalize_feature_categories(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Harmonize known categorical spelling differences.
    """

    output = df.copy()

    for column, replacements in CATEGORY_ALIASES.items():

        if column in output.columns:
            output[column] = output[column].replace(
                replacements
            )

    return output


# ======================================================================
# Data loading
# ======================================================================


def load_data(
    feature_path: PathLike,
    abundance_path: PathLike,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load nanoparticle feature data and protein abundance data.

    Parameters
    ----------
    feature_path
        Path to Data_1.csv.

    abundance_path
        Path to Data_2.csv.

    Returns
    -------
    feature_df
        Nanoparticle physicochemical and experimental features.

    abundance_df
        Wide-format protein abundance table.
    """

    feature_df = pd.read_csv(
        feature_path
    )

    abundance_df = pd.read_csv(
        abundance_path
    )

    # Check required columns
    _require_columns(
        feature_df,
        ["NP_ID"] + FEATURE_COLS,
        "Feature dataset",
    )

    _require_columns(
        abundance_df,
        ["NP_ID"],
        "Abundance dataset",
    )

    # Check NP_ID uniqueness
    _check_unique_np_ids(
        feature_df,
        "Feature dataset",
    )

    _check_unique_np_ids(
        abundance_df,
        "Abundance dataset",
    )

    # Harmonize known feature categories
    feature_df = canonicalize_feature_categories(
        feature_df
    )

    # Identify protein columns
    protein_cols = [
        column
        for column in abundance_df.columns
        if column != "NP_ID"
    ]

    if not protein_cols:
        raise ValueError(
            "No protein abundance columns were found."
        )

    # Convert protein abundance values to numeric
    for column in protein_cols:

        abundance_df[column] = pd.to_numeric(
            abundance_df[column],
            errors="coerce",
        )

    # Verify that every abundance sample has feature data
    missing_feature_ids = (
        set(abundance_df["NP_ID"])
        - set(feature_df["NP_ID"])
    )

    if missing_feature_ids:

        examples = sorted(
            missing_feature_ids
        )[:10]

        raise ValueError(
            "Some abundance samples do not have matching "
            f"feature rows. Examples: {examples}"
        )

    return feature_df, abundance_df


# ======================================================================
# Stratification helper
# ======================================================================


def make_quantile_bins(
    counts: np.ndarray,
    q: int = 5,
) -> np.ndarray:
    """
    Create quantile-based labels for stratified splitting.

    If quantile binning cannot create at least two groups,
    a single zero-valued group is returned.
    """

    counts = np.asarray(
        counts
    )

    try:

        bins = pd.qcut(
            counts,
            q=q,
            duplicates="drop",
            labels=False,
        )

        labels = (
            pd.Series(bins)
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        if len(np.unique(labels)) >= 2:
            return labels

    except (ValueError, TypeError):
        pass

    return np.zeros(
        len(counts),
        dtype=int,
    )


# ======================================================================
# Train/test splitting
# ======================================================================


def split_abundance_data(
    abundance_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split NP samples into model-development and held-out test sets.

    Stratification is based on the number of detected proteins
    in each NP sample.
    """

    protein_cols = [
        column
        for column in abundance_df.columns
        if column != "NP_ID"
    ]

    detected_counts = (
        abundance_df[protein_cols]
        > MIN_DETECT_THRESHOLD
    ).sum(axis=1)

    strat_labels = make_quantile_bins(
        detected_counts.to_numpy()
    )

    if len(np.unique(strat_labels)) >= 2:
        stratify = strat_labels
    else:
        stratify = None

    train_ids, test_ids = train_test_split(
        abundance_df["NP_ID"],
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    train_set = set(
        train_ids
    )

    test_set = set(
        test_ids
    )

    abundance_train = (
        abundance_df[
            abundance_df["NP_ID"].isin(train_set)
        ]
        .reset_index(drop=True)
    )

    abundance_test = (
        abundance_df[
            abundance_df["NP_ID"].isin(test_set)
        ]
        .reset_index(drop=True)
    )

    return (
        abundance_train,
        abundance_test,
    )


# ======================================================================
# Protein-panel selection
# ======================================================================


def select_protein_panel(
    abundance_train: pd.DataFrame,
    min_nps: int = MIN_NPS_PER_PROTEIN,
    coverage_target: float = COVERAGE_TARGET,
) -> ProteinPanelResult:
    """
    Select the representative protein panel using development data only.

    Steps
    -----
    1. Remove proteins detected in fewer than ``min_nps`` samples.
    2. Rank eligible proteins by total abundance.
    3. Retain the smallest panel reaching ``coverage_target`` cumulative
       abundance among the eligible proteins.
    """

    protein_cols = [
        column
        for column in abundance_train.columns
        if column != "NP_ID"
    ]

    clean = (
        abundance_train[protein_cols]
        .clip(lower=0.0)
        .fillna(0.0)
    )

    # Total abundance of each protein
    total_abundance = clean.sum(
        axis=0
    )

    # Number of NPs in which each protein is detected
    detection_frequency = (
        clean > MIN_DETECT_THRESHOLD
    ).sum(axis=0)

    stats = pd.DataFrame(
        {
            "sum_abundance": total_abundance,
            "n_nps_detected": detection_frequency,
        }
    )

    stats["detection_fraction"] = (
        stats["n_nps_detected"]
        / max(
            len(abundance_train),
            1,
        )
    )

    # Frequency filter
    eligible = stats[
        stats["n_nps_detected"]
        >= min_nps
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No proteins remain after detection-frequency filtering."
        )

    eligible_total = float(
        eligible["sum_abundance"].sum()
    )

    all_total = float(
        stats["sum_abundance"].sum()
    )

    if eligible_total <= 0:

        raise ValueError(
            "Eligible proteins have zero total abundance."
        )

    # Rank proteins by abundance
    ranked_abundance = (
        eligible["sum_abundance"]
        .sort_values(
            ascending=False
        )
    )

    # Cumulative abundance among eligible proteins
    cumulative = (
        ranked_abundance.cumsum()
        / eligible_total
    )

    threshold_indices = np.flatnonzero(
        cumulative.to_numpy()
        >= coverage_target
    )

    if len(threshold_indices):

        k = int(
            threshold_indices[0]
            + 1
        )

    else:

        k = len(
            cumulative
        )

    panel = list(
        cumulative.index[:k]
    )

    # Fraction of all abundance represented by proteins
    # surviving the frequency filter
    if all_total > 0:

        eligible_fraction_of_all = (
            eligible_total
            / all_total
        )

    else:

        eligible_fraction_of_all = np.nan

    stats = stats.sort_values(
        "sum_abundance",
        ascending=False,
    )

    return ProteinPanelResult(
        panel=panel,
        stats=stats,
        cumulative_abundance=cumulative,
        eligible_abundance_fraction_of_all=(
            eligible_fraction_of_all
        ),
    )


# ======================================================================
# Target construction
# ======================================================================


def build_targets(
    abundance_df: pd.DataFrame,
    panel: Sequence[str],
    include_other_bin: bool = True,
    include_other_in_presence: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
]:
    """
    Construct adsorption and abundance model targets.

    Parameters
    ----------
    abundance_df
        Wide-format protein abundance table.

    panel
        Selected individual proteins.

    include_other_bin
        If True, abundance outside the selected panel is pooled
        into an OTHER category.

    include_other_in_presence
        If True, OTHER is also included in the adsorption target.
        This reproduces the original notebook architecture.

        If False, presence contains only individual proteins,
        while abundance retains the OTHER category.

    Returns
    -------
    Y_presence
        Binary protein adsorption/detection targets.

    Y_abundance
        Relative abundance distributions.

    presence_columns
        Names corresponding to Y_presence columns.

    abundance_columns
        Names corresponding to Y_abundance columns.
    """

    panel = list(
        panel
    )

    missing_panel = [
        column
        for column in panel
        if column not in abundance_df.columns
    ]

    if missing_panel:

        raise ValueError(
            "Selected proteins are missing from the abundance "
            f"dataset. Examples: {missing_panel[:10]}"
        )

    # Selected protein abundance
    panel_abundance = (
        abundance_df[panel]
        .clip(lower=0.0)
        .fillna(0.0)
    )

    # Binary adsorption labels
    panel_presence = (
        panel_abundance
        > MIN_DETECT_THRESHOLD
    ).astype(
        np.float32
    )

    # --------------------------------------------------------------
    # No OTHER bin
    # --------------------------------------------------------------

    if not include_other_bin:

        denominator = (
            panel_abundance
            .sum(axis=1)
            .replace(
                0.0,
                np.nan,
            )
        )

        abundance_fraction = (
            panel_abundance
            .div(
                denominator,
                axis=0,
            )
            .fillna(0.0)
        )

        return (
            panel_presence.to_numpy(
                dtype=np.float32
            ),
            abundance_fraction.to_numpy(
                dtype=np.float32
            ),
            panel.copy(),
            panel.copy(),
        )

    # --------------------------------------------------------------
    # Build OTHER bin
    # --------------------------------------------------------------

    all_protein_cols = [
        column
        for column in abundance_df.columns
        if column != "NP_ID"
    ]

    total_all = (
        abundance_df[
            all_protein_cols
        ]
        .clip(lower=0.0)
        .fillna(0.0)
        .sum(axis=1)
    )

    total_panel = (
        panel_abundance
        .sum(axis=1)
    )

    leftover = (
        total_all
        - total_panel
    ).clip(
        lower=0.0
    )

    abundance_plus_other = pd.concat(
        [
            panel_abundance,
            leftover.rename(
                "OTHER"
            ),
        ],
        axis=1,
    )

    denominator = (
        abundance_plus_other
        .sum(axis=1)
        .replace(
            0.0,
            np.nan,
        )
    )

    abundance_fraction = (
        abundance_plus_other
        .div(
            denominator,
            axis=0,
        )
        .fillna(0.0)
    )

    abundance_columns = (
        panel
        + ["OTHER"]
    )

    # --------------------------------------------------------------
    # Presence target
    # --------------------------------------------------------------

    if include_other_in_presence:

        other_presence = (
            leftover
            > MIN_DETECT_THRESHOLD
        ).astype(
            np.float32
        )

        presence_df = pd.concat(
            [
                panel_presence,
                other_presence.rename(
                    "OTHER"
                ),
            ],
            axis=1,
        )

        presence_columns = (
            abundance_columns.copy()
        )

    else:

        presence_df = (
            panel_presence
        )

        presence_columns = (
            panel.copy()
        )

    return (
        presence_df.to_numpy(
            dtype=np.float32
        ),
        abundance_fraction.to_numpy(
            dtype=np.float32
        ),
        presence_columns,
        abundance_columns,
    )


# ======================================================================
# Feature alignment
# ======================================================================


def align_features_to_ids(
    feature_df: pd.DataFrame,
    np_ids: Sequence[str],
) -> pd.DataFrame:
    """
    Align Data_1 feature rows to a specified NP_ID order.
    """

    id_frame = pd.DataFrame(
        {
            "NP_ID": list(
                np_ids
            )
        }
    )

    aligned = id_frame.merge(
        feature_df,
        on="NP_ID",
        how="left",
        validate="one_to_one",
    )

    missing_rows = (
        aligned[FEATURE_COLS]
        .isna()
        .all(axis=1)
    )

    if missing_rows.any():

        missing_ids = aligned.loc[
            missing_rows,
            "NP_ID",
        ].tolist()

        raise ValueError(
            "Feature rows are missing for some NP_ID values. "
            f"Examples: {missing_ids[:10]}"
        )

    return aligned


# ======================================================================
# Feature preprocessing
# ======================================================================


class FeaturePreprocessor:
    """
    Preprocess nanoparticle and experimental features.

    Numeric variables
    -----------------
    Converted to numeric, imputed using training-set medians,
    and standardized using a StandardScaler fitted on training data only.

    Ordinal variables
    -----------------
    Converted to integer levels according to ORDINAL_ORDER.

    Nominal variables
    -----------------
    One-hot encoded using categories observed in model-development data.
    New categories at prediction time become all-zero dummy blocks.
    """

    def __init__(
        self,
    ) -> None:

        self.numeric_medians_: pd.Series | None = None
        self.scaler_: StandardScaler | None = None

        self.nominal_dummy_columns_: List[str] | None = None

        self.feature_names_: List[str] | None = None

        self.is_fitted_: bool = False

    @staticmethod
    def _encode_ordinal(
        series: pd.Series,
        ordered_categories: Sequence[str],
    ) -> pd.Series:
        """
        Convert ordinal categories to numeric values.
        """

        mapping = {
            category: index
            for index, category
            in enumerate(
                ordered_categories
            )
        }

        unknown = (
            series.notna()
            & ~series.isin(
                mapping.keys()
            )
        )

        if unknown.any():

            examples = sorted(
                series.loc[
                    unknown
                ]
                .astype(str)
                .unique()
            )

            raise ValueError(
                "Unknown ordinal category encountered: "
                f"{examples[:10]}"
            )

        return (
            series
            .map(mapping)
            .fillna(0)
            .astype(
                np.float32
            )
        )

    def fit(
        self,
        feature_train_df: pd.DataFrame,
    ) -> "FeaturePreprocessor":
        """
        Fit preprocessing parameters using development data only.
        """

        _require_columns(
            feature_train_df,
            ["NP_ID"] + FEATURE_COLS,
            "Training feature dataset",
        )

        df = canonicalize_feature_categories(
            feature_train_df
        )

        # ----------------------------------------------------------
        # Numeric block
        # ----------------------------------------------------------

        numeric = pd.DataFrame(
            index=df.index
        )

        for column in NUMERIC_COLS:

            numeric[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        self.numeric_medians_ = (
            numeric.median()
        )

        numeric = numeric.fillna(
            self.numeric_medians_
        )

        self.scaler_ = (
            StandardScaler()
        )

        self.scaler_.fit(
            numeric[
                NUMERIC_COLS
            ].to_numpy(
                dtype=np.float32
            )
        )

        # ----------------------------------------------------------
        # Nominal block
        # ----------------------------------------------------------

        nominal = pd.get_dummies(
            df[
                NOMINAL_COLS
            ].astype(str),
            dummy_na=False,
            dtype=np.float32,
        )

        self.nominal_dummy_columns_ = list(
            nominal.columns
        )

        self.feature_names_ = (
            NUMERIC_COLS
            + ORDINAL_COLS
            + self.nominal_dummy_columns_
        )

        self.is_fitted_ = True

        return self

    def transform(
        self,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform samples using the frozen training-fitted preprocessor.
        """

        if not self.is_fitted_:

            raise RuntimeError(
                "FeaturePreprocessor must be fitted "
                "before transform()."
            )

        assert self.numeric_medians_ is not None
        assert self.scaler_ is not None
        assert self.nominal_dummy_columns_ is not None
        assert self.feature_names_ is not None

        _require_columns(
            feature_df,
            ["NP_ID"] + FEATURE_COLS,
            "Feature dataset",
        )

        df = canonicalize_feature_categories(
            feature_df
        )

        # ----------------------------------------------------------
        # Numeric block
        # ----------------------------------------------------------

        numeric = pd.DataFrame(
            index=df.index
        )

        for column in NUMERIC_COLS:

            numeric[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        numeric = numeric.fillna(
            self.numeric_medians_
        )

        numeric_scaled = pd.DataFrame(
            self.scaler_.transform(
                numeric[
                    NUMERIC_COLS
                ].to_numpy(
                    dtype=np.float32
                )
            ),
            columns=NUMERIC_COLS,
            index=df.index,
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Ordinal block
        # ----------------------------------------------------------

        ordinal = pd.DataFrame(
            index=df.index
        )

        for column in ORDINAL_COLS:

            ordinal[column] = (
                self._encode_ordinal(
                    df[column],
                    ORDINAL_ORDER[
                        column
                    ],
                )
            )

        # ----------------------------------------------------------
        # Nominal block
        # ----------------------------------------------------------

        nominal = pd.get_dummies(
            df[
                NOMINAL_COLS
            ].astype(str),
            dummy_na=False,
            dtype=np.float32,
        )

        # Ensure test/external data have exactly the same
        # one-hot columns as training data.
        nominal = nominal.reindex(
            columns=(
                self.nominal_dummy_columns_
            ),
            fill_value=0.0,
        )

        # ----------------------------------------------------------
        # Combine blocks
        # ----------------------------------------------------------

        encoded = pd.concat(
            [
                df[
                    ["NP_ID"]
                ].reset_index(
                    drop=True
                ),
                numeric_scaled.reset_index(
                    drop=True
                ),
                ordinal.reset_index(
                    drop=True
                ),
                nominal.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        return encoded

    def fit_transform(
        self,
        feature_train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit preprocessing and transform training data.
        """

        return (
            self.fit(
                feature_train_df
            )
            .transform(
                feature_train_df
            )
        )

    @property
    def feature_names(
        self,
    ) -> List[str]:
        """
        Return encoded feature names.
        """

        if (
            not self.is_fitted_
            or self.feature_names_ is None
        ):

            raise RuntimeError(
                "FeaturePreprocessor has not been fitted."
            )

        return (
            self.feature_names_.copy()
        )


# ======================================================================
# Complete model-data preparation
# ======================================================================


def prepare_model_data(
    feature_path: PathLike,
    abundance_path: PathLike,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
    min_nps: int = MIN_NPS_PER_PROTEIN,
    coverage_target: float = COVERAGE_TARGET,
    include_other_bin: bool = True,
    include_other_in_presence: bool = True,
) -> PreparedData:
    """
    Run the complete data-preparation workflow.

    The default ``include_other_in_presence=True`` preserves the
    output dimensionality of the original notebook.
    """

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    feature_df, abundance_df = load_data(
        feature_path,
        abundance_path,
    )

    # --------------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------------

    (
        abundance_train,
        abundance_test,
    ) = split_abundance_data(
        abundance_df,
        test_size=test_size,
        seed=seed,
    )

    # --------------------------------------------------------------
    # Select proteins using development data only
    # --------------------------------------------------------------

    panel_result = select_protein_panel(
        abundance_train,
        min_nps=min_nps,
        coverage_target=coverage_target,
    )

    panel = (
        panel_result.panel
    )

    # --------------------------------------------------------------
    # Build training targets
    # --------------------------------------------------------------

    (
        Y_presence_train,
        Y_abundance_train,
        presence_columns,
        abundance_columns,
    ) = build_targets(
        abundance_train,
        panel,
        include_other_bin=include_other_bin,
        include_other_in_presence=(
            include_other_in_presence
        ),
    )

    # --------------------------------------------------------------
    # Build test targets
    # --------------------------------------------------------------

    (
        Y_presence_test,
        Y_abundance_test,
        presence_columns_test,
        abundance_columns_test,
    ) = build_targets(
        abundance_test,
        panel,
        include_other_bin=include_other_bin,
        include_other_in_presence=(
            include_other_in_presence
        ),
    )

    if (
        presence_columns
        != presence_columns_test
    ):

        raise RuntimeError(
            "Train/test presence output columns do not match."
        )

    if (
        abundance_columns
        != abundance_columns_test
    ):

        raise RuntimeError(
            "Train/test abundance output columns do not match."
        )

    # --------------------------------------------------------------
    # Align feature rows
    # --------------------------------------------------------------

    train_ids = (
        abundance_train[
            "NP_ID"
        ].tolist()
    )

    test_ids = (
        abundance_test[
            "NP_ID"
        ].tolist()
    )

    feature_train_raw = (
        align_features_to_ids(
            feature_df,
            train_ids,
        )
    )

    feature_test_raw = (
        align_features_to_ids(
            feature_df,
            test_ids,
        )
    )

    # --------------------------------------------------------------
    # Fit preprocessing on development data only
    # --------------------------------------------------------------

    preprocessor = (
        FeaturePreprocessor()
    )

    feature_train_encoded = (
        preprocessor.fit_transform(
            feature_train_raw
        )
    )

    feature_test_encoded = (
        preprocessor.transform(
            feature_test_raw
        )
    )

    feature_names = (
        preprocessor.feature_names
    )

    # --------------------------------------------------------------
    # Convert to NumPy matrices
    # --------------------------------------------------------------

    X_train = (
        feature_train_encoded[
            feature_names
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_test = (
        feature_test_encoded[
            feature_names
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------------
    # Return everything needed downstream
    # --------------------------------------------------------------

    return PreparedData(
        X_train=X_train,
        X_test=X_test,

        Y_presence_train=(
            Y_presence_train
        ),
        Y_presence_test=(
            Y_presence_test
        ),

        Y_abundance_train=(
            Y_abundance_train
        ),
        Y_abundance_test=(
            Y_abundance_test
        ),

        train_ids=train_ids,
        test_ids=test_ids,

        feature_names=(
            feature_names
        ),

        panel=panel,

        presence_columns=(
            presence_columns
        ),

        abundance_columns=(
            abundance_columns
        ),

        preprocessor=(
            preprocessor
        ),

        panel_result=(
            panel_result
        ),

        feature_train_df=(
            feature_train_encoded
        ),

        feature_test_df=(
            feature_test_encoded
        ),

        abundance_train_df=(
            abundance_train
        ),

        abundance_test_df=(
            abundance_test
        ),
    )


# ======================================================================
# External-validation feature preparation
# ======================================================================


def prepare_external_features(
    external_df: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
) -> Tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """
    Transform external-validation features using the exact same
    preprocessor fitted on model-development data.

    The external dataset may also contain protein abundance columns.
    Only NP_ID and the model feature columns are used here.
    """

    _require_columns(
        external_df,
        ["NP_ID"] + FEATURE_COLS,
        "External validation dataset",
    )

    feature_subset = external_df[
        ["NP_ID"] + FEATURE_COLS
    ].copy()

    encoded = (
        preprocessor.transform(
            feature_subset
        )
    )

    feature_names = (
        preprocessor.feature_names
    )

    X_external = (
        encoded[
            feature_names
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    return (
        X_external,
        encoded,
    )