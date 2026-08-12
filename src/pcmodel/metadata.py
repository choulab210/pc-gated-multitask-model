"""
Protein Metadata Utilities
==========================

This module manages protein annotation used by the protein-corona model.

Responsibilities
----------------
- Define the functional protein categories.
- Assign protein names to functional categories.
- Load reproducible protein metadata from CSV.
- Build UniProt ID -> protein name mappings.
- Build UniProt ID -> functional category mappings.
- Validate metadata against the model protein panel.

Important
---------
Model training and external validation should NOT depend on a live
UniProt query.

Protein names should be retrieved once and saved to:

    data/protein_metadata.csv

After that, all analyses should use the saved metadata file so that
results remain reproducible and can run offline on HPCC.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd


# ======================================================================
# Functional categories
# ======================================================================


CATEGORY_ORDER = [
    "Apolipoproteins",
    "Coagulation/Fibrinogen",
    "Complement System",
    "Cytoskeletal",
    "Immunoglobulins",
    "Metabolic Enzymes",
    "Protease/Inhibitors",
    "Transport/Binding",
    "Other/Mixed",
]


# ======================================================================
# Protein-name category assignment
# ======================================================================


def categorize_protein_name(
    name: str,
) -> str:
    """
    Assign a protein name to one of the nine functional categories.

    The rules reproduce the category logic used in the original
    two-head model notebook.

    Parameters
    ----------
    name
        Protein name, normally the UniProt recommended protein name.

    Returns
    -------
    str
        Functional protein category.
    """

    if (
        not name
        or name == "<UNKNOWN>"
    ):
        return "Other/Mixed"

    n = str(
        name
    ).lower()

    # --------------------------------------------------------------
    # Apolipoproteins
    # --------------------------------------------------------------

    if (
        "apolipoprotein" in n
        or n.startswith("apo ")
    ):
        return "Apolipoproteins"

    # --------------------------------------------------------------
    # Coagulation / Fibrinogen
    # --------------------------------------------------------------

    if (
        "fibrinogen" in n
        or "coagulation factor" in n
        or "prothrombin" in n
        or "tissue factor" in n
        or "vitamin k-dependent" in n
        or "heparin cofactor" in n
        or "antiplasmin" in n
        or "antithrombin" in n
        or "platelet factor" in n
        or "platelet basic protein" in n
        or "platelet glycoprotein" in n
    ):
        return "Coagulation/Fibrinogen"

    # --------------------------------------------------------------
    # Complement System
    # --------------------------------------------------------------

    if (
        "complement" in n
        or "properdin" in n
        or "ficolin" in n
        or re.search(
            r"\bc[1-9][a-z0-9\-]*\b",
            n,
        )
    ):

        if "c-reactive" not in n:

            return "Complement System"

    # --------------------------------------------------------------
    # Cytoskeletal
    # --------------------------------------------------------------

    if any(
        keyword in n
        for keyword in [
            "keratin",
            "actin",
            "tubulin",
            "myosin",
            "filamin",
            "tropomyosin",
            "vimentin",
            "spectrin",
            "alpha-actinin",
            "profilin",
            "gelsolin",
            "talin",
            "annexin",
            "cofilin",
            "vinculin",
            "moesin",
            "ezrin",
            "integrin",
            "band 3 anion",
        ]
    ):
        return "Cytoskeletal"

    # --------------------------------------------------------------
    # Immunoglobulins
    # --------------------------------------------------------------

    if "immunoglobulin" in n:

        return "Immunoglobulins"

    # --------------------------------------------------------------
    # Metabolic Enzymes
    # --------------------------------------------------------------

    if any(
        keyword in n
        for keyword in [
            "kinase",
            "dehydrogenase",
            "isomerase",
            "oxidase",
            "peroxidase",
            "transferase",
            "reductase",
            "synthase",
            "phosphatase",
            "hydrolase",
            "esterase",
            "enolase",
            "lysozyme",
        ]
    ):
        return "Metabolic Enzymes"

    # --------------------------------------------------------------
    # Protease / Inhibitors
    # --------------------------------------------------------------

    if any(
        keyword in n
        for keyword in [
            "protease",
            "proteinase",
            "inhibitor",
            "antitrypsin",
            "antichymotrypsin",
            "serpin",
            "kininogen",
            "plasminogen",
            "kallikrein",
            "carboxypeptidase",
        ]
    ):
        return "Protease/Inhibitors"

    # --------------------------------------------------------------
    # Transport / Binding
    # --------------------------------------------------------------

    if any(
        keyword in n
        for keyword in [
            "albumin",
            "transferrin",
            "haptoglobin",
            "hemopexin",
            "ceruloplasmin",
            "transthyretin",
            "vitamin d-binding",
            "retinol-binding",
            "binding protein",
            "binding-protein",
            "galectin",
            "lipopolysaccharide-binding",
            "insulin-like growth factor-binding",
            "alpha-2-macroglobulin",
            "alpha-1-acid glycoprotein",
            "vitronectin",
            "clusterin",
            "fibronectin",
            "thrombospondin",
            "histidine-rich glycoprotein",
            "beta-2-glycoprotein",
            "alpha-2-hs-glycoprotein",
            "alpha-1b-glycoprotein",
            "ambp",
            "selenoprotein p",
            "pigment epithelium-derived factor",
            "cartilage oligomeric matrix",
        ]
    ):
        return "Transport/Binding"

    return "Other/Mixed"


# ======================================================================
# Build metadata
# ======================================================================


def build_metadata_dataframe(
    protein_ids: Sequence[str],
    id_to_name: Mapping[str, str],
) -> pd.DataFrame:
    """
    Construct the protein metadata table.

    Parameters
    ----------
    protein_ids
        UniProt accessions in model-panel order.

    id_to_name
        UniProt accession -> protein name mapping.

    Returns
    -------
    pandas.DataFrame
        Columns:

            accession
            protein_name
            category
    """

    rows = []

    for accession in protein_ids:

        name = id_to_name.get(
            accession,
            "<UNKNOWN>",
        )

        category = (
            categorize_protein_name(
                name
            )
        )

        rows.append(
            {
                "accession": accession,
                "protein_name": name,
                "category": category,
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Save metadata
# ======================================================================


def save_protein_metadata(
    metadata: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Save protein metadata to CSV.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        path,
        index=False,
    )


# ======================================================================
# Load metadata
# ======================================================================


def load_protein_metadata(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load and validate protein metadata CSV.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Protein metadata file not found: {path}"
        )

    metadata = pd.read_csv(
        path
    )

    required_columns = {
        "accession",
        "protein_name",
        "category",
    }

    missing_columns = (
        required_columns
        - set(
            metadata.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Protein metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------------
    # Normalize text
    # --------------------------------------------------------------

    metadata[
        "accession"
    ] = (
        metadata[
            "accession"
        ]
        .astype(str)
        .str.strip()
    )

    metadata[
        "protein_name"
    ] = (
        metadata[
            "protein_name"
        ]
        .fillna("<UNKNOWN>")
        .astype(str)
        .str.strip()
    )

    metadata[
        "category"
    ] = (
        metadata[
            "category"
        ]
        .fillna("Other/Mixed")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------------
    # Check duplicated accessions
    # --------------------------------------------------------------

    duplicated = metadata[
        metadata[
            "accession"
        ].duplicated(
            keep=False
        )
    ]

    if not duplicated.empty:

        duplicated_ids = (
            duplicated[
                "accession"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate UniProt accessions found in protein metadata: "
            f"{duplicated_ids}"
        )

    # --------------------------------------------------------------
    # Unknown categories become Other/Mixed
    # --------------------------------------------------------------

    metadata.loc[
        ~metadata[
            "category"
        ].isin(
            CATEGORY_ORDER
        ),
        "category",
    ] = "Other/Mixed"

    return metadata


# ======================================================================
# Mapping helpers
# ======================================================================


def metadata_to_mappings(
    metadata: pd.DataFrame,
) -> Tuple[
    Dict[str, str],
    Dict[str, str],
]:
    """
    Convert protein metadata into lookup dictionaries.

    Returns
    -------
    id_to_name
        UniProt accession -> protein name

    id_to_category
        UniProt accession -> functional category
    """

    id_to_name = dict(
        zip(
            metadata[
                "accession"
            ],
            metadata[
                "protein_name"
            ],
        )
    )

    id_to_category = dict(
        zip(
            metadata[
                "accession"
            ],
            metadata[
                "category"
            ],
        )
    )

    return (
        id_to_name,
        id_to_category,
    )


# ======================================================================
# Panel validation
# ======================================================================


def validate_metadata_for_panel(
    metadata: pd.DataFrame,
    panel: Sequence[str],
    *,
    require_all: bool = True,
) -> Dict[str, object]:
    """
    Check metadata coverage of a model protein panel.

    Parameters
    ----------
    metadata
        Protein metadata dataframe.

    panel
        Model UniProt protein panel.

    require_all
        Raise an exception if any proteins are missing.

    Returns
    -------
    dict
        Coverage summary.
    """

    panel = list(
        panel
    )

    metadata_ids = set(
        metadata[
            "accession"
        ]
    )

    missing = [
        accession
        for accession in panel
        if accession
        not in metadata_ids
    ]

    extra = [
        accession
        for accession in metadata_ids
        if accession
        not in set(panel)
    ]

    summary = {
        "panel_size": len(
            panel
        ),

        "metadata_size": len(
            metadata
        ),

        "matched": (
            len(panel)
            - len(missing)
        ),

        "missing": missing,

        "extra": extra,
    }

    if (
        require_all
        and missing
    ):

        raise ValueError(
            "Protein metadata does not cover the complete "
            f"model panel. Missing {len(missing)} protein(s): "
            f"{missing}"
        )

    return summary


# ======================================================================
# Category audit
# ======================================================================


def category_audit(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the number and percentage of proteins in each category.
    """

    counts = (
        metadata[
            "category"
        ]
        .value_counts()
        .reindex(
            CATEGORY_ORDER,
            fill_value=0,
        )
    )

    total = len(
        metadata
    )

    percentages = (
        counts
        / total
        * 100.0
        if total > 0
        else counts.astype(float)
    )

    audit = pd.DataFrame(
        {
            "n_proteins":
                counts.astype(int),

            "percentage":
                percentages.round(2),
        }
    )

    audit.index.name = (
        "category"
    )

    return audit