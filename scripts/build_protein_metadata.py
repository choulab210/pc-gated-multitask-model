"""
Build Protein Metadata
======================

This script creates:

    data/protein_metadata.csv

The model protein panel is read from the saved final-model checkpoint.
Protein names are retrieved once from UniProt and then stored locally.

After this file has been created, model evaluation and external
validation should use the local CSV rather than querying UniProt again.

Run from project root:

    python scripts/build_protein_metadata.py
"""

from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
import torch

from pcmodel.metadata import (
    CATEGORY_ORDER,
    build_metadata_dataframe,
    category_audit,
    save_protein_metadata,
    validate_metadata_for_panel,
)


# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "twohead_model_checkpoint.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "protein_metadata.csv"
)


# ======================================================================
# UniProt configuration
# ======================================================================

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
)

BATCH_SIZE = 25

MAX_RETRIES = 3

RETRY_DELAY_SECONDS = 2


# ======================================================================
# Expected category counts
# ======================================================================
#
# These are used only as a sanity check against the current
# 174-protein model panel.
#
# They do NOT control category assignment.
# ======================================================================

EXPECTED_CATEGORY_COUNTS = {
    "Apolipoproteins": 15,
    "Coagulation/Fibrinogen": 19,
    "Complement System": 25,
    "Cytoskeletal": 22,
    "Immunoglobulins": 24,
    "Metabolic Enzymes": 6,
    "Protease/Inhibitors": 15,
    "Transport/Binding": 28,
    "Other/Mixed": 20,
}


# ======================================================================
# Helpers
# ======================================================================


def batch_list(
    values: List[str],
    batch_size: int,
):
    """
    Yield successive batches.
    """

    for start in range(
        0,
        len(values),
        batch_size,
    ):

        yield values[
            start:
            start + batch_size
        ]


def get_recommended_name(
    entry: Dict,
) -> str:
    """
    Extract the recommended protein name from a UniProt JSON entry.

    Falls back to submitted-name information when a recommended name
    is unavailable.
    """

    description = entry.get(
        "proteinDescription",
        {},
    )

    # --------------------------------------------------------------
    # Recommended name
    # --------------------------------------------------------------

    recommended = description.get(
        "recommendedName"
    )

    if recommended:

        full_name = (
            recommended
            .get(
                "fullName",
                {}
            )
            .get(
                "value"
            )
        )

        if full_name:

            return str(
                full_name
            ).strip()

    # --------------------------------------------------------------
    # Submission name fallback
    # --------------------------------------------------------------

    submission_names = (
        description.get(
            "submissionNames",
            []
        )
    )

    if submission_names:

        full_name = (
            submission_names[0]
            .get(
                "fullName",
                {}
            )
            .get(
                "value"
            )
        )

        if full_name:

            return str(
                full_name
            ).strip()

    return "<UNKNOWN>"


# ======================================================================
# UniProt query
# ======================================================================


def query_uniprot_batch(
    accessions: List[str],
) -> Dict[str, str]:
    """
    Query UniProt for one batch of accession IDs.

    Returns
    -------
    dict
        accession -> recommended protein name
    """

    query = " OR ".join(
        f"accession:{accession}"
        for accession
        in accessions
    )

    parameters = {
        "query": (
            f"({query})"
        ),
        "fields": (
            "accession,protein_name"
        ),
        "format": "json",
        "size": len(
            accessions
        ),
    }

    url = (
        UNIPROT_URL
        + "?"
        + urlencode(
            parameters
        )
    )

    request = Request(
        url,
        headers={
            "User-Agent":
                "protein-corona-model/0.1"
        },
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            with urlopen(
                request,
                timeout=60,
            ) as response:

                payload = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

            mapping = {}

            for entry in payload.get(
                "results",
                []
            ):

                accession = entry.get(
                    "primaryAccession"
                )

                if not accession:

                    continue

                mapping[
                    accession
                ] = get_recommended_name(
                    entry
                )

            return mapping

        except (
            HTTPError,
            URLError,
            TimeoutError,
        ) as error:

            last_error = error

            print(
                f"UniProt request failed "
                f"(attempt {attempt}/"
                f"{MAX_RETRIES}): {error}"
            )

            if (
                attempt
                < MAX_RETRIES
            ):

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        "UniProt query failed after "
        f"{MAX_RETRIES} attempts."
    ) from last_error


def query_uniprot(
    accessions: List[str],
) -> Dict[str, str]:
    """
    Retrieve protein names for the complete model panel.
    """

    id_to_name: Dict[
        str,
        str,
    ] = {}

    batches = list(
        batch_list(
            accessions,
            BATCH_SIZE,
        )
    )

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        print(
            f"Querying UniProt batch "
            f"{batch_number}/{len(batches)} "
            f"({len(batch)} proteins)..."
        )

        result = query_uniprot_batch(
            batch
        )

        id_to_name.update(
            result
        )

        # Be polite to the remote API.
        time.sleep(
            0.5
        )

    return id_to_name


# ======================================================================
# Main
# ======================================================================


def main() -> None:

    print("=" * 70)
    print("BUILD PROTEIN METADATA")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Check checkpoint
    # ------------------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Final model checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    panel = list(
        checkpoint[
            "panel"
        ]
    )

    print()
    print(
        "Model panel size:",
        len(panel),
    )

    # ------------------------------------------------------------------
    # Query UniProt
    # ------------------------------------------------------------------

    print()
    print(
        "Retrieving protein names from UniProt..."
    )

    id_to_name = query_uniprot(
        panel
    )

    print()
    print(
        "Names retrieved:",
        len(id_to_name),
    )

    # ------------------------------------------------------------------
    # Identify missing names
    # ------------------------------------------------------------------

    missing_names = [
        accession
        for accession in panel
        if accession
        not in id_to_name
    ]

    if missing_names:

        print()
        print(
            "WARNING:"
        )

        print(
            f"{len(missing_names)} accession(s) "
            "were not returned by UniProt:"
        )

        for accession in missing_names:

            print(
                "  ",
                accession,
            )

        # Keep the panel complete.
        for accession in missing_names:

            id_to_name[
                accession
            ] = "<UNKNOWN>"

    # ------------------------------------------------------------------
    # Build metadata dataframe
    # ------------------------------------------------------------------

    metadata = (
        build_metadata_dataframe(
            panel,
            id_to_name,
        )
    )

    # ------------------------------------------------------------------
    # Validate panel coverage
    # ------------------------------------------------------------------

    validation = (
        validate_metadata_for_panel(
            metadata,
            panel,
            require_all=True,
        )
    )

    print()
    print(
        "Metadata matched:",
        validation[
            "matched"
        ],
        "/",
        validation[
            "panel_size"
        ],
    )

    # ------------------------------------------------------------------
    # Category audit
    # ------------------------------------------------------------------

    audit = category_audit(
        metadata
    )

    print()
    print("=" * 70)
    print("CATEGORY AUDIT")
    print("=" * 70)

    print(
        audit.to_string()
    )

    # ------------------------------------------------------------------
    # Compare with expected category counts
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CATEGORY COUNT CHECK")
    print("=" * 70)

    counts_match = True

    for category in CATEGORY_ORDER:

        actual = int(
            audit.loc[
                category,
                "n_proteins",
            ]
        )

        expected = (
            EXPECTED_CATEGORY_COUNTS.get(
                category
            )
        )

        status = (
            "OK"
            if actual == expected
            else "CHECK"
        )

        if actual != expected:

            counts_match = False

        print(
            f"{category:25s} "
            f"expected={expected:3d} "
            f"actual={actual:3d} "
            f"{status}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_protein_metadata(
        metadata,
        OUTPUT_PATH,
    )

    print()
    print("=" * 70)
    print("METADATA SAVED")
    print("=" * 70)

    print(
        OUTPUT_PATH
    )

    print()
    print(
        metadata.head(
            10
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    print()

    if counts_match:

        print(
            "Protein metadata created successfully "
            "and category counts match the expected panel."
        )

    else:

        print(
            "Protein metadata was created successfully, "
            "but one or more category counts differ from "
            "the expected model categories."
        )

        print(
            "Do NOT change the rules yet. "
            "We should inspect the mismatched proteins first."
        )


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    main()