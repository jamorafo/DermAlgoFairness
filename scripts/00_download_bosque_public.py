#!/usr/bin/env python3
"""Download helper for the published public BOSQUE dataset.

The revised public results should use the published Harvard Dataverse version
of BOSQUE rather than earlier internal working copies.

Dataset DOI:
    doi:10.7910/DVN/AQEPIN

This script prepares the destination directory and records the intended source.
The actual download can be completed manually from Harvard Dataverse or via the
Dataverse API, depending on server network access.
"""

from __future__ import annotations

import argparse
from pathlib import Path


BOSQUE_DOI = "doi:10.7910/DVN/AQEPIN"
DATAVERSE_LANDING_PAGE = (
    "https://dataverse.harvard.edu/dataset.xhtml?"
    "persistentId=doi:10.7910/DVN/AQEPIN"
)
DATAVERSE_API_METADATA = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/"
    "?persistentId=doi:10.7910/DVN/AQEPIN"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the published public BOSQUE dataset directory."
    )
    parser.add_argument(
        "--out",
        default="data/external/bosque_public",
        help="Destination directory for the published public BOSQUE dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    readme_path = out_dir / "README_BOSQUE_PUBLIC.txt"
    readme_path.write_text(
        f"""Published public BOSQUE dataset

Reference DOI:
{BOSQUE_DOI}

This directory is reserved for the published public version of the BOSQUE dataset
used in the revised public results workflow.

Recommended source:
{DATAVERSE_LANDING_PAGE}

Dataverse API metadata endpoint:
{DATAVERSE_API_METADATA}

Expected use:
- download the published dataset files from Harvard Dataverse;
- keep raw files under this directory;
- do not commit dataset files to Git;
- record the access date, file names, and checksums for official runs.
""",
        encoding="utf-8",
    )

    print(f"Prepared directory: {out_dir}")
    print(f"Wrote: {readme_path}")
    print()
    print("Dataset DOI:")
    print(f"  {BOSQUE_DOI}")
    print()
    print("Dataverse landing page:")
    print(f"  {DATAVERSE_LANDING_PAGE}")
    print()
    print("Dataverse API metadata endpoint:")
    print(f"  {DATAVERSE_API_METADATA}")


if __name__ == "__main__":
    main()
