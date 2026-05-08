#!/usr/bin/env python3
"""Download the published public BOSQUE dataset from Harvard Dataverse.

Dataset DOI:
    doi:10.7910/DVN/AQEPIN

Expected public files:
    BOSQUE_metadata.tab
    BOSQUE_test_set.zip

Files are downloaded to:
    data/external/bosque_public/

Dataset files are not committed to Git.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path


BOSQUE_DOI = "doi:10.7910/DVN/AQEPIN"
DATAVERSE_API_METADATA = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/"
    "?persistentId=doi:10.7910/DVN/AQEPIN"
)
FILE_DOWNLOAD_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
USER_AGENT = "Mozilla/5.0 DermAlgoFairness-revision-script"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the published public BOSQUE dataset from Harvard Dataverse."
    )
    parser.add_argument(
        "--out",
        default="data/external/bosque_public",
        help="Destination directory for the published public BOSQUE dataset.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract ZIP files after download.",
    )
    return parser.parse_args()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download_file(file_id: int, filename: str, out_dir: Path) -> Path:
    out_path = out_dir / filename

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already exists, skipping: {out_path}")
        return out_path

    url = FILE_DOWNLOAD_URL.format(file_id=file_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    print(f"Downloading {filename} from file id {file_id}...")
    with urllib.request.urlopen(request) as response, out_path.open("wb") as f:
        f.write(response.read())

    print(f"Wrote: {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    extract_dir = out_dir / zip_path.stem

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Extraction directory already exists, skipping: {extract_dir}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"Extracted to: {extract_dir}")


def write_readme(out_dir: Path, files: list[dict]) -> None:
    readme_path = out_dir / "README_BOSQUE_PUBLIC.txt"
    lines = [
        "Published public BOSQUE dataset",
        "",
        f"Reference DOI: {BOSQUE_DOI}",
        f"Dataverse API metadata endpoint: {DATAVERSE_API_METADATA}",
        "",
        "Downloaded files:",
    ]

    for item in files:
        f = item["dataFile"]
        lines.append(f"- {f['filename']} | file_id={f['id']} | size={f.get('filesize', 'NA')}")

    lines.extend(
        [
            "",
            "These files correspond to the published public version of BOSQUE.",
            "They are used as the reference dataset for the revised public results workflow.",
            "Dataset files are intentionally not committed to Git.",
            "",
        ]
    )

    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {readme_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_json(DATAVERSE_API_METADATA)
    files = metadata["data"]["latestVersion"]["files"]

    print(f"Dataset DOI: {BOSQUE_DOI}")
    print(f"Found {len(files)} files")

    downloaded = []
    for item in files:
        f = item["dataFile"]
        path = download_file(f["id"], f["filename"], out_dir)
        downloaded.append(path)

    write_readme(out_dir, files)

    if args.extract:
        for path in downloaded:
            if path.suffix.lower() == ".zip":
                extract_zip(path, out_dir)


if __name__ == "__main__":
    main()
