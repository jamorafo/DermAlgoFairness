"""Dataset loading utilities for DermAlgoFairness."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from dermalgo.paths import resolve_project_path


def normalize_label(value: object) -> str:
    """Normalize labels and metadata values to lowercase strings."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def skin_group_from_phototype(
    phototype: object,
    light_phototypes: Iterable[int] = (1, 2, 3),
    dark_phototypes: Iterable[int] = (4, 5, 6),
) -> str:
    """Map Fitzpatrick skin phototype to light/dark group."""
    value = str(phototype).strip().upper()

    roman_to_int = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
    }

    if value in roman_to_int:
        numeric = roman_to_int[value]
    else:
        try:
            numeric = int(value)
        except ValueError:
            return "unknown"

    if numeric in set(light_phototypes):
        return "light"
    if numeric in set(dark_phototypes):
        return "dark"
    return "unknown"


def load_bosque_public(
    bosque_root: str | Path = "data/external/bosque_public",
    metadata_filename: str = "BOSQUE_metadata.tab",
    image_subdir: str = "BOSQUE_test_set/BOSQUE_test_set",
    light_phototypes: Iterable[int] = (1, 2, 3),
    dark_phototypes: Iterable[int] = (4, 5, 6),
) -> pd.DataFrame:
    """Load the published public BOSQUE dataset metadata and resolve image paths.

    Returns a dataframe with one row per image and standardized columns:
    - file_name
    - image_id
    - image_path
    - clinical_label
    - histology_label
    - benign_malignant
    - skin_phototype
    - skin_group
    """
    root = resolve_project_path(bosque_root)
    metadata_path = root / metadata_filename
    image_dir = root / image_subdir

    if not metadata_path.exists():
        raise FileNotFoundError(f"BOSQUE metadata not found: {metadata_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"BOSQUE image directory not found: {image_dir}")

    df = pd.read_csv(metadata_path, sep="\t")

    required_columns = {
        "file_name",
        "image_id",
        "skin_phototype",
        "clinical_label",
        "histology_label",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required BOSQUE metadata columns: {missing}")

    df = df.copy()
    df["file_name"] = df["file_name"].astype(str)

    # Public metadata stores file_name without extension.
    df["image_path"] = df["file_name"].apply(lambda name: str(image_dir / f"{name}.jpg"))
    df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())

    df["clinical_label"] = df["clinical_label"].map(normalize_label)
    df["histology_label"] = df["histology_label"].map(normalize_label)

    # The public metadata provides clinical_label as benign/malignant.
    df["benign_malignant"] = df["clinical_label"]

    df["skin_group"] = df["skin_phototype"].apply(
        lambda value: skin_group_from_phototype(
            value,
            light_phototypes=light_phototypes,
            dark_phototypes=dark_phototypes,
        )
    )

    missing_images = df.loc[~df["image_exists"], ["file_name", "image_path"]]
    if not missing_images.empty:
        preview = missing_images.head(10).to_string(index=False)
        raise FileNotFoundError(f"Some BOSQUE images are missing:\n{preview}")

    return df


def summarize_bosque_public(df: pd.DataFrame) -> dict[str, object]:
    """Return basic public BOSQUE dataset counts."""
    return {
        "n_rows": int(len(df)),
        "n_images": int(df["image_path"].nunique()),
        "clinical_label_counts": df["clinical_label"].value_counts(dropna=False).to_dict(),
        "histology_label_counts": df["histology_label"].value_counts(dropna=False).to_dict(),
        "skin_phototype_counts": df["skin_phototype"].value_counts(dropna=False).to_dict(),
        "skin_group_counts": df["skin_group"].value_counts(dropna=False).to_dict(),
        "clinical_label_by_skin_group": (
            df.groupby(["skin_group", "clinical_label"]).size().to_dict()
        ),
    }
