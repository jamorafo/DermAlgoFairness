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
        "lesion_nature",
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

    df["lesion_nature"] = df["lesion_nature"].map(normalize_label)
    df["benign_malignant"] = df["lesion_nature"]
    df["label_binary"] = df["benign_malignant"].map({"benign": 0, "malignant": 1})

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


def load_ham10000(
    ham10000_root: str | Path = "data/raw",
    metadata_filename: str = "HAM10000_metadata.csv",
    benign_labels: Iterable[str] = ("nv", "bkl", "df", "vasc"),
    malignant_labels: Iterable[str] = ("mel", "bcc", "akiec"),
) -> pd.DataFrame:
    """Load HAM10000 metadata and resolve image paths.

    Returns a dataframe with one row per image and standardized columns:
    - lesion_id
    - image_id
    - image_path
    - dx
    - benign_malignant
    - label_binary

    label_binary convention:
    - 0 = benign
    - 1 = malignant
    """
    root = resolve_project_path(ham10000_root)
    metadata_path = root / metadata_filename

    if not metadata_path.exists():
        raise FileNotFoundError(f"HAM10000 metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_columns = {"lesion_id", "image_id", "dx"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required HAM10000 metadata columns: {missing}")

    benign_set = {normalize_label(x) for x in benign_labels}
    malignant_set = {normalize_label(x) for x in malignant_labels}

    df = df.copy()
    df["dx"] = df["dx"].map(normalize_label)
    df["image_path"] = df["image_id"].apply(lambda image_id: str(root / f"{image_id}.jpg"))
    df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())

    def map_binary_label(dx: str) -> str:
        if dx in benign_set:
            return "benign"
        if dx in malignant_set:
            return "malignant"
        return "unknown"

    df["benign_malignant"] = df["dx"].apply(map_binary_label)
    df["label_binary"] = df["benign_malignant"].map({"benign": 0, "malignant": 1})

    unknown = df[df["benign_malignant"] == "unknown"]
    if not unknown.empty:
        raise ValueError(
            "Some HAM10000 diagnoses were not mapped to benign/malignant: "
            f"{sorted(unknown['dx'].unique())}"
        )

    missing_images = df.loc[~df["image_exists"], ["image_id", "image_path"]]
    if not missing_images.empty:
        preview = missing_images.head(10).to_string(index=False)
        raise FileNotFoundError(f"Some HAM10000 images are missing:\n{preview}")

    return df


def summarize_ham10000(df: pd.DataFrame) -> dict[str, object]:
    """Return basic HAM10000 dataset counts."""
    return {
        "n_rows": int(len(df)),
        "n_images": int(df["image_path"].nunique()),
        "dx_counts": df["dx"].value_counts(dropna=False).to_dict(),
        "benign_malignant_counts": df["benign_malignant"].value_counts(dropna=False).to_dict(),
        "label_binary_counts": df["label_binary"].value_counts(dropna=False).to_dict(),
    }


def split_ham10000(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_fraction_of_temp: float = 0.5,
    seed: int = 1,
    stratify_column: str = "label_binary",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split HAM10000 into train, validation, and internal test sets.

    The split is performed before any oversampling.

    With the default values:
    - 80% train
    - 10% validation
    - 10% internal test
    """
    from sklearn.model_selection import train_test_split

    if stratify_column not in df.columns:
        raise ValueError(f"Stratify column not found: {stratify_column}")

    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_column],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=validation_fraction_of_temp,
        random_state=seed,
        stratify=temp_df[stratify_column],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def oversample_training_dataframe(
    train_df: pd.DataFrame,
    label_column: str = "label_binary",
    seed: int = 1,
) -> pd.DataFrame:
    """Randomly oversample the training dataframe only.

    This duplicates rows from minority classes in the training set until all
    classes have the same number of rows as the majority class.

    It must be called only after train/validation/test splitting.
    """
    if label_column not in train_df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    counts = train_df[label_column].value_counts(dropna=False)
    max_count = int(counts.max())

    sampled_parts = []
    for label_value, group in train_df.groupby(label_column, dropna=False):
        replace = len(group) < max_count
        sampled = group.sample(
            n=max_count,
            replace=replace,
            random_state=seed,
        )
        sampled_parts.append(sampled)

    sampled_df = (
        pd.concat(sampled_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return sampled_df


def summarize_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "label_binary",
) -> pd.DataFrame:
    """Return class counts for train, validation, and test splits."""
    rows = []
    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        counts = split_df[label_column].value_counts(dropna=False).sort_index()
        for label, n in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "n": int(n),
                }
            )
    return pd.DataFrame(rows)
