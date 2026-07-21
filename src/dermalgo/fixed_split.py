"""Utilities for loading the immutable lesion-grouped HAM10000 split."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from dermalgo.paths import resolve_project_path


REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "lesion_id",
    "dx",
    "benign_malignant",
    "label_binary",
    "fold",
    "split",
}

EXPECTED_SPLITS = {
    "train",
    "validation",
    "test",
}


def calculate_sha256(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file."""
    resolved_path = Path(path)

    digest = hashlib.sha256()

    with resolved_path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _normalize_identifier(series: pd.Series) -> pd.Series:
    """Normalize identifier values for validation."""
    return series.astype(str).str.strip()


def validate_fixed_split_manifest(
    ham_df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """Validate correspondence between HAM10000 and the split manifest."""
    missing_manifest_columns = sorted(
        REQUIRED_MANIFEST_COLUMNS
        - set(manifest.columns)
    )

    if missing_manifest_columns:
        raise ValueError(
            "The fixed-split manifest is missing required columns: "
            f"{missing_manifest_columns}"
        )

    required_ham_columns = {
        "image_id",
        "lesion_id",
        "dx",
        "benign_malignant",
        "label_binary",
    }

    missing_ham_columns = sorted(
        required_ham_columns
        - set(ham_df.columns)
    )

    if missing_ham_columns:
        raise ValueError(
            "The HAM10000 dataframe is missing required columns: "
            f"{missing_ham_columns}"
        )

    if manifest["image_id"].isna().any():
        raise ValueError(
            "The fixed-split manifest contains missing image_id values."
        )

    if manifest["lesion_id"].isna().any():
        raise ValueError(
            "The fixed-split manifest contains missing lesion_id values."
        )

    if manifest["image_id"].duplicated().any():
        duplicates = (
            manifest.loc[
                manifest["image_id"].duplicated(keep=False),
                "image_id",
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            "The fixed-split manifest contains duplicate image_id values. "
            f"Examples: {duplicates}"
        )

    observed_splits = set(
        manifest["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    if observed_splits != EXPECTED_SPLITS:
        raise ValueError(
            "Unexpected partition labels in the fixed-split manifest: "
            f"{sorted(observed_splits)}"
        )

    ham_image_ids = set(
        _normalize_identifier(
            ham_df["image_id"]
        )
    )

    manifest_image_ids = set(
        _normalize_identifier(
            manifest["image_id"]
        )
    )

    missing_from_manifest = sorted(
        ham_image_ids
        - manifest_image_ids
    )

    extra_in_manifest = sorted(
        manifest_image_ids
        - ham_image_ids
    )

    if missing_from_manifest:
        raise ValueError(
            f"{len(missing_from_manifest)} HAM10000 images are absent "
            "from the fixed-split manifest. Examples: "
            f"{missing_from_manifest[:10]}"
        )

    if extra_in_manifest:
        raise ValueError(
            f"{len(extra_in_manifest)} manifest images are absent from "
            "the loaded HAM10000 metadata. Examples: "
            f"{extra_in_manifest[:10]}"
        )

    ham_validation = ham_df[
        [
            "image_id",
            "lesion_id",
            "dx",
            "benign_malignant",
            "label_binary",
        ]
    ].copy()

    manifest_validation = manifest[
        [
            "image_id",
            "lesion_id",
            "dx",
            "benign_malignant",
            "label_binary",
        ]
    ].copy()

    for frame in [
        ham_validation,
        manifest_validation,
    ]:
        frame["image_id"] = _normalize_identifier(
            frame["image_id"]
        )

        frame["lesion_id"] = _normalize_identifier(
            frame["lesion_id"]
        )

        frame["dx"] = (
            frame["dx"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        frame["benign_malignant"] = (
            frame["benign_malignant"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        frame["label_binary"] = pd.to_numeric(
            frame["label_binary"],
            errors="raise",
        ).astype(int)

    comparison = ham_validation.merge(
        manifest_validation,
        on="image_id",
        how="inner",
        suffixes=("_ham", "_manifest"),
        validate="one_to_one",
    )

    variables_to_compare = [
        "lesion_id",
        "dx",
        "benign_malignant",
        "label_binary",
    ]

    for variable in variables_to_compare:
        mismatch = (
            comparison[f"{variable}_ham"]
            != comparison[f"{variable}_manifest"]
        )

        if mismatch.any():
            examples = (
                comparison.loc[
                    mismatch,
                    [
                        "image_id",
                        f"{variable}_ham",
                        f"{variable}_manifest",
                    ],
                ]
                .head(10)
                .to_dict(orient="records")
            )

            raise ValueError(
                f"The manifest and HAM10000 metadata disagree on "
                f"{variable}. Examples: {examples}"
            )

    lesion_partition_counts = (
        manifest.groupby(
            "lesion_id",
            dropna=False,
        )["split"]
        .nunique(dropna=False)
    )

    cross_partition_lesions = lesion_partition_counts[
        lesion_partition_counts > 1
    ]

    if not cross_partition_lesions.empty:
        raise ValueError(
            f"{len(cross_partition_lesions)} lesions occur in more "
            "than one partition."
        )


def load_fixed_ham10000_split(
    ham_df: pd.DataFrame,
    manifest_path: str | Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
    str,
]:
    """
    Load and apply the immutable lesion-grouped HAM10000 manifest.

    Returns
    -------
    train_df
        Fixed HAM10000 training partition.
    validation_df
        Fixed HAM10000 validation partition.
    test_df
        Fixed HAM10000 internal-test partition.
    resolved_manifest_path
        Absolute path to the split manifest.
    manifest_sha256
        SHA-256 checksum of the manifest used.
    """
    resolved_manifest_path = resolve_project_path(
        manifest_path
    )

    if not resolved_manifest_path.exists():
        raise FileNotFoundError(
            "Fixed HAM10000 split manifest not found: "
            f"{resolved_manifest_path}"
        )

    manifest = pd.read_csv(
        resolved_manifest_path
    )

    validate_fixed_split_manifest(
        ham_df,
        manifest,
    )

    split_assignment = manifest[
        [
            "image_id",
            "fold",
            "split",
        ]
    ].copy()

    split_assignment["image_id"] = _normalize_identifier(
        split_assignment["image_id"]
    )

    split_assignment["split"] = (
        split_assignment["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    merged = ham_df.copy()

    merged["image_id"] = _normalize_identifier(
        merged["image_id"]
    )

    merged = merged.merge(
        split_assignment,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        missing = (
            merged.loc[
                merged["split"].isna(),
                "image_id",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "Some HAM10000 images did not receive a partition. "
            f"Examples: {missing}"
        )

    train_df = (
        merged[
            merged["split"] == "train"
        ]
        .reset_index(drop=True)
    )

    validation_df = (
        merged[
            merged["split"] == "validation"
        ]
        .reset_index(drop=True)
    )

    test_df = (
        merged[
            merged["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    if (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        != len(ham_df)
    ):
        raise RuntimeError(
            "The fixed partitions do not reconstruct the full "
            "HAM10000 dataframe."
        )

    manifest_sha256 = calculate_sha256(
        resolved_manifest_path
    )

    return (
        train_df,
        validation_df,
        test_df,
        resolved_manifest_path,
        manifest_sha256,
    )


def summarize_fixed_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize image, lesion, and outcome counts by partition."""
    rows = []

    for split_name, frame in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        label_counts = (
            frame["label_binary"]
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "split": split_name,
                "n_images": int(len(frame)),
                "n_lesions": int(
                    frame["lesion_id"].nunique()
                ),
                "n_benign_images": int(
                    label_counts.get(0, 0)
                ),
                "n_malignant_images": int(
                    label_counts.get(1, 0)
                ),
                "malignant_image_proportion": float(
                    frame["label_binary"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)
