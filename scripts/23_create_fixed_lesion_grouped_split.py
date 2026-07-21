#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create the immutable lesion-grouped HAM10000 split.

The split seed is obtained from config/random_seeds.json. Ten lesion-grouped
folds are first generated using StratifiedGroupKFold. The validation and test
folds are then selected deterministically by minimizing, in order:

1. the maximum deviation of split-specific malignant prevalence from the
   overall HAM10000 malignant prevalence;
2. the total prevalence deviation;
3. the maximum deviation from the intended 80/10/10 image allocation;
4. the maximum deviation from the intended 80/10/10 lesion allocation.

Selection uses only lesion identifiers, outcome labels, and partition sizes.
No model predictions or performance estimates are used.

Images remain the prediction unit. All images associated with one lesion are
assigned to exactly one partition.
"""

from __future__ import annotations

import hashlib
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dermalgo.config import load_config
from dermalgo.data import load_ham10000
from dermalgo.seeds import get_fixed_split_seed


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "resnet50.yaml"

SPLIT_SEED = get_fixed_split_seed()
N_FOLDS = 10

TARGET_PROPORTIONS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}

SPLIT_DIR = ROOT / "splits"
AUDIT_DIR = ROOT / "outputs" / "audit"

MANIFEST_PATH = (
    SPLIT_DIR
    / "ham10000_lesion_grouped_fixed.csv"
)

SUMMARY_PATH = (
    AUDIT_DIR
    / "ham10000_fixed_split_summary.csv"
)

SEARCH_PATH = (
    AUDIT_DIR
    / "ham10000_fixed_split_fold_pair_search.csv"
)

METADATA_PATH = (
    AUDIT_DIR
    / "ham10000_fixed_split_metadata.json"
)


def calculate_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_ham10000(df: pd.DataFrame) -> None:
    """Validate columns and lesion-level outcome consistency."""
    required_columns = {
        "image_id",
        "lesion_id",
        "dx",
        "benign_malignant",
        "label_binary",
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required HAM10000 columns: "
            f"{missing}"
        )

    for column in [
        "image_id",
        "lesion_id",
        "label_binary",
    ]:
        if df[column].isna().any():
            raise ValueError(
                f"Missing values detected in {column}."
            )

    if df["image_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["image_id"].duplicated(keep=False),
                "image_id",
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Duplicate image identifiers detected. "
            f"Examples: {duplicates}"
        )

    lesion_binary_counts = (
        df.groupby("lesion_id")["label_binary"]
        .nunique(dropna=False)
    )

    inconsistent_binary = lesion_binary_counts[
        lesion_binary_counts > 1
    ]

    if not inconsistent_binary.empty:
        examples = (
            inconsistent_binary.index
            .astype(str)
            .tolist()[:10]
        )

        raise ValueError(
            "Some lesions contain multiple binary labels. "
            f"Examples: {examples}"
        )

    lesion_dx_counts = (
        df.groupby("lesion_id")["dx"]
        .nunique(dropna=False)
    )

    inconsistent_dx = lesion_dx_counts[
        lesion_dx_counts > 1
    ]

    if not inconsistent_dx.empty:
        examples = (
            inconsistent_dx.index
            .astype(str)
            .tolist()[:10]
        )

        raise ValueError(
            "Some lesions contain multiple diagnosis codes. "
            f"Examples: {examples}"
        )


def assign_grouped_folds(df: pd.DataFrame) -> np.ndarray:
    """Generate ten fixed stratified lesion-grouped folds."""
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SPLIT_SEED,
    )

    fold_assignment = np.full(
        len(df),
        -1,
        dtype=int,
    )

    x_placeholder = np.zeros(
        shape=(len(df), 1),
        dtype=np.float32,
    )

    y = (
        df["label_binary"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        df["lesion_id"]
        .astype(str)
        .to_numpy()
    )

    for fold_number, (_, held_out_indices) in enumerate(
        splitter.split(
            x_placeholder,
            y,
            groups,
        )
    ):
        if np.any(
            fold_assignment[held_out_indices] != -1
        ):
            raise RuntimeError(
                "At least one image was assigned to "
                "multiple folds."
            )

        fold_assignment[
            held_out_indices
        ] = fold_number

    if np.any(fold_assignment == -1):
        raise RuntimeError(
            "At least one image was not assigned "
            "to a fold."
        )

    return fold_assignment


def make_split_assignment(
    folds: np.ndarray,
    test_fold: int,
    validation_fold: int,
) -> np.ndarray:
    """Map two held-out folds to validation and test."""
    split = np.full(
        len(folds),
        "train",
        dtype=object,
    )

    split[
        folds == validation_fold
    ] = "validation"

    split[
        folds == test_fold
    ] = "test"

    return split


def create_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize image, lesion, and outcome balance."""
    total_images = len(frame)

    total_lesions = (
        frame["lesion_id"]
        .nunique()
    )

    overall_malignant_proportion = float(
        frame["label_binary"].mean()
    )

    rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_frame = frame[
            frame["split"] == split_name
        ]

        n_images = len(split_frame)

        n_lesions = (
            split_frame["lesion_id"]
            .nunique()
        )

        n_benign = int(
            (
                split_frame["label_binary"]
                == 0
            ).sum()
        )

        n_malignant = int(
            (
                split_frame["label_binary"]
                == 1
            ).sum()
        )

        malignant_proportion = (
            n_malignant / n_images
            if n_images > 0
            else np.nan
        )

        rows.append(
            {
                "split": split_name,
                "n_images": int(n_images),
                "image_proportion": (
                    n_images / total_images
                ),
                "n_lesions": int(n_lesions),
                "lesion_proportion": (
                    n_lesions / total_lesions
                ),
                "n_benign_images": n_benign,
                "n_malignant_images": n_malignant,
                "malignant_image_proportion": (
                    malignant_proportion
                ),
                (
                    "absolute_deviation_from_"
                    "overall_malignant_proportion"
                ): abs(
                    malignant_proportion
                    - overall_malignant_proportion
                ),
            }
        )

    return pd.DataFrame(rows)


def score_summary(
    summary: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Return the prespecified lexicographic balance score."""
    indexed = summary.set_index("split")

    prevalence_deviations = (
        indexed[
            "absolute_deviation_from_"
            "overall_malignant_proportion"
        ]
        .to_numpy(dtype=float)
    )

    image_deviations = np.asarray(
        [
            abs(
                float(
                    indexed.loc[
                        split_name,
                        "image_proportion",
                    ]
                )
                - TARGET_PROPORTIONS[split_name]
            )
            for split_name in [
                "train",
                "validation",
                "test",
            ]
        ],
        dtype=float,
    )

    lesion_deviations = np.asarray(
        [
            abs(
                float(
                    indexed.loc[
                        split_name,
                        "lesion_proportion",
                    ]
                )
                - TARGET_PROPORTIONS[split_name]
            )
            for split_name in [
                "train",
                "validation",
                "test",
            ]
        ],
        dtype=float,
    )

    return (
        float(
            prevalence_deviations.max()
        ),
        float(
            prevalence_deviations.sum()
        ),
        float(
            image_deviations.max()
        ),
        float(
            lesion_deviations.max()
        ),
    )


def select_validation_and_test_folds(
    df: pd.DataFrame,
) -> tuple[
    int,
    int,
    pd.DataFrame,
]:
    """Select the best ordered validation/test fold pair."""
    candidate_rows = []

    best_key = None
    best_test_fold = None
    best_validation_fold = None

    for test_fold, validation_fold in permutations(
        range(N_FOLDS),
        2,
    ):
        candidate = df.copy()

        candidate["split"] = (
            make_split_assignment(
                candidate["fold"].to_numpy(),
                test_fold=test_fold,
                validation_fold=validation_fold,
            )
        )

        summary = create_summary(candidate)

        (
            maximum_prevalence_deviation,
            total_prevalence_deviation,
            maximum_image_deviation,
            maximum_lesion_deviation,
        ) = score_summary(summary)

        key = (
            maximum_prevalence_deviation,
            total_prevalence_deviation,
            maximum_image_deviation,
            maximum_lesion_deviation,
            test_fold,
            validation_fold,
        )

        candidate_rows.append(
            {
                "split_seed": SPLIT_SEED,
                "test_fold": test_fold,
                "validation_fold": validation_fold,
                "maximum_prevalence_deviation": (
                    maximum_prevalence_deviation
                ),
                "total_prevalence_deviation": (
                    total_prevalence_deviation
                ),
                "maximum_image_proportion_deviation": (
                    maximum_image_deviation
                ),
                "maximum_lesion_proportion_deviation": (
                    maximum_lesion_deviation
                ),
            }
        )

        if (
            best_key is None
            or key < best_key
        ):
            best_key = key
            best_test_fold = test_fold
            best_validation_fold = validation_fold

    if (
        best_test_fold is None
        or best_validation_fold is None
    ):
        raise RuntimeError(
            "No valid validation/test fold pair was found."
        )

    candidate_results = (
        pd.DataFrame(candidate_rows)
        .sort_values(
            [
                "maximum_prevalence_deviation",
                "total_prevalence_deviation",
                "maximum_image_proportion_deviation",
                "maximum_lesion_proportion_deviation",
                "test_fold",
                "validation_fold",
            ]
        )
        .reset_index(drop=True)
    )

    candidate_results.insert(
        0,
        "rank",
        np.arange(
            1,
            len(candidate_results) + 1,
        ),
    )

    return (
        int(best_test_fold),
        int(best_validation_fold),
        candidate_results,
    )


def validate_manifest(
    manifest: pd.DataFrame,
) -> None:
    """Ensure images and lesions do not cross partitions."""
    expected_splits = {
        "train",
        "validation",
        "test",
    }

    observed_splits = set(
        manifest["split"].unique()
    )

    if observed_splits != expected_splits:
        raise RuntimeError(
            "Unexpected partition labels: "
            f"{sorted(observed_splits)}"
        )

    if manifest["image_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate image identifiers exist in "
            "the final manifest."
        )

    lesion_partition_counts = (
        manifest.groupby("lesion_id")["split"]
        .nunique()
    )

    cross_partition_lesions = (
        lesion_partition_counts[
            lesion_partition_counts > 1
        ]
    )

    if not cross_partition_lesions.empty:
        raise RuntimeError(
            f"{len(cross_partition_lesions)} lesions "
            "cross partitions."
        )

    for split_a, split_b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        images_a = set(
            manifest.loc[
                manifest["split"] == split_a,
                "image_id",
            ].astype(str)
        )

        images_b = set(
            manifest.loc[
                manifest["split"] == split_b,
                "image_id",
            ].astype(str)
        )

        lesions_a = set(
            manifest.loc[
                manifest["split"] == split_a,
                "lesion_id",
            ].astype(str)
        )

        lesions_b = set(
            manifest.loc[
                manifest["split"] == split_b,
                "lesion_id",
            ].astype(str)
        )

        if images_a.intersection(images_b):
            raise RuntimeError(
                f"Image overlap between {split_a} "
                f"and {split_b}."
            )

        if lesions_a.intersection(lesions_b):
            raise RuntimeError(
                f"Lesion overlap between {split_a} "
                f"and {split_b}."
            )


def main() -> None:
    """Create and document the final fixed source split."""
    config = load_config(
        CONFIG_PATH
    )

    ham_df = load_ham10000(
        ham10000_root=(
            config["data"]["ham10000_root"]
        ),
        metadata_filename=Path(
            config["data"]["ham10000_metadata"]
        ).name,
        benign_labels=(
            config["data"]["benign_labels"]
        ),
        malignant_labels=(
            config["data"]["malignant_labels"]
        ),
    ).copy()

    validate_ham10000(
        ham_df
    )

    ham_df["fold"] = assign_grouped_folds(
        ham_df
    )

    (
        test_fold,
        validation_fold,
        candidate_results,
    ) = select_validation_and_test_folds(
        ham_df
    )

    ham_df["split"] = make_split_assignment(
        ham_df["fold"].to_numpy(),
        test_fold=test_fold,
        validation_fold=validation_fold,
    )

    manifest_columns = [
        "image_id",
        "lesion_id",
        "dx",
        "benign_malignant",
        "label_binary",
        "fold",
        "split",
    ]

    manifest = (
        ham_df[manifest_columns]
        .sort_values(
            [
                "split",
                "lesion_id",
                "image_id",
            ]
        )
        .reset_index(drop=True)
    )

    validate_manifest(
        manifest
    )

    summary = create_summary(
        ham_df
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    candidate_results.to_csv(
        SEARCH_PATH,
        index=False,
    )

    manifest_sha256 = calculate_sha256(
        MANIFEST_PATH
    )

    metadata = {
        "dataset": "HAM10000",
        "manifest_path": str(
            MANIFEST_PATH.relative_to(ROOT)
        ),
        "manifest_sha256": manifest_sha256,
        "splitting_algorithm": (
            "StratifiedGroupKFold followed by "
            "prespecified fold-pair balance selection"
        ),
        "grouping_variable": "lesion_id",
        "stratification_variable": "label_binary",
        "split_seed": SPLIT_SEED,
        "n_folds": N_FOLDS,
        "selected_test_fold": test_fold,
        "selected_validation_fold": (
            validation_fold
        ),
        "selection_criteria": [
            (
                "minimum maximum malignant-prevalence "
                "deviation"
            ),
            (
                "minimum total malignant-prevalence "
                "deviation"
            ),
            (
                "minimum maximum image-allocation "
                "deviation"
            ),
            (
                "minimum maximum lesion-allocation "
                "deviation"
            ),
            "fold number for deterministic tie-breaking",
        ],
        "selection_uses_model_predictions": False,
        "prediction_unit": "image",
        "source_uncertainty_resampling_unit": (
            "lesion"
        ),
        "n_images": int(
            len(manifest)
        ),
        "n_lesions": int(
            manifest["lesion_id"].nunique()
        ),
        "n_cross_partition_images": 0,
        "n_cross_partition_lesions": 0,
        "split_summary": summary.to_dict(
            orient="records"
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Fixed lesion-grouped HAM10000 split")
    print("-----------------------------------")
    print(
        f"Policy-derived split seed: {SPLIT_SEED}"
    )
    print(
        f"Selected test fold: {test_fold}"
    )
    print(
        "Selected validation fold: "
        f"{validation_fold}"
    )
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Validation")
    print("----------")
    print("Cross-partition images:  0")
    print("Cross-partition lesions: 0")

    print()
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Summary:  {SUMMARY_PATH}")
    print(f"Search:   {SEARCH_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    print()
    print(
        f"Manifest SHA-256: {manifest_sha256}"
    )


if __name__ == "__main__":
    main()
