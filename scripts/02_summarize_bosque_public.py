#!/usr/bin/env python3
"""Summarize the published public BOSQUE dataset.

This script reads the public BOSQUE metadata and image files, then writes
summary tables for class balance, skin phototype distribution, and subgroup
prevalence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dermalgo.data import load_bosque_public
from dermalgo.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize public BOSQUE dataset.")
    parser.add_argument(
        "--bosque-root",
        default="data/external/bosque_public",
        help="Root directory for the published public BOSQUE dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/tables",
        help="Directory where summary CSV files will be written.",
    )
    return parser.parse_args()


def write_count_table(series: pd.Series, out_path: Path, index_name: str) -> None:
    table = (
        series.value_counts(dropna=False)
        .rename_axis(index_name)
        .reset_index(name="n")
        .sort_values(index_name)
    )
    table.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()

    df = load_bosque_public(bosque_root=args.bosque_root)
    out_dir = ensure_dir(args.out_dir)

    dataset_summary_path = out_dir / "bosque_public_dataset_summary.csv"
    label_counts_path = out_dir / "bosque_public_label_counts.csv"
    phototype_counts_path = out_dir / "bosque_public_phototype_counts.csv"
    skin_group_counts_path = out_dir / "bosque_public_skin_group_counts.csv"
    label_by_group_path = out_dir / "bosque_public_label_by_skin_group.csv"

    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": "BOSQUE_public",
                "n_rows": len(df),
                "n_images": df["image_path"].nunique(),
                "n_missing_images": int((~df["image_exists"]).sum()),
            }
        ]
    )
    dataset_summary.to_csv(dataset_summary_path, index=False)

    write_count_table(df["clinical_label"], label_counts_path, "clinical_label")
    write_count_table(df["skin_phototype"], phototype_counts_path, "skin_phototype")
    write_count_table(df["skin_group"], skin_group_counts_path, "skin_group")

    label_by_group = (
        df.groupby(["skin_group", "clinical_label"])
        .size()
        .reset_index(name="n")
        .sort_values(["skin_group", "clinical_label"])
    )
    label_by_group.to_csv(label_by_group_path, index=False)

    print(f"Wrote: {dataset_summary_path}")
    print(f"Wrote: {label_counts_path}")
    print(f"Wrote: {phototype_counts_path}")
    print(f"Wrote: {skin_group_counts_path}")
    print(f"Wrote: {label_by_group_path}")

    print()
    print("Dataset summary:")
    print(dataset_summary.to_string(index=False))

    print()
    print("Clinical label by skin group:")
    print(label_by_group.to_string(index=False))


if __name__ == "__main__":
    main()
