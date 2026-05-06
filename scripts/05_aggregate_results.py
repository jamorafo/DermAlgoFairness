#!/usr/bin/env python3
"""Aggregate evaluation metrics across seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate model evaluation CSV files.")
    parser.add_argument("--model", required=True, help="Model name prefix, e.g. resnet50.")
    parser.add_argument("--tables-dir", default="outputs/tables")
    return parser.parse_args()


def aggregate_overall(model: str, tables_dir: Path) -> None:
    rows = []
    for path in sorted(tables_dir.glob(f"evaluation_metrics_{model}_seed*.csv")):
        df = pd.read_csv(path)
        seed = int(path.name.split("seed")[1].split("_")[0])
        df.insert(0, "seed", seed)
        df.insert(1, "source_file", path.name)
        rows.append(df)

    if not rows:
        raise SystemExit(f"No overall metric files found for model: {model}")

    all_df = pd.concat(rows, ignore_index=True)
    summary = (
        all_df.groupby("dataset")[
            ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    all_path = tables_dir / f"{model}_all_seed_metrics.csv"
    summary_path = tables_dir / f"{model}_all_seed_metrics_summary.csv"

    all_df.to_csv(all_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {all_path}")
    print(f"Wrote: {summary_path}")
    print(summary.to_string(index=False))


def aggregate_subgroups(model: str, tables_dir: Path) -> None:
    rows = []
    for path in sorted(tables_dir.glob(f"evaluation_metrics_by_subgroup_{model}_seed*.csv")):
        df = pd.read_csv(path)
        seed = int(path.name.split("seed")[1].split("_")[0])
        df.insert(0, "seed", seed)
        df.insert(1, "source_file", path.name)
        rows.append(df)

    if not rows:
        print(f"No subgroup metric files found for model: {model}")
        return

    all_df = pd.concat(rows, ignore_index=True)
    summary = (
        all_df.groupby(["subgroup_column", "subgroup"])[
            ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    all_path = tables_dir / f"{model}_all_seed_subgroup_metrics.csv"
    summary_path = tables_dir / f"{model}_all_seed_subgroup_metrics_summary.csv"

    all_df.to_csv(all_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print(f"Wrote: {all_path}")
    print(f"Wrote: {summary_path}")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)

    aggregate_overall(args.model, tables_dir)
    aggregate_subgroups(args.model, tables_dir)


if __name__ == "__main__":
    main()
