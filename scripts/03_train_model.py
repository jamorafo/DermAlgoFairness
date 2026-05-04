#!/usr/bin/env python3
"""Train a dermatology classifier from a YAML configuration.

This script is the standardized replacement for the historical training notebooks.
The first version validates configuration loading and records the intended run.
The full training loop will be added incrementally.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from dermalgo.config import load_config, require_keys
from dermalgo.paths import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model YAML configuration, e.g. configs/resnet50.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to run. If omitted, the first seed in the config is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and write a run manifest without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    require_keys(
        config,
        ["model", "data", "split", "training", "evaluation", "outputs"],
        context=args.config,
    )

    seeds = config["training"].get("seeds", [])
    if args.seed is None:
        if not seeds:
            raise ValueError("No seed provided and no seeds found in config.")
        seed = int(seeds[0])
    else:
        seed = int(args.seed)

    model_name = config["model"]["name"]
    logs_dir = ensure_dir(config["outputs"]["logs_dir"])

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "model_name": model_name,
        "seed": seed,
        "dry_run": bool(args.dry_run),
        "model": config["model"],
        "data": config["data"],
        "split": config["split"],
        "training": config["training"],
        "evaluation": config["evaluation"],
    }

    manifest_path = logs_dir / f"train_manifest_{model_name}_seed{seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Input size: {config['model']['input_size']}")
    print(f"Training dataset: {config['data']['train_dataset']}")
    print(f"External test dataset: {config['data']['external_test_dataset']}")
    print(f"Dry run: {args.dry_run}")
    print(f"Wrote manifest: {manifest_path}")

    if args.dry_run:
        return

    raise NotImplementedError(
        "Training is not implemented yet. Use --dry-run for configuration validation."
    )


if __name__ == "__main__":
    main()
