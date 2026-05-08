"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")

    return config


def require_keys(config: dict[str, Any], keys: list[str], context: str = "config") -> None:
    """Validate that required top-level keys exist in a config dictionary."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing required keys in {context}: {missing}")
