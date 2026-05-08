"""Path utilities for project scripts."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root unless already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = resolve_project_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
