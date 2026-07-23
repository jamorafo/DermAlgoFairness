"""Central random-seed policy for the DermAlgoFairness workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "config" / "random_seeds.json"


def load_seed_policy(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    validate: bool = True,
) -> dict:
    """Load the repository seed policy and optionally verify its derivation."""
    path = Path(policy_path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Random-seed policy not found: {path}"
        )

    policy = json.loads(
        path.read_text(encoding="utf-8")
    )

    required = {
        "policy_version",
        "master_seed",
        "derivation_order",
        "derived_seeds",
        "fixed_split_seed",
        "training_seeds",
        "analysis_seeds",
    }

    missing = sorted(
        required - set(policy)
    )

    if missing:
        raise ValueError(
            f"Seed policy is missing fields: {missing}"
        )

    if validate:
        validate_seed_policy(policy)

    return policy


def validate_seed_policy(policy: dict) -> None:
    """Reproduce all derived values from the documented master seed."""
    master_seed = int(
        policy["master_seed"]
    )

    derivation_order = list(
        policy["derivation_order"]
    )

    children = np.random.SeedSequence(
        master_seed
    ).spawn(
        len(derivation_order)
    )

    reproduced = {
        name: int(
            child.generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )
        for name, child in zip(
            derivation_order,
            children,
        )
    }

    documented = {
        str(name): int(value)
        for name, value in policy[
            "derived_seeds"
        ].items()
    }

    if reproduced != documented:
        raise ValueError(
            "The documented seed values cannot be reproduced "
            "from the master seed and derivation order.\n"
            f"Reproduced: {reproduced}\n"
            f"Documented: {documented}"
        )

    expected_training = [
        documented[f"training_run_{run}"]
        for run in range(1, 6)
    ]

    observed_training = [
        int(value)
        for value in policy["training_seeds"]
    ]

    if observed_training != expected_training:
        raise ValueError(
            "training_seeds does not match the five derived "
            "training-run values."
        )

    expected_split = documented[
        "fixed_ham10000_split"
    ]

    if int(
        policy["fixed_split_seed"]
    ) != expected_split:
        raise ValueError(
            "fixed_split_seed does not match the derived "
            "fixed_ham10000_split value."
        )

    for name, value in policy[
        "analysis_seeds"
    ].items():
        if int(value) != documented[name]:
            raise ValueError(
                f"Analysis seed {name} does not match its "
                "derived value."
            )


def get_training_seeds() -> list[int]:
    """Return the five prespecified training-randomization values."""
    policy = load_seed_policy()

    return [
        int(value)
        for value in policy["training_seeds"]
    ]


def get_fixed_split_seed() -> int:
    """Return the immutable HAM10000 split seed."""
    policy = load_seed_policy()

    return int(
        policy["fixed_split_seed"]
    )


def get_analysis_seed(name: str) -> int:
    """Return one named analysis seed."""
    policy = load_seed_policy()

    analysis_seeds = policy[
        "analysis_seeds"
    ]

    if name not in analysis_seeds:
        raise KeyError(
            f"Unknown analysis seed: {name}. "
            f"Available values: {sorted(analysis_seeds)}"
        )

    return int(
        analysis_seeds[name]
    )
