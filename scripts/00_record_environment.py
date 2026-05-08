#!/usr/bin/env python3
"""Record the computational environment for an official experiment run."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_command(command: list[str]) -> str:
    """Run a shell command and return stdout, or an informative error string."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "COMMAND_NOT_FOUND"

    output = result.stdout.strip()
    error = result.stderr.strip()

    if result.returncode != 0:
        return f"RETURN_CODE={result.returncode}; STDERR={error}"

    return output


def main() -> None:
    output_dir = Path("outputs/logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_branch": run_command(["git", "branch", "--show-current"]),
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_command(["git", "status", "--short"]),
        "docker_version": run_command(["docker", "--version"]),
        "nvidia_smi": run_command(["nvidia-smi"]),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }

    try:
        import tensorflow as tf

        record["tensorflow_version"] = tf.__version__
        record["tensorflow_gpus"] = [str(device) for device in tf.config.list_physical_devices("GPU")]
    except Exception as exc:
        record["tensorflow_error"] = repr(exc)

    output_path = output_dir / "environment_record.json"
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
