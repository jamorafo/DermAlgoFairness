#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Running expensive interval TAC/ETC estimation..."
"$PYTHON_BIN" scripts/12_interval_tac_etc_analysis.py

echo "Done."
