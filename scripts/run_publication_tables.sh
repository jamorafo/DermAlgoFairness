#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Generating publication tables..."

"$PYTHON_BIN" scripts/14_make_interval_tac_etc_publication_table.py

if [ -f scripts/15_make_publication_latex_tables_01_to_03.py ]; then
  "$PYTHON_BIN" scripts/15_make_publication_latex_tables_01_to_03.py
fi

"$PYTHON_BIN" scripts/16_make_pr_documentation_tables.py
"$PYTHON_BIN" scripts/17_make_source_pr_diagnostics.py
"$PYTHON_BIN" scripts/18_make_target_pr_diagnostics.py

echo "Done."
