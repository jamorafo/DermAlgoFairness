#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "Generating publication figures..."

if command -v Rscript >/dev/null 2>&1 && [ -f scripts/19_make_sorted_bosque_gap_forest.R ]; then
  Rscript scripts/19_make_sorted_bosque_gap_forest.R
else
  echo "Rscript not available or R figure script missing. Skipping R figures."
fi

echo "Done."
