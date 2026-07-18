#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="/workspaces/DermAlgoFairness"
PYTHON="$ROOT/.venv-wholeham/bin/python"
LOGDIR="$ROOT/outputs/logs"
STATUS="$LOGDIR/final_10000_bootstraps_status.txt"

cd "$ROOT"

mkdir -p "$LOGDIR"

trap '
rc=$?
echo "FAILED $(date -Is) exit=$rc command=$BASH_COMMAND" \
  | tee -a "$STATUS"
exit "$rc"
' ERR

run_step() {
    local name="$1"
    shift

    echo
    echo "============================================================"
    echo "START $name: $(date -Is)"
    echo "COMMAND: $*"
    echo "============================================================"

    "$@" 2>&1 \
      | tee "$LOGDIR/${name}.log"

    echo "FINISH $name: $(date -Is)"
}

echo "STARTED $(date -Is)" > "$STATUS"

cat > "$LOGDIR/final_10000_bootstraps_metadata.txt" <<'EOF'
policy_version=1.0
master_seed=104729
training_seeds=1,2,3,4,5
primary_interval_bootstrap_seed=1517787898
primary_interval_bootstrap_replicates=10000
seed_aware_subgroup_gap_seed=3484139151
seed_aware_subgroup_gap_replicates=10000
architecture_level_bh_seed=2729206535
architecture_level_bh_replicates=10000
classification_threshold=0.5
execution=sequential
EOF

run_step \
  final_12_interval_tac_etc_10000 \
  "$PYTHON" -u \
  scripts/12_interval_tac_etc_analysis.py

run_step \
  final_14_interval_publication_table \
  "$PYTHON" -u \
  scripts/14_make_interval_tac_etc_publication_table.py

run_step \
  final_17_source_precision \
  "$PYTHON" -u \
  scripts/17_make_source_pr_diagnostics.py

run_step \
  final_18_target_precision \
  "$PYTHON" -u \
  scripts/18_make_target_pr_diagnostics.py

run_step \
  final_06_subgroup_gaps_10000 \
  "$PYTHON" -u \
  scripts/06_compare_bosque_subgroups.py \
    --n-boot 10000 \
    --threshold 0.5 \
    --seed 3484139151

run_step \
  final_20_subgroup_publication_outputs \
  "$PYTHON" -u \
  scripts/20_make_seed_aware_gap_publication_outputs.py

run_step \
  final_08_descriptive_figures \
  "$PYTHON" -u \
  scripts/08_plot_results.py

for required in \
  outputs/tables/interval_tac_etc_by_seed.csv \
  outputs/tables/interval_tac_etc_summary.csv \
  outputs/tables/interval_tac_etc_consistency.csv \
  outputs/tables/bosque_light_dark_gap_by_model_seed.csv \
  outputs/tables/bosque_light_dark_gap_summary_by_architecture.csv \
  outputs/publication_tables/table_03_seed_aware_light_dark_gaps.tex \
  outputs/publication_tables/table_04_interval_tac_etc_consistency.tex \
  outputs/publication_tables/table_07_source_pr_diagnostics.tex \
  outputs/publication_tables/table_08_target_pr_diagnostics.tex
do
    if [[ ! -s "$required" ]]; then
        echo "ERROR: missing or empty output: $required"
        exit 1
    fi
done

sha256sum \
  outputs/tables/interval_tac_etc_by_seed.csv \
  outputs/tables/interval_tac_etc_consistency.csv \
  outputs/tables/bosque_light_dark_gap_by_model_seed.csv \
  outputs/tables/bosque_light_dark_gap_summary_by_architecture.csv \
  outputs/publication_tables/table_03_seed_aware_light_dark_gaps.tex \
  outputs/publication_tables/table_04_interval_tac_etc_consistency.tex \
  > "$LOGDIR/final_10000_result_checksums.sha256"

echo "FINISHED SUCCESSFULLY $(date -Is)" \
  | tee -a "$STATUS"
