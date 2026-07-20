# -------------------------------------------------------------------------
# Reproduction and validation driver for all R publication figures.
#
# Usage from the repository root:
#   Rscript --vanilla scripts/R/00_run_all_publication_figures.R
#   Rscript --vanilla scripts/R/00_run_all_publication_figures.R --check-only
#
# The figure scripts only read finalized CSV outputs. They do not fit models
# or repeat the bootstrap analyses.
# -------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check-only" %in% args

required_root_files <- c(
  file.path("scripts", "R", "figure_theme.R"),
  file.path("outputs", "tables", "interval_tac_etc_by_seed.csv")
)

missing_root_files <- required_root_files[!file.exists(required_root_files)]

if (length(missing_root_files) > 0L) {
  stop(
    "Run this script from the repository root. Missing: ",
    paste(missing_root_files, collapse = ", ")
  )
}

figure_scripts <- c(
  file.path("scripts", "R", "01_seed_aware_gap_figures.R"),
  file.path("scripts", "R", "02_interval_decision_summary.R"),
  file.path("scripts", "R", "03_internal_external_figures.R"),
  file.path("scripts", "R", "04_architecture_bh_sensitivity_figure.R"),
  file.path("scripts", "R", "05_finite_orp_precision_figure.R"),
  file.path("scripts", "R", "06_mean_gap_heatmap.R")
)

missing_scripts <- figure_scripts[!file.exists(figure_scripts)]

if (length(missing_scripts) > 0L) {
  stop(
    "Missing R figure scripts: ",
    paste(missing_scripts, collapse = ", ")
  )
}

figure_basenames <- c(
  "figure_bosque_recall_gap_seed_aware",
  "figure_bosque_auc_pr_gap_seed_aware",
  "figure_bosque_f1_gap_seed_aware",
  "figure_bosque_precision_gap_seed_aware",
  "figure_interval_tac_etc_consistency",
  "figure_internal_external_f1",
  "figure_internal_external_auc_pr",
  "figure_architecture_level_bh_sensitivity",
  "figure_finite_orp_precision_primary_metrics",
  "figure_bosque_light_dark_mean_gap_heatmap"
)

expected_outputs <- as.vector(
  outer(
    file.path("outputs", "figures-r", figure_basenames),
    c(".pdf", ".png"),
    paste0
  )
)

if (!check_only) {
  rscript_name <- if (.Platform$OS.type == "windows") {
    "Rscript.exe"
  } else {
    "Rscript"
  }

  rscript <- file.path(R.home("bin"), rscript_name)

  if (!file.exists(rscript)) {
    stop("Could not locate Rscript at: ", rscript)
  }

  for (script in figure_scripts) {
    message("")
    message("Running: ", script)

    status <- system2(
      command = rscript,
      args = c("--vanilla", script)
    )

    if (!identical(status, 0L)) {
      stop(
        "Figure script failed with status ",
        status,
        ": ",
        script
      )
    }
  }
} else {
  message("Check-only mode: existing figures will not be regenerated.")
}

exists_flag <- file.exists(expected_outputs)
sizes <- rep(NA_real_, length(expected_outputs))

if (any(exists_flag)) {
  sizes[exists_flag] <- file.info(expected_outputs[exists_flag])$size
}

valid_flag <- exists_flag & !is.na(sizes) & sizes > 0

audit <- data.frame(
  file = expected_outputs,
  exists = exists_flag,
  bytes = sizes,
  valid = valid_flag,
  stringsAsFactors = FALSE
)

cat("\nR publication figure audit\n")
cat(strrep("-", 80), "\n", sep = "")
print(audit, row.names = FALSE)
cat(strrep("-", 80), "\n", sep = "")

n_valid <- sum(audit$valid)
n_total <- nrow(audit)

if (n_valid != n_total) {
  invalid <- audit$file[!audit$valid]

  stop(
    "Figure audit failed: ",
    n_valid,
    "/",
    n_total,
    " valid files. Missing or empty: ",
    paste(invalid, collapse = ", ")
  )
}

cat(
  "PASS: ",
  n_valid,
  "/",
  n_total,
  " expected PDF/PNG files exist and are non-empty.\n",
  sep = ""
)

cat(
  "PASS: 10 figure basenames are available in both publication formats.\n"
)
