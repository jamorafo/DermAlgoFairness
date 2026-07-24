# -------------------------------------------------------------------------
# Render the nine publication LaTeX tables from finalized canonical CSVs.
#
# This script is a publication-rendering layer only. It does not train models,
# generate predictions, or repeat any bootstrap analysis.
#
# Usage:
#   Rscript --vanilla scripts/R/07_make_publication_tables.R
#   Rscript --vanilla scripts/R/07_make_publication_tables.R \
#       --output-dir=/tmp/dermalgo-r-table-preview
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)

input_dir <- file.path(
  "outputs",
  "publication_tables"
)

output_dir <- input_dir

output_argument <- grep(
  "^--output-dir=",
  args,
  value = TRUE
)

if (length(output_argument) > 1L) {
  stop("Specify at most one --output-dir argument.")
}

if (length(output_argument) == 1L) {
  output_dir <- sub(
    "^--output-dir=",
    "",
    output_argument
  )
}

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

###############################################################################
# General helpers
###############################################################################

read_table_csv <- function(filename) {
  path <- file.path(
    input_dir,
    filename
  )

  if (!file.exists(path)) {
    stop("Missing canonical CSV: ", path)
  }

  read_csv(
    path,
    show_col_types = FALSE,
    progress = FALSE,
    name_repair = "minimal"
  )
}

assert_columns <- function(
  data,
  expected,
  table_name
) {
  if (!identical(names(data), expected)) {
    stop(
      table_name,
      " has unexpected columns.\nExpected: ",
      paste(expected, collapse = ", "),
      "\nObserved: ",
      paste(names(data), collapse = ", ")
    )
  }
}

assert_rows <- function(
  data,
  expected,
  table_name
) {
  if (nrow(data) != expected) {
    stop(
      table_name,
      " has ",
      nrow(data),
      " rows; expected ",
      expected,
      "."
    )
  }
}

normalise_publication_prose <- function(x) {
  x <- as.character(x)

  x <- gsub(
    "model--seed",
    "architecture--run",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "training-seed",
    "training-run",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "seed-specific",
    "run-specific",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "random seeds",
    "independently randomized training runs",
    x,
    fixed = TRUE
  )

  x
}

latex_escape_plain <- function(x) {
  x <- normalise_publication_prose(x)

  x <- gsub(
    "&",
    "\\\\&",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "%",
    "\\\\%",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "#",
    "\\\\#",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "_",
    "\\\\_",
    x,
    fixed = TRUE
  )

  x
}

latex_escape_preserve_math <- function(x) {
  if (
    length(x) != 1L ||
    is.na(x)
  ) {
    return("")
  }

  x <- as.character(x)

  dollar_positions <- gregexpr(
    "$",
    x,
    fixed = TRUE
  )[[1]]

  if (
    length(dollar_positions) == 1L &&
    dollar_positions[[1]] == -1L
  ) {
    return(
      latex_escape_plain(x)
    )
  }

  if (
    length(dollar_positions) %% 2L != 0L
  ) {
    stop(
      "Unbalanced dollar-delimited mathematics in: ",
      x
    )
  }

  output <- ""
  cursor <- 1L

  for (
    index in seq.int(
      1L,
      length(dollar_positions),
      by = 2L
    )
  ) {
    open_position <- dollar_positions[[index]]
    close_position <- dollar_positions[[index + 1L]]

    plain_text <- if (
      open_position > cursor
    ) {
      substr(
        x,
        cursor,
        open_position - 1L
      )
    } else {
      ""
    }

    math_text <- substr(
      x,
      open_position,
      close_position
    )

    output <- paste0(
      output,
      latex_escape_plain(plain_text),
      math_text
    )

    cursor <- close_position + 1L
  }

  trailing_text <- if (
    cursor <= nchar(x)
  ) {
    substr(
      x,
      cursor,
      nchar(x)
    )
  } else {
    ""
  }

  paste0(
    output,
    latex_escape_plain(trailing_text)
  )
}

format_mean_sd <- function(x) {
  pieces <- strsplit(
    as.character(x),
    " ± ",
    fixed = TRUE
  )[[1]]

  if (length(pieces) != 2L) {
    stop(
      "Expected a mean ± SD value, found: ",
      x
    )
  }

  paste0(
    pieces[[1]],
    " $\\pm$ ",
    pieces[[2]]
  )
}

format_three <- function(x) {
  sprintf(
    "%.3f",
    as.numeric(x)
  )
}

format_four <- function(x) {
  sprintf(
    "%.4f",
    as.numeric(x)
  )
}

latex_row <- function(values) {
  paste0(
    paste(
      values,
      collapse = " & "
    ),
    " \\\\"
  )
}

table_shell <- function(
  caption,
  label,
  tabular,
  header,
  body,
  note,
  tabcolsep
) {
  c(
    "\\begin{table}[h!]",
    "\\centering",
    "\\scriptsize",
    paste0(
      "\\caption{",
      caption,
      "}"
    ),
    paste0(
      "\\label{",
      label,
      "}"
    ),
    paste0(
      "\\setlength{\\tabcolsep}{",
      tabcolsep,
      "}"
    ),
    "\\renewcommand{\\arraystretch}{1.08}",
    paste0(
      "\\begin{tabular}{",
      tabular,
      "}"
    ),
    "\\toprule",
    latex_row(header),
    "\\midrule",
    body,
    "\\bottomrule",
    "\\end{tabular}",
    "\\begin{flushleft}",
    paste0(
      "\\footnotesize Notes: ",
      note
    ),
    "\\end{flushleft}",
    "\\end{table}"
  )
}

write_table <- function(
  filename,
  lines
) {
  path <- file.path(
    output_dir,
    filename
  )

  writeLines(
    lines,
    path,
    useBytes = TRUE
  )

  if (
    !file.exists(path) ||
    file.info(path)$size <= 0
  ) {
    stop(
      "Failed to create: ",
      path
    )
  }

  message("Saved: ", path)
}

###############################################################################
# Table 01
###############################################################################

render_table_01 <- function() {
  data <- read_table_csv(
    "table_01_internal_external_performance.csv"
  )

  expected_columns <- c(
    "model",
    "dataset",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc_roc",
    "auc_pr"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 01"
  )

  assert_rows(
    data,
    10L,
    "Table 01"
  )

  architectures <- unique(
    data$model
  )

  metric_columns <- c(
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc_roc",
    "auc_pr"
  )

  body <- character()

  for (
    architecture_index in seq_along(
      architectures
    )
  ) {
    architecture <- architectures[[
      architecture_index
    ]]

    block <- data[
      data$model == architecture,
      ,
      drop = FALSE
    ]

    if (nrow(block) != 2L) {
      stop(
        "Table 01 expected two ORPs for ",
        architecture,
        "."
      )
    }

    for (row_index in seq_len(nrow(block))) {
      architecture_label <- if (
        row_index == 1L
      ) {
        latex_escape_plain(
          architecture
        )
      } else {
        ""
      }

      metric_values <- vapply(
        metric_columns,
        function(column) {
          format_mean_sd(
            block[[column]][[row_index]]
          )
        },
        character(1)
      )

      body <- c(
        body,
        latex_row(
          c(
            architecture_label,
            latex_escape_plain(
              block$dataset[[row_index]]
            ),
            metric_values
          )
        )
      )
    }

    if (
      architecture_index <
      length(architectures)
    ) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "Internal source and external target performance",
      "across five independently randomized training runs."
    ),
    label = "tab:internal-external-performance",
    tabular = "llrrrrrrr",
    header = c(
      "\\textbf{Architecture}",
      "\\textbf{Evaluation ORP}",
      "\\textbf{Acc.}",
      "\\textbf{Prec.}",
      "\\textbf{Recall}",
      "\\textbf{Spec.}",
      "\\textbf{F1}",
      "\\textbf{AUC-ROC}",
      "\\textbf{AUC-PR}"
    ),
    body = body,
    note = paste(
      "Values are mean $\\pm$ standard deviation across five",
      "independently randomized training runs.",
      "HAM10000 internal denotes the fixed lesion-grouped held-out",
      "source ORP ($990$ images from $747$ lesions).",
      "BOSQUE external denotes the external target ORP ($n=151$).",
      "Point estimates are computed for the malignant class under",
      "the locked binary diagnostic task."
    ),
    tabcolsep = "3.5pt"
  )

  write_table(
    "table_01_internal_external_performance.tex",
    lines
  )
}

###############################################################################
# Table 02
###############################################################################

render_table_02 <- function() {
  data <- read_table_csv(
    "table_02_bosque_subgroup_performance.csv"
  )

  expected_columns <- c(
    "model",
    "subgroup",
    "n",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc_roc",
    "auc_pr"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 02"
  )

  assert_rows(
    data,
    10L,
    "Table 02"
  )

  architectures <- unique(
    data$model
  )

  subgroup_labels <- c(
    light = "Light phototype",
    dark = "Dark phototype"
  )

  metric_columns <- c(
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auc_roc",
    "auc_pr"
  )

  body <- character()

  for (
    architecture_index in seq_along(
      architectures
    )
  ) {
    architecture <- architectures[[
      architecture_index
    ]]

    block <- data[
      data$model == architecture,
      ,
      drop = FALSE
    ]

    if (nrow(block) != 2L) {
      stop(
        "Table 02 expected two subgroups for ",
        architecture,
        "."
      )
    }

    for (row_index in seq_len(nrow(block))) {
      subgroup <- as.character(
        block$subgroup[[row_index]]
      )

      if (
        !subgroup %in%
        names(subgroup_labels)
      ) {
        stop(
          "Unexpected Table 02 subgroup: ",
          subgroup
        )
      }

      architecture_label <- if (
        row_index == 1L
      ) {
        latex_escape_plain(
          architecture
        )
      } else {
        ""
      }

      metric_values <- vapply(
        metric_columns,
        function(column) {
          format_mean_sd(
            block[[column]][[row_index]]
          )
        },
        character(1)
      )

      body <- c(
        body,
        latex_row(
          c(
            architecture_label,
            subgroup_labels[[subgroup]],
            as.character(
              block$n[[row_index]]
            ),
            metric_values
          )
        )
      )
    }

    if (
      architecture_index <
      length(architectures)
    ) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "BOSQUE target-side performance by phototype group",
      "across five independently randomized training runs."
    ),
    label = "tab:bosque-subgroup-performance",
    tabular = "llrrrrrrrr",
    header = c(
      "\\textbf{Architecture}",
      "\\textbf{Subgroup}",
      "\\textbf{$n$}",
      "\\textbf{Acc.}",
      "\\textbf{Prec.}",
      "\\textbf{Recall}",
      "\\textbf{Spec.}",
      "\\textbf{F1}",
      "\\textbf{AUC-ROC}",
      "\\textbf{AUC-PR}"
    ),
    body = body,
    note = paste(
      "Values are mean $\\pm$ standard deviation across five",
      "independently randomized training runs.",
      "The target ORPs contain $105$ light-phototype and",
      "$46$ dark-phototype images.",
      "BOSQUE contains one image per lesion.",
      "These summaries describe performance under the observed",
      "BOSQUE subgroup conditions and do not by themselves",
      "establish population-level generalizability."
    ),
    tabcolsep = "3.2pt"
  )

  write_table(
    "table_02_bosque_subgroup_performance.tex",
    lines
  )
}

###############################################################################
# Table 03
###############################################################################

render_table_03 <- function() {
  data <- read_table_csv(
    "table_03_seed_aware_light_dark_gaps.csv"
  )

  expected_columns <- c(
    "Metric",
    "Architecture",
    "n_seeds",
    "mean_light",
    "mean_dark",
    "mean_gap",
    "sd_gap",
    "n_positive_gap",
    "n_ci_light_higher",
    "n_ci_includes_zero",
    "n_ci_dark_higher",
    "ci_direction_counts"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 03"
  )

  assert_rows(
    data,
    20L,
    "Table 03"
  )

  if (any(data$n_seeds != 5L)) {
    stop(
      "Table 03 must summarize five training runs per architecture."
    )
  }

  body <- character()

  for (row_index in seq_len(nrow(data))) {
    metric_label <- if (
      row_index == 1L ||
      data$Metric[[row_index]] !=
        data$Metric[[row_index - 1L]]
    ) {
      latex_escape_plain(
        data$Metric[[row_index]]
      )
    } else {
      ""
    }

    body <- c(
      body,
      latex_row(
        c(
          metric_label,
          latex_escape_plain(
            data$Architecture[[row_index]]
          ),
          format_three(
            data$mean_light[[row_index]]
          ),
          format_three(
            data$mean_dark[[row_index]]
          ),
          format_three(
            data$mean_gap[[row_index]]
          ),
          format_three(
            data$sd_gap[[row_index]]
          ),
          paste0(
            data$n_positive_gap[[row_index]],
            "/",
            data$n_seeds[[row_index]]
          ),
          as.character(
            data$ci_direction_counts[[row_index]]
          )
        )
      )
    )

    if (
      row_index < nrow(data) &&
      data$Metric[[row_index + 1L]] !=
        data$Metric[[row_index]]
    ) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "Seed-aware BOSQUE light--dark performance gaps",
      "for the primary metrics."
    ),
    label = "tab:seed-aware-light-dark-gaps",
    tabular = "llrrrrrr",
    header = c(
      "\\textbf{Metric}",
      "\\textbf{Architecture}",
      "\\textbf{Light}",
      "\\textbf{Dark}",
      "\\textbf{Mean gap}",
      "\\textbf{SD gap}",
      "\\textbf{Positive}",
      "\\textbf{CI $+/0/-$}"
    ),
    body = body,
    note = paste(
      "The gap is light-phototype performance minus",
      "dark-phototype performance.",
      "Light, dark, mean gap, and SD gap summarize five",
      "independently randomized training runs within each architecture.",
      "``Positive'' gives the number of runs with a positive",
      "point-estimate gap.",
      "CI $+/0/-$ gives the number of run-specific 95\\%",
      "percentile-bootstrap intervals lying entirely above zero,",
      "including zero, or lying entirely below zero, respectively.",
      "Each interval uses 10{,}000 image-level bootstrap replicates",
      "within the observed BOSQUE light ($n=105$) and dark ($n=46$)",
      "ORPs.",
      "Intervals are conditional on the locked system and on",
      "image-level independence.",
      "No $p$-values or multiplicity adjustments are used."
    ),
    tabcolsep = "3.5pt"
  )

  write_table(
    "table_03_seed_aware_light_dark_gaps.tex",
    lines
  )
}

###############################################################################
# Table 04
###############################################################################

render_table_04 <- function() {
  data <- read_table_csv(
    "table_04_interval_tac_etc_consistency.csv"
  )

  expected_columns <- c(
    "Metric group",
    "Metric",
    "Target condition",
    "Adequate and preserved",
    "Inconclusive",
    "Not adequate and not preserved",
    "Preserved but inadequate",
    "Adequate but not preserved",
    "Evidentially unresolved",
    "Total model--seed decisions"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 04"
  )

  assert_rows(
    data,
    21L,
    "Table 04"
  )

  if (
    any(
      data[[
        "Total model--seed decisions"
      ]] != 25L
    )
  ) {
    stop(
      "Table 04 must summarize 25 locked systems per row."
    )
  }

  body <- character()

  for (row_index in seq_len(nrow(data))) {
    new_group <- (
      row_index == 1L ||
      data[["Metric group"]][[row_index]] !=
        data[["Metric group"]][[row_index - 1L]]
    )

    new_metric <- (
      row_index == 1L ||
      new_group ||
      data$Metric[[row_index]] !=
        data$Metric[[row_index - 1L]]
    )

    group_label <- if (new_group) {
      latex_escape_plain(
        data[["Metric group"]][[row_index]]
      )
    } else {
      ""
    }

    metric_label <- if (new_metric) {
      latex_escape_plain(
        data$Metric[[row_index]]
      )
    } else {
      ""
    }

    body <- c(
      body,
      latex_row(
        c(
          group_label,
          metric_label,
          latex_escape_plain(
            data[["Target condition"]][[row_index]]
          ),
          data[["Adequate and preserved"]][[row_index]],
          data$Inconclusive[[row_index]],
          data[[
            "Not adequate and not preserved"
          ]][[row_index]],
          data[[
            "Preserved but inadequate"
          ]][[row_index]],
          data[[
            "Adequate but not preserved"
          ]][[row_index]],
          data[[
            "Evidentially unresolved"
          ]][[row_index]],
          data[[
            "Total model--seed decisions"
          ]][[row_index]]
        )
      )
    )

    if (
      row_index < nrow(data) &&
      (
        data[["Metric group"]][[row_index + 1L]] !=
          data[["Metric group"]][[row_index]] ||
        data$Metric[[row_index + 1L]] !=
          data$Metric[[row_index]]
      )
    ) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "Interval-based target adequacy and performance-preservation",
      "decisions across locked architecture--run systems."
    ),
    label = "tab:interval-tac-etc-consistency",
    tabular = "lllrrrrrrr",
    header = c(
      "\\textbf{Group}",
      "\\textbf{Metric}",
      "\\textbf{Target}",
      "\\textbf{Adeq. + pres.}",
      "\\textbf{Inconc.}",
      "\\textbf{Not adeq. + not pres.}",
      "\\textbf{Pres. but inad.}",
      "\\textbf{Adeq. but not pres.}",
      "\\textbf{Unresolved}",
      "\\textbf{Total}"
    ),
    body = body,
    note = paste(
      "Each row summarizes 25 locked architecture--run decisions,",
      "corresponding to five architectures and five independently",
      "randomized training runs.",
      "For the overall BOSQUE condition, preservation denotes ETC",
      "transportability relative to the HAM10000 held-out source estimate.",
      "For the light- and dark-phototype conditions, preservation denotes",
      "comparison with the overall HAM10000 source benchmark and is not",
      "subgroup transportability.",
      "Primary metrics drive the main interpretation; secondary metrics",
      "are descriptive.",
      "Decisions are based on 95\\% intervals."
    ),
    tabcolsep = "3pt"
  )

  write_table(
    "table_04_interval_tac_etc_consistency.tex",
    lines
  )
}

###############################################################################
# Table 05
###############################################################################

render_table_05 <- function() {
  data <- read_table_csv(
    "table_05_orp_pr_assessment.csv"
  )

  expected_columns <- c(
    "Claim",
    "ORP",
    "Estimand",
    "Sampling or evaluation basis",
    "PR status",
    "Scope of inference",
    "Limitation"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 05"
  )

  assert_rows(
    data,
    6L,
    "Table 05"
  )

  body <- character()

  for (row_index in seq_len(nrow(data))) {
    values <- vapply(
      expected_columns,
      function(column) {
        latex_escape_preserve_math(
          data[[column]][[row_index]]
        )
      },
      character(1)
    )

    body <- c(
      body,
      latex_row(values)
    )

    if (row_index < nrow(data)) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "Objective Reference Point and Predictive Representativity",
      "assessment by validation claim."
    ),
    label = "tab:orp-pr-assessment",
    tabular = paste0(
      "p{2.5cm}",
      "p{2.5cm}",
      "p{2.1cm}",
      "p{3.4cm}",
      "p{2.5cm}",
      "p{3.3cm}",
      "p{3.2cm}"
    ),
    header = c(
      "\\textbf{Claim}",
      "\\textbf{ORP}",
      "\\textbf{Estimand}",
      "\\textbf{Sampling/evaluation basis}",
      "\\textbf{PR status}",
      "\\textbf{Scope of inference}",
      "\\textbf{Limitation}"
    ),
    body = body,
    note = paste(
      "PR denotes Predictive Representativity.",
      "The table separates the evidential status of each validation",
      "objective before TAC/ETC decision rules are applied.",
      "A PR-adequate objective is not necessarily adequate or transported;",
      "it only means that the ORP, estimand, estimator, and uncertainty",
      "procedure are aligned with the stated scope of inference."
    ),
    tabcolsep = "3pt"
  )

  write_table(
    "table_05_orp_pr_assessment.tex",
    lines
  )
}

###############################################################################
# Table 06
###############################################################################

render_table_06 <- function() {
  data <- read_table_csv(
    "table_06_uncertainty_sources.csv"
  )

  expected_columns <- c(
    "Uncertainty source",
    "Applies to",
    "Meaning",
    "Treatment",
    "Not treated as"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 06"
  )

  assert_rows(
    data,
    8L,
    "Table 06"
  )

  body <- character()

  for (row_index in seq_len(nrow(data))) {
    values <- vapply(
      expected_columns,
      function(column) {
        latex_escape_preserve_math(
          data[[column]][[row_index]]
        )
      },
      character(1)
    )

    body <- c(
      body,
      latex_row(values)
    )

    if (row_index < nrow(data)) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "Uncertainty sources distinguished in the",
      "ORP-based validation analysis."
    ),
    label = "tab:uncertainty-sources",
    tabular = paste0(
      "p{3.0cm}",
      "p{3.0cm}",
      "p{4.2cm}",
      "p{4.2cm}",
      "p{3.7cm}"
    ),
    header = c(
      "\\textbf{Uncertainty source}",
      "\\textbf{Applies to}",
      "\\textbf{Meaning}",
      "\\textbf{Treatment}",
      "\\textbf{Not treated as}"
    ),
    body = body,
    note = paste(
      "Evaluation uncertainty is conditional on a locked predictive",
      "system and is estimated from the corresponding non-augmented ORP.",
      "Training stochasticity concerns variation in the learned function",
      "across independently randomized training runs and is summarized",
      "separately from ORP sampling uncertainty.",
      "Training augmentation and oversampling are model-development",
      "procedures, not mechanisms for increasing the effective evaluation",
      "sample size."
    ),
    tabcolsep = "4pt"
  )

  write_table(
    "table_06_uncertainty_sources.tex",
    lines
  )
}

###############################################################################
# Table 07
###############################################################################

render_table_07 <- function() {
  data <- read_table_csv(
    "table_07_source_pr_diagnostics.csv"
  )

  expected_columns <- c(
    "Metric",
    "n_model_seed",
    "n_source_images",
    "n_source_lesions",
    "mean_source",
    "sd_across_systems",
    "mean_half_width",
    "max_half_width"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 07"
  )

  assert_rows(
    data,
    7L,
    "Table 07"
  )

  if (
    any(data$n_model_seed != 25L) ||
    any(data$n_source_images != 990L) ||
    any(data$n_source_lesions != 747L)
  ) {
    stop(
      "Table 07 source counts are inconsistent with the locked ORP."
    )
  }

  body <- character()

  for (row_index in seq_len(nrow(data))) {
    body <- c(
      body,
      latex_row(
        c(
          latex_escape_plain(
            data$Metric[[row_index]]
          ),
          data$n_model_seed[[row_index]],
          data$n_source_images[[row_index]],
          data$n_source_lesions[[row_index]],
          format_three(
            data$mean_source[[row_index]]
          ),
          format_three(
            data$sd_across_systems[[row_index]]
          ),
          format_three(
            data$mean_half_width[[row_index]]
          ),
          format_three(
            data$max_half_width[[row_index]]
          )
        )
      )
    )
  }

  lines <- table_shell(
    caption = paste(
      "Finite-ORP precision of HAM10000 held-out",
      "source performance estimates."
    ),
    label = "tab:source-pr-diagnostics",
    tabular = "lrrrrrrr",
    header = c(
      "\\textbf{Metric}",
      "\\textbf{$n$ systems}",
      "\\textbf{Images}",
      "\\textbf{Lesions}",
      "\\textbf{Mean}",
      "\\textbf{SD systems}",
      "\\textbf{Mean HW}",
      "\\textbf{Max HW}"
    ),
    body = body,
    note = paste(
      "Each row summarizes 25 locked architecture--run systems evaluated",
      "on the fixed HAM10000 source ORP of $990$ images from $747$ lesions.",
      "HW denotes the half-width of the 95\\% percentile-bootstrap interval.",
      "Source intervals use 10{,}000 lesion-cluster bootstrap replicates,",
      "preserving all images belonging to a sampled lesion.",
      "SD systems describes variation across architectures and training runs",
      "and is not a sampling standard error.",
      "Interval width is reported descriptively without imposing an",
      "additional post hoc precision threshold.",
      "These summaries do not establish clinical adequacy, bias control,",
      "or population representativity."
    ),
    tabcolsep = "3pt"
  )

  write_table(
    "table_07_source_pr_diagnostics.tex",
    lines
  )
}

###############################################################################
# Table 08
###############################################################################

render_table_08 <- function() {
  data <- read_table_csv(
    "table_08_target_pr_diagnostics.csv"
  )

  expected_columns <- c(
    "Metric",
    "Target condition",
    "n_model_seed",
    "n_target",
    "mean_target",
    "sd_across_systems",
    "mean_half_width",
    "max_half_width"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 08"
  )

  assert_rows(
    data,
    21L,
    "Table 08"
  )

  if (any(data$n_model_seed != 25L)) {
    stop(
      "Table 08 must summarize 25 locked systems."
    )
  }

  target_sizes <- unique(
    data[
      ,
      c(
        "Target condition",
        "n_target"
      )
    ]
  )

  expected_targets <- c(
    Overall = 151L,
    "Light phototype" = 105L,
    "Dark phototype" = 46L
  )

  if (nrow(target_sizes) != length(expected_targets)) {
    stop(
      "Table 08 must contain exactly three distinct target ORPs."
    )
  }

  unexpected_conditions <- setdiff(
    as.character(
      target_sizes[[
        "Target condition"
      ]]
    ),
    names(expected_targets)
  )

  if (length(unexpected_conditions) > 0L) {
    stop(
      "Unexpected Table 08 target condition(s): ",
      paste(
        unexpected_conditions,
        collapse = ", "
      )
    )
  }

  for (condition in names(expected_targets)) {
    observed_values <- target_sizes$n_target[
      target_sizes[[
        "Target condition"
      ]] == condition
    ]

    if (
      length(observed_values) != 1L ||
      as.integer(observed_values[[1]]) !=
        expected_targets[[condition]]
    ) {
      stop(
        "Table 08 target count mismatch for ",
        condition,
        ": expected ",
        expected_targets[[condition]],
        ", observed ",
        paste(
          observed_values,
          collapse = ", "
        ),
        "."
      )
    }
  }

  body <- character()

  for (row_index in seq_len(nrow(data))) {
    metric_label <- if (
      row_index == 1L ||
      data$Metric[[row_index]] !=
        data$Metric[[row_index - 1L]]
    ) {
      latex_escape_plain(
        data$Metric[[row_index]]
      )
    } else {
      ""
    }

    body <- c(
      body,
      latex_row(
        c(
          metric_label,
          latex_escape_plain(
            data[[
              "Target condition"
            ]][[row_index]]
          ),
          data$n_model_seed[[row_index]],
          data$n_target[[row_index]],
          format_three(
            data$mean_target[[row_index]]
          ),
          format_three(
            data$sd_across_systems[[row_index]]
          ),
          format_three(
            data$mean_half_width[[row_index]]
          ),
          format_three(
            data$max_half_width[[row_index]]
          )
        )
      )
    )

    if (
      row_index < nrow(data) &&
      data$Metric[[row_index + 1L]] !=
        data$Metric[[row_index]]
    ) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "Finite-ORP precision of BOSQUE overall and",
      "phototype-specific target performance estimates."
    ),
    label = "tab:target-pr-diagnostics",
    tabular = "llrrrrrr",
    header = c(
      "\\textbf{Metric}",
      "\\textbf{Target}",
      "\\textbf{$n$ systems}",
      "\\textbf{$n_T$}",
      "\\textbf{Mean}",
      "\\textbf{SD systems}",
      "\\textbf{Mean HW}",
      "\\textbf{Max HW}"
    ),
    body = body,
    note = paste(
      "Each row summarizes 25 locked architecture--run systems.",
      "The BOSQUE ORPs contain $151$ images overall,",
      "$105$ light-phototype images, and $46$ dark-phototype images.",
      "HW denotes the half-width of the 95\\% percentile-bootstrap interval.",
      "Target intervals use 10{,}000 image-level bootstrap replicates;",
      "because BOSQUE contains one image per lesion, this is also a",
      "lesion-level resampling scheme.",
      "SD systems describes architecture and training-run variation and",
      "is not a sampling standard error.",
      "Interval width is reported descriptively without imposing an",
      "additional post hoc precision threshold.",
      "These summaries do not establish clinical adequacy or",
      "generalizability beyond BOSQUE."
    ),
    tabcolsep = "3pt"
  )

  write_table(
    "table_08_target_pr_diagnostics.tex",
    lines
  )
}

###############################################################################
# Table 09
###############################################################################

render_table_09 <- function() {
  data <- read_table_csv(
    "table_09_architecture_level_bh_sensitivity.csv"
  )

  expected_columns <- c(
    "Metric",
    "Architecture",
    "n_seeds",
    "observed_mean_gap",
    "sd_seed_gaps",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "n_boot_valid",
    "p_value_raw",
    "p_value_bh",
    "p_value_by",
    "bh_significant",
    "by_significant"
  )

  assert_columns(
    data,
    expected_columns,
    "Table 09"
  )

  assert_rows(
    data,
    20L,
    "Table 09"
  )

  if (
    any(data$n_seeds != 5L) ||
    any(data$n_boot_valid != 10000L)
  ) {
    stop(
      "Table 09 must document five runs and 10,000 bootstrap replicates."
    )
  }

  body <- character()

  for (row_index in seq_len(nrow(data))) {
    metric_label <- if (
      row_index == 1L ||
      data$Metric[[row_index]] !=
        data$Metric[[row_index - 1L]]
    ) {
      latex_escape_plain(
        data$Metric[[row_index]]
      )
    } else {
      ""
    }

    confidence_interval <- paste0(
      "[",
      format_three(
        data$bootstrap_ci_low[[row_index]]
      ),
      ", ",
      format_three(
        data$bootstrap_ci_high[[row_index]]
      ),
      "]"
    )

    body <- c(
      body,
      latex_row(
        c(
          metric_label,
          latex_escape_plain(
            data$Architecture[[row_index]]
          ),
          format_three(
            data$observed_mean_gap[[row_index]]
          ),
          format_three(
            data$sd_seed_gaps[[row_index]]
          ),
          confidence_interval,
          format_four(
            data$p_value_raw[[row_index]]
          ),
          format_four(
            data$p_value_bh[[row_index]]
          ),
          format_four(
            data$p_value_by[[row_index]]
          ),
          latex_escape_plain(
            data$bh_significant[[row_index]]
          )
        )
      )
    )

    if (
      row_index < nrow(data) &&
      data$Metric[[row_index + 1L]] !=
        data$Metric[[row_index]]
    ) {
      body <- c(
        body,
        "\\addlinespace"
      )
    }
  }

  lines <- table_shell(
    caption = paste(
      "Architecture-level multiplicity-adjusted BOSQUE",
      "light--dark subgroup sensitivity analysis."
    ),
    label = "tab:architecture-bh-sensitivity",
    tabular = "llrrrrrrl",
    header = c(
      "\\textbf{Metric}",
      "\\textbf{Architecture}",
      "\\textbf{Mean gap}",
      "\\textbf{SD runs}",
      "\\textbf{95\\% CI}",
      "\\textbf{$p$}",
      "\\textbf{$p_{\\mathrm{BH}}$}",
      "\\textbf{$p_{\\mathrm{BY}}$}",
      "\\textbf{BH $<0.05$}"
    ),
    body = body,
    note = paste(
      "The architecture-level estimand is the mean light-minus-dark",
      "performance gap across the five locked run-specific systems.",
      "Positive values indicate higher performance in the light-phototype",
      "group.",
      "BOSQUE light and dark images were resampled separately;",
      "within each bootstrap replicate, the same sampled image indices",
      "were applied to all 25 systems.",
      "The 95\\% intervals are percentile-bootstrap intervals.",
      "Two-sided $p$-values were obtained from the null-centred",
      "bootstrap distribution.",
      "Benjamini--Hochberg adjustment was applied to the prespecified",
      "family of 20 architecture--primary-metric tests.",
      "Benjamini--Yekutieli adjusted values are reported as a conservative",
      "dependence sensitivity analysis.",
      "Inference is conditional on the five observed training runs and",
      "on image-level independence within the BOSQUE ORP.",
      "This secondary analysis does not replace the locked-system TAC,",
      "ETC, benchmark-preservation, or interval results."
    ),
    tabcolsep = "3.3pt"
  )

  write_table(
    "table_09_architecture_level_bh_sensitivity.tex",
    lines
  )
}

###############################################################################
# Render all nine tables
###############################################################################

render_table_01()
render_table_02()
render_table_03()
render_table_04()
render_table_05()
render_table_06()
render_table_07()
render_table_08()
render_table_09()

###############################################################################
# Final output audit
###############################################################################

expected_outputs <- file.path(
  output_dir,
  c(
    "table_01_internal_external_performance.tex",
    "table_02_bosque_subgroup_performance.tex",
    "table_03_seed_aware_light_dark_gaps.tex",
    "table_04_interval_tac_etc_consistency.tex",
    "table_05_orp_pr_assessment.tex",
    "table_06_uncertainty_sources.tex",
    "table_07_source_pr_diagnostics.tex",
    "table_08_target_pr_diagnostics.tex",
    "table_09_architecture_level_bh_sensitivity.tex"
  )
)

for (path in expected_outputs) {
  if (
    !file.exists(path) ||
    file.info(path)$size <= 0
  ) {
    stop(
      "Missing or empty rendered table: ",
      path
    )
  }

  text <- paste(
    readLines(
      path,
      warn = FALSE,
      encoding = "UTF-8"
    ),
    collapse = "\n"
  )

  required_markers <- c(
    "\\begin{table}[h!]",
    "\\caption{",
    "\\label{",
    "\\toprule",
    "\\midrule",
    "\\bottomrule",
    "\\end{table}"
  )

  missing_markers <- required_markers[
    !vapply(
      required_markers,
      function(marker) {
        grepl(
          marker,
          text,
          fixed = TRUE
        )
      },
      logical(1)
    )
  ]

  if (length(missing_markers) > 0L) {
    stop(
      basename(path),
      " is missing marker(s): ",
      paste(
        missing_markers,
        collapse = ", "
      )
    )
  }
}

cat(
  "\nPASS: 9/9 publication LaTeX tables were rendered and audited.\n"
)
