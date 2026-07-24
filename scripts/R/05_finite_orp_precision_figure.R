# -------------------------------------------------------------------------
# Hybrid finite-ORP precision figure for the four primary metrics.
#
# Individual points show interval half-width for each of the 25 locked
# architecture--run systems. Boxplots summarize the distribution, while
# solid and open markers show the mean and maximum half-width.
#
# Inputs:
#   outputs/tables/interval_tac_etc_by_seed.csv
#   outputs/publication_tables/table_07_source_pr_diagnostics.csv
#   outputs/publication_tables/table_08_target_pr_diagnostics.csv
#
# Outputs:
#   figure_finite_orp_precision_primary_metrics.pdf
#   figure_finite_orp_precision_primary_metrics.png
#
# Plotting only. This script does not repeat bootstrap estimation and does
# not impose an additional post hoc precision threshold.
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
})

source(
  file.path(
    "scripts",
    "R",
    "figure_theme.R"
  )
)

interval_file <- file.path(
  "outputs",
  "tables",
  "interval_tac_etc_by_seed.csv"
)

source_contract_file <- file.path(
  "outputs",
  "publication_tables",
  "table_07_source_pr_diagnostics.csv"
)

target_contract_file <- file.path(
  "outputs",
  "publication_tables",
  "table_08_target_pr_diagnostics.csv"
)

output_dir <- Sys.getenv(
  "DERMALGO_FIGURE_OUTPUT_DIR",
  unset = file.path(
    "outputs",
    "figures-r"
  )
)

required_files <- c(
  interval_file,
  source_contract_file,
  target_contract_file
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0L) {
  stop(
    "Missing finalized input file(s):\n",
    paste0(
      "  - ",
      missing_files,
      collapse = "\n"
    )
  )
}

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

interval_data <- read_csv(
  interval_file,
  show_col_types = FALSE,
  progress = FALSE
)

source_contract <- read_csv(
  source_contract_file,
  show_col_types = FALSE,
  progress = FALSE
)

target_contract <- read_csv(
  target_contract_file,
  show_col_types = FALSE,
  progress = FALSE
)

###############################################################################
# Validate finalized per-system interval input
###############################################################################

required_interval_columns <- c(
  "model",
  "seed",
  "metric",
  "metric_group",
  "target_condition",
  "n_source",
  "n_source_clusters",
  "n_target",
  "source_bootstrap_unit",
  "target_bootstrap_unit",
  "n_boot_requested",
  "n_boot_valid",
  "source_ci_low",
  "source_ci_high",
  "target_ci_low",
  "target_ci_high"
)

missing_interval_columns <- setdiff(
  required_interval_columns,
  names(interval_data)
)

if (length(missing_interval_columns) > 0L) {
  stop(
    "The finalized interval file is missing columns: ",
    paste(
      missing_interval_columns,
      collapse = ", "
    )
  )
}

primary_metrics <- c(
  "recall",
  "auc_pr",
  "f1",
  "precision"
)

metric_labels <- c(
  recall = "Sensitivity",
  auc_pr = "AUC-PR",
  f1 = "F1-score",
  precision = "Precision"
)

target_conditions <- c(
  "BOSQUE overall",
  "BOSQUE light",
  "BOSQUE dark"
)

analysis_data <- interval_data |>
  filter(
    metric_group == "Primary",
    metric %in% primary_metrics,
    target_condition %in% target_conditions
  )

expected_rows <- (
  5L *
  5L *
  length(primary_metrics) *
  length(target_conditions)
)

if (nrow(analysis_data) != expected_rows) {
  stop(
    "Expected ",
    expected_rows,
    " finalized model--run--metric--target rows, found ",
    nrow(analysis_data),
    "."
  )
}

if (
  any(analysis_data$n_source != 990L) ||
  any(analysis_data$n_source_clusters != 747L)
) {
  stop(
    "The source ORP must contain 990 images from 747 lesions."
  )
}

expected_target_sizes <- c(
  "BOSQUE overall" = 151L,
  "BOSQUE light" = 105L,
  "BOSQUE dark" = 46L
)

for (condition in names(expected_target_sizes)) {
  observed <- unique(
    analysis_data$n_target[
      analysis_data$target_condition == condition
    ]
  )

  if (
    length(observed) != 1L ||
    as.integer(observed[[1]]) !=
      expected_target_sizes[[condition]]
  ) {
    stop(
      "Unexpected target size for ",
      condition,
      "."
    )
  }
}

if (
  any(analysis_data$n_boot_requested != 10000L) ||
  any(analysis_data$n_boot_valid != 10000L)
) {
  stop(
    "All plotted intervals must use 10,000 valid bootstrap replicates."
  )
}

if (
  any(analysis_data$source_bootstrap_unit != "lesion") ||
  any(
    analysis_data$target_bootstrap_unit !=
      "image_or_lesion"
  )
) {
  stop(
    "Unexpected source or target bootstrap unit."
  )
}

configured_training_seeds <- c(
  347535239,
  1431725881,
  3368960919,
  4236551057,
  780568470
)

observed_training_seeds <- sort(
  unique(
    analysis_data$seed
  )
)

if (
  !identical(
    observed_training_seeds,
    sort(configured_training_seeds)
  )
) {
  stop(
    "The plotted systems do not match the configured training seeds."
  )
}

###############################################################################
# Select the canonical per-system source interval
#
# The interval analysis repeated the source bootstrap within each target
# condition, producing small Monte Carlo differences among three intervals
# referring to the same locked source system. Canonical Table 07 was generated
# from the source intervals attached to the BOSQUE-overall rows. We apply the
# same reduction rule here so that the distributional figure reproduces the
# published source summaries exactly.
###############################################################################

source_points <- analysis_data |>
  filter(
    target_condition == "BOSQUE overall"
  ) |>
  transmute(
    model,
    seed,
    metric,
    condition = "HAM10000 held-out",
    half_width = (
      source_ci_high -
        source_ci_low
    ) / 2
  )

if (
  nrow(source_points) != 100L ||
  anyDuplicated(
    source_points[
      c(
        "model",
        "seed",
        "metric"
      )
    ]
  )
) {
  stop(
    "The canonical BOSQUE-overall reduction must yield ",
    "100 unique source system--metric rows."
  )
}

target_points <- analysis_data |>
  transmute(
    model,
    seed,
    metric,
    condition = recode(
      as.character(
        target_condition
      ),
      "BOSQUE overall" = "BOSQUE overall",
      "BOSQUE light" = "BOSQUE light",
      "BOSQUE dark" = "BOSQUE dark"
    ),
    half_width = (
      target_ci_high -
        target_ci_low
    ) / 2
  )

if (nrow(source_points) != 100L) {
  stop(
    "Expected 100 source system--metric rows, found ",
    nrow(source_points),
    "."
  )
}

if (nrow(target_points) != 300L) {
  stop(
    "Expected 300 target system--metric rows, found ",
    nrow(target_points),
    "."
  )
}

plot_data <- bind_rows(
  source_points,
  target_points
)

if (
  nrow(plot_data) != 400L ||
  any(!is.finite(plot_data$half_width)) ||
  any(plot_data$half_width < 0)
) {
  stop(
    "Invalid hybrid precision plotting data."
  )
}

###############################################################################
# Aggregate mean and maximum and cross-check Tables 07 and 08
###############################################################################

summary_data <- plot_data |>
  group_by(
    metric,
    condition
  ) |>
  summarise(
    n_systems = n(),
    mean_half_width = mean(
      half_width
    ),
    max_half_width = max(
      half_width
    ),
    .groups = "drop"
  )

if (
  nrow(summary_data) != 16L ||
  any(summary_data$n_systems != 25L)
) {
  stop(
    "Each metric--ORP combination must contain 25 systems."
  )
}

required_source_contract <- c(
  "Metric",
  "n_model_seed",
  "n_source_images",
  "n_source_lesions",
  "mean_half_width",
  "max_half_width"
)

required_target_contract <- c(
  "Metric",
  "Target condition",
  "n_model_seed",
  "n_target",
  "mean_half_width",
  "max_half_width"
)

if (
  length(
    setdiff(
      required_source_contract,
      names(source_contract)
    )
  ) > 0L ||
  length(
    setdiff(
      required_target_contract,
      names(target_contract)
    )
  ) > 0L
) {
  stop(
    "Tables 07 or 08 do not contain the expected contract columns."
  )
}

source_contract_comparison <- source_contract |>
  transmute(
    metric = recode(
      as.character(
        Metric
      ),
      "Recall / sensitivity" = "recall",
      "AUC-PR" = "auc_pr",
      "F1-score" = "f1",
      "Precision" = "precision",
      .default = NA_character_
    ),
    condition = "HAM10000 held-out",
    contract_mean = mean_half_width,
    contract_max = max_half_width
  ) |>
  filter(
    metric %in% primary_metrics
  )

target_contract_comparison <- target_contract |>
  transmute(
    metric = recode(
      as.character(
        Metric
      ),
      "Recall / sensitivity" = "recall",
      "AUC-PR" = "auc_pr",
      "F1-score" = "f1",
      "Precision" = "precision",
      .default = NA_character_
    ),
    condition = recode(
      as.character(
        `Target condition`
      ),
      "Overall" = "BOSQUE overall",
      "Light phototype" = "BOSQUE light",
      "Dark phototype" = "BOSQUE dark",
      .default = NA_character_
    ),
    contract_mean = mean_half_width,
    contract_max = max_half_width
  ) |>
  filter(
    metric %in% primary_metrics,
    condition %in% c(
      "BOSQUE overall",
      "BOSQUE light",
      "BOSQUE dark"
    )
  )

contract_comparison <- summary_data |>
  left_join(
    bind_rows(
      source_contract_comparison,
      target_contract_comparison
    ),
    by = c(
      "metric",
      "condition"
    )
  )

if (
  anyNA(
    contract_comparison$contract_mean
  ) ||
  anyNA(
    contract_comparison$contract_max
  )
) {
  stop(
    "Could not match every hybrid summary to Tables 07 and 08."
  )
}

comparison_tolerance <- 1e-10

if (
  any(
    abs(
      contract_comparison$mean_half_width -
        contract_comparison$contract_mean
    ) > comparison_tolerance
  ) ||
  any(
    abs(
      contract_comparison$max_half_width -
        contract_comparison$contract_max
    ) > comparison_tolerance
  )
) {
  stop(
    "Hybrid half-width summaries do not reproduce Tables 07 and 08."
  )
}

###############################################################################
# Publication labels and ordering
###############################################################################

condition_order <- c(
  "HAM10000 held-out",
  "BOSQUE overall",
  "BOSQUE light",
  "BOSQUE dark"
)

condition_labels <- c(
  "HAM10000 held-out" =
    "HAM10000 held-out\n(990 images; 747 lesions)",
  "BOSQUE overall" =
    "BOSQUE overall\n(n = 151)",
  "BOSQUE light" =
    "BOSQUE light\n(n = 105)",
  "BOSQUE dark" =
    "BOSQUE dark\n(n = 46)"
)

plot_data <- plot_data |>
  mutate(
    metric = factor(
      metric,
      levels = primary_metrics,
      labels = unname(
        metric_labels[
          primary_metrics
        ]
      )
    ),
    condition = factor(
      condition,
      levels = rev(
        condition_order
      ),
      labels = unname(
        condition_labels[
          rev(
            condition_order
          )
        ]
      )
    )
  )

summary_data <- summary_data |>
  mutate(
    metric = factor(
      metric,
      levels = primary_metrics,
      labels = unname(
        metric_labels[
          primary_metrics
        ]
      )
    ),
    condition = factor(
      condition,
      levels = rev(
        condition_order
      ),
      labels = unname(
        condition_labels[
          rev(
            condition_order
          )
        ]
      )
    )
  )

summary_points <- bind_rows(
  summary_data |>
    transmute(
      metric,
      condition,
      summary_type = "Mean half-width",
      half_width = mean_half_width
    ),
  summary_data |>
    transmute(
      metric,
      condition,
      summary_type = "Maximum half-width",
      half_width = max_half_width
    )
) |>
  mutate(
    summary_type = factor(
      summary_type,
      levels = c(
        "Mean half-width",
        "Maximum half-width"
      )
    )
  )

###############################################################################
# Hybrid figure
###############################################################################

x_upper <- max(
  0.25,
  ceiling(
    max(
      plot_data$half_width,
      na.rm = TRUE
    ) * 20
  ) / 20
)

figure <- ggplot(
  plot_data,
  aes(
    x = half_width,
    y = condition
  )
) +
  geom_boxplot(
    width = 0.48,
    outlier.shape = NA,
    size = 0.5,
    fill = "white",
    colour = "grey70"
  ) +
  geom_point(
    position = position_jitter(
      width = 0,
      height = 0.10,
      seed = 1517787898
    ),
    size = 1.55,
    alpha = 0.68,
    colour = "grey45"
  ) +
  geom_point(
    data = summary_points,
    aes(
      x = half_width,
      y = condition,
      shape = summary_type
    ),
    inherit.aes = FALSE,
    size = 3.0,
    stroke = 1.0,
    colour = "grey10"
  ) +
  facet_wrap(
    vars(metric),
    ncol = 2
  ) +
  scale_shape_manual(
    values = c(
      "Mean half-width" = 16,
      "Maximum half-width" = 1
    ),
    drop = FALSE
  ) +
  scale_x_continuous(
    breaks = pretty_breaks(
      n = 4
    ),
    labels = label_number(
      accuracy = 0.01,
      trim = TRUE
    ),
    expand = expansion(
      mult = c(
        0.015,
        0.035
      )
    )
  ) +
  coord_cartesian(
    xlim = c(
      0,
      x_upper + 0.01
    ),
    clip = "off"
  ) +
  labs(
    x = "95% percentile-bootstrap interval half-width",
    y = NULL,
    shape = NULL
  ) +
  theme_publication(
    base_size = 10.5
  ) +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    panel.spacing.x = grid::unit(
      1.6,
      "lines"
    ),
    panel.spacing.y = grid::unit(
      1.1,
      "lines"
    ),
    strip.text = element_text(
      face = "bold"
    ),
    axis.text.x = element_text(
      size = 8.8
    ),
    axis.text.y = element_text(
      size = 9.1
    )
  )

pdf_device <- if (
  capabilities("cairo")
) {
  grDevices::cairo_pdf
} else {
  grDevices::pdf
}

pdf_path <- file.path(
  output_dir,
  "figure_finite_orp_precision_primary_metrics.pdf"
)

png_path <- file.path(
  output_dir,
  "figure_finite_orp_precision_primary_metrics.png"
)

ggsave(
  filename = pdf_path,
  plot = figure,
  width = 8.4,
  height = 6.4,
  units = "in",
  device = pdf_device,
  bg = "white"
)

ggsave(
  filename = png_path,
  plot = figure,
  width = 8.4,
  height = 6.4,
  units = "in",
  dpi = 400,
  bg = "white"
)

for (path in c(pdf_path, png_path)) {
  if (
    !file.exists(path) ||
    file.info(path)$size <= 0
  ) {
    stop(
      "Missing or empty hybrid Figure 05 output: ",
      path
    )
  }
}

message("Saved: ", pdf_path)
message("Saved: ", png_path)
message(
  "Completed the hybrid finite-ORP precision figure."
)
