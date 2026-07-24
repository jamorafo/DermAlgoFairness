# -------------------------------------------------------------------------
# Architecture-level multiplicity-adjusted subgroup sensitivity figure.
#
# Input:
#   outputs/publication_tables/
#     table_09_architecture_level_bh_sensitivity.csv
#
# Outputs:
#   figure_architecture_level_bh_sensitivity.pdf
#   figure_architecture_level_bh_sensitivity.png
#
# Plotting only. This script reads finalized results and does not repeat
# bootstrap estimation or multiplicity adjustment.
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

input_file <- file.path(
  "outputs",
  "publication_tables",
  "table_09_architecture_level_bh_sensitivity.csv"
)

output_dir <- Sys.getenv(
  "DERMALGO_FIGURE_OUTPUT_DIR",
  unset = file.path(
    "outputs",
    "figures-r"
  )
)

if (!file.exists(input_file)) {
  stop(
    "Finalized Table 09 CSV was not found: ",
    input_file
  )
}

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

raw_data <- read_csv(
  input_file,
  show_col_types = FALSE,
  progress = FALSE
)

required_columns <- c(
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

missing_columns <- setdiff(
  required_columns,
  names(raw_data)
)

if (length(missing_columns) > 0L) {
  stop(
    "Table 09 is missing required columns: ",
    paste(
      missing_columns,
      collapse = ", "
    )
  )
}

if (nrow(raw_data) != 20L) {
  stop(
    "Expected 20 architecture--metric rows, found ",
    nrow(raw_data),
    "."
  )
}

if (
  any(raw_data$n_seeds != 5L) ||
  any(raw_data$n_boot_valid != 10000L)
) {
  stop(
    "Table 09 must document five training runs and ",
    "10,000 bootstrap replicates per row."
  )
}

if (
  anyDuplicated(
    raw_data[
      c(
        "Metric",
        "Architecture"
      )
    ]
  )
) {
  stop(
    "Duplicate architecture--metric rows were found."
  )
}

if (
  any(!is.finite(raw_data$observed_mean_gap)) ||
  any(!is.finite(raw_data$bootstrap_ci_low)) ||
  any(!is.finite(raw_data$bootstrap_ci_high)) ||
  any(!is.finite(raw_data$p_value_bh))
) {
  stop(
    "Table 09 contains missing or non-finite plotted values."
  )
}

if (
  any(
    raw_data$bootstrap_ci_low >
      raw_data$observed_mean_gap
  ) ||
  any(
    raw_data$bootstrap_ci_high <
      raw_data$observed_mean_gap
  )
) {
  stop(
    "At least one Table 09 interval does not contain its point estimate."
  )
}

bh_flag <- tolower(
  trimws(
    as.character(
      raw_data$bh_significant
    )
  )
) %in% c(
  "yes",
  "true",
  "1"
)

if (
  any(
    bh_flag !=
      (
        raw_data$p_value_bh < 0.05
      )
  )
) {
  stop(
    "BH significance labels are inconsistent with adjusted p-values."
  )
}

metric_levels <- c(
  "Recall / sensitivity",
  "AUC-PR",
  "F1-score",
  "Precision"
)

architecture_levels <- rev(
  c(
    "ResNet50",
    "DenseNet121",
    "MobileNetV2",
    "EfficientNetV2B0",
    "VGG16"
  )
)

plot_data <- raw_data |>
  transmute(
    Metric,
    Architecture,
    mean_gap = observed_mean_gap,
    ci_low = bootstrap_ci_low,
    ci_high = bootstrap_ci_high,
    p_bh = p_value_bh,
    BH_status = if_else(
      bh_flag,
      "Retained after BH adjustment",
      "Not retained after BH adjustment"
    )
  ) |>
  mutate(
    Metric = factor(
      Metric,
      levels = metric_levels
    ),
    Architecture = factor(
      Architecture,
      levels = architecture_levels
    ),
    BH_status = factor(
      BH_status,
      levels = c(
        "Retained after BH adjustment",
        "Not retained after BH adjustment"
      )
    )
  )

if (
  anyNA(plot_data$Metric) ||
  anyNA(plot_data$Architecture)
) {
  stop(
    "Unexpected metric or architecture labels were found."
  )
}

status_colours <- c(
  "Retained after BH adjustment" = "#D55E00",
  "Not retained after BH adjustment" = "#6B7280"
)

observed_range <- range(
  c(
    plot_data$ci_low,
    plot_data$ci_high,
    0
  ),
  finite = TRUE
)

padding <- max(
  0.03,
  diff(observed_range) * 0.05
)

x_limits <- c(
  observed_range[[1]] - padding,
  observed_range[[2]] + padding
)

figure <- ggplot(
  plot_data,
  aes(
    x = mean_gap,
    y = Architecture,
    colour = BH_status
  )
) +
  geom_vline(
    xintercept = 0,
    size = 0.55,
    linetype = "dashed",
    colour = "grey25"
  ) +
  geom_segment(
    aes(
      x = ci_low,
      xend = ci_high,
      yend = Architecture
    ),
    size = 0.8
  ) +
  geom_point(
    size = 2.8,
    stroke = 0
  ) +
  facet_wrap(
    vars(Metric),
    ncol = 2
  ) +
  scale_colour_manual(
    values = status_colours,
    drop = TRUE
  ) +
  scale_x_continuous(
    breaks = pretty_breaks(
      n = 6
    ),
    labels = label_number(
      accuracy = 0.01,
      trim = TRUE
    )
  ) +
  coord_cartesian(
    xlim = x_limits,
    clip = "off"
  ) +
  labs(
    x = "Mean performance difference (light - dark)",
    y = NULL,
    colour = NULL
  ) +
  theme_publication() +
  theme(
    legend.position = "top",
    strip.text = element_text(
      face = "bold"
    ),
    panel.spacing = grid::unit(
      1.1,
      "lines"
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
  "figure_architecture_level_bh_sensitivity.pdf"
)

png_path <- file.path(
  output_dir,
  "figure_architecture_level_bh_sensitivity.png"
)

ggsave(
  filename = pdf_path,
  plot = figure,
  width = 7.2,
  height = 6.6,
  units = "in",
  device = pdf_device,
  bg = "white"
)

ggsave(
  filename = png_path,
  plot = figure,
  width = 7.2,
  height = 6.6,
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
      "Missing or empty Figure 04 output: ",
      path
    )
  }
}

message("Saved: ", pdf_path)
message("Saved: ", png_path)
message(
  "Completed the architecture-level BH sensitivity figure."
)
