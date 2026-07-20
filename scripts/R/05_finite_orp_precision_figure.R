# -------------------------------------------------------------------------
# Finite-ORP precision figure for the four primary performance metrics.
#
# Inputs:
#   outputs/tables/source_pr_diagnostics_by_model_seed.csv
#   outputs/tables/target_pr_diagnostics_by_model_seed.csv
#
# Output:
#   outputs/figures-r/figure_finite_orp_precision_primary_metrics.pdf
#   outputs/figures-r/figure_finite_orp_precision_primary_metrics.png
#
# This script performs plotting only. It does not rerun bootstrap estimation.
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(forcats)
})

source(file.path("scripts", "R", "figure_theme.R"))

source_file <- file.path(
  "outputs", "tables", "source_pr_diagnostics_by_model_seed.csv"
)

target_file <- file.path(
  "outputs", "tables", "target_pr_diagnostics_by_model_seed.csv"
)

output_dir <- file.path("outputs", "figures-r")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

missing_files <- c(source_file, target_file)[
  !file.exists(c(source_file, target_file))
]

if (length(missing_files) > 0L) {
  stop(
    "Missing required input file(s):\n",
    paste0("  - ", missing_files, collapse = "\n"),
    "\nThe finalized diagnostic CSVs must be present locally."
  )
}

source_data <- read_csv(source_file, show_col_types = FALSE)
target_data <- read_csv(target_file, show_col_types = FALSE)

source_required <- c(
  "model", "seed", "metric", "n_source",
  "source_half_width", "source_half_width_tolerance"
)

target_required <- c(
  "model", "seed", "metric", "target_condition", "n_target",
  "target_half_width", "target_half_width_tolerance"
)

missing_source <- setdiff(source_required, names(source_data))
missing_target <- setdiff(target_required, names(target_data))

if (length(missing_source) > 0L) {
  stop(
    "Source diagnostic CSV is missing columns: ",
    paste(missing_source, collapse = ", ")
  )
}

if (length(missing_target) > 0L) {
  stop(
    "Target diagnostic CSV is missing columns: ",
    paste(missing_target, collapse = ", ")
  )
}

primary_metrics <- c("recall", "auc_pr", "f1", "precision")

metric_labels <- c(
  recall = "Sensitivity",
  auc_pr = "AUC-PR",
  f1 = "F1-score",
  precision = "Precision"
)

condition_levels <- c(
  "HAM10000 held-out",
  "BOSQUE overall",
  "BOSQUE light",
  "BOSQUE dark"
)

condition_labels <- c(
  "HAM10000 held-out" = "HAM10000 held-out\n(n = 1002)",
  "BOSQUE overall" = "BOSQUE overall\n(n = 151)",
  "BOSQUE light" = "BOSQUE light\n(n = 105)",
  "BOSQUE dark" = "BOSQUE dark\n(n = 46)"
)

source_plot <- source_data |>
  filter(metric %in% primary_metrics) |>
  transmute(
    model,
    seed,
    metric,
    condition = "HAM10000 held-out",
    n_orp = n_source,
    half_width = source_half_width,
    tolerance = source_half_width_tolerance
  )

target_plot <- target_data |>
  filter(metric %in% primary_metrics) |>
  transmute(
    model,
    seed,
    metric,
    condition = recode(
      as.character(target_condition),
      "BOSQUE overall" = "BOSQUE overall",
      "BOSQUE light" = "BOSQUE light",
      "BOSQUE dark" = "BOSQUE dark"
    ),
    n_orp = n_target,
    half_width = target_half_width,
    tolerance = target_half_width_tolerance
  )

plot_data <- bind_rows(source_plot, target_plot) |>
  mutate(
    metric = factor(
      metric,
      levels = primary_metrics,
      labels = unname(metric_labels[primary_metrics])
    ),
    condition = factor(
      condition,
      levels = condition_levels,
      labels = unname(condition_labels[condition_levels])
    ),
    precision_ratio = half_width / tolerance,
    tolerance_status = if_else(
      precision_ratio <= 1,
      "Within tolerance",
      "Exceeds tolerance"
    ),
    tolerance_status = factor(
      tolerance_status,
      levels = c("Within tolerance", "Exceeds tolerance")
    )
  )

expected_rows <- 4L * 4L * 25L

if (nrow(plot_data) != expected_rows) {
  stop(
    "Expected ", expected_rows,
    " primary-metric ORP rows, found ", nrow(plot_data), "."
  )
}

status_colours <- c(
  "Within tolerance" = "#4D4D4D",
  "Exceeds tolerance" = "#D55E00"
)

figure <- ggplot(
  plot_data,
  aes(
    x = precision_ratio,
    y = condition
  )
) +
  geom_vline(
    xintercept = 1,
    linetype = "dashed",
    linewidth = 0.6,
    colour = "grey25"
  ) +
  geom_boxplot(
    width = 0.48,
    outlier.shape = NA,
    linewidth = 0.55,
    fill = "white",
    colour = "grey55"
  ) +
  geom_jitter(
    aes(colour = tolerance_status),
    width = 0,
    height = 0.11,
    size = 1.75,
    alpha = 0.85
  ) +
  facet_wrap(
    vars(metric),
    ncol = 2
  ) +
  scale_colour_manual(
    values = status_colours,
    drop = FALSE
  ) +
  scale_x_continuous(
    breaks = seq(0, 2.5, by = 0.5),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    x = "95% CI half-width / documentation tolerance",
    y = NULL,
    colour = NULL
  ) +
  coord_cartesian(
    xlim = c(0, max(2.5, max(plot_data$precision_ratio) * 1.04)),
    clip = "off"
  ) +
  theme_publication(base_size = 10.5) +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 9.3)
  )

pdf_device <- if (capabilities("cairo")) {
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
  width = 7.4,
  height = 6.2,
  units = "in",
  device = pdf_device,
  bg = "white"
)

ggsave(
  filename = png_path,
  plot = figure,
  width = 7.4,
  height = 6.2,
  units = "in",
  dpi = 400,
  bg = "white"
)

if (!file.exists(pdf_path) || !file.exists(png_path)) {
  stop("One or more expected figure files were not created.")
}

message("Saved: ", pdf_path)
message("Saved: ", png_path)
message("Completed the finite-ORP precision figure.")
