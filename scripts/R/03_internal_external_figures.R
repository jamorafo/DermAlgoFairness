# -------------------------------------------------------------------------
# R alternatives to the Python internal-versus-external performance figures.
#
# Input:
#   outputs/tables/interval_tac_etc_by_seed.csv
#
# Outputs:
#   outputs/figures-r/figure_internal_external_f1.pdf
#   outputs/figures-r/figure_internal_external_f1.png
#   outputs/figures-r/figure_internal_external_auc_pr.pdf
#   outputs/figures-r/figure_internal_external_auc_pr.png
#
# Plotting only: no model fitting and no bootstrap recomputation.
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
})

source(file.path("scripts", "R", "figure_theme.R"))

input_file <- file.path(
  "outputs",
  "tables",
  "interval_tac_etc_by_seed.csv"
)

output_dir <- file.path("outputs", "figures-r")

if (!file.exists(input_file)) {
  stop(
    "Input file was not found: ",
    input_file,
    "\nRun this script from the repository root."
  )
}

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

results <- read_csv(
  input_file,
  show_col_types = FALSE
)

required_columns <- c(
  "model",
  "seed",
  "metric",
  "metric_group",
  "target_condition",
  "source_performance",
  "target_performance",
  "tau"
)

missing_columns <- setdiff(required_columns, names(results))

if (length(missing_columns) > 0L) {
  stop(
    "The input file is missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

architecture_levels <- c(
  "ResNet50",
  "DenseNet121",
  "MobileNetV2",
  "EfficientNetV2B0",
  "VGG16"
)

architecture_positions <- c(
  "ResNet50" = 5,
  "DenseNet121" = 4,
  "MobileNetV2" = 3,
  "EfficientNetV2B0" = 2,
  "VGG16" = 1
)

condition_colours <- c(
  "HAM10000 source" = "#0072B2",
  "BOSQUE target" = "#D55E00"
)

condition_shapes <- c(
  "HAM10000 source" = 16,
  "BOSQUE target" = 17
)

metric_specs <- list(
  f1 = list(
    filename = "figure_internal_external_f1",
    x_label = "F1-score"
  ),
  auc_pr = list(
    filename = "figure_internal_external_auc_pr",
    x_label = "AUC-PR"
  )
)

make_source_target_plot <- function(metric_name, specification) {
  plot_data <- results |>
    filter(
      metric_group == "Primary",
      target_condition == "BOSQUE overall",
      metric == metric_name
    ) |>
    mutate(
      model = factor(model, levels = architecture_levels),
      architecture_position = unname(
        architecture_positions[as.character(model)]
      ),
      seed_offset = (as.numeric(seed) - 3) * 0.065,
      y_position = architecture_position + seed_offset
    )

  if (nrow(plot_data) != 25L) {
    stop(
      "Expected 25 architecture-seed rows for metric '",
      metric_name,
      "', but found ",
      nrow(plot_data),
      "."
    )
  }

  if (anyNA(plot_data$architecture_position)) {
    stop(
      "An unexpected architecture label was found for metric '",
      metric_name,
      "'."
    )
  }

  threshold_values <- unique(plot_data$tau)

  if (length(threshold_values) != 1L) {
    stop(
      "Expected one adequacy threshold for metric '",
      metric_name,
      "', but found: ",
      paste(threshold_values, collapse = ", ")
    )
  }

  point_data <- bind_rows(
    plot_data |>
      transmute(
        model,
        seed,
        y_position,
        condition = "HAM10000 source",
        performance = source_performance
      ),
    plot_data |>
      transmute(
        model,
        seed,
        y_position,
        condition = "BOSQUE target",
        performance = target_performance
      )
  ) |>
    mutate(
      condition = factor(
        condition,
        levels = c("HAM10000 source", "BOSQUE target")
      )
    )

  observed_minimum <- min(
    c(
      plot_data$source_performance,
      plot_data$target_performance,
      threshold_values
    ),
    na.rm = TRUE
  )

  lower_limit <- max(
    0,
    floor((observed_minimum - 0.06) * 10) / 10
  )

  ggplot() +
    geom_vline(
      xintercept = threshold_values,
      aes(linetype = "Target adequacy threshold"),
      linewidth = 0.65,
      colour = "grey25"
    ) +
    geom_segment(
      data = plot_data,
      aes(
        x = source_performance,
        xend = target_performance,
        y = y_position,
        yend = y_position
      ),
      linewidth = 0.75,
      colour = "grey72",
      lineend = "round"
    ) +
    geom_point(
      data = point_data,
      aes(
        x = performance,
        y = y_position,
        colour = condition,
        shape = condition
      ),
      size = 2.7,
      stroke = 0
    ) +
    scale_colour_manual(
      values = condition_colours,
      drop = FALSE
    ) +
    scale_shape_manual(
      values = condition_shapes,
      drop = FALSE
    ) +
    scale_linetype_manual(
      values = c("Target adequacy threshold" = "dashed")
    ) +
    scale_x_continuous(
      limits = c(lower_limit, 1),
      breaks = breaks_pretty(n = 6),
      labels = label_number(
        accuracy = 0.01,
        trim = TRUE
      ),
      expand = expansion(mult = c(0.015, 0.025))
    ) +
    scale_y_continuous(
      breaks = unname(architecture_positions),
      labels = names(architecture_positions),
      limits = c(0.65, 5.35),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    labs(
      x = specification$x_label,
      y = NULL,
      colour = NULL,
      shape = NULL,
      linetype = NULL
    ) +
    guides(
      colour = guide_legend(
        order = 1,
        override.aes = list(size = 3)
      ),
      shape = guide_legend(
        order = 1,
        override.aes = list(size = 3)
      ),
      linetype = guide_legend(order = 2)
    ) +
    theme_publication(base_size = 11) +
    theme(
      axis.text.y = element_text(
        face = "bold",
        colour = "grey15"
      ),
      legend.position = "top",
      legend.box = "horizontal",
      legend.spacing.x = grid::unit(0.35, "cm"),
      panel.grid.major.y = element_line(
        linewidth = 0.35,
        colour = "grey92"
      ),
      plot.margin = margin(7, 10, 7, 8)
    )
}

pdf_device <- if (capabilities("cairo")) {
  grDevices::cairo_pdf
} else {
  grDevices::pdf
}

for (metric_name in names(metric_specs)) {
  specification <- metric_specs[[metric_name]]

  figure <- make_source_target_plot(
    metric_name,
    specification
  )

  pdf_path <- file.path(
    output_dir,
    paste0(specification$filename, ".pdf")
  )

  png_path <- file.path(
    output_dir,
    paste0(specification$filename, ".png")
  )

  ggsave(
    filename = pdf_path,
    plot = figure,
    width = 6.7,
    height = 4.5,
    units = "in",
    device = pdf_device,
    bg = "white"
  )

  ggsave(
    filename = png_path,
    plot = figure,
    width = 6.7,
    height = 4.5,
    units = "in",
    dpi = 400,
    bg = "white"
  )

  message("Saved: ", pdf_path)
  message("Saved: ", png_path)
}

message("Completed the internal-versus-external R figures.")
