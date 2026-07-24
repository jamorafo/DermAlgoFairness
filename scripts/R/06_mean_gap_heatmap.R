# -------------------------------------------------------------------------
# Architecture-level mean BOSQUE light-dark performance-gap heatmap.
#
# Input:
#   outputs/tables/bosque_light_dark_gap_summary_by_architecture.csv
#
# Outputs:
#   outputs/figures-r/figure_bosque_light_dark_mean_gap_heatmap.pdf
#   outputs/figures-r/figure_bosque_light_dark_mean_gap_heatmap.png
#
# Plotting only: no model fitting or bootstrap computation is repeated.
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
  "bosque_light_dark_gap_summary_by_architecture.csv"
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

gap_data <- read_csv(
  input_file,
  show_col_types = FALSE
)

required_columns <- c(
  "metric_group",
  "model_label",
  "metric",
  "n_seeds",
  "mean_gap"
)

missing_columns <- setdiff(
  required_columns,
  names(gap_data)
)

if (length(missing_columns) > 0L) {
  stop(
    "The input file is missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

architecture_order <- c(
  "ResNet50",
  "DenseNet121",
  "MobileNetV2",
  "EfficientNetV2B0",
  "VGG16"
)

metric_order <- c(
  "recall",
  "auc_pr",
  "f1",
  "precision",
  "accuracy",
  "specificity",
  "auc_roc"
)

metric_labels <- c(
  recall = "Sensitivity",
  auc_pr = "AUC-PR",
  f1 = "F1-score",
  precision = "Precision",
  accuracy = "Accuracy",
  specificity = "Specificity",
  auc_roc = "AUC-ROC"
)

group_order <- c(
  "Primary",
  "Secondary"
)

plot_data <- gap_data |>
  filter(
    metric %in% metric_order,
    model_label %in% architecture_order,
    metric_group %in% group_order
  ) |>
  mutate(
    model_label = factor(
      model_label,
      levels = rev(architecture_order)
    ),
    metric = factor(
      metric,
      levels = metric_order,
      labels = unname(metric_labels[metric_order])
    ),
    metric_group = factor(
      metric_group,
      levels = group_order
    ),
    cell_label = sprintf(
      "%+.2f",
      mean_gap
    )
  )

expected_rows <- (
  length(architecture_order) *
  length(metric_order)
)

if (nrow(plot_data) != expected_rows) {
  stop(
    "Expected ",
    expected_rows,
    " architecture-metric rows, found ",
    nrow(plot_data),
    "."
  )
}

if (any(plot_data$n_seeds != 5L)) {
  stop(
    "At least one architecture-metric row does not contain five training runs."
  )
}

if (
  anyDuplicated(
    plot_data[c("model_label", "metric")]
  )
) {
  stop(
    "Duplicate architecture-metric rows were found."
  )
}

fill_limit <- max(
  0.05,
  max(abs(plot_data$mean_gap), na.rm = TRUE)
)

label_colour_threshold <- 0.62 * fill_limit

plot_data <- plot_data |>
  mutate(
    label_colour = if_else(
      abs(mean_gap) >= label_colour_threshold,
      "white",
      "black"
    )
  )

figure <- ggplot(
  plot_data,
  aes(
    x = metric,
    y = model_label,
    fill = mean_gap
  )
) +
  geom_tile(
    size = 0.7,
    colour = "white"
  ) +
  geom_text(
    aes(
      label = cell_label,
      colour = label_colour
    ),
    size = 4.2,
    fontface = "bold",
    show.legend = FALSE
  ) +
  facet_grid(
    cols = vars(metric_group),
    scales = "free_x",
    space = "free_x"
  ) +
  scale_fill_gradient2(
    low = "#D55E00",
    mid = "white",
    high = "#0072B2",
    midpoint = 0,
    limits = c(-fill_limit, fill_limit),
    breaks = pretty_breaks(n = 5),
    labels = label_number(
      accuracy = 0.01,
      trim = TRUE
    ),
    oob = squish,
    name = NULL
  ) +
  scale_colour_manual(
    values = c(
      black = "grey10",
      white = "white"
    )
  ) +
  scale_x_discrete(
    drop = TRUE,
    expand = c(0, 0)
  ) +
  scale_y_discrete(
    drop = TRUE,
    expand = c(0, 0)
  ) +
  guides(
    fill = guide_colourbar(
      direction = "horizontal",
      barwidth = grid::unit(4.2, "cm"),
      barheight = grid::unit(0.42, "cm"),
      ticks = TRUE,
      frame.colour = "grey55",
    )
  ) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_publication(
    base_size = 10.5
  ) +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(
      colour = "grey55",
      fill = NA,
      size = 0.45
    ),
    panel.spacing.x = grid::unit(0.18, "cm"),
    strip.background = element_rect(
      fill = "grey94",
      colour = "grey55",
      size = 0.45
    ),
    strip.text = element_text(
      face = "bold",
      size = 10.5,
      margin = margin(
        t = 6,
        r = 6,
        b = 6,
        l = 6
      )
    ),
    axis.text.x = element_text(
      angle = 32,
      hjust = 1,
      vjust = 1,
      face = "bold",
      size = 9.2
    ),
    axis.text.y = element_text(
      face = "bold",
      size = 9.6,
      margin = margin(r = 5)
    ),
    legend.position = "top",
    legend.justification = "center",
    legend.box.just = "center",
    legend.margin = margin(
      t = 0,
      r = 0,
      b = 7,
      l = 0
    ),
    plot.margin = margin(
      t = 6,
      r = 8,
      b = 6,
      l = 8
    )
  )

pdf_path <- file.path(
  output_dir,
  "figure_bosque_light_dark_mean_gap_heatmap.pdf"
)

png_path <- file.path(
  output_dir,
  "figure_bosque_light_dark_mean_gap_heatmap.png"
)

pdf_device <- if (capabilities("cairo")) {
  grDevices::cairo_pdf
} else {
  grDevices::pdf
}

ggsave(
  filename = pdf_path,
  plot = figure,
  width = 8.2,
  height = 5.0,
  units = "in",
  device = pdf_device,
  bg = "white"
)

ggsave(
  filename = png_path,
  plot = figure,
  width = 8.2,
  height = 5.0,
  units = "in",
  dpi = 400,
  bg = "white"
)

if (!file.exists(pdf_path)) {
  stop("The PDF output was not created.")
}

if (!file.exists(png_path)) {
  stop("The PNG output was not created.")
}

message("Saved: ", pdf_path)
message("Saved: ", png_path)
message("Completed the mean-gap heatmap.")
