# -------------------------------------------------------------------------
# Publication-ready R alternatives to the seed-aware BOSQUE light-dark
# performance-gap figures.
#
# Input:
#   outputs/tables/bosque_light_dark_gap_by_model_seed.csv
#
# Outputs:
#   outputs/figures-r/figure_bosque_<metric>_gap_seed_aware.pdf
#   outputs/figures-r/figure_bosque_<metric>_gap_seed_aware.png
#
# Plotting only: this script does not rerun model training or bootstrapping.
# Figures intentionally contain no title, subtitle, or caption.
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
  "bosque_light_dark_gap_by_model_seed.csv"
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
  "seed",
  "metric",
  "gap_light_minus_dark",
  "bootstrap_ci_low",
  "bootstrap_ci_high",
  "n_boot_valid"
)

missing_columns <- setdiff(required_columns, names(gap_data))

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

seed_config_file <- file.path(
  "config",
  "random_seeds.json"
)

if (!file.exists(seed_config_file)) {
  stop(
    "Seed configuration was not found: ",
    seed_config_file
  )
}

seed_config_lines <- readLines(
  seed_config_file,
  warn = FALSE
)

training_seed_lines <- grep(
  '"training_run_[1-5]"[[:space:]]*:',
  seed_config_lines,
  value = TRUE
)

if (length(training_seed_lines) != 5L) {
  stop(
    "Expected five configured training-run values, found ",
    length(training_seed_lines),
    "."
  )
}

training_run_numbers <- as.integer(
  sub(
    '.*"training_run_([1-5])".*',
    "\\1",
    training_seed_lines
  )
)

training_seed_values <- sub(
  '.*:[[:space:]]*([0-9]+),?[[:space:]]*$',
  "\\1",
  training_seed_lines
)

ordering <- order(training_run_numbers)

training_run_numbers <- training_run_numbers[
  ordering
]

training_seed_values <- training_seed_values[
  ordering
]

if (
  !identical(
    training_run_numbers,
    1:5
  ) ||
  anyDuplicated(
    training_seed_values
  )
) {
  stop(
    "Configured training-run values are incomplete or duplicated."
  )
}

training_run_labels <- setNames(
  paste(
    "Run",
    training_run_numbers
  ),
  training_seed_values
)

run_levels <- paste(
  "Run",
  5:1
)

primary_data <- gap_data |>
  filter(metric_group == "Primary") |>
  mutate(
    model_label = factor(
      model_label,
      levels = architecture_levels
    ),
    seed_key = format(
      seed,
      scientific = FALSE,
      trim = TRUE
    ),
    seed_label = factor(
      unname(
        training_run_labels[
          seed_key
        ]
      ),
      levels = run_levels
    ),
    direction = case_when(
      bootstrap_ci_low > 0 ~ "95% CI above 0",
      bootstrap_ci_high < 0 ~ "95% CI below 0",
      TRUE ~ "95% CI includes 0"
    ),
    direction = factor(
      direction,
      levels = c(
        "95% CI above 0",
        "95% CI includes 0",
        "95% CI below 0"
      )
    )
  )

if (
  anyNA(
    primary_data$seed_label
  ) ||
  dplyr::n_distinct(
    primary_data$seed_key
  ) != 5L
) {
  unknown_values <- sort(
    unique(
      primary_data$seed_key[
        is.na(
          primary_data$seed_label
        )
      ]
    )
  )

  stop(
    "Figure 01 could not map all training values to ",
    "Run 1--Run 5. Unmapped values: ",
    paste(
      unknown_values,
      collapse = ", "
    )
  )
}

metric_specs <- list(
  recall = list(
    filename = "figure_bosque_recall_gap_seed_aware",
    x_label = "Sensitivity difference (light - dark)"
  ),
  auc_pr = list(
    filename = "figure_bosque_auc_pr_gap_seed_aware",
    x_label = "AUC-PR difference (light - dark)"
  ),
  f1 = list(
    filename = "figure_bosque_f1_gap_seed_aware",
    x_label = "F1-score difference (light - dark)"
  ),
  precision = list(
    filename = "figure_bosque_precision_gap_seed_aware",
    x_label = "Precision difference (light - dark)"
  )
)

make_gap_plot <- function(metric_name, specification) {
  plot_data <- primary_data |>
    filter(metric == metric_name)

  if (nrow(plot_data) != 25L) {
    stop(
      "Expected 25 architecture-run rows for metric '",
      metric_name,
      "', but found ",
      nrow(plot_data),
      "."
    )
  }

  bootstrap_counts <- unique(plot_data$n_boot_valid)

  if (
    length(bootstrap_counts) != 1L ||
    bootstrap_counts != 10000L
  ) {
    warning(
      "Unexpected bootstrap count for metric '",
      metric_name,
      "': ",
      paste(bootstrap_counts, collapse = ", ")
    )
  }

  ggplot(
    plot_data,
    aes(
      x = gap_light_minus_dark,
      y = seed_label,
      colour = direction
    )
  ) +
    geom_vline(
      xintercept = 0,
      size = 0.55,
      linetype = "dashed",
      colour = "grey25"
    ) +
    geom_errorbar(
      aes(
        xmin = bootstrap_ci_low,
        xmax = bootstrap_ci_high
      ),
      orientation = "y",
      width = 0,
      size = 0.8
    ) +
    geom_point(
      size = 2.7,
      stroke = 0
    ) +
    facet_grid(
      rows = vars(model_label),
      scales = "free_y",
      space = "free_y",
      switch = "y"
    ) +
    scale_colour_manual(
      values = publication_colours,
      drop = TRUE
    ) +
    scale_x_continuous(
      breaks = scales::breaks_pretty(n = 6),
      labels = scales::label_number(
        accuracy = 0.01,
        trim = TRUE
      ),
      expand = expansion(mult = c(0.04, 0.05))
    ) +
    labs(
      x = specification$x_label,
      y = NULL,
      colour = NULL
    ) +
    guides(
      colour = guide_legend(
        override.aes = list(
          size = 3
        )
      )
    ) +
    coord_cartesian(
      clip = "off"
    ) +
    theme_publication()
}

pdf_device <- if (capabilities("cairo")) {
  grDevices::cairo_pdf
} else {
  grDevices::pdf
}

for (metric_name in names(metric_specs)) {
  specification <- metric_specs[[metric_name]]

  figure <- make_gap_plot(
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
    width = 7.2,
    height = 7.2,
    units = "in",
    device = pdf_device,
    bg = "white"
  )

  ggsave(
    filename = png_path,
    plot = figure,
    width = 7.2,
    height = 7.2,
    units = "in",
    dpi = 400,
    bg = "white"
  )

  message("Saved: ", pdf_path)
  message("Saved: ", png_path)
}

message("Completed all seed-aware R figures.")
