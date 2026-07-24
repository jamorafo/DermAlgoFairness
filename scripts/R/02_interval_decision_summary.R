# -------------------------------------------------------------------------
# Publication-ready R alternative to the Python TAC/ETC consistency figure.
#
# Input:
#   outputs/publication_tables/table_04_interval_tac_etc_consistency.csv
#
# Outputs:
#   outputs/figures-r/figure_interval_tac_etc_consistency.pdf
#   outputs/figures-r/figure_interval_tac_etc_consistency.png
#
# Plotting only: no bootstrap estimation is repeated.
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(forcats)
  library(scales)
})

source(file.path("scripts", "R", "figure_theme.R"))

input_file <- file.path(
  "outputs",
  "publication_tables",
  "table_04_interval_tac_etc_consistency.csv"
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

decision_data <- read_csv(
  input_file,
  show_col_types = FALSE
)

required_columns <- c(
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

missing_columns <- setdiff(required_columns, names(decision_data))

if (length(missing_columns) > 0L) {
  stop(
    "The input file is missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

metric_levels <- c(
  "Precision",
  "F1-score",
  "AUC-PR",
  "Recall / sensitivity"
)

target_levels <- c(
  "Overall",
  "Light phototype",
  "Dark phototype"
)

target_labels <- c(
  "Overall" = "Overall: TAC + ETC",
  "Light phototype" = "Light phototype: TAC + benchmark preservation",
  "Dark phototype" = "Dark phototype: TAC + benchmark preservation"
)

evidence_levels <- c(
  "Joint claim supported",
  "Inconclusive",
  "Joint claim not supported"
)

evidence_colours <- c(
  "Joint claim supported" = "#0072B2",
  "Inconclusive" = "#737373",
  "Joint claim not supported" = "#D55E00"
)

plot_data <- decision_data |>
  filter(`Metric group` == "Primary") |>
  transmute(
    metric = factor(
      Metric,
      levels = metric_levels
    ),
    target = factor(
      `Target condition`,
      levels = target_levels
    ),
    target_label = factor(
      unname(target_labels[as.character(`Target condition`)]),
      levels = unname(target_labels[target_levels])
    ),
    `Joint claim supported` = `Adequate and preserved`,
    Inconclusive = Inconclusive + `Evidentially unresolved`,
    `Joint claim not supported` =
      `Not adequate and not preserved` +
      `Preserved but inadequate` +
      `Adequate but not preserved`,
    expected_total = `Total model--seed decisions`
  ) |>
  pivot_longer(
    cols = all_of(evidence_levels),
    names_to = "evidence",
    values_to = "count"
  ) |>
  mutate(
    evidence = factor(
      evidence,
      levels = evidence_levels
    )
  )

validation <- plot_data |>
  group_by(metric, target_label) |>
  summarise(
    observed_total = sum(count),
    expected_total = first(expected_total),
    .groups = "drop"
  )

if (any(validation$observed_total != validation$expected_total)) {
  stop(
    "At least one metric-target row does not sum to the expected total."
  )
}

if (any(validation$expected_total != 25L)) {
  warning(
    "At least one row has a total other than 25 locked systems."
  )
}

figure <- ggplot(
  plot_data,
  aes(
    x = count,
    y = metric,
    fill = evidence
  )
) +
  geom_col(
    width = 0.66,
    colour = "white",
    size = 0.45
  ) +
  geom_text(
    aes(
      label = ifelse(count > 0, count, "")
    ),
    position = position_stack(vjust = 0.5),
    colour = "white",
    fontface = "bold",
    size = 3.5
  ) +
  facet_wrap(
    vars(target_label),
    ncol = 1
  ) +
  scale_fill_manual(
    values = evidence_colours,
    breaks = evidence_levels,
    drop = TRUE
  ) +
  scale_x_continuous(
    breaks = seq(0, 25, by = 5),
    limits = c(0, 25),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = "Locked systems (out of 25)",
    y = NULL,
    fill = NULL
  ) +
  theme_publication(base_size = 10.5) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    legend.box.just = "left",
    axis.text.y = element_text(
      face = "bold",
      colour = "grey15"
    ),
    strip.text = element_text(
      face = "bold",
      hjust = 0,
      size = 10.8,
      margin = margin(b = 5)
    ),
    panel.spacing.y = grid::unit(0.75, "lines"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(8, 12, 8, 8)
  )

pdf_device <- if (capabilities("cairo")) {
  grDevices::cairo_pdf
} else {
  grDevices::pdf
}

pdf_path <- file.path(
  output_dir,
  "figure_interval_tac_etc_consistency.pdf"
)

png_path <- file.path(
  output_dir,
  "figure_interval_tac_etc_consistency.png"
)

ggsave(
  filename = pdf_path,
  plot = figure,
  width = 7.2,
  height = 6.8,
  units = "in",
  device = pdf_device,
  bg = "white"
)

ggsave(
  filename = png_path,
  plot = figure,
  width = 7.2,
  height = 6.8,
  units = "in",
  dpi = 400,
  bg = "white"
)

message("Saved: ", pdf_path)
message("Saved: ", png_path)
message("Completed the R decision-summary figure.")
