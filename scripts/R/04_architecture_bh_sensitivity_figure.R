# -------------------------------------------------------------------------
# Architecture-level BH sensitivity figure.
# Plotting only: values are the finalized 10,000-replicate results.
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

source(file.path("scripts", "R", "figure_theme.R"))

output_dir <- file.path("outputs", "figures-r")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

csv_text <- paste(
  "Metric,Architecture,mean_gap,ci_low,ci_high,p_bh,bh_significant",
  "Recall / sensitivity,ResNet50,0.156,-0.062,0.367,0.2261,No",
  "Recall / sensitivity,DenseNet121,0.091,-0.117,0.295,0.4370,No",
  "Recall / sensitivity,MobileNetV2,-0.012,-0.228,0.208,0.9101,No",
  "Recall / sensitivity,EfficientNetV2B0,0.128,-0.085,0.346,0.3106,No",
  "Recall / sensitivity,VGG16,0.040,-0.135,0.218,0.6982,No",
  "AUC-PR,ResNet50,0.185,0.047,0.360,0.1467,No",
  "AUC-PR,DenseNet121,0.131,-0.006,0.295,0.2146,No",
  "AUC-PR,MobileNetV2,0.235,0.073,0.420,0.0970,No",
  "AUC-PR,EfficientNetV2B0,0.165,0.032,0.336,0.1725,No",
  "AUC-PR,VGG16,0.159,0.011,0.330,0.1888,No",
  "F1-score,ResNet50,0.173,-0.012,0.395,0.2146,No",
  "F1-score,DenseNet121,0.108,-0.061,0.306,0.3106,No",
  "F1-score,MobileNetV2,0.129,-0.037,0.325,0.2261,No",
  "F1-score,EfficientNetV2B0,0.148,-0.014,0.347,0.2166,No",
  "F1-score,VGG16,0.079,-0.071,0.255,0.4056,No",
  "Precision,ResNet50,0.161,-0.002,0.356,0.2146,No",
  "Precision,DenseNet121,0.132,-0.028,0.341,0.2261,No",
  "Precision,MobileNetV2,0.352,0.159,0.559,0.0160,Yes",
  "Precision,EfficientNetV2B0,0.173,-0.013,0.398,0.2146,No",
  "Precision,VGG16,0.167,-0.031,0.394,0.2261,No",
  sep = "\n"
)

plot_data <- read_csv(
  I(csv_text),
  show_col_types = FALSE
) |>
  mutate(
    Metric = factor(
      Metric,
      levels = c(
        "Recall / sensitivity",
        "AUC-PR",
        "F1-score",
        "Precision"
      )
    ),
    Architecture = factor(
      Architecture,
      levels = rev(c(
        "ResNet50",
        "DenseNet121",
        "MobileNetV2",
        "EfficientNetV2B0",
        "VGG16"
      ))
    ),
    BH_status = if_else(
      bh_significant == "Yes",
      "Retained after BH adjustment",
      "Not retained after BH adjustment"
    ),
    BH_status = factor(
      BH_status,
      levels = c(
        "Retained after BH adjustment",
        "Not retained after BH adjustment"
      )
    )
  )

if (nrow(plot_data) != 20L) {
  stop("Expected 20 architecture-metric rows, found ", nrow(plot_data), ".")
}

status_colours <- c(
  "Retained after BH adjustment" = "#D55E00",
  "Not retained after BH adjustment" = "#6B7280"
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
    linewidth = 0.55,
    linetype = "dashed",
    colour = "grey25"
  ) +
  geom_errorbar(
    aes(
      xmin = ci_low,
      xmax = ci_high
    ),
    orientation = "y",
    width = 0,
    linewidth = 0.8,
    lineend = "round"
  ) +
  geom_point(
    size = 2.8
  ) +
  facet_wrap(
    vars(Metric),
    ncol = 2
  ) +
  scale_colour_manual(
    values = status_colours,
    drop = FALSE
  ) +
  scale_x_continuous(
    breaks = seq(-0.2, 0.6, by = 0.2),
    limits = c(-0.25, 0.60),
    labels = scales::label_number(
      accuracy = 0.01,
      trim = TRUE
    )
  ) +
  labs(
    x = "Mean performance difference (light - dark)",
    y = NULL,
    colour = NULL
  ) +
  theme_publication() +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold"),
    panel.spacing = grid::unit(1.1, "lines")
  )

pdf_device <- if (capabilities("cairo")) {
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

message("Saved: ", pdf_path)
message("Saved: ", png_path)
message("Completed the architecture-level BH sensitivity figure.")
