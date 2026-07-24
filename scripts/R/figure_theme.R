# -------------------------------------------------------------------------
# Shared publication theme for the R alternatives to the Python figures.
# Figures intentionally contain no title, subtitle, or caption; those belong
# in the LaTeX document.
# -------------------------------------------------------------------------

library(ggplot2)

publication_colours <- c(
  "95% CI above 0" = "#0072B2",
  "95% CI includes 0" = "#6B7280",
  "95% CI below 0" = "#D55E00"
)

theme_publication <- function(base_size = 10.5, base_family = "sans") {
  theme_minimal(
    base_size = base_size,
    base_family = base_family
  ) +
    theme(
      axis.title.x = element_text(
        face = "bold",
        margin = margin(t = 8)
      ),
      axis.title.y = element_blank(),
      axis.text = element_text(colour = "grey15"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(
        size = 0.35,
        colour = "grey88"
      ),
      strip.background = element_blank(),
      strip.text.y.left = element_text(
        angle = 0,
        face = "bold",
        colour = "grey15",
        margin = margin(r = 8)
      ),
      strip.placement = "outside",
      panel.spacing.y = grid::unit(0.45, "lines"),
      legend.position = "top",
      legend.justification = "left",
      legend.box.just = "left",
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.5),
      legend.margin = margin(b = 4),
      plot.margin = margin(8, 12, 8, 10)
    )
}
