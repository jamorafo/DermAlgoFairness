# ------------------------------------------------------------------------------
# 03_figure2_PR_by_model.R
# Reproduces Figure 2 from the paper: plots Predictive Representativity (PR)
# metrics across models and skin tone groups (Light vs Dark).
# Output: PR barplots per model, saved as PDF and EPS files.
# ------------------------------------------------------------------------------

# Clear workspace
rm(list=ls(all=TRUE))

# ------------------------------------------------------------------------------
# Define project paths (adjust `setwd` to your own local directory)
# ------------------------------------------------------------------------------
setwd("/Users/andresmorales/Documents/HAM10000/src/")

outPath      <- "../output/"
outPathGraph <- "../output/graph/"
inPath       <- "../input/"
srcPath      <- "../src/"

# ------------------------------------------------------------------------------
# Load required libraries
# ------------------------------------------------------------------------------
library(ggplot2)
library(dplyr)
library(cowplot) # For arranging multiple plots

# ------------------------------------------------------------------------------
# Load and preprocess input metrics data
# ------------------------------------------------------------------------------

fileName <- "/tables/detailed_metrics_full.csv"
inFile   <- paste(outPath, fileName, sep = "")
df_metrics <- read.csv(inFile)

# Rename columns for clarity
colnames(df_metrics) <- c("Model","Metric","HAM10000","Overall","Light","Dark",
                         "PR_Light","PR_Dark","Z_Statistic","P_Value","Significance")

# Define the custom importance order for plotting metrics
importance_order <- c(
  "Precision (Malignant)", "Sensitivity", "AUC-PR", "Specificity",
  "Accuracy", "AUC-ROC", "F1-Score (Malignant)"
)


# Filter and reorder metrics by importance
df_metrics <- df_metrics %>%
  filter(Metric %in% importance_order) %>%
  mutate(Metric = factor(Metric, levels = importance_order)) %>%
  arrange(factor(Metric, levels = importance_order)) %>%
  mutate(Metric = forcats::fct_rev(Metric)) # Reverse for plotting

# ------------------------------------------------------------------------------
# Function to create a barplot of PR metrics for a given model
# ------------------------------------------------------------------------------
create_plot <- function(data, model_name) {
  ggplot(data, aes(y = Metric)) +
    # Bars for Light Skin
    geom_bar(aes(x = `PR_Light`, fill = "Light Skin"),
             stat = "identity", position = position_nudge(y = 0.2), width = 0.4) +
    # Bars for Dark Skin
    geom_bar(aes(x = `PR_Dark`, fill = "Dark Skin"),
             stat = "identity", position = position_nudge(y = -0.2), width = 0.4) +
    
    # Add vertical line at x=0
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    
    # Customize labels and theme
    scale_fill_manual(values = c("Light Skin" = "#ff7f0e", "Dark Skin" = "#1f77b4")) +
    labs(
      x = expression(PR[M[i]]),
      y = NULL,
      fill = NULL,
      title = model_name
    ) +
    theme_minimal() +
    labs(fill = NULL) + # Ensure "fill" is removed
    theme(
      axis.text.y = element_text(size = 15),
      axis.title.x = element_text(size = 15),
      plot.title = element_text(size = 17, face = "bold"),
      legend.position = "none" # Remove individual legends
    )
}

# ------------------------------------------------------------------------------
# Prepare model ordering and factor alignment
# ------------------------------------------------------------------------------
model_order <- df_metrics %>%                         
  pull(Model) %>%                        
  unique()  # Preserve appearance order

df_metrics <- df_metrics %>%
  mutate(
    Model  = factor(Model, levels = model_order),     # <- key line
    Metric = factor(Metric, levels = importance_order)
  ) %>%
  arrange(Model, Metric)                              # keeps both orders

# ------------------------------------------------------------------------------
# Generate individual plots for each model
# ------------------------------------------------------------------------------
plots <- split(df_metrics, df_metrics$Model) %>%      # keeps order!
  lapply(function(d) create_plot(d, unique(d$Model)))

# ------------------------------------------------------------------------------
# Create a shared legend
# ------------------------------------------------------------------------------
g <- ggplot(df_metrics, aes(y = Metric)) +
  geom_bar(aes(x = `PR_Light`, fill = "Light Skin"), stat = "identity") +
  geom_bar(aes(x = `PR_Dark`, fill = "Dark Skin"), stat = "identity") +
  scale_fill_manual(values = c("Light Skin" = "#ff7f0e", "Dark Skin" = "#1f77b4")) + # Ensure "fill" is removed
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 15), # Adjust legend text size
    legend.title = element_blank() # Ensure no legend title appears
  )


shared_legend <- get_legend(g)

# ------------------------------------------------------------------------------
# Layout Option 1: 2-column grid + shared legend (18x14)
# ------------------------------------------------------------------------------
final_plot <- cowplot::plot_grid(
  plotlist = plots,
  ncol = 2,
  align = "v"
)

# Add the shared legend to the bottom
final_plot_with_legend <- cowplot::plot_grid(
  final_plot,
  shared_legend,
  ncol = 1,
  rel_heights = c(1, 0.1)
)


fileName <- "PRms.pdf"
outFile   <-  paste(outPath, fileName, sep = "")
pdf(outFile,width = 18,height = 14)
par(mar=c(1,2,1,1))
# Save or display the final plot
print(final_plot_with_legend)
dev.off()

fileName <- "PRms.eps"
outFile   <- paste(outPath, fileName, sep = "")
setEPS()
postscript(outFile,width = 18,height = 14,horizontal = FALSE)
par(mar=c(1,2,1,1))
# Save or display the final plot
print(final_plot_with_legend)
dev.off()


# ------------------------------------------------------------------------------
# Layout Option 2: 1-column grid + shared legend (7x14)
# ------------------------------------------------------------------------------
final_plot <- cowplot::plot_grid(
  plotlist = plots,
  ncol = 1,
  align = "v"
)

# Add the shared legend to the bottom
final_plot_with_legend <- cowplot::plot_grid(
  final_plot,
  shared_legend,
  ncol = 1,
  rel_heights = c(1, 0.1)
)


fileName <- "PRms_1.pdf"
outFile   <- paste(outPath, fileName, sep = "")
pdf(outFile,width = 7,height = 14)
par(mar=c(1,2,1,1))
# Save or display the final plot
print(final_plot_with_legend)
dev.off()



fileName <- "PRms_1.eps"
outFile   <- paste(outPath, fileName, sep = "")
setEPS()
postscript(outFile,width = 7,height = 14,horizontal = FALSE)
par(mar=c(1,2,1,1)) #c(bottom, left, top, right)
# Save or display the final plot
print(final_plot_with_legend)
dev.off()





