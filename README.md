# DermAlgoFairness

**Predictive Representativity: Uncovering Racial Bias in AI-Based Skin Cancer Detection**

This repository contains the codebase, notebooks, and figures supporting the publication on **Predictive Representativity (PR)** — a framework for fairness auditing in medical AI, applied to skin cancer diagnosis models trained on the HAM10000 dataset.

## Project Summary

We introduce **Predictive Representativity (PR)** and an **External Transportability Criterion (ETC)** to audit how well machine learning models generalize fairness across demographic groups, particularly between light and dark skin phototypes.

Our case study evaluates several convolutional neural network (CNN) classifiers on the **BOSQUE** test dataset from Colombia, highlighting how models trained on demographically balanced datasets (like HAM10000) may underperform on underrepresented populations.

---

## Repository Structure

```bash
DermAlgoFairness/
├── notebooks/              # Jupyter notebooks for training and validation
│   ├── 01_Training_*.ipynb     # Per-model training workflows
│   └── 02_Validation_*.ipynb   # Per-model evaluation and PR metric analysis
│
├── src/                   # Custom R and Python scripts
│   └── 03_figure2_PR_by_model.R   # R script to generate Figure 2 (PR by model & skin tone)
│
├── output/
│   ├── figures/           # Output plots (PDF/EPS)
│   │   ├── PRms.pdf
│   │   ├── PRms_1.pdf
│   └── tables/            # Classification reports and detailed metrics
│       ├── classification_report_test_model_final_*.csv
│       ├── detailed_metrics_test_model_final_*.csv
│       └── detailed_metrics_full.csv
│
├── dockerfile             # Docker setup for GPU-enabled reproducibility
└── README.md              # Project documentation (you are here)


