# DermAlgoFairness

**Predictive Representativity: Uncovering Racial Bias in AI-Based Skin Cancer Detection**

This repository contains the codebase, notebooks, and figures supporting the publication on **Predictive Representativity (PR)** — a framework for fairness auditing in medical AI, applied to skin cancer diagnosis models trained on the HAM10000 dataset [1].

## Project Summary

We introduce **Predictive Representativity (PR)** and an **External Transportability Criterion (ETC)** to audit how well machine learning models generalize fairness across demographic groups, particularly between light and dark skin phototypes.

Our case study evaluates several convolutional neural network (CNN) classifiers on the **BOSQUE** test dataset from Colombia [2], highlighting how models trained on demographically balanced datasets (like HAM10000 [3]) may underperform on underrepresented populations.

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
```

---

## References

1. **Morales-Forero, A., Rueda, L. J., Herrera, R., Bassetto, S., & Coatanea, E.** (2025). *Predictive Representativity: Uncovering Racial Bias in AI-Based Skin Cancer Detection.* *arXiv preprint.* [https://arxiv.org/abs/2507.14176v1](https://arxiv.org/abs/2507.14176v1)
  
2. **Jaramillo Arboleda, A., Sánchez Zapata, M. J., Rueda Jaime, L. J., Morales-Forero, A., & Bassetto, S.** (2025). *BOSQUE Test Set.* *Harvard Dataverse.* [https://doi.org/10.7910/DVN/AQEPIN](https://doi.org/10.7910/DVN/AQEPIN)

3. **Tschandl, P., Rosendahl, C., & Kittler, H.** (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* *Scientific Data, 5*, 180161. [https://doi.org/10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

## Revised public results workflow

The branch `arxiv-results-revision-2026` contains the standardized workflow for revising and extending the public arXiv-related results using the published public version of the BOSQUE dataset from Harvard Dataverse.

The revised workflow focuses on scripted experiment execution, explicit environment documentation, saved per-image predictions, repeated runs to quantify training stochasticity, and reproducible statistical analysis.

See [`docs/arxiv_results_revision.md`](docs/arxiv_results_revision.md) for details.
