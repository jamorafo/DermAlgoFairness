# Revised Dissertation and Publication Workflow

## Purpose

This track documents the revised empirical workflow used for the dissertation
case study and for a potential revised publication.

It preserves the dermatology application introduced in the public preprint
while strengthening:

- evaluation-population definition;
- dataset provenance;
- locked-system definition;
- training repetition;
- uncertainty quantification;
- subgroup analysis;
- multiplicity assessment;
- publication-output provenance.

## HAM10000 source ORP

The finalized held-out source evaluation set contains:

- 990 images;
- 747 lesions;
- a lesion-grouped split;
- lesion-cluster bootstrap resampling.

The locked split is:

- `splits/ham10000_lesion_grouped_fixed.csv`

Expected SHA-256:

- `1bc5cfb103f79a9fdb87f035b22818a911a442632a9c1154ab72fe90fb5f4b48`

## BOSQUE target ORPs

The published public BOSQUE analysis set contains:

- 151 images overall;
- 105 light-phototype images;
- 46 dark-phototype images;
- one image per lesion.

The target ORPs are:

- BOSQUE overall;
- BOSQUE light;
- BOSQUE dark.

Metric-specific eligible denominators are:

- all lesions for accuracy-like metrics: 105 light and 46 dark;
- malignant lesions for recall: 77 light and 21 dark;
- benign lesions for specificity: 28 light and 25 dark.

These denominators, rather than the nominal total of 151 lesions alone, determine the power and rare-failure exposure of the ORP for each question.

## Locked predictive systems

Five architecture families are evaluated:

- ResNet50
- DenseNet121
- MobileNetV2
- EfficientNetV2B0
- VGG16

Each architecture has five independently randomized training runs.

The complete evaluation therefore contains 25 locked
architecture--run systems.

Configured training and analysis seeds are recorded in:

- `config/random_seeds.json`

A training run is part of the locked-system identity. It is not treated merely
as a disposable computational seed.

## Python estimation layer

Python performs:

- public BOSQUE acquisition and verification;
- locked HAM10000 split handling;
- model training;
- saved per-image prediction generation;
- source and target performance estimation;
- light-minus-dark subgroup-gap analysis;
- interval-based target adequacy analysis;
- performance-preservation analysis;
- architecture-level multiplicity sensitivity;
- canonical statistical CSV generation.

Important scripts include:

- `scripts/00_download_bosque_public.py`
- `scripts/02_summarize_bosque_public.py`
- `scripts/03_train_model.py`
- `scripts/04_evaluate_model.py`
- `scripts/06_compare_bosque_subgroups.py`
- `scripts/12_interval_tac_etc_analysis.py`
- `scripts/20_make_seed_aware_gap_publication_outputs.py`
- `scripts/21_make_architecture_level_bh_sensitivity.py`
- `scripts/23_create_fixed_lesion_grouped_split.py`
- `scripts/24_regenerate_final_publication_tables.py`
- `scripts/25_generate_orp_sizing_power.py`

## R publication-rendering layer

R reads finalized CSV files and renders publication outputs.

Important files include:

- `scripts/R/00_run_all_publication_figures.R`
- `scripts/R/01_seed_aware_gap_figures.R`
- `scripts/R/02_interval_decision_summary.R`
- `scripts/R/03_internal_external_figures.R`
- `scripts/R/04_architecture_bh_sensitivity_figure.R`
- `scripts/R/05_finite_orp_precision_figure.R`
- `scripts/R/06_mean_gap_heatmap.R`
- `scripts/R/07_make_publication_tables.R`
- `scripts/R/figure_theme.R`

The R layer does not:

- train models;
- regenerate predictions;
- recreate the locked split;
- repeat bootstrap estimation;
- perform an independent second multiplicity analysis.

## Statistical reporting

The finalized interval analyses use 10,000 bootstrap replicates.

### Primary metrics

- recall or sensitivity;
- AUC-PR;
- F1-score;
- precision.

### Secondary metrics

- accuracy;
- specificity;
- AUC-ROC.

### Reported analyses

- source performance;
- target performance;
- source-to-target degradation;
- target adequacy;
- performance preservation;
- light-minus-dark subgroup gaps;
- finite-ORP interval precision;
- architecture-level BH and BY multiplicity sensitivity;
- metric-specific ORP power, minimum-detectable-difference, TAC-sizing, and rare-failure diagnostics.

No architecture--metric light-minus-dark contrast is presented as significant
after BH or BY adjustment in the finalized architecture-level sensitivity
table.

Finite-ORP interval half-widths are reported descriptively.

No additional post hoc precision threshold is imposed.

## Canonical outputs

### Statistical source tables

- `outputs/tables/`

### Publication CSV and LaTeX tables

- `outputs/publication_tables/`

The canonical package contains nine CSV files and nine corresponding LaTeX
tables.

### Publication figures

- `outputs/figures-r/`

The canonical package contains ten figure basenames, each rendered as PDF and
PNG.

### ORP-sizing outputs

- `outputs/orp_sizing/csv/` — numerical planning and diagnostic outputs;
- `outputs/orp_sizing/tables/` — generated LaTeX tables;
- `outputs/orp_sizing/figures/` — PDF and PNG power and rare-failure curves;
- `outputs/orp_sizing/orp_sizing_metadata.json` — assumptions, methods, counts, and software versions.

These outputs are generated independently of model training and bootstrap estimation.

## Regeneration boundary

Publication tables and figures may be regenerated from finalized CSV files
without repeating the expensive estimation layer.

Model training, prediction generation, fixed-split creation, and 10,000-run
bootstrap analyses should be repeated only when the scientific analysis
changes.

Examples include changes to:

- the estimand;
- the target population;
- the source population;
- the locked system;
- the prediction artifacts;
- the bootstrap design;
- the multiplicity procedure.

Formatting, captions, terminology, or figure styling do not require
re-estimation.

ORP-sizing calculations may also be regenerated independently because they use prespecified counts and planning assumptions rather than prediction-level model outputs. See [`orp_sizing_power.md`](orp_sizing_power.md).

## Relationship with arXiv v1

This workflow is a revision and extension of the original dermatology case
study.

It is not an exact reproduction of the arXiv v1 results.

Important revisions include:

- the public finalized BOSQUE analysis set;
- the locked lesion-grouped source split;
- five independently randomized runs per architecture;
- explicit source and target ORPs;
- interval-based adequacy and preservation decisions;
- seed-aware subgroup-gap intervals;
- architecture-level BH and BY sensitivity;
- separation of Python estimation from R publication rendering.

Read [`results_version_map.md`](results_version_map.md) before comparing
revised numerical outputs with the original preprint.

<!-- ORP-SIZING-WORKFLOW -->
