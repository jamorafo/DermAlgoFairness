# DermAlgoFairness

**Predictive Representativity and external validation in AI-based
skin-cancer detection**

This repository supports two related but distinct result generations:

1. the original workflow associated with the public preprint
   *Predictive Representativity: Uncovering Racial Bias in AI-Based Skin
   Cancer Detection* (`arXiv:2507.14176v1`);
2. a revised workflow developed for the dissertation case study and a
   potential revised publication.

The revised workflow preserves the scientific continuity of the original
study while strengthening experimental repetition, dataset provenance,
uncertainty quantification, multiplicity analysis, and publication-output
reproducibility.

The revised numerical outputs are not exact reproductions of the original
arXiv v1 tables and figures.

## Choose the correct workflow

### Original arXiv v1 workflow

Use this track to inspect the notebooks, source files, tables, and figures
associated with the original public preprint.

Main locations:

- `notebooks/`
- `src/`
- `output/tables/`
- `output/figures/`

Documentation:

- [`docs/arxiv_v1_workflow.md`](docs/arxiv_v1_workflow.md)

### Revised dissertation and publication workflow

Use this track for the locked, repeated, uncertainty-aware external-validation
analysis used in the dissertation case study.

Main locations:

- `config/`
- `configs/`
- `splits/`
- `scripts/`
- `scripts/R/`
- `outputs/tables/`
- `outputs/publication_tables/`
- `outputs/figures-r/`

Documentation:

- [`docs/revised_dissertation_workflow.md`](docs/revised_dissertation_workflow.md)

### Relationship between the two tracks

Before comparing results across versions, read:

- [`docs/results_version_map.md`](docs/results_version_map.md)

The former revision-documentation entry point is retained at:

- [`docs/arxiv_results_revision.md`](docs/arxiv_results_revision.md)

## Scientific scope

The project evaluates supervised dermoscopic-image classifiers trained using
HAM10000 and externally evaluated using the BOSQUE test set.

The central question is whether predictive performance and subgroup
performance remain adequate and transportable when a locked system is
evaluated in a population different from its development source.

Five CNN architecture families are included:

- ResNet50
- DenseNet121
- MobileNetV2
- EfficientNetV2B0
- VGG16

## Repository structure

### Original arXiv v1 assets

- `notebooks/` — original training, validation, reporting, and Predictive
  Representativity notebooks.
- `src/` — original supporting source files.
- `output/tables/` — historical arXiv-era tables.
- `output/figures/` — historical arXiv-era figures.

### Revised dissertation and publication assets

- `config/` — registered training and analysis seeds.
- `configs/` — architecture-specific configurations.
- `splits/` — locked lesion-grouped HAM10000 split.
- `scripts/` — Python estimation and statistical-analysis workflow.
- `scripts/R/` — R publication-table and figure rendering.
- `outputs/tables/` — finalized statistical-analysis CSV files.
- `outputs/publication_tables/` — canonical CSV and LaTeX tables.
- `outputs/figures-r/` — canonical PDF and PNG figures.
- `docs/` — documentation for both result generations.

The singular directory `output/` belongs to the historical workflow. The
plural directory `outputs/` belongs to the revised workflow.

## Reproducibility boundary

The revised publication layer is separated into two parts.

### Python estimation layer

Python performs:

- model execution;
- prediction generation;
- bootstrap estimation;
- subgroup analysis;
- multiplicity analysis;
- statistical summarization;
- generation of finalized CSV files.

### R rendering layer

R renders publication tables and figures from finalized CSV files.

The R layer does not:

- retrain models;
- regenerate predictions;
- recreate the locked split;
- repeat bootstrap estimation;
- perform an independent second statistical analysis.

Expensive analyses should not be repeated merely to modify table formatting,
captions, terminology, or figure presentation.

## Version identification

Branches are mutable development references. Reproducible reporting should
identify:

- the exact Git commit;
- the locked-split SHA-256 hash;
- the configured training and analysis seeds;
- the software environment;
- the relevant prediction and output manifests.

An immutable historical tag for the exact arXiv v1 code state has not yet been
assigned. Such a tag should be created only after the corresponding historical
commit has been identified and verified.

## References

1. Morales-Forero, A., Rueda, L. J., Herrera, R., Bassetto, S., and
   Coatanea, E. (2025). *Predictive Representativity: Uncovering Racial Bias
   in AI-Based Skin Cancer Detection*. `arXiv:2507.14176v1`.

2. Jaramillo Arboleda, A., Sánchez Zapata, M. J., Rueda Jaime, L. J.,
   Morales-Forero, A., and Bassetto, S. (2025). *BOSQUE Test Set*.
   DOI: `10.7910/DVN/AQEPIN`.

3. Tschandl, P., Rosendahl, C., and Kittler, H. (2018).
   *The HAM10000 dataset*. DOI: `10.1038/sdata.2018.161`.
