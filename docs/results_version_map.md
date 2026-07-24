# Results Version Map

## Purpose

`DermAlgoFairness` contains two scientifically related but numerically
distinct result generations.

This document prevents the original public-preprint workflow from being
confused with the revised dissertation and publication workflow.

## Original arXiv v1 track

### Scientific role

The original empirical case study associated with
`arXiv:2507.14176v1`.

### Main workflow

- notebook-based model training;
- notebook-based internal and external validation;
- original Predictive Representativity analysis;
- original reporting scripts.

### Main locations

- `notebooks/`
- `src/`
- `output/tables/`
- `output/figures/`

### BOSQUE set

The public preprint reports an independently collected external test set of
167 dermoscopic images.

### Result interpretation

These assets preserve the historical workflow and its associated outputs.

They should not be silently overwritten by revised dissertation outputs.

## Revised dissertation and publication track

### Scientific role

A revised empirical case study used in the dissertation and intended to
support subsequent publication reporting.

### Main workflow

- scripted data preparation;
- locked lesion-grouped source splitting;
- five independent training runs per architecture;
- saved per-image predictions;
- 10,000-replicate bootstrap analyses;
- target adequacy and performance-preservation analysis;
- seed-aware subgroup-gap analysis;
- architecture-level BH and BY multiplicity sensitivity;
- deterministic R publication rendering.

### Main locations

- `config/`
- `configs/`
- `splits/`
- `scripts/`
- `scripts/R/`
- `outputs/tables/`
- `outputs/publication_tables/`
- `outputs/figures-r/`

### HAM10000 source ORP

The revised held-out source evaluation set contains:

- 990 images;
- 747 lesions;
- a locked lesion-grouped split;
- lesion-cluster bootstrap resampling.

### BOSQUE target ORPs

The revised public analysis set contains:

- 151 images overall;
- 105 light-phototype images;
- 46 dark-phototype images;
- one image per lesion.

### Locked systems

Five architecture families are evaluated using five independently randomized
training runs each.

This produces 25 locked architecture--run systems.

## Important differences

### Dataset version and analysis population

The original and revised tracks do not use identical BOSQUE analysis
populations.

### Training repetition

The revised track explicitly evaluates five independently randomized training
runs per architecture.

### Statistical uncertainty

The revised track uses finalized 10,000-replicate bootstrap analyses.

### Multiplicity

The revised track includes architecture-level BH and BY sensitivity analysis.

### Publication rendering

The revised track separates Python estimation from R table-and-figure
rendering.

## Interpretation rule

The revised results extend and strengthen the original empirical case study,
but they are not a number-for-number reproduction of arXiv v1.

When citing a numerical result, identify the track that generated it.

Do not combine an original arXiv table with a revised figure without
documenting differences in:

- dataset version;
- analysis population;
- locked system;
- training run;
- estimator;
- bootstrap design;
- multiplicity treatment;
- output provenance.

## Version control

The branches `arxiv-results-revision-2026` and `r-publication-figures`
document stages of the revision process.

Branch names are not immutable provenance identifiers.

A reproducible citation should report:

1. the exact Git commit;
2. the SHA-256 hash of the locked split;
3. `config/random_seeds.json`;
4. the relevant prediction and output manifests;
5. the Python and R environment records.

No immutable tag currently identifies the exact historical arXiv v1 code
state.
