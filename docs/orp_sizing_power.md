# ORP Sizing and Evidential-Capacity Analysis

## Purpose

This analysis quantifies whether the BOSQUE objective reference point (ORP)
contains enough eligible observations to resolve prespecified subgroup
performance differences or expose rare failures. It complements, but does not
replace, the bootstrap confidence intervals, TAC/ETC decisions, or
architecture-level multiplicity analysis.

The sizing analysis addresses three distinct planning questions:

1. How much power do the observed light- and dark-phototype denominators provide
   for an absolute difference in a binary performance metric?
2. What is the minimum detectable absolute difference under conventional power
   targets?
3. How many independent eligible observations are needed to have a specified
   probability of observing at least one rare failure?

## Metric-specific BOSQUE denominators

The finalized BOSQUE analysis set contains 151 lesions, but the relevant
denominator depends on the metric:

| Performance question | Light | Dark | Eligible units |
|---|---:|---:|---|
| Accuracy-like metric | 105 | 46 | All lesions |
| Recall / sensitivity | 77 | 21 | Malignant lesions |
| Specificity | 28 | 25 | Benign lesions |

Under the stated planning assumptions, the observed denominators provide:

| Performance question | Power for a 0.10 gap | Power for a 0.15 gap | MDD at 80% power |
|---|---:|---:|---:|
| Accuracy-like metric | 29.6% | 53.9% | 0.212 |
| Recall / sensitivity | 17.6% | 31.5% | 0.307 |
| Specificity | 15.0% | 26.2% | 0.348 |

MDD denotes the minimum detectable absolute light-minus-dark performance
difference.

## Planning assumptions

The subgroup-gap calculations use a large-sample two-proportion planning
approximation with:

- independent eligible lesions;
- light-condition performance fixed at 0.85;
- dark-condition performance equal to `0.85 - delta`;
- two-sided `alpha = 0.05`;
- conventional 80% and 90% power targets.

The rare-failure calculations use the exact binomial complement

\[
P(\text{at least one observed failure}) = 1-(1-q)^n,
\]

where `q` is the failure probability among eligible units and `n` is the number
of independent eligible units.

For the 46 dark-phototype lesions, the probability of observing at least one
failure is approximately 37.0% when `q = 1%` and 4.5% when `q = 0.1%`.

## Interpretation

These calculations are prospective and retrospective evidential-capacity
diagnostics. They do not:

- test whether an observed subgroup difference is real;
- convert a non-significant or multiplicity-adjusted result into evidence of
  equality;
- replace bootstrap uncertainty;
- correct selection bias;
- establish that BOSQUE represents Colombia or another deployment population;
- apply directly to nonlinear metrics such as F1-score or AUC-PR without a
  metric-specific simulation design.

They show that the available ORP can reveal large performance problems but has
limited resolution for moderate subgroup differences and rare failures,
especially for recall and specificity in the dark-phototype condition.

## Reproduction

From the repository root:

```bash
python scripts/25_generate_orp_sizing_power.py
```

Required Python packages are already part of the repository scientific stack:

- NumPy
- pandas
- SciPy
- statsmodels
- Matplotlib

The analysis does not train models, regenerate predictions, recreate the locked
split, rerun the 10,000-replicate bootstrap, or recompute multiplicity results.

## Canonical outputs

The script generates:

```text
outputs/orp_sizing/
├── csv/
│   ├── orp_current_rare_event_detection.csv
│   ├── orp_current_subgroup_power.csv
│   ├── orp_prospective_subgroup_sample_sizes.csv
│   ├── orp_rare_failure_sample_sizes.csv
│   └── orp_tac_sample_sizes.csv
├── figures/
│   ├── figure_orp_current_subgroup_power.pdf
│   ├── figure_orp_current_subgroup_power.png
│   ├── figure_orp_rare_event_detection.pdf
│   └── figure_orp_rare_event_detection.png
├── tables/
│   ├── table_orp_current_power.tex
│   ├── table_orp_evidential_capacity_summary.tex
│   └── table_orp_prospective_sizing.tex
└── orp_sizing_metadata.json
```

The metadata file records the numerical assumptions, current denominators,
methods, and software versions.

## Relationship to the dissertation

The dissertation uses:

- a concise evidential-capacity subsection and compact summary table in the
  dermatology case-study chapter;
- full derivations, operating-characteristic curves, prospective sample-size
  examples, and sensitivity discussion in the technical appendix.

The repository is the reproducibility source for the calculations and generated
artifacts. The dissertation remains the authoritative source for the final
narrative interpretation and cross-references.
