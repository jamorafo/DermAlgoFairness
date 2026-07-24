# Revised Public Results Workflow

This file is retained as the stable documentation entry point referenced by
earlier versions of the repository README.

The repository now distinguishes two result generations:

1. the original workflow associated with `arXiv:2507.14176v1`;
2. the revised dissertation and publication workflow.

## Documentation map

- [`results_version_map.md`](results_version_map.md) explains the relationship
  between the two result generations.
- [`arxiv_v1_workflow.md`](arxiv_v1_workflow.md) documents the preserved
  original notebook-based workflow.
- [`revised_dissertation_workflow.md`](revised_dissertation_workflow.md)
  documents the current locked, repeated, uncertainty-aware workflow.

## Continuity with the public manuscript

The revised experiments preserve the central scientific question of the
public preprint:

> Do skin-cancer classifiers trained using HAM10000 maintain acceptable and
> equitable predictive performance when externally evaluated using BOSQUE?

The revised workflow strengthens:

- public dataset provenance;
- lesion-grouped source evaluation;
- repeated model training;
- saved prediction artifacts;
- bootstrap uncertainty;
- subgroup-gap reporting;
- target adequacy and performance-preservation analysis;
- multiplicity sensitivity;
- deterministic table and figure rendering.

These changes can alter the numerical results.

Revised outputs must therefore be identified as revised dissertation or
publication results rather than silently substituted for the original arXiv
v1 outputs.

## Development history

The revision was developed through branches including:

- `arxiv-results-revision-2026`
- `r-publication-figures`

Branch names document development stages but are not immutable provenance
identifiers.

Reproducible reporting should use exact Git commit hashes and output
manifests.
