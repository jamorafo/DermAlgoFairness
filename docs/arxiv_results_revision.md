# Revised Public Results Workflow

This document describes the standardized workflow for revising and extending the public results associated with the `DermAlgoFairness` project and its arXiv manuscript.

The revision is based on the published and publicly available version of the BOSQUE dataset. This version is preferred because it provides a cleaner and more citable data source for reproducibility than earlier internal working versions used during project development.

The goal is to preserve continuity with the original public results while improving reproducibility, traceability, and reporting quality.

## Scope

This workflow applies to the supervised dermatology classification experiments in which models are trained on HAM10000 and externally evaluated on the published public version of the BOSQUE dataset.

It covers:

- dataset provenance;
- preprocessing;
- model training;
- external validation;
- subgroup evaluation;
- statistical analysis;
- experiment tracking;
- generated tables and figures;
- repository and container reproducibility.

## Reporting and reproducibility conventions

The revised workflow is intended to align with internationally recognized conventions for machine-learning and medical-AI studies, including:

- TRIPOD+AI for prediction model reporting;
- PROBAST+AI for risk-of-bias and applicability assessment;
- STARD-AI where the evaluation is interpreted as diagnostic accuracy assessment;
- ML reproducibility checklist principles for datasets, code, environment, hyperparameters, seeds, metrics, and compute documentation.

These standards are used as methodological guidance for reporting and reproducibility.

## Relationship with the public arXiv version

The revised experiments are intended to provide a more reproducible and standardized version of the public results associated with the arXiv manuscript.

The main objectives are:

- to use the published public BOSQUE dataset version whenever possible;
- to make the experimental pipeline easier to reproduce from the repository;
- to separate code, data, model artifacts, predictions, tables, and figures;
- to document the computational environment;
- to save per-image predictions for official evaluations;
- to regenerate statistical tables and figures from scripts;
- to report uncertainty and subgroup-level performance transparently.

## Repository versioning

This work is developed on the branch:

```text
arxiv-results-revision-2026
