# Original arXiv v1 Workflow

## Reference

This track documents the repository assets associated with:

> Morales-Forero, A., Rueda, L. J., Herrera, R., Bassetto, S., and
> Coatanea, E. (2025). *Predictive Representativity: Uncovering Racial Bias
> in AI-Based Skin Cancer Detection*. `arXiv:2507.14176v1`.

The public preprint reports an independently collected BOSQUE external test
set of 167 dermoscopic images and evaluates five CNN architecture families.

## Architecture families

- ResNet50
- DenseNet121
- MobileNetV2
- EfficientNetV2B0
- VGG16

## Historical training notebooks

- `notebooks/01_Training_DenseNet121.ipynb`
- `notebooks/01_Training_EfficientNetV2B0.ipynb`
- `notebooks/01_Training_MobileNetV2.ipynb`
- `notebooks/01_Training_ResNet50.ipynb`
- `notebooks/01_Training_VGG16.ipynb`

## Historical validation notebooks

- `notebooks/02_Validation_DenseNet121.ipynb`
- `notebooks/02_Validation_EfficientNetV2B0.ipynb`
- `notebooks/02_Validation_MobileNetV2.ipynb`
- `notebooks/02_Validation_ResNet50.ipynb`
- `notebooks/02_Validation_VGG16.ipynb`

## Historical reporting and PR analysis

- `notebooks/03_Reporting.ipynb`
- `notebooks/06_Predictive_Representativity_Analysis.ipynb`
- `notebooks/BOSQUE_EDA.ipynb`
- `src/03_figure2_PR_by_model.R`

## Historical outputs

### Tables

- `output/tables/`

### Figures

- `output/figures/`

The singular directory `output/` identifies the original workflow.

The plural directory `outputs/` identifies the revised dissertation and
publication workflow.

## Reproducibility status

This track is retained as a historical publication workflow.

Its notebooks may depend on:

- the software environment used during the original study;
- the original data organization;
- historical path conventions;
- model artifacts used during the original evaluation;
- notebook execution order.

It should therefore be treated as an archival workflow rather than assumed to
be a one-command reproduction under the current environment.

## Preservation policy

The legacy notebooks, source files, tables, and figures must not be deleted or
silently overwritten by the revised workflow.

Revised analyses should write to the plural `outputs/` hierarchy.

The current files under:

- `outputs/publication_tables/`
- `outputs/figures-r/`

must not be described as exact reproductions of the arXiv v1 outputs.

## Historical commit and tag

The repository currently has no immutable tag identifying the exact code state
used for the arXiv v1 submission.

Before assigning such a tag:

1. identify the historical commit containing the matching notebooks and
   output files;
2. compare its artifacts with the public manuscript;
3. verify that the commit contains no confidential data;
4. assign an annotated tag that describes the manuscript version and
   preservation scope.

Until that verification is complete, the repository should describe this
track as the preserved historical workflow without claiming exact
commit-level reconstruction.
