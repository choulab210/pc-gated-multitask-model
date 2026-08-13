# Protein Corona Prediction Model

This repository contains the code and reproducible analysis workflow for a gated two-head neural-network framework that jointly predicts protein adsorption and quantitative protein corona composition from nanoparticle physicochemical properties and experimental conditions.

## Project Overview

The model uses a shared neural-network encoder with two prediction heads:

- **Adsorption head**: predicts whether individual proteins are detected in the protein corona.
- **Abundance head**: predicts the relative protein corona composition.

An adsorption-guided gating mechanism links the two tasks by using predicted adsorption probabilities to constrain abundance prediction.

## Dataset

The modeling dataset contains 397 nanoparticle–protein corona samples derived from a curated Protein Corona Database.

The workflow includes:

- 317 model-development samples
- 80 independent held-out test samples
- 174 individually modeled proteins
- 1 residual `OTHER` category
- 175 total model outputs
- 12 raw nanoparticle and experimental variables
- 55 encoded model inputs

Protein-panel selection and feature preprocessing are performed using model-development data only.

## Repository Structure

```text
protein-corona-model/
├── data/
├── figures/
│   ├── main/
│   └── supplementary/
├── results/
├── scripts/
│   ├── figures/
│   ├── tables/
│   ├── train_final.py
│   ├── run_external_validation.py
│   ├── run_applicability.py
│   ├── run_interpretation.py
│   ├── run_shap.py
│   ├── run_ablation.py
│   ├── run_grouped_validation.py
│   └── run_benchmarks.py
├── src/
│   └── pcmodel/
│       ├── data.py
│       ├── models.py
│       ├── training.py
│       ├── metrics.py
│       ├── validation.py
│       ├── metadata.py
│       ├── applicability.py
│       └── interpretation.py
├── tables/
└── pyproject.toml
```

## Installation

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
pip install -e .
```

## Main Analysis Workflow

### Train the Final Model

```powershell
python scripts/train_final.py
```

### External Validation

```powershell
python scripts/run_external_validation.py
```

### Feature-Grouped Validation

```powershell
python scripts/run_grouped_validation.py
```

### SHAP Interpretation

```powershell
python scripts/run_shap.py
```

### Architecture Ablation

```powershell
python scripts/run_ablation.py
```

### Conventional Machine-Learning Benchmarks

```powershell
python scripts/run_benchmarks.py
```

### Applicability-Domain Analysis

```powershell
python scripts/run_applicability.py
```

## Generate Manuscript Figures

Figure-generation scripts are located in:

```text
scripts/figures/
```

For example, generate Figure 1 using:

```powershell
python scripts/figures/plot_figure1_dataset.py
```

Generated main figures are saved under:

```text
figures/main/
```

Supplementary figures are saved under:

```text
figures/supplementary/
```

## Generate Supplementary Tables

Table-generation scripts are located in:

```text
scripts/tables/
```

For example, generate the individual-nanoparticle external-validation table using:

```powershell
python scripts/tables/tableS5_external_validation_by_np.py
```

Generated tables are saved under:

```text
tables/
```

## Reproducibility

A fixed random seed is used where applicable.

The complete dataset contains 397 NP–PC samples and is divided into 317 model-development samples and 80 held-out test samples.

The split is stratified according to quantile bins of the number of detected proteins per nanoparticle.

Protein-panel selection is performed using model-development data only. Proteins detected in fewer than 10 model-development samples are excluded, and the smallest protein panel accounting for 99% of cumulative abundance is retained.

This procedure results in 174 individually modeled proteins. Protein abundance outside the selected panel is aggregated into an `OTHER` category.

Feature preprocessing is also fitted using model-development data only and then applied unchanged to held-out and external-validation datasets.

## Model Inputs

The model uses 12 raw nanoparticle physicochemical and experimental variables:

- Particle size
- Zeta potential
- Surface-modification type
- Surface-modification charge
- Nanoparticle type
- Nanoparticle subtype
- Zeta-potential charge category
- Zeta-potential measurement solvent
- Hydrodynamic-size measurement solvent
- Incubation time
- Agitation
- Washing steps

After preprocessing, these variables are represented by 55 encoded model inputs.

## Model Architecture

The final model consists of a shared multilayer perceptron encoder with three hidden layers followed by:

- an adsorption prediction head
- an abundance prediction head

The abundance head is coupled to the adsorption head through an adsorption-guided soft-gating mechanism.

The gate down-weights abundance predictions for proteins with lower predicted adsorption probabilities before normalization of the final protein corona composition.

The final shared hidden layers contain 320 neurons each with ReLU activation and dropout.

## Model Evaluation

Adsorption prediction is evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- Matthews correlation coefficient
- AUROC
- AUPRC

Quantitative protein corona composition is evaluated using:

- Per-protein Pearson correlation
- 1-TVD
- Cosine similarity

Held-out performance uncertainty is estimated using 10,000 nanoparticle-level bootstrap resamples.

## External Validation

The final gated model is evaluated on eight independent NP–PC samples without model retraining.

External validation includes:

- overall adsorption performance
- per-nanoparticle performance
- protein functional-category performance
- quantitative protein corona composition comparison

Evaluation is restricted to proteins shared between the trained model panel and the external validation dataset.

## Data and Code Availability

The source code used for:

- data preprocessing
- model training
- hyperparameter optimization
- model evaluation
- architecture comparison
- feature-grouped validation
- applicability-domain analysis
- SHAP interpretation
- external validation
- manuscript figure generation
- supplementary table generation

is maintained in this repository.

Processed datasets required to reproduce the manuscript analyses will be made publicly available with the associated publication.

## Citation

Citation information will be added following publication of the manuscript.

## Contact

Wei-Chun Chou  
Email: weichun.chou@ucr.edu
Department of Environmental Sciences  
University of California, Riverside