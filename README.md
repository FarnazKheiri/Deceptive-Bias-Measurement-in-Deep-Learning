# Quantifying Shortcut Reliance in Cancer Classification


This repository contains the official implementation of the Deceptive Pipeline, a novel measurement framework designed to quantify site-specific shortcut reliance in deep learning models for histopathology.
The research specifically investigates how artifacts in medical imaging datasets (such as TCGA) can lead to "shortcut learning," where models rely on irrelevent features for cancer classification.

## Research Overview
Our work proposes a new bias measurement metric, the Deceptive Signal ($S$), to address the limitations of existing fairness metrics in medical AI. By systematically isolating site-specific features across multiple architectures (EfficientNet, ResNet, and DenseNet).

**Key Contributions:**
- Deceptive Pipeline: A methodology for isolating shortcuts through controlled class-site exclusion.
- Cross-Architecture Synthesis: Evaluation of susceptibility to bias across EfficientNet-B0, ResNet50V2, and DenseNet121.
- Deceptive Signal ($S$): A robust metric for quantifying latent bias gain.

## Repository Structure
- `models.py`: Definitions for all CNN architectures with specific preprocessing and layer-freezing logic.
- `data_full_exclusion.py`: Phase 1 logic for establishing baseline error rates on external centers.
- `data_partial_inclusion.py`: Phase 2 logic for shortcut isolation and signal measurement.
- `slide_names.csv`: The list of specific TCGA UUIDs used to ensure reproducibility of our cohorts.

## Data Acquisition
Due to size constraints, raw histopathology images are not hosted in this repository.

 1. Download the LUAD and LUSC datasets from the [GDC Data Portal](https://portal.gdc.cancer.gov/).
 2. Use the provided `slide_names.csv` to filter for the exact slides used in our 3-center, 4-center, and 5-center configurations.

## Getting Started
1. Installation
```bash
pip install tensorflow sklearn fairlearn numpy
