# Quantifying Shortcut Reliance in Cancer Classification


This repository contains the official implementation of the Deceptive Pipeline, a novel measurement framework designed to quantify site-specific shortcut reliance in deep learning models for histopathology.
The research specifically investigates how artifacts in medical imaging datasets (such as TCGA) can lead to "shortcut learning," where models rely on irrelevent features for cancer classification.

## Research Overview
Our work proposes a new bias measurement metric, the Deceptive Signal ($S$), to address the limitations of existing fairness metrics in medical AI. By systematically isolating site-specific features across multiple architectures (EfficientNet, ResNet, and DenseNet).

Key Contributions:
- Deceptive Pipeline: A methodology for isolating shortcuts through controlled class-site exclusion.
- Cross-Architecture Synthesis: Evaluation of susceptibility to bias across EfficientNet-B0, ResNet50V2, and DenseNet121.
- Deceptive Signal ($S$): A robust metric for quantifying latent bias gain.
