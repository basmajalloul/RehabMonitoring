# Regime-Aware Evaluation for Rehabilitation Monitoring

This repository accompanies the manuscript:

**"Regime-Aware Evaluation of Explainable Biomechanical Representations
for Rehabilitation Monitoring"**

It provides the experimental code used to evaluate deployment regimes
and model families on public rehabilitation datasets.

------------------------------------------------------------------------

## Overview

This project investigates how **deployment regime** (generalized,
per-exercise, individualized) influences model performance in
rehabilitation movement assessment.

Rather than focusing solely on model architecture, we analyze how:

-   Evaluation protocol\
-   Dataset structure\
-   Representation design\
-   Model capacity

interact under realistic clinical constraints.

> Model ranking is regime-dependent.\
> Architectural depth does not uniformly guarantee improved performance
> under small, heterogeneous rehabilitation datasets.

------------------------------------------------------------------------

## Repository Structure

### Notebooks

-   `REHAB24_6.ipynb`\
    Regime-aware evaluation on the REHAB24-6 dataset.

-   `IntelliRehabDS.ipynb`\
    Regime-aware evaluation on the IntelliRehabDS dataset.

-   `REHAB24_6_cross_method_XGB_LSTM_RF_SVM.ipynb`\
    Cross-model validation (RF, XGB, SVM, LSTM) and stress testing on
    REHAB24-6.

------------------------------------------------------------------------

## Experimental Framework

### Deployment Regimes

1.  **Generalized** -- Subject-disjoint evaluation across exercises\
2.  **Per-Exercise** -- Exercise-specific modeling\
3.  **Individualized** -- Subject--exercise pair modeling

------------------------------------------------------------------------

### Models Evaluated

Primary comparison: - Logistic Regression (LR) - Dual-Attention Temporal
Convolutional Network (DA-TCN)

Cross-model validation: - Random Forest (RF) - XGBoost (XGB) - Support
Vector Machine (SVM) - Long Short-Term Memory (LSTM)

------------------------------------------------------------------------

## Evaluation Metrics

-   Balanced Accuracy (BA)
-   Area Under the Curve (AUC)
-   Wilcoxon signed-rank tests
-   Median performance deltas
-   Percentage of regime-wise model dominance

------------------------------------------------------------------------

## Robustness Analysis

Time-shuffle stress testing was conducted to assess whether models rely
on temporal structure.

Additional robustness checks include: - Jitter - Frame dropout -
Occlusion simulation

------------------------------------------------------------------------

## Interpretability

-   SHAP-based feature attribution (LR)
-   Temporal and joint attention visualization (DA-TCN)
-   Post-hoc biomechanical validation using MQM metrics:
    -   Pose Similarity Metric (PSM)
    -   Movement Smoothness Index (MSI)
    -   Joint Deviation (JD)

------------------------------------------------------------------------

## Data

This repository uses the public datasets:

-   **REHAB24-6**
-   **IntelliRehabDS**

Datasets are not included in this repository. Please obtain them from
their official sources.

------------------------------------------------------------------------

## Requirements

Typical dependencies:

-   Python ≥ 3.9
-   NumPy
-   Pandas
-   Scikit-learn
-   XGBoost
-   TensorFlow / Keras
-   SHAP
-   Matplotlib

------------------------------------------------------------------------

## Reproducibility

1.  Download dataset.
2.  Adjust dataset paths in notebooks.
3.  Run notebooks for regime-aware modeling and validation.

All experiments use fixed random seeds.

------------------------------------------------------------------------

## Citation

If you use this code, please cite the associated manuscript.

------------------------------------------------------------------------

## License

Provided for academic and research use.
