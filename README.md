# Regime-Aware Evaluation of Skeleton-Based Rehabilitation Movement Assessment

This repository accompanies the manuscript:

**Deployment-Aware Evaluation of Interpretable Skeletal Representations for Rehabilitation Movement Assessment**

The repository contains the experimental notebooks used to evaluate how deployment regime, model family, temporal structure, robustness, and interpretability influence skeleton-based rehabilitation movement assessment across two public datasets: **REHAB24-6** and **IntelliRehabDS**.

## Overview

The study evaluates rehabilitation movement classification under three distinct deployment settings:

- **Generalized evaluation**: subject-disjoint evaluation on previously unseen users.
- **Task-specific evaluation**: exercise- or gesture-specific models evaluated on unseen users.
- **Individualized evaluation**: within-subject, subject-task modeling using user-specific calibration data.

The experiments use a harmonized, temporally normalized joint-angle representation in which each movement repetition is represented as a `100 × 10` trajectory. The same representation is used across model families to enable controlled comparisons.

The primary comparison is between:

- **Logistic Regression (LR)**
- **Dual-Attention Temporal Convolutional Network (DA-TCN)**

Additional validation includes:

- Random Forest (RF)
- Support Vector Machine (SVM)
- XGBoost (XGB)
- Long Short-Term Memory network (LSTM)

The repository also includes temporal-order stress testing, controlled skeleton perturbations, movement-quality measures, SHAP-based attribution, and DA-TCN attention analysis.

## Repository Structure

```text
RehabMonitoring/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_REHAB24_6_primary_analysis.ipynb
    ├── 02_IntelliRehabDS_primary_analysis.ipynb
    └── 03_cross_method_temporal_stress_validation.ipynb
```

### `01_REHAB24_6_primary_analysis.ipynb`

Primary analysis for the **REHAB24-6** dataset.

Includes:

- dataset loading and preprocessing;
- skeleton harmonization;
- joint-angle extraction;
- temporal normalization to 100 movement phases;
- generalized evaluation;
- task-specific evaluation;
- individualized evaluation;
- LR and DA-TCN comparison;
- balanced accuracy and ROC-AUC evaluation;
- statistical comparison between LR and DA-TCN;
- movement-quality measures;
- LR SHAP analysis;
- DA-TCN attention analysis;
- controlled robustness experiments.

### `02_IntelliRehabDS_primary_analysis.ipynb`

Primary analysis for the **IntelliRehabDS** dataset.

Includes the same core evaluation pipeline as the REHAB24-6 notebook:

- preprocessing and skeleton harmonization;
- joint-angle representation;
- generalized, task-specific, and individualized evaluation;
- LR and DA-TCN comparison;
- balanced accuracy and ROC-AUC evaluation;
- statistical analysis;
- movement-quality measures;
- SHAP and attention analysis;
- robustness testing.

### `03_cross_method_temporal_stress_validation.ipynb`

Cross-model validation and temporal-order stress testing for both datasets.

This notebook evaluates:

- Random Forest;
- SVM;
- XGBoost;
- LSTM;

under:

- generalized evaluation;
- individualized evaluation;
- original temporally ordered trajectories;
- temporally shuffled trajectories.

For individualized temporal-shuffle analysis, repeated subject-task results are aggregated at participant level before statistical testing. Exact two-sided sign-flip tests are then applied across subjects, with Holm correction across the four model families within each dataset.

The notebook reproduces the cross-method and temporal-shuffling results reported in the manuscript.

## Joint-Angle Representation

Both datasets are mapped to a common skeletal representation and transformed into anatomically interpretable joint-angle trajectories.

Each repetition is temporally normalized to **100 movement phases**, producing an input representation of:

```text
100 time-normalized phases × 10 joint-angle channels
```

Flattened models operate on the phase-indexed representation as a fixed feature vector, while sequential models retain the explicit temporal dimension.

The representation is designed to support comparison across heterogeneous skeletal sources while preserving movement-phase information.

## Evaluation Regimes

### Generalized

Models are trained and evaluated using subject-disjoint folds.

No participant appearing in the test fold is present in the corresponding training fold.

This setting represents deployment to previously unseen users without user-specific calibration data.

### Task-Specific

Models are trained independently for each exercise or gesture while preserving subject-disjoint evaluation.

This setting reduces movement heterogeneity while retaining cold-start generalization to unseen users.

### Individualized

Models are fitted separately within eligible subject-task pairs using repeated train/test splits.

This setting represents user-specific calibration and should not be interpreted as unseen-subject or longitudinal generalization.

Sparse subject-task groups that do not contain sufficient samples from both correctness classes are excluded before training.

## Models

### Logistic Regression

Logistic Regression uses L2 regularization and class-balanced weighting.

For generalized and task-specific evaluation, the inverse regularization strength `C` is selected inside the outer training partition using subject-disjoint inner cross-validation over:

```text
{0.01, 0.03, 0.1, 0.3, 1, 3, 10}
```

For individualized evaluation, `C = 1.0` is fixed across subject-task pairs.

Features are standardized using statistics fitted on the training partition only.

### DA-TCN

The DA-TCN is a temporal convolutional model with angle- and temporal-attention mechanisms.

Hyperparameter selection is performed using training data only, following the protocol described in the manuscript.

### Additional Model Families

The cross-method validation notebook evaluates fixed implementations of:

- **Random Forest**: 300 trees, class-balanced weighting;
- **SVM**: RBF kernel, `C = 1`, `gamma = scale`, class-balanced weighting;
- **XGBoost**: 300 estimators, maximum depth 5, learning rate 0.05;
- **LSTM**: single 64-unit recurrent layer, dropout 0.30, Adam optimization.

## Evaluation Metrics

### Balanced Accuracy

Balanced Accuracy (BA) is the primary metric:

\[
\mathrm{BA}
=
\frac{1}{2}
\left(
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
+
\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}
\right)
\]

It gives equal weight to sensitivity and specificity and is therefore appropriate for unequal class distributions.

### ROC-AUC

The area under the receiver operating characteristic curve (ROC-AUC) is reported as a secondary discrimination metric.

ROC-AUC evaluates ranking performance across classification thresholds, whereas BA is computed from class predictions at the fixed decision threshold used by the models.

## Statistical Analysis

The manuscript uses paired statistical tests that match the evaluation unit of each experiment.

Depending on the analysis, these include:

- paired Wilcoxon signed-rank tests;
- bootstrap confidence intervals for paired median differences;
- exact subject-level sign-flip tests;
- Benjamini-Hochberg false-discovery-rate correction;
- Holm correction for cross-model temporal-shuffle comparisons.

For individualized primary comparisons, results are paired by subject-task pair after averaging repeated splits.

For the subject-aware temporal-shuffle sensitivity analysis, pair-level differences are first averaged within participant before exact sign-flip testing.

## Temporal-Order Stress Test

The temporal-shuffling experiment tests whether predictive performance depends on the ordering of movement phases.

A deterministic frame permutation is applied consistently across all angle channels of each repetition.

Models are then retrained and evaluated using the same split indices and model seeds used for the corresponding original-order experiment.

The reported shuffle effect is:

```text
ΔBA = BA_shuffled − BA_original
```

Negative values indicate degradation after temporal-order disruption.

The experiment measures predictive sensitivity to phase ordering; it is not intended as a direct measure of biomechanical temporal reasoning.

## Robustness Analysis

The primary notebooks also evaluate test-time sensitivity to controlled skeleton perturbations.

Perturbation families include:

- spatial jitter;
- joint dropout;
- burst occlusion;
- frame dropping.

Models are trained once on the clean training partition and then held fixed across all corruption levels.

The corresponding uncorrupted test performance is therefore the experiment-specific clean baseline for the robustness trajectories.

The experiments evaluate downstream predictive stability under controlled skeletal degradation.

## Movement-Quality Measures

Three post-hoc movement-quality measures are computed independently of the classifiers:

- **Pose Similarity Metric (PSM)**
- **Movement Smoothness Index (MSI)**
- **Joint Deviation (JD)**

These descriptors characterize complementary aspects of movement execution relative to task-specific reference trajectories.

Correct and incorrect repetitions are compared using two-sided Mann-Whitney U tests, both globally and at task level.

## Model Attribution

### Logistic Regression

SHAP values are used to identify phase-specific joint-angle features that contribute most strongly to LR predictions.

### DA-TCN

Angle and temporal attention weights are analyzed to inspect how the temporal model distributes internal weighting across movement features and phases.

SHAP and attention are treated as complementary explanation mechanisms and are not interpreted as numerically equivalent measures of feature importance.

Together with the movement-quality measures, these analyses support inspection of both movement deviations and the features associated with model decisions.

## Datasets

This repository uses two public rehabilitation datasets:

- **REHAB24-6**
- **IntelliRehabDS**

The datasets are **not redistributed in this repository**.

Please obtain each dataset from its official source and update the local paths in the corresponding notebook before execution.

## Reproducibility

Recommended workflow:

1. Clone this repository.
2. Create a Python environment.
3. Install the required dependencies.
4. Download REHAB24-6 and IntelliRehabDS from their official sources.
5. Update dataset paths in the notebooks.
6. Run the primary dataset notebooks.
7. Run the cross-method temporal-stress notebook.

Example:

```bash
git clone https://github.com/basmajalloul/RehabMonitoring.git
cd RehabMonitoring

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

## Main Dependencies

The analysis uses standard scientific Python packages including:

```text
numpy
pandas
scipy
scikit-learn
xgboost
tensorflow
shap
matplotlib
seaborn
optuna
jupyter
```

Exact package versions may be provided in `requirements.txt` for reproducible execution.

## Notes on Randomness

Fixed seeds are used wherever supported.

Some neural-network experiments may still show small numerical differences across environments because of stochastic optimization, hardware, TensorFlow/CUDA versions, or nondeterministic GPU operations.

For robustness experiments, the clean baseline corresponds to the specific trained model instance used throughout the corruption trajectory and may therefore differ slightly from an independently trained clean model reported elsewhere in the analysis.

## Outputs

The notebooks produce manuscript-level outputs including:

- fold-level generalized results;
- subject-task individualized results;
- cross-model summary tables;
- temporal-shuffle sensitivity statistics;
- robustness trajectories;
- bootstrap confidence intervals;
- movement-quality summaries;
- SHAP attribution results;
- attention summaries and visualizations.

Intermediate checkpoints may be generated during long-running experiments to allow safe resumption.

## Scope

The code is intended to reproduce the experimental framework described in the associated manuscript.

The individualized regime represents within-subject evaluation with available calibration data rather than longitudinal deployment.

Robustness perturbations are synthetic test-time perturbations of skeletal trajectories and should be interpreted as controlled sensitivity tests.

## Citation

If you use this repository, please cite the associated manuscript.

```bibtex
@article{jalloul_deployment_aware_rehabilitation,
  title   = {Deployment-Aware Evaluation of Interpretable Skeletal Representations for Rehabilitation Movement Assessment},
  author  = {Jalloul, Basma and others},
  journal = {Multimedia Tools and Applications},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

Please update the citation metadata once the final publication details and DOI are available.

## License

This repository is provided for academic and research use.

Please consult the licenses and terms of use of the original datasets separately.
