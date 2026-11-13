# Deployment-Aware Rehabilitation Movement Assessment

This repository accompanies the manuscript **Deployment-Aware Evaluation of Interpretable Skeletal Representations for Rehabilitation Movement Assessment**. The study compares movement classification for unseen participants, for a known exercise or gesture, and for participants with labelled calibration repetitions.

The reproducibility package contains the executed notebooks, matched train/test and validation splits, held-out predictions, and exported results for REHAB24-6 and IntelliRehabDS. **Start with the saved predictions to verify the manuscript scores.** This check runs on a CPU and needs only the exported files.

## Repository contents

```text
reproducibility/
├── metric_audit.json
├── notebooks/
│   ├── REHAB24-6_executed.ipynb
│   └── IntelliRehabDS_executed.ipynb
└── data/
    ├── rehab/
    └── intelli/
```

The paths below assume that the uploaded package is inside `reproducibility/`. If you unpacked its contents at the repository root, use the root as the package folder.

## Find the results behind the manuscript

Paths in this table are relative to each dataset folder, `data/rehab/` or `data/intelli/`. The `per_exercise/` directory holds the per-gesture results for IntelliRehabDS.

| To inspect | Files |
|---|---|
| Main performance table and full metric summaries | `manuscript/metric_summary_all_regimes.csv` |
| Held-out scores and classifications for each regime | `{generalized,per_exercise,individualized}/predictions.csv` and `unit_metrics.csv` |
| Exercise- or gesture-specific performance | `per_exercise/metric_summary_by_exercise.csv` |
| Matched model comparisons | `manuscript/paired_comparisons_all_regimes.csv`, `per_exercise/across_exercise_comparison.csv`, and `individualized/participant_sensitivity.csv` |
| Outer and inner split membership, unit counts, and exclusions | `splits/` and `data_audit/` |
| Hyperparameters selected for each fit and candidate results | `manuscript/selected_parameters_all_regimes.csv` and `tuning_candidates.csv` in each regime |
| Movement-quality descriptor comparisons | `generalized/mqm/mqm_class_comparisons.csv` |
| Individualized attribution summaries | `individualized/explanations/` |
| Test-time skeleton corruption experiments | `generalized/robustness/severity_summary.csv` and `slopes.csv` |
| Additional classifiers and temporal-order stress tests | `supplement_cross_method/metric_summary.csv` and `shuffle_participant_sensitivity.csv` |
| Run configuration, software versions, and fingerprints | `manifest.json` |

## Verify the reported metrics

Each prediction row identifies the held-out repetition (`trial_id`), model, and evaluation unit (`unit`). `y_true` is the true label, `p_correct` is the predicted probability of correct execution, and `y_hat` is the predicted class. Correct execution is class `1`. The decision threshold is `p_correct >= 0.5`.

The three model identifiers are `LR_angles` for trajectory logistic regression, `LR_features` for descriptor logistic regression, and `DA_TCN` for the dual-attention temporal convolutional network.

| Regime | REHAB24-6 | IntelliRehabDS | How to aggregate |
|---|---:|---:|---|
| Generalized | 5 outer folds | 5 outer folds | Average the metrics across subject-disjoint outer folds. |
| Per-exercise or per-gesture | 30 eligible exercise–fold units | 42 eligible gesture–fold units | Average the metrics across eligible task–fold units. For task-specific results, average within each task. |
| Individualized | 46 participant–exercise pairs | 16 participant–gesture pairs | Average the 30 holdouts within each pair, then give each pair equal weight. |

`metric_audit.json` records the reconstruction of accuracy, balanced accuracy, precision, sensitivity, specificity, F1, and area under the receiver operating characteristic curve from the saved predictions. The largest absolute discrepancy in the audit was approximately `2.22e-16`.

From the repository root, install the audit dependencies with `python -m pip install pandas numpy scikit-learn`, then run the check below. If the package folders are at the repository root, set `package = Path(".")`.

```bash
python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

package = Path("reproducibility")
metrics = [
    "accuracy", "balanced_accuracy", "precision",
    "sensitivity", "specificity", "f1", "auc",
]

for dataset in ("rehab", "intelli"):
    data = package / "data" / dataset
    exported = pd.read_csv(data / "manuscript/metric_summary_all_regimes.csv")

    for regime in ("generalized", "per_exercise", "individualized"):
        predictions = pd.read_csv(data / regime / "predictions.csv")
        key = ["unit", "model", "trial_id"]
        assert not predictions.duplicated(key).any()
        assert predictions.groupby(["unit", "trial_id"])["model"].nunique().eq(3).all()

        rows = []
        for (unit, model), group in predictions.groupby(["unit", "model"]):
            actual = group["y_true"].to_numpy() == 1
            predicted = group["p_correct"].to_numpy() >= 0.5
            np.testing.assert_array_equal(predicted, group["y_hat"].to_numpy())

            tp = np.count_nonzero(actual & predicted)
            tn = np.count_nonzero(~actual & ~predicted)
            fp = np.count_nonzero(~actual & predicted)
            fn = np.count_nonzero(actual & ~predicted)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            rows.append({
                "unit": unit,
                "model": model,
                "subject": group["subject"].iloc[0],
                "exercise": group["exercise"].iloc[0],
                "accuracy": (tp + tn) / len(group),
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "auc": roc_auc_score(actual, group["p_correct"]),
            })

        calculated = pd.DataFrame(rows)
        expected = pd.read_csv(data / regime / "unit_metrics.csv")
        index = ["unit", "model"]
        observed = calculated.set_index(index)[metrics].sort_index()
        expected = expected.set_index(index)[metrics].sort_index()
        pd.testing.assert_index_equal(observed.index, expected.index)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-10)

        if regime == "individualized":
            calculated = (
                calculated.groupby(["model", "subject", "exercise"])[metrics]
                .mean()
                .reset_index()
            )

        expected_summary = exported[exported["regime"] == regime]
        groups = calculated.groupby("model")[metrics]
        for column, observed_summary in (("mean", groups.mean()), ("sd", groups.std())):
            reference = expected_summary.pivot(index="model", columns="metric", values=column)
            reference = reference.loc[observed_summary.index, metrics]
            np.testing.assert_allclose(observed_summary, reference, rtol=0, atol=1e-10)

        print(f"{dataset} / {regime}: seven metrics, means and SDs verified")
PY
```

For the pooled ROC curves and confusion counts in the two cross-subject regimes, combine held-out predictions by model within each regime. For individualized figures, average the holdout results within each eligible pair before averaging across pairs. A pooled ROC area can differ from the mean of fold-level ROC areas in the performance table.

## Rerun the analyses

1. Download the public datasets from [REHAB24-6](https://zenodo.org/records/13305826) and [IntelliRehabDS](https://zenodo.org/records/4610859).
2. Open the corresponding executed notebook. For REHAB24-6, set `DATA_ROOT` to the directory containing `Segmentation.csv`, `joints_names.txt`, and `2d_joints/`. For IntelliRehabDS, set `DATA_ROOT` to the extracted `Simplified/` directory and update `PROJECT_ROOT` if using the notebook's default path construction. Set `RESULTS_ROOT` to a writable location for the new run.
3. Use a CUDA-capable PyTorch environment for the full experiment. The manifests record the original Python and package versions. The notebooks use NumPy, pandas, SciPy, scikit-learn, PyTorch, Optuna, Matplotlib, joblib, and tqdm, with XGBoost for the supplementary model comparisons.
4. Run from the setup cell with `PROFILE = "full"`. Leave `RUN_CROSS_METHOD = True` to include the additional classifiers and the temporal-shuffle experiment. The `preview` profile uses smaller search budgets and fewer repetitions for checking setup.
5. Compare the new exports with the corresponding manifests, split files, and metric summaries. Neural training can vary with hardware and software versions.

Both notebooks use ten joint-angle trajectories over 100 normalized frames. The two logistic regression variants and the temporal network are tuned independently within the outer training data in all three regimes. Individualized fits use labelled repetitions from the same participant and task. The corruption analysis reuses the clean generalized fits for each model.

The public dataset records provide the original recordings and their terms of use. Please cite the accompanying manuscript and the relevant dataset publication when using these materials.
