# Model Zoo and Training Recipes

This appendix summarizes what models are currently trained, where they live, and how they are trained.

## 1. Current model zoo (`src/emotions/`)

| Task | GNN | Baselines |
|---|---|---|
| Binary classification | `BinarySpatioTemporalGNN` | Mean, SVM (RBF), LightGBM, MLP |
| Multiclass classification | `MulticlassSpatioTemporalGNN` | Mean, SVM (OVR), LightGBM, MLP |
| Single-target regression | `SpatioTemporalHeteroGNN` | Mean, SVR, LightGBM regressor, MLP regressor |

## 2. GNN training recipes by task

## 2.1 Binary

Script: `src/emotions/binary/train_binary.py`

- Loss: `binary_cross_entropy_with_logits`
- Prediction for metrics: `sigmoid(logits)`
- Fold label generation:
  - threshold from train split (`mean`/`median`/fixed)
  - `target > threshold` -> class 1
- Metrics: accuracy, balanced accuracy, f1, precision, recall, auc
- Best checkpoint selected by validation loss

## 2.2 Multiclass

Script: `src/emotions/multiclass/train_multiclass.py`

- Loss: `cross_entropy`
- Modes:
  - `emotion-id`
  - `va-quadrant` (LL/LH/HL/HH using train-fold thresholds)
- Predictions: `softmax(logits)`
- Metrics: accuracy, balanced accuracy, macro/weighted precision/recall/f1/auc

## 2.3 Regression

Script: `src/emotions/regression/train_regression.py`

- Loss: `mse_loss`
- Output: scalar per graph
- Metrics: MAE, CCC, Spearman (+ loss logged)

## 3. Baseline training behavior

Binary baselines (`src/emotions/binary/baseline_model_binary.py`):
- Mean: constant positive-class prior
- SVM: RBF `SVC(probability=True)` + StandardScaler
- LightGBM: `LGBMClassifier`
- MLP: 2-layer `MLPClassifier` + StandardScaler

Multiclass baselines (`src/emotions/multiclass/baseline_model_multiclass.py`):
- Mean prior classifier
- SVM OVR + probability alignment to full class set
- LightGBM multiclass/binary objective as needed
- MLP classifier + StandardScaler

Regression baselines (`src/emotions/baseline_model.py`):
- Mean regressor
- SVR (RBF) via `MultiOutputRegressor`
- LightGBM regressor per target
- MLP regressor via `MultiOutputRegressor`

## 4. Split strategies and evaluation protocols

Common splitters from `src/emotions/splits.py`:
- `subject_loo`
- `recording_loo`
- `combined_loo`
- `recording_kfold`

Important protocol guarantees:
- train/val/test group separation by strategy
- fold-specific thresholding for binary/VA tasks
- train-only scaling
- strict split matching for baseline vs GNN comparisons (binary pipeline enforces fold-signature identity)

## 5. Main orchestrators

## 5.1 Task-level runners

- Binary: `python src/emotions/binary/train_binary.py --config ...`
- Multiclass: `python src/emotions/multiclass/train_multiclass.py --config ...`
- Regression: `python src/emotions/regression/train_regression.py --config ...`

## 5.2 Full suite runner

- `src/emotions/suite/run_hci_experiment_suite.py`
- One wrapper config runs:
  - snapshot build + EDA summary
  - selected task experiments
  - suite-level comparison CSVs + plots

Default wrapper config:
- `src/emotions/suite/configs/run_hci_experiment_suite.yaml`

## 5.3 GNN ablation runner

- `src/emotions/gnn_improvement_experiments/run_gnn_ablation_suite.py`
- Generates one-factor variants (depth, pooling, edge weights, `kt/ks`, conv type, target aggregation, early stopping)
- Runs valence/arousal-focused suite variants and summarizes deltas

## 6. Historical predecessor (for context)

`src/gnext/*`:
- early next-point gaze prediction stack
- GraphSAGE-centered model (`NextPointGNN`)
- important for project history, not the current benchmark path

## 7. Minimal command cookbook

```bash
# Binary task
python src/emotions/binary/train_binary.py \
  --config src/emotions/binary/configs/train_binary_hci_tagging.yaml

# Multiclass task
python src/emotions/multiclass/train_multiclass.py \
  --config src/emotions/multiclass/configs/train_multiclass_hci_tagging.yaml

# Regression task
python src/emotions/regression/train_regression.py \
  --config src/emotions/regression/configs/train_regression_hci_tagging.yaml

# Full HCI suite
python src/emotions/suite/run_hci_experiment_suite.py \
  --config src/emotions/suite/configs/run_hci_experiment_suite.yaml

# Focused GNN ablations
python src/emotions/gnn_improvement_experiments/run_gnn_ablation_suite.py \
  --base-config src/emotions/gnn_improvement_experiments/configs/run_hci_experiment_suite_small.yaml
```
