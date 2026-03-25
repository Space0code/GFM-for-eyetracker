# RETAIN Suite Summary (2026-03-25 14:10:00)
Run used the "optimal" wrapper config (`num_layers=10`, `pooling=mean_max`, `edge_weights=true`, early stopping on) and enabled only:
- `binary_emotion_valence_emotion-elicitation`
- `binary_emotion_arousal_emotion-elicitation`

## Run Status
- Suite status in `suite_experiment_registry.csv`: both experiments marked as `success`.
- This run is the successful rerun of the earlier `RETAIN_2026-03-05_16-11-26` attempt (which failed with `IndexError: list index out of range`).
- Non-fatal DataLoader pin-memory thread failures occurred and were automatically recovered with safe-loader fallback (`Retrying fold in safe-loader mode`), then each run finished with `Training complete!`.

## What Was Produced
- `classification_master_comparison.csv` with aggregated metrics for both `recording_loo` and `subject_loo`.
- Per-experiment run directories with logs, figures, and split-level summaries.
- `regression_master_comparison.csv` is empty (expected, because regression experiments were disabled in this wrapper config).

## Results (Current Run)

### binary_emotion_valence_emotion-elicitation (`recording_loo`)

| model | accuracy | balanced_accuracy | f1 | auc |
|---|---:|---:|---:|---:|
| Mean | 0.4042 | 0.5000 | 0.4616 | 0.5000 |
| SVM | 0.5886 | 0.5235 | 0.5225 | 0.5416 |
| LightGBM | 0.5900 | 0.5082 | 0.5249 | 0.5393 |
| MLP | 0.5887 | 0.5104 | 0.5160 | 0.5217 |
| GNN | 0.5854 | 0.5118 | 0.5429 | 0.5181 |

### binary_emotion_valence_emotion-elicitation (`subject_loo`)

| model | accuracy | balanced_accuracy | f1 | auc |
|---|---:|---:|---:|---:|
| Mean | 0.5341 | 0.5000 | 0.6917 | 0.5000 |
| SVM | 0.6334 | 0.6304 | 0.6738 | 0.6915 |
| LightGBM | 0.6058 | 0.6037 | 0.6476 | 0.6829 |
| MLP | 0.6166 | 0.6094 | 0.6487 | 0.6697 |
| GNN | 0.6392 | 0.6354 | 0.6277 | 0.7373 |

### binary_emotion_arousal_emotion-elicitation (`recording_loo`)

| model | accuracy | balanced_accuracy | f1 | auc |
|---|---:|---:|---:|---:|
| Mean | 0.4032 | 0.5000 | 0.2934 | 0.5000 |
| SVM | 0.5785 | 0.5558 | 0.5245 | 0.5782 |
| LightGBM | 0.5862 | 0.5619 | 0.5178 | 0.5860 |
| MLP | 0.5975 | 0.5852 | 0.5304 | 0.6166 |
| GNN | 0.5730 | 0.5553 | 0.5127 | 0.6058 |

### binary_emotion_arousal_emotion-elicitation (`subject_loo`)

| model | accuracy | balanced_accuracy | f1 | auc |
|---|---:|---:|---:|---:|
| Mean | 0.3670 | 0.5000 | 0.1447 | 0.5000 |
| SVM | 0.5540 | 0.5726 | 0.5099 | 0.6174 |
| LightGBM | 0.5696 | 0.5678 | 0.5157 | 0.6068 |
| MLP | 0.5835 | 0.5640 | 0.5142 | 0.5944 |
| GNN | 0.5589 | 0.5754 | 0.5473 | 0.6205 |

## Comparison vs `suite_FULL` (overlap only)
Reference run: `results/suite_FULL/2026-03-05_13-25-29` (`recording_loo`, overlapping tasks only).

### GNN delta vs `suite_FULL` (`recording_loo`)

| suite_experiment_id | accuracy (new vs full) | balanced_accuracy (new vs full) | f1 (new vs full) | auc (new vs full) |
|---|---:|---:|---:|---:|
| binary_emotion_valence_emotion-elicitation | 0.5854 vs 0.5808 (`+0.0046`) | 0.5118 vs 0.5169 (`-0.0052`) | 0.5429 vs 0.5266 (`+0.0163`) | 0.5181 vs 0.5223 (`-0.0041`) |
| binary_emotion_arousal_emotion-elicitation | 0.5730 vs 0.5882 (`-0.0152`) | 0.5553 vs 0.5856 (`-0.0303`) | 0.5127 vs 0.5197 (`-0.0070`) | 0.6058 vs 0.6230 (`-0.0172`) |

## Brief interpretation
- The rerun is complete and valid.
- On `recording_loo`, GNN is not best in balanced accuracy for either task (`SVM` wins valence, `MLP` wins arousal).
- On `subject_loo`, GNN has the best balanced accuracy for both tasks.
