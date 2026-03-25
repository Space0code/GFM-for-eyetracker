# RETAIN Suite Summary (2026-03-05 13:25:29)
Various classification and regression experiments run on the full hci-tagging subject set (no subject filter).

## Regression Check
- Verdict: Regression quality is still modest overall, but stronger than the subset RETAIN run.
- Best observed regression run by MAE: `regression_emotion_arousal_emotion-elicitation` with `GNN` (MAE=1.7903, CCC=0.1291, Spearman=0.2008).
- Metric bests among non-Mean models: CCC -> `regression_emotion_arousal_emotion-elicitation` `MLP` (0.1392); Spearman -> `regression_emotion_arousal_emotion-elicitation` `MLP` (0.2382); MAE (lower better) -> `regression_emotion_arousal_emotion-elicitation` `GNN` (1.7903).

## Table 1: GNN Across All Classification Experiments

| suite_experiment_id | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| binary_emotion_valence_emotion-elicitation | 0.5808 | 0.5169 | 0.5266 |
| binary_emotion_arousal_emotion-elicitation | 0.5882 | 0.5856 | 0.5197 |
| binary_emotion_control_emotion-elicitation | 0.6323 | 0.5340 | 0.5834 |
| binary_emotion_predictability_emotion-elicitation | 0.5442 | 0.5352 | 0.6090 |
| binary_tag_agree_image-tagging-1 | 0.6037 | 0.6945 | 0.3533 |
| binary_tag_agree_image-tagging-2 | 0.5583 | 0.7163 | 0.3706 |
| binary_tag_agree_video-tagging | 0.5848 | 0.5151 | 0.3584 |
| binary_tag_agree_pooled-tagging | 0.6188 | 0.6724 | 0.3837 |
| binary_tag_valid_image-tagging-1 | 0.5340 | N/A | 0.3456 |
| binary_tag_valid_image-tagging-2 | 0.5399 | N/A | 0.3380 |
| binary_tag_valid_video-tagging | 0.5995 | N/A | 0.3528 |
| binary_tag_valid_pooled-tagging | 0.5872 | N/A | 0.3665 |
| multiclass_emotion_id_emotion-elicitation | 0.2070 | 0.1704 | N/A |
| multiclass_va_quadrant_emotion-elicitation | 0.3959 | 0.3229 | N/A |

## Table 2: Per-Experiment (Mean vs GNN vs Best of SVM/LightGBM/MLP by Balanced Accuracy)

### binary_emotion_valence_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.4042 | 0.5000 | 0.4616 |
| GNN | 0.5808 | 0.5169 | 0.5266 |
| SVM (best non-GNN by balanced_accuracy) | 0.5886 | 0.5235 | 0.5225 |

### binary_emotion_arousal_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.4032 | 0.5000 | 0.2934 |
| GNN | 0.5882 | 0.5856 | 0.5197 |
| MLP (best non-GNN by balanced_accuracy) | 0.5975 | 0.5852 | 0.5304 |

### binary_emotion_control_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.6559 | 0.5000 | 0.6119 |
| GNN | 0.6323 | 0.5340 | 0.5834 |
| LightGBM (best non-GNN by balanced_accuracy) | 0.6210 | 0.5374 | 0.5672 |

### binary_emotion_predictability_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5340 | 0.5000 | 0.6735 |
| GNN | 0.5442 | 0.5352 | 0.6090 |
| SVM (best non-GNN by balanced_accuracy) | 0.5193 | 0.5263 | 0.6128 |

### binary_tag_agree_image-tagging-1

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5085 | 0.5000 | 0.0000 |
| GNN | 0.6037 | 0.6945 | 0.3533 |
| SVM (best non-GNN by balanced_accuracy) | 0.6567 | 0.7424 | 0.3606 |

### binary_tag_agree_image-tagging-2

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.3557 | 0.5000 | 0.0056 |
| GNN | 0.5583 | 0.7163 | 0.3706 |
| SVM (best non-GNN by balanced_accuracy) | 0.6428 | 0.6947 | 0.3784 |

### binary_tag_agree_video-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.1823 | 0.5000 | 0.1837 |
| GNN | 0.5848 | 0.5151 | 0.3584 |
| SVM (best non-GNN by balanced_accuracy) | 0.4830 | 0.6428 | 0.3222 |

### binary_tag_agree_pooled-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5264 | 0.5000 | 0.0000 |
| GNN | 0.6188 | 0.6724 | 0.3837 |
| SVM (best non-GNN by balanced_accuracy) | 0.6290 | 0.6515 | 0.3588 |

### binary_tag_valid_image-tagging-1

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.2500 | N/A | 0.2500 |
| GNN | 0.5340 | N/A | 0.3456 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### binary_tag_valid_image-tagging-2

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.2143 | N/A | 0.2143 |
| GNN | 0.5399 | N/A | 0.3380 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### binary_tag_valid_video-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.1786 | N/A | 0.1786 |
| GNN | 0.5995 | N/A | 0.3528 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### binary_tag_valid_pooled-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.2024 | N/A | 0.2024 |
| GNN | 0.5872 | N/A | 0.3665 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### multiclass_emotion_id_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.0629 | 0.1117 | N/A |
| GNN | 0.2070 | 0.1704 | N/A |
| MLP (best non-GNN by balanced_accuracy) | 0.1846 | 0.1526 | N/A |

### multiclass_va_quadrant_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.1868 | 0.2083 | N/A |
| GNN | 0.3959 | 0.3229 | N/A |
| MLP (best non-GNN by balanced_accuracy) | 0.3833 | 0.3215 | N/A |

