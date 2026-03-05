# RETAIN Suite Summary (2026-03-05 13:04:55)
Various classification and regression experiments run on a subset of hci-tagging database.

## Regression Check
- Verdict: Regression experiments appear to have failed to reach quality performance.
- Best observed regression run: `regression_emotion_arousal_emotion-elicitation` with `MLP` (MAE=1.9273, CCC=0.0932, Spearman=0.1119).
- Metric bests among non-Mean models: CCC -> `regression_emotion_arousal_emotion-elicitation` `MLP` (0.0932); Spearman -> `regression_emotion_valence_emotion-elicitation` `MLP` (0.1173); MAE (lower better) -> `regression_emotion_arousal_emotion-elicitation` `MLP` (1.9273).

## Table 1: GNN Across All Classification Experiments

| suite_experiment_id | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| binary_emotion_valence_emotion-elicitation | 0.5475 | 0.5502 | 0.4741 |
| binary_emotion_arousal_emotion-elicitation | 0.5211 | 0.4958 | 0.5710 |
| binary_emotion_control_emotion-elicitation | 0.6443 | 0.6813 | 0.6359 |
| binary_emotion_predictability_emotion-elicitation | 0.5356 | 0.4891 | 0.6134 |
| binary_tag_agree_image-tagging-1 | 0.6963 | 0.7598 | 0.3588 |
| binary_tag_agree_image-tagging-2 | 0.6460 | 0.8016 | 0.3686 |
| binary_tag_agree_video-tagging | 0.6313 | 0.6799 | 0.4157 |
| binary_tag_agree_pooled-tagging | 0.6675 | 0.7308 | 0.4014 |
| binary_tag_valid_image-tagging-1 | 0.5794 | N/A | 0.3669 |
| binary_tag_valid_image-tagging-2 | 0.5891 | N/A | 0.3808 |
| binary_tag_valid_video-tagging | 0.6073 | N/A | 0.4057 |
| binary_tag_valid_pooled-tagging | 0.6149 | N/A | 0.4047 |
| multiclass_emotion_id_emotion-elicitation | 0.1801 | 0.1803 | 0.0510 |
| multiclass_va_quadrant_emotion-elicitation | 0.3284 | 0.2866 | 0.1884 |

## Table 2: Per-Experiment (Mean vs GNN vs Best of SVM/LightGBM/MLP by Balanced Accuracy)

### binary_emotion_valence_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.2478 | 0.5000 | 0.2595 |
| GNN | 0.5475 | 0.5502 | 0.4741 |
| LightGBM (best non-GNN by balanced_accuracy) | 0.5492 | 0.5598 | 0.4765 |

### binary_emotion_arousal_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5732 | 0.5000 | 0.6866 |
| GNN | 0.5211 | 0.4958 | 0.5710 |
| LightGBM (best non-GNN by balanced_accuracy) | 0.5542 | 0.5220 | 0.6030 |

### binary_emotion_control_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5629 | 0.5000 | 0.6567 |
| GNN | 0.6443 | 0.6813 | 0.6359 |
| MLP (best non-GNN by balanced_accuracy) | 0.6207 | 0.6561 | 0.5969 |

### binary_emotion_predictability_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5734 | 0.5000 | 0.6868 |
| GNN | 0.5356 | 0.4891 | 0.6134 |
| MLP (best non-GNN by balanced_accuracy) | 0.5230 | 0.5005 | 0.5827 |

### binary_tag_agree_image-tagging-1

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5769 | 0.5000 | 0.0000 |
| GNN | 0.6963 | 0.7598 | 0.3588 |
| MLP (best non-GNN by balanced_accuracy) | 0.6868 | 0.7752 | 0.3601 |

### binary_tag_agree_image-tagging-2

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5692 | 0.5000 | 0.0000 |
| GNN | 0.6460 | 0.8016 | 0.3686 |
| MLP (best non-GNN by balanced_accuracy) | 0.6594 | 0.7929 | 0.3677 |

### binary_tag_agree_video-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.1731 | 0.5000 | 0.1786 |
| GNN | 0.6313 | 0.6799 | 0.4157 |
| MLP (best non-GNN by balanced_accuracy) | 0.5472 | 0.6670 | 0.3666 |

### binary_tag_agree_pooled-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5539 | 0.5000 | 0.0000 |
| GNN | 0.6675 | 0.7308 | 0.4014 |
| MLP (best non-GNN by balanced_accuracy) | 0.6653 | 0.7669 | 0.3724 |

### binary_tag_valid_image-tagging-1

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5000 | N/A | 0.5000 |
| GNN | 0.5794 | N/A | 0.3669 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### binary_tag_valid_image-tagging-2

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.2143 | N/A | 0.2143 |
| GNN | 0.5891 | N/A | 0.3808 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### binary_tag_valid_video-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.2500 | N/A | 0.2500 |
| GNN | 0.6073 | N/A | 0.4057 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### binary_tag_valid_pooled-tagging

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.5000 | N/A | 0.5000 |
| GNN | 0.6149 | N/A | 0.4047 |
| N/A (best non-GNN by balanced_accuracy) | N/A | N/A | N/A |

_Note: balanced_accuracy unavailable for SVM/LightGBM/MLP in source file._

### multiclass_emotion_id_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.0215 | 0.0500 | 0.0039 |
| GNN | 0.1801 | 0.1803 | 0.0510 |
| MLP (best non-GNN by balanced_accuracy) | 0.1808 | 0.1768 | 0.0528 |

### multiclass_va_quadrant_emotion-elicitation

| model | accuracy | balanced_accuracy | f1 |
|---|---:|---:|---:|
| Mean | 0.2796 | 0.1711 | 0.0837 |
| GNN | 0.3284 | 0.2866 | 0.1884 |
| LightGBM (best non-GNN by balanced_accuracy) | 0.3415 | 0.2952 | 0.2045 |
