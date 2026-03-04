# Binary Classification for Emotion Recognition on eSEEd_v2 dataset

This module extends the emotion recognition framework to support binary classification tasks.

## Overview

The original framework predicts emotion intensities on a 0-10 scale (regression). This binary classification module simplifies the task to:
- **Label 0**: Emotion intensity ≤ threshold
- **Label 1**: Emotion intensity > threshold

This approach is useful when many samples have zero intensity and the goal is to detect emotion presence/absence rather than predict exact intensity.

## Features

- **Single emotion focus**: Train models to predict one emotion at a time
- **Configurable threshold**: Adjust the intensity threshold for binary classification
- **Wrapper architecture**: Extends existing datasets without modifying core code
- **Full model support**: Binary versions of all baseline models + GNN
- **Binary metrics**: Accuracy, Precision, Recall, F1, AUC-ROC

## File Structure

```
src/emotions/binary/
├── __init__.py
├── data_binary.py              # Binary dataset wrappers
├── model_binary.py             # Binary GNN model
├── baseline_model_binary.py    # Binary baseline models
├── metrics_binary.py           # Binary classification metrics
├── train_binary.py             # Main training script
├── configs/
│   └── train_binary.yaml       # Configuration file
└── README.md                   # This file
```

## Quick Start

### 1. Configure Your Experiment

Edit `configs/train_binary.yaml`:

```yaml
# Select target emotion and threshold
binary_task:
  target_emotion: "emotion-anger"  # Options: emotion-anger, emotion-tenderness, emotion-sadness, emotion-disgust
  threshold: 0.0  # Intensity threshold for binary classification

# Select models to run
run_experiments:
  baselines: true
  gnn: true

# Other settings (dataset, CV strategy, etc.)
```

### 2. Run Training

```bash
# Activate conda environment
conda activate gfm

# Run binary classification training
python src/emotions/binary/train_binary.py --config src/emotions/binary/configs/train_binary.yaml
```

### 3. View Results

Results are saved to `results/binary/<timestamp>/`:
- `training_log.txt`: Full training log
- `<strategy>/summary.csv`: Performance metrics by fold
- `<strategy>/<fold_id>/`: Per-fold results and model checkpoints

## Configuration Options

### Binary Task Configuration

```yaml
binary_task:
  target_emotion: "emotion-anger"  # Which emotion to predict
  threshold: 0.0                   # Binary threshold
```

### Available Emotions
- `emotion-anger`
- `emotion-tenderness`
- `emotion-sadness`
- `emotion-disgust`

### Threshold Selection
- `threshold: 0.0` → Detect any non-zero emotion intensity (most common)
- `threshold: 5.0` → Detect high-intensity emotions only
- The threshold determines class balance

### Cross-Validation Strategies

```yaml
cross_validation:
  strategies:
    - subject_loo     # Leave-one-subject-out
    - recording_loo   # Leave-one-recording-out
    - combined_loo    # Leave-one-pair-out
  val_size: 1
  random_state: 42
```

### Model Configuration

**GNN Model:**
```yaml
gnn:
  model:
    in_channels: 5
    hidden_channels: 64
    # Binary output is automatic (out_channels=1)
  training:
    num_epochs: 100
    batch_size: 32
    learning_rate: 0.001
```

**Baseline Models:**
```yaml
baselines:
  models:
    - Mean        # Predicts mean probability
    - SVM         # SVM classifier with RBF kernel
    - LightGBM    # Gradient boosting classifier
    - MLP         # Neural network classifier
```

## Binary Metrics

The module computes standard binary classification metrics:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (requires probability outputs)

Metrics are computed at two levels:
1. **Sample-level**: Metrics on individual windows
2. **Pair-aggregated**: Metrics averaged per (subject, recording) pair

## Implementation Details

### Data Wrappers

**BinarySpacioTemporalDataset**: Wraps graph dataset
- Converts multi-emotion targets to binary
- Selects single target emotion
- Keeps graph structure intact

**BinaryTabularSample**: Wraps tabular samples
- Converts features to binary labels
- Preserves metadata (subject, recording)

### Model Architecture

**BinarySpatioTemporalGNN**: Binary classification GNN
- Inherits from base `SpatioTemporalHeteroGNN`
- Output: Single probability value [0, 1]
- Loss: Binary cross-entropy

**Binary Baselines**:
- All use sklearn classifiers (not regressors)
- Output probabilities via `predict_proba()`
- Support same hyperparameters as regression versions

### Loss Function

**GNN**: Binary cross-entropy with logits (model outputs raw logits)
```python
loss = F.binary_cross_entropy_with_logits(logits, targets)
# Sigmoid applied only during evaluation to get probabilities
prob = torch.sigmoid(logits)
```

**Baselines**: Use sklearn's built-in loss functions

## Class Distribution

The module prints class distribution at startup:

```
Class distribution: 2400 negative (<=threshold), 2600 positive (>threshold) [52.0%]
```

This helps verify the threshold creates balanced classes (~50%/50%).

## Example Workflow

1. **Start with emotion-anger and threshold=0**
   ```yaml
   binary_task:
     target_emotion: "emotion-anger"
     threshold: 0.0
   ```

2. **Run subject-level cross-validation**
   ```yaml
   cross_validation:
     strategies: [subject_loo]
   ```

3. **Compare models**
   ```bash
   python src/emotions/binary/train_binary.py
   ```

4. **Try other emotions**
   - Change `target_emotion` in config
   - Rerun training

5. **Experiment with thresholds**
   - Try `threshold: 1.0`, `5.0`, etc.
   - Compare performance

## Differences from Regression Framework

| Aspect | Regression | Binary Classification |
|--------|-----------|----------------------|
| **Output** | 4 emotions × [0-10] | Single probability [0-1] |
| **Loss** | MSE | Binary cross-entropy |
| **Metrics** | MAE, MSE, CCC, Spearman | Accuracy, F1, AUC |
| **Models** | SVR, Regressor | SVC, Classifier |
| **Target** | All emotions | One emotion |

## Tips

- **Start simple**: Use threshold=0 and emotion-anger
- **Check class balance**: Ensure ~50%/50% split
- **Compare all models**: Different models work better for different emotions
- **Use subject_loo**: Most realistic evaluation for new subjects
- **Monitor overfitting**: Compare train vs. validation accuracy

## Troubleshooting

**Issue**: Low accuracy (~50%)
- **Solution**: Models may be predicting mean. Try increasing training epochs or adjusting learning rate.

**Issue**: Class imbalance warning
- **Solution**: Adjust threshold to balance classes or use a different emotion.

**Issue**: Import errors
- **Solution**: Ensure conda environment `gfm` is activated and project root is in PYTHONPATH.

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in config (e.g., from 32 to 16).

## Next Steps

After running binary classification:
1. Compare results with regression approach
2. Try different emotions to see which are easier to predict
3. Experiment with threshold values
4. Consider multi-class classification (low/medium/high intensity)
5. Analyze misclassified samples to understand model failures

## Citation

If you use this binary classification module in your research, please cite the original GFM framework and mention the binary extension.
