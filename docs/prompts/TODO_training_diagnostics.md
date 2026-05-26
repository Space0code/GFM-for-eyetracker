# TODO: Training Diagnostics For Table-6 Experiments

Goal: extend the Table-6 training pipeline so diagnostic artifacts are available for thesis analysis, while keeping the main diploma text concise.

## Scope

- Focus on training diagnostics, not final evaluation metrics.
- Support Table-6 3-class valence/arousal experiments.
- Track richer diagnostics during training/evaluation, but report only the most interpretable subset in the main thesis.
- Put additional diagnostic tables/plots in the appendix when useful.

## Diagnostics To Track

For all neural models:

- `train_loss`
- `val_loss`
- `best_epoch`
- `best_val_loss`
- `early_stopped`
- `epoch_runtime_seconds`
- `learning_rate`
- `grad_norm_mean`
- `grad_norm_max`

For MLP and GazeMAE_MLP:

- Keep loss curves and best epoch.
- Do not add representation diagnostics unless they are needed later.

For GNNs:

- graph embedding variance at the graph-level representation before the classification head;
- mean pairwise cosine similarity of graph embeddings;
- logit mean, standard deviation, minimum, maximum and range;
- prediction entropy from softmax probabilities;
- learned edge-weight mean, standard deviation, minimum and maximum per relation:
  `temporal_forward`, `temporal_backward`, `spatial`, `fixation`;
- optional attention-pooling entropy if attention weights are easy to expose cleanly.

## Implementation Notes

- Add diagnostic collection to the multiclass GNN path, not only the binary path.
- Use the existing `return_graph_embedding=True` model interface for graph embeddings.
- Save fold-level GNN diagnostics next to the existing `gnn_training_history.csv`.
- Keep per-epoch diagnostics lightweight. If full GNN diagnostics are too expensive every epoch, compute them on validation data at a configurable interval and always at the best checkpoint.
- Save:
  - per-fold diagnostics CSV;
  - aggregate diagnostics CSV;
  - compact plots for loss, best epoch, embedding variance/cosine similarity, logit spread and prediction entropy;
  - optional appendix plots for edge-weight distributions.
- Preserve the rule that test data is only used after model selection.

## Diploma Reporting Plan

Main text:

- mention train/validation loss curves;
- mention best epoch / early stopping;
- mention only the core GNN diagnostics: embedding variance, cosine similarity, logit spread and prediction entropy;
- state that diagnostics are auxiliary analyses, not part of the model architecture.

Appendix:

- edge-weight distributions by relation;
- additional fold-level diagnostic plots;
- any detailed representation-collapse plots.

## Acceptance Criteria

- Table-6 3-class quick runs still produce the existing metric outputs.
- GNN folds produce diagnostic artifacts without breaking existing result aggregation.
- Aggregated diagnostic tables can be used directly for thesis figures/tables.
- Main-text thesis TODO can be replaced by a concise description without over-reporting.
