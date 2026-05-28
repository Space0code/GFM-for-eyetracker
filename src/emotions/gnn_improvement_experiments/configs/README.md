# GNN Improvement Experiment Configs

This directory is the canonical config home for scripts under
`src/emotions/gnn_improvement_experiments`.

## Which script uses which config

- `run_quick_v1_v2_comparison.py`
  - Default wrapper config:
    `quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`
  - Base trainer configs used by that wrapper:
    - `quick_v1_v2/quick_v1_v2_train_binary_hci_tagging.yaml`
    - `quick_v1_v2/quick_v1_v2_train_multiclass_hci_tagging.yaml`

- `ablations/run_signal_ablation_suite.py`
  - Default wrapper config:
    `quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`
  - Use `--base-config` with the low/high wrapper for additional low/high
    ablation runs.

## Notes

- Keep quick-run and active ablation configs here to avoid cross-folder path ambiguity.
- In filenames, `hci_tagging` means the MAHNOB-HCI-TAGGING dataset. It does
  not mean the image/video/pooled tagging experiment scopes unless the wrapper
  explicitly selects those scopes.
- If you update trainer defaults for quick/ablation runs, update files in
  `quick_v1_v2/` first.
- Regression is still supported by the main suite config and trainer, but the
  quick/Table-6 wrappers do not declare a regression base config because they do
  not run regression experiments.
- For quick v1/v2 comparison runs, shared architecture choices belong in the
  wrapper config `quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`
  under `global_overrides.gnn.model`. In particular, switch both GNN_v1 and
  GNN_v2 between GCN and GAT by changing only `conv_type` there.
- The Table-6 multiclass quick configs enable `benchmarking` by default. During
  a normal quick-comparison run, the multiclass trainer writes per-fold
  `model_benchmark.json` files with parameter counts, training time, inference
  time, and the fold's accuracy/macro-F1. Each trainer strategy also writes
  `model_benchmark_raw.csv` and `model_benchmark_summary.csv`. The quick
  comparison runner then aggregates these into `tables/model_benchmark_raw.csv`,
  `tables/model_benchmark_summary.csv`, and the thesis-facing
  `tables/main_model_complexity_report.csv`/`.md`.
- Benchmark timings are measured during the current run, not recovered from old
  `.log` files. Training time covers model fitting for the fold, not dataset or
  graph construction. Inference time is measured after training on the fold's
  test split with warmup and repeated timed passes. For `GazeMAE_MLP`, raw
  artifacts include both cached-embedding head-only inference and a synthetic
  frozen-encoder-plus-head forward timing; the main report uses the encoder-plus-
  head timing when available.
- The old `run_gnn_ablation_suite.py` hyperparameter sweep was archived under
  `archive/src/emotions/gnn_improvement_experiments/`.
