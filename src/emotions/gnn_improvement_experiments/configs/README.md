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
    - `quick_v1_v2/quick_v1_v2_train_regression_hci_tagging.yaml`

- `run_gnn_ablation_suite.py`
  - Default wrapper config:
    `run_hci_experiment_suite_small.yaml`
  - This wrapper also points to the same `quick_v1_v2/` base trainer configs.

## Notes

- Keep quick-run and ablation configs here to avoid cross-folder path ambiguity.
- If you update trainer defaults for quick/ablation runs, update files in
  `quick_v1_v2/` first.
