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

- `run_gnn_ablation_suite.py`
  - Default wrapper config:
    `run_hci_experiment_suite_small.yaml`
  - This wrapper points to the same binary/multiclass `quick_v1_v2/` base
    trainer configs.

## Notes

- Keep quick-run and ablation configs here to avoid cross-folder path ambiguity.
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
