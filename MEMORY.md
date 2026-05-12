# Project Memory

This file is the persistent working memory for Codex sessions in this repository. Read it at the beginning of every conversation before making assumptions about recent work, plans, or decisions. Update it whenever a conversation changes project direction, locks in a decision, discovers important experimental context, or creates a follow-up plan.

## How To Use This File
- Keep entries concise and factual.
- Prefer dated bullets with enough context to understand why the note matters.
- Move outdated notes to `Archived Notes` instead of deleting useful history.
- Do not store secrets, credentials, private tokens, or sensitive subject-level data.
- When updating this file, preserve existing notes unless they are clearly obsolete.

## Current Focus
- Build a general graph foundation model (GFM) for eye-tracking data that can infer physiological and psychological states.
- Develop the model step by step, compare against classical ML baselines, and keep experiments reproducible.
- Current diploma framing is narrower than the long-term GFM goal: develop and evaluate a spatio-temporal GNN for emotion/affective-state recognition from MAHNOB-HCI-TAGGING.

## Locked Decisions
- Use the `gfm` conda environment for Python work.
- Ignore files under `archive/`.
- Treat `diploma_knowledge_base.md` as the live project knowledge base as of 2026-05-02; keep it updated when project direction, architecture decisions, experiment context, or writing plans change. The older `diploma_knowledge_base_02_05_2026.md` was renamed to `diploma_knowledge_base.md` in git.
- Keep scripts configurable with sensible defaults and log final arguments at startup.
- Normalize confusion-matrix rows to per-class percentages and use a fixed color scale from 0.0 to 1.0.
- Use the `Blues` color scheme for heatmaps.

## Recent Notes
- 2026-05-04: Created this memory file and added repo instructions to read it at the start of each conversation and update it when important project context changes.
- 2026-05-05: User wants next model work to prioritize incremental, modular upgrades: MLP pooling/fusion of spatial and temporal node representations, MLP pooling of nodes into graph embeddings, separate temporal forward/backward/spatial edge types, and learned edge weights from `[t_i, t_j, x_i, x_j, y_i, y_j]` using an MLP with layers `6 -> 6 -> 4 -> 2 -> 1`.
- 2026-05-05: Updated edge-weight plan for fastest path to a stronger working model: use relation features such as `[t_i, t_j, delta_t, delta_x, delta_y, distance]`; normalize edge weights per target node; use separate weight MLPs for spatial vs temporal edges; use the same temporal weight MLP for forward/backward edges with direction encoded.
- 2026-05-05: Locked edge-weight details: allow signed weights, but normalize signed incoming weights per target node by signed score magnitude; use spatial MLP `6 -> 6 -> 4 -> 2 -> 1`; use temporal MLP `7 -> 6 -> 4 -> 2 -> 1` by adding a direction feature.
- 2026-05-05: Implemented GNN v2 architecture in four commits: frozen `SpatioTemporalHeteroGNNV1`, v2 split temporal forward/backward graph schema, relation concat+MLP fusion, attention graph pooling, and learned signed normalized edge weights. Added quick Table-6 arousal comparison runner at `src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py`.
- 2026-05-07: Quick v1/v2 comparison now includes `Random` and `Majority` multiclass baselines by default. Comparison plots use the fixed display order `Random`, `Majority`, `GNN_v1`, `GNN_v2`, `MLP`, then remaining trained models alphabetically.
- 2026-05-07: Quick v1/v2 comparison is YAML-first: subjects, recordings, CV settings, graph settings, cache settings, and epochs should be controlled in the suite YAML by default. CLI flags only apply optional explicit overrides.
- 2026-05-07: Quick v1/v2 comparison groups all requested baseline models into one suite/trainer invocation so baselines share one dataset load and one CV split construction. GNN v1 and GNN v2 remain separate invocations because their architecture configs differ.
- 2026-05-07: Debugged quick v1/v2 comparison failure from PyTorch Inductor during `torch.compile` backward graph compilation on dynamic PyG batches. The quick runner now disables `use_torch_compile` by default, offers `--use-torch-compile` for explicit re-enable, and mirrors stdout/stderr to `quick_v1_v2_comparison.log` in each timestamped results folder.
- 2026-05-07: Completed LOSO Table-6 arousal quick comparison with `--models GNN_v1,GNN_v2,LightGBM,random,majority --num-epochs 30 --cv-strategy subject_loo`. Results folder: `results/quick_v1_v2_comparison/2026-05-07_14-54-32`. Aggregated balanced accuracy: `GNN_v2=0.4740`, `GNN_v1=0.4603`, `LightGBM=0.3854`, `Majority=0.3333`, `Random=0.3068`.
- 2026-05-11: Quick v1/v2 comparison now accepts multiple comma-separated CV strategies in one command, for example `--cv-strategy subject_loo,recording_loo` or `--cv-strategy subject_kfold,recording_kfold --n-splits 3`. The summary CSV stores one row per `(cv_strategy, model)`, ranking plots show one panel per strategy, and multi-strategy confusion matrices are saved as `figures/confusion_matrices_<strategy>.png`.
- 2026-05-11: Quick v1/v2 comparison now saves label-distribution/class-balance outputs for completed runs: `tables/label_distribution_by_fold.csv`, `tables/label_distribution_aggregate.csv`, `plots/label_distribution_counts.png`, and `plots/label_distribution_proportions.png`.
- 2026-05-11: Added explicit `dataset.exclude_subjects` support across snapshot building, graph loading, tabular sample building, and cache keys. Default HCI configs now conservatively exclude `P9`, `P12`, and `P15` to align with the MAHNOB paper's excluded participants under current uncertainty about the reduced local ET-only copy. Added root report `MAHNOB_dataset_report.md` with verified local counts and open uncertainties.
- 2026-05-11: `dataset.min_samples_per_window` now controls both baseline window filtering and GNN graph-window filtering. Previously GNN still hardcoded the minimum to `max(kt, ks) + 1`; current HCI/Table-6 suite defaults were raised to `60`.
- 2026-05-12: Added explicit dual pooling controls for GNN v2. `graph_pooling` now controls node-to-graph pooling (`mean`/`mean_max`/`attention`), and new `relation_pooling` controls per-layer spatial/temporal relation fusion (`attention` or `mlp`). Defaults are now attention for both in v2, while `pooling` is kept as a backward-compatible alias for graph pooling.
- 2026-05-12: Updated pooling defaults per user preference for active v2 work: graph/head pooling defaults to attention, and relation pooling defaults to concat+MLP (`relation_pooling: mlp`). Added optional `head_pooling` config alias that maps to graph-level pooling for clearer intent, and wired these defaults into quick v1/v2 comparison overrides and base suite configs.
- 2026-05-12: Consolidated quick-comparison config ownership under `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/`. `run_quick_v1_v2_comparison.py` now defaults to `.../quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`; that wrapper and `run_hci_experiment_suite_small.yaml` both point to local quick trainer configs (`quick_v1_v2_train_binary|multiclass|regression_hci_tagging.yaml`). Archived unused legacy config `src/emotions/multiclass/configs/train_multiclass_hci_tagging_emotion_id_legacy.yaml` to `archive/src/emotions/multiclass/configs/`.
- 2026-05-12: Simplified quick v1/v2 comparison config ownership: Python variant overrides now only select variant identity (`model_version`, `graph_version`, edge-weight mode, and baseline/GNN enable flags). Shared architecture knobs, especially `gnn.model.conv_type`, live in `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`. Default v1/v2 comparison uses the same convolution; switch both between `GCNConv` and `GATConv` by changing that wrapper YAML. `GATConv` ignores scalar edge weights in both v1 and v2 model code.
- 2026-05-12: Prepared the active quick v1/v2 Table-6 comparison wrapper for a GAT run by setting `global_overrides.gnn.model.conv_type: GATConv` in `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`.

## Open Plans
- Next priority: inspect LOSO quick comparison confusion matrices and ranking plot from `results/quick_v1_v2_comparison/2026-05-07_14-54-32`, then decide whether to tune v2 or broaden to ablations.

## Experiment Context
- For questions about recent experiments, trainings, models, or data, check the latest git commit(s) and explicitly state the assumption that the user likely means the most recently modified experiment context.

## Archived Notes
- None yet.
