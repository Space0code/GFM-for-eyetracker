# Project Memory

This file is the persistent working memory for Codex sessions in this repository.
Read it at the beginning of every conversation before making assumptions about
recent work, plans, or decisions.

## Source Map

- `AGENTS.md`: stable repo instructions for assistants.
- `MEMORY.md`: current state, locked decisions, recent high-signal context, and next actions.
- `MAHNOB_dataset_report.md`: canonical local MAHNOB-HCI dataset inventory, exclusions, counts, and uncertainty notes.
- `diploma_knowledge_base.md`: thesis-facing synthesis, architecture rationale, literature notes, and writing structure.
- `docs/experiment_log.md`: detailed experiment/config/run history that should not clutter memory or the thesis knowledge base.
- `docs/diploma_reference_archive.md`: non-central reference notes moved out of the main thesis knowledge base.
- `docs/journal.md`: human-readable progress reports and experiment interpretation.

## How To Use This File

- Keep entries concise and factual.
- Store only current or high-signal context here; move detailed run history to `docs/experiment_log.md`.
- Do not store secrets, credentials, private tokens, or sensitive subject-level data.
- Update this file when a conversation changes project direction, locks in a decision, records important experimental context, or creates a follow-up plan.

## Current Focus

- Long-term goal: build a general graph foundation model (GFM) for eye-tracking data.
- Diploma scope: develop and evaluate a spatio-temporal GNN for emotion/affective-state recognition from MAHNOB-HCI-TAGGING.
- Current practical aim: make the GNN strong and interpretable enough to beat local baselines and ideally approach or improve on the MAHNOB eye-gaze paper result.
- Diploma writing has started in Overleaf, covering motivation, related work, and theory. Broader multi-dataset/GFM-style experiments are deferred to post-diploma work.

## Locked Decisions

- Use the `gfm` conda environment for Python work.
- Ignore files under `archive/`.
- Treat `diploma_knowledge_base.md` as the live thesis/project knowledge base as of 2026-05-02.
- Treat `MAHNOB_dataset_report.md` as the canonical source for local MAHNOB-HCI counts and subject-exclusion rationale.
- Keep scripts configurable with sensible defaults; terminal scripts should run as `python <script_name>.py` where feasible and log final arguments at startup.
- Normalize confusion-matrix rows to per-class percentages, use fixed color scale `[0.0, 1.0]`, and use the `Blues` color scheme for heatmaps.
- For questions about recent experiments, trainings, models, or data, check the latest git commits and explicitly state the assumption that the user likely means the most recently modified experiment context.

## Current Data Assumptions

- MAHNOB-HCI-TAGGING is the main diploma dataset.
- The repository uses a reduced local eye-tracking-focused copy, not a fully verifiable copy of the original full multimodal release.
- Default HCI configs conservatively exclude `P9`, `P12`, and `P15` via `dataset.exclude_subjects`; in the current local emotion ET copy, `P15` is already absent, so the effective exclusions are `P9` and `P12`.
- Current default training footprint after exclusion and suite filtering is documented in `MAHNOB_dataset_report.md`: 22 subjects, 436 labeled `(subject, recording)` groups, 20 emotional recordings, 2,639,048 suite-default rows, and 5,158 usable 10-second windows.
- `dataset.min_samples_per_window` controls both baseline and GNN graph-window filtering; current HCI/Table-6 suite defaults use `60`.

## Current Model Direction

- Active GNN v2 work is incremental and modular:
  - separate `temporal_forward`, `temporal_backward`, and `spatial` edge types;
  - relation-specific node representations;
  - relation fusion with concat+MLP (`relation_pooling: mlp`);
  - graph/head pooling with attention by default;
  - learned signed edge weights normalized per target node.
- Preferred edge-weight features: `[t_i, t_j, delta_t, delta_x, delta_y, distance]`, with a direction feature for temporal edges.
- Spatial edge-weight MLP: `6 -> 6 -> 4 -> 2 -> 1`.
- Temporal edge-weight MLP: `7 -> 6 -> 4 -> 2 -> 1`; forward/backward temporal edges share this MLP.
- Current Table-6 GNN v2 defaults after valence depth checks: `num_layers: 3`, `early_stopping_patience: 20`, validation-loss checkpointing via `best_model.pt`.
- 2026-05-13 mentor meeting clarified pooling: keep MLP fusion/pooling at node level, but use attention pooling at graph level rather than MLP graph pooling.

## Recent High-Signal Results

- 2026-05-07 LOSO Table-6 arousal quick comparison (`results/quick_v1_v2_comparison/2026-05-07_14-54-32`): balanced accuracy `GNN_v2=0.4740`, `GNN_v1=0.4603`, `LightGBM=0.3854`, `Majority=0.3333`, `Random=0.3068`.
- 2026-05-12 focused Table-6 valence subject-kfold matrix (`results/table6_valence_v2_checks/2026-05-12_16-09-19`): convergence comparison balanced accuracy `GNN_v1=0.5237`, current 10-layer weighted-GCN `GNN_v2=0.4983`, `LightGBM=0.4358`, `MLP=0.4175`.
- 2026-05-12 valence v2 matrix: depth sweep favored 3 layers (`0.5285`) over 1 (`0.5180`), 5 (`0.5175`), and 10 (`0.5120`); architecture sweep favored weighted GCN (`0.5107`) over unweighted GAT (`0.4967`) and unweighted GCN (`0.4960`).
- 2026-05-12 valence convergence: long fixed training without early stopping overfits; training loss keeps decreasing while validation loss worsens. Use validation-loss early stopping and prioritize shallower v2 depths.
- 2026-05-12 low-vs-high Table-6 subject-kfold (`results/quick_v1_v2_comparison/2026-05-12_18-14-03`): arousal balanced accuracy `GNN_v1=0.5211`, `MLP=0.5144`, `LightGBM=0.5046`, `GNN_v2=0.5000`; valence `GNN_v1=0.6507`, `GNN_v2=0.6234`, `LightGBM=0.6052`, `MLP=0.5890`.
- 2026-05-13 arousal low-vs-high with low-class downsampling (`results/quick_v1_v2_comparison/2026-05-13_08-48-31`): raw `{0,2}` -> encoded `{0,1}`, counts before `{0: 2360, 2: 928}`, after `{0: 928, 2: 928}`, balanced accuracy `GNN_v1=0.5472`, `MLP=0.5424`, `GNN_v2=0.5348`, `LightGBM=0.5278`.
- 2026-05-13 low/high Table-6 valence 7-fold subject-kfold (`results/quick_v1_v2_comparison/2026-05-13_09-36-39`): balanced accuracy `GNN_v2=0.6646`, `GNN_v1=0.6473`, `LightGBM=0.6104`, `MLP=0.6003`.
- 2026-05-13 label-noise proxy analysis (`results/label_noise_analysis/2026-05-13_table6_self_report_alignment`): mismatch between emotion-id-derived Table-6 targets and participant self-report rating buckets was arousal 3-class `48.6%`, arousal low/high `29.7%`, valence 3-class `27.5%`, valence low/high `3.3%`; this supports using low/high valence as the cleanest current target.

## Current Config Context

- Quick v1/v2 configs live under `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/`.
- The quick runner defaults to `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`.
- Shared architecture settings, especially `gnn.model.conv_type`, live in the wrapper YAML; Python variant overrides should only select variant identity.
- `GATConv` ignores scalar edge weights in both v1 and v2 model code, so weighted-edge comparisons should use `GCNConv` unless intentionally testing unweighted attention behavior.
- Quick comparison supports multiple comma-separated CV strategies, class-balance diagnostics, multiclass training-history aggregation, task selection through `quick_comparison.table6_tasks`, and held-out test-loss plots.

## Open Plans

- Near-term: use 3-layer v2 with validation-loss early stopping for the next core Table-6 experiments.
- Convergence follow-up from 2026-05-13 mentor meeting: run more than 3 folds, plot train/val/test loss per fold and/or aggregated with uncertainty bands, reduce LR by 10x or 100x and train longer; if still unstable, try DropEdge, PairNorm, or GraphNorm.
- Preliminarily test adding eye-tracker-to-eyes distance and fixation ID. Keep them only if they help; otherwise mention the negative/neutral preliminary result in the diploma and omit them from the main model.
- Diploma writing decision from 2026-05-13: current RV/PV research questions are acceptable for now, may later be shortened/merged, and explicit hypotheses are not needed.
- Before final reported runs, verify that preprocessing, normalization, label transforms, and any resampling/downsampling are fitted or decided using train-fold information only where applicable.
- Inspect LOSO quick comparison confusion matrices and ranking plot from `results/quick_v1_v2_comparison/2026-05-07_14-54-32`.
- Decide whether the next step is tuning v2 further, running clean ablations, or aligning the experiment story with diploma writing needs after convergence follow-up.
- Keep detailed future run notes in `docs/experiment_log.md`; keep thesis-facing conclusions in `diploma_knowledge_base.md`.

## Archived Notes

- 2026-05-13: Repo cleanup archived the legacy unified regression workflow (`train.py`, `train_gnn.py`, `param_search.py`) under `archive/src/emotions/legacy_unified_regression/`. Active training entrypoints are now the suite runner, task-specific binary/multiclass/regression trainers, and the quick Table-6 runner.
- Detailed May 2026 quick-runner and Table-6 experiment history was moved to `docs/experiment_log.md` on 2026-05-13 to keep this memory file short.
- Documentation source-of-truth cleanup on 2026-05-13 shortened `AGENTS.md`, `MEMORY.md`, and `diploma_knowledge_base.md` while preserving detailed run history in `docs/experiment_log.md`.
