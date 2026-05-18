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
- `docs/diploma_gnn_v2_design_notes.md`: thesis-support notes focused on the newest GNN v2 component decomposition, including distance/fixation features and relation-level architecture rationale.

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
- A thesis-support design note now exists at `docs/diploma_gnn_v2_design_notes.md`; before relying on it, ask whether any newer architecture changes should be reflected.

## Locked Decisions

- Use the `gfm` conda environment for Python work.
- When an extra Python visualization/analysis package would materially help, ask before installing it; if approved, install it into the `gfm` conda environment with `pip`. On 2026-05-14, Plotly was approved and installed for interactive graph-window visualization.
- Ignore files under `archive/`.
- Treat `diploma_knowledge_base.md` as the live thesis/project knowledge base as of 2026-05-02.
- Treat `MAHNOB_dataset_report.md` as the canonical source for local MAHNOB-HCI counts and subject-exclusion rationale.
- Keep scripts configurable with sensible defaults; terminal scripts should run as `python <script_name>.py` where feasible and log final arguments at startup.
- Normalize confusion-matrix rows to per-class percentages, use fixed color scale `[0.0, 1.0]`, and use the `Blues` color scheme for heatmaps.
- For questions about recent experiments, trainings, models, or data, check the latest git commits and explicitly state the assumption that the user likely means the most recently modified experiment context.
- When making git commits, keep them granular by functionality or goal and write high-quality commit messages that are clear but not overly wordy.
- If a user prompt, comment, or question appears strange, nonsensical, or based on a possibly mistaken premise, say so directly and ask what was meant instead of forcing a sensible interpretation.

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
- 2026-05-14 GNN v2 signal extension: `distance-avg`, `fixation-duration`, `delta_distance` edge features, and sequential same-fixation edges are now intended to be enabled by default. They remain configurable for ablations. Do not use raw `fixation-index` as a numeric node feature. Keep fixation meta-nodes for future work.
- 2026-05-14 fixation-edge check: current GNN v2 has no exact duplicate edge pairs within any relation, but every sequential same-fixation edge overlaps with the corresponding temporal-forward/backward edge because `kt >= 1`. Treat this as an intentional multi-relation overlap unless ablations show overemphasis.
- 2026-05-15 GNN v2 time/edge-scale update: `time-window-normalized = (time_i - time_window_start) / window_length_from_config` is the intended node and baseline time feature; do not use per-window observed min-max normalization or absolute recording time. Learned edge features `t_i`, `t_j`, and `delta_t` use the same window-local normalized time. Active quick configs enable `use_relative_time: true` and `standardize_edge_features: true`; edge scalers are train-fold-only, relation-family-specific, and leave temporal direction unscaled.
- 2026-05-15 fixation all-to-all check on locally available HCI raw-one-format P1 sections 2/5: 10s windows had ~562 nodes on average; sequential same-fixation edges averaged ~1,015 directed edges/window, while all-to-all same-`fixation-index` edges averaged ~37,006 and reached 228,072 in one window. Treat full same-fixation cliques as likely too dense/dominant unless capped or replaced by fixation meta-nodes/top-k-within-fixation.
- 2026-05-18 GazeMAE transfer baseline decision: use frozen pretrained GazeMAE position and velocity encoders packaged locally as encoder-only weights under `models/gazemae/`, split each 10s MAHNOB window into 2s chunks at 500 Hz, pool chunk embeddings with mean+std into 512 features, and train only a PyTorch MLP head. Clip raw MAHNOB coordinates to the actual screen resolution `1280x800` with no scaling and no mean normalization; do not scale `y` to 1024 because that would distort position and velocity magnitudes.

## Recent High-Signal Results

- 2026-05-07 LOSO Table-6 arousal quick comparison (`results/quick_v1_v2_comparison/2026-05-07_14-54-32`): balanced accuracy `GNN_v2=0.4740`, `GNN_v1=0.4603`, `LightGBM=0.3854`, `Majority=0.3333`, `Random=0.3068`.
- 2026-05-12 focused Table-6 valence subject-kfold matrix (`results/table6_valence_v2_checks/2026-05-12_16-09-19`): convergence comparison balanced accuracy `GNN_v1=0.5237`, current 10-layer weighted-GCN `GNN_v2=0.4983`, `LightGBM=0.4358`, `MLP=0.4175`.
- 2026-05-12 valence v2 matrix: depth sweep favored 3 layers (`0.5285`) over 1 (`0.5180`), 5 (`0.5175`), and 10 (`0.5120`); architecture sweep favored weighted GCN (`0.5107`) over unweighted GAT (`0.4967`) and unweighted GCN (`0.4960`).
- 2026-05-12 valence convergence: long fixed training without early stopping overfits; training loss keeps decreasing while validation loss worsens. Use validation-loss early stopping and prioritize shallower v2 depths.
- 2026-05-12 low-vs-high Table-6 subject-kfold (`results/quick_v1_v2_comparison/2026-05-12_18-14-03`): arousal balanced accuracy `GNN_v1=0.5211`, `MLP=0.5144`, `LightGBM=0.5046`, `GNN_v2=0.5000`; valence `GNN_v1=0.6507`, `GNN_v2=0.6234`, `LightGBM=0.6052`, `MLP=0.5890`.
- 2026-05-13 arousal low-vs-high with low-class downsampling (`results/quick_v1_v2_comparison/2026-05-13_08-48-31`): raw `{0,2}` -> encoded `{0,1}`, counts before `{0: 2360, 2: 928}`, after `{0: 928, 2: 928}`, balanced accuracy `GNN_v1=0.5472`, `MLP=0.5424`, `GNN_v2=0.5348`, `LightGBM=0.5278`.
- 2026-05-13 low/high Table-6 valence 7-fold subject-kfold (`results/quick_v1_v2_comparison/2026-05-13_09-36-39`): balanced accuracy `GNN_v2=0.6646`, `GNN_v1=0.6473`, `LightGBM=0.6104`, `MLP=0.6003`.
- 2026-05-13 label-noise proxy analysis (`results/label_noise_analysis/2026-05-13_table6_self_report_alignment`): mismatch between emotion-id-derived Table-6 targets and participant self-report rating buckets was arousal 3-class `48.6%`, arousal low/high `29.7%`, valence 3-class `27.5%`, valence low/high `3.3%`; this supports using low/high valence as the cleanest current target.
- 2026-05-18 Table-6 valence 3-class quick comparison with newly implemented self-contained `GazeMAE_MLP` (`results/quick_v1_v2_comparison/2026-05-18_12-06-36`), requested models only and aligned baseline/GNN subject folds in one suite invocation: balanced accuracy `GazeMAE_MLP=0.5060`, `GNN_v2=0.5026`, `Random=0.3355`, `Majority=0.3333`. The run completed end-to-end after fixing the quick runner/trainer fold alignment path; GazeMAE embeddings loaded from cache and GNN graph dataset built successfully.

## Current Config Context

- Quick v1/v2 configs live under `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/`.
- Thesis-facing implementation notes for the frozen `GazeMAE_MLP` transfer baseline are in `docs/appendix/gazemae_mlp_baseline.md`; use this as the detailed code-grounded source when writing the diploma comparison section.
- The quick runner defaults to `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`.
- Shared architecture settings, especially `gnn.model.conv_type`, live in the wrapper YAML; Python variant overrides should only select variant identity.
- `GATConv` ignores scalar edge weights in both v1 and v2 model code, so weighted-edge comparisons should use `GCNConv` unless intentionally testing unweighted attention behavior.
- Quick comparison supports multiple comma-separated CV strategies, class-balance diagnostics, multiclass training-history aggregation, task selection through `quick_comparison.table6_tasks`, and held-out test-loss plots.
- Quick comparison training-progress loss plots include an aggregated split-panel plot, an aggregated combined train/validation/test plot, and per-fold combined loss plots under `plots/losses/`; aggregated loss plots use darker `mean ± std` bands and lighter min-max bands.
- `GazeMAE_MLP` is available in the quick comparison runner as a baseline model alias (`gazemae`, `gazemae_mlp`, `GazeMAE_MLP`). It uses the same suite snapshots, CV splits, Table-6 mappings, summaries, label-distribution tables, loss plots, and confusion-matrix plotting as other multiclass baselines. The Table-6 low/high wrapper lives at `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_low_high.yaml`.
- GazeMAE embedding cache reuse is intentionally stable across timestamped quick/suite runs: when a suite snapshot manifest is available, the cache key uses `snapshot_hash`/`snapshot_cache_key` rather than the generated snapshot CSV path or modification time. The key also includes GazeMAE checkpoint file hashes and preprocessing settings, so embeddings are reused only for identical data/model/preprocessing identities. This was verified with a repeated small quick-comparison run.
- GazeMAE runtime is now self-contained in this repository: `src/emotions/gazemae_model.py` implements the minimal inference encoder, and `models/gazemae/*-encoder-state.pt` stores converted encoder+bottleneck state dicts. The quick configs no longer depend on `/home/ppg/eyetracking/gazemae` at runtime.
- As of 2026-05-14, new multiclass `summary.csv` files keep only `metric_type=aggregated`; the redundant `emotion_multiclass` row was removed, and multiclass plotting reads the `aggregated` row directly.
- 2026-05-14 3-class quick comparison run `results/quick_v1_v2_comparison/2026-05-14_13-26-54` failed during `GNN_v1` subject-kfold 2 with `Pin memory thread exited unexpectedly`; `GNN_v2` never ran. Multiclass GNN training now retries this DataLoader failure with `num_workers=0`, `pin_memory=false`, `persistent_workers=false`, and the quick 3-class wrapper uses those safe defaults.

## Open Plans

- Near-term: use 3-layer v2 with validation-loss early stopping for the next core Table-6 experiments.
- Run full comparable `GazeMAE_MLP` quick comparison for low/high Table-6 targets after the 2026-05-18 3-class valence run. Frame it as a pretrained gaze representation transfer baseline, not a full SOTA reproduction.
- Convergence follow-up from 2026-05-13 mentor meeting: run more than 3 folds, plot train/val/test loss per fold and/or aggregated with uncertainty bands, reduce LR by 10x or 100x and train longer; if still unstable, try DropEdge, PairNorm, or GraphNorm.
- Preliminarily test the default extended GNN v2 with `distance-avg`, `fixation-duration`, `delta_distance` edge features, and sequential same-fixation edges against the old core-feature version via ablations.
- Diploma writing decision from 2026-05-13: current RV/PV research questions are acceptable for now, may later be shortened/merged, and explicit hypotheses are not needed.
- Before final reported runs, verify that preprocessing, normalization, label transforms, and any resampling/downsampling are fitted or decided using train-fold information only where applicable.
- Inspect LOSO quick comparison confusion matrices and ranking plot from `results/quick_v1_v2_comparison/2026-05-07_14-54-32`.
- Decide whether the next step is tuning v2 further, running clean ablations, or aligning the experiment story with diploma writing needs after convergence follow-up.
- Keep detailed future run notes in `docs/experiment_log.md`; keep thesis-facing conclusions in `diploma_knowledge_base.md`.

## Visualization Ideas To Implement Later

- Method figures: graph construction from one 10-second gaze window; GNN v2 architecture diagram; raw MAHNOB-to-window-to-graph pipeline.
- GNN graph-window visualization notebook should show PyG `HeteroData` windows with node positions from `x-avg`/`y-avg`, node size from mean left/right pupil size, local node-index labels starting at 0, and distinct curved edge colors for relation types (`temporal`, `spatial`, `temporal_forward`, `temporal_backward`, optionally `fixation`) so multi-relational edges between the same nodes remain visible. Edge alpha defaults to `0.60`; hover still includes `distance-avg` in centimeters. For plotting only, exact/reverse duplicates within the same relation are collapsed by default, while diagnostics report both directed and displayed edge counts. For readability, it defaults to one reproducibly random `(subject, recording)` subset, shorter 2-second visualization windows, and draws all nodes instead of arbitrary subsets of a dense 10-second training window.
- Data figures: class distributions, subject/recording usable-window coverage, signal distributions and missingness for gaze, pupil, `distance-avg`, and `fixation-duration`.
- Representation figures: PCA/UMAP of raw window features, GNN graph embeddings, and possibly GazeMAE embeddings; always compare coloring by target label and by subject to expose possible subject confounds.
- Result figures: row-normalized confusion matrices using `Blues` and fixed `[0, 1]` scale; model ranking plots with fold uncertainty; train/validation/test loss curves.
- Interpretability case studies: one correct and one incorrect window with gaze path, prediction, true label, attention weights, and optionally learned edge weights. Treat these as model-attribution views, not causal evidence.

## Archived Notes

- 2026-05-13: Repo cleanup archived the legacy unified regression workflow (`train.py`, `train_gnn.py`, `param_search.py`) under `archive/src/emotions/legacy_unified_regression/`. Active training entrypoints are now the suite runner, task-specific binary/multiclass/regression trainers, and the quick Table-6 runner.
- Detailed May 2026 quick-runner and Table-6 experiment history was moved to `docs/experiment_log.md` on 2026-05-13 to keep this memory file short.
- Documentation source-of-truth cleanup on 2026-05-13 shortened `AGENTS.md`, `MEMORY.md`, and `diploma_knowledge_base.md` while preserving detailed run history in `docs/experiment_log.md`.
