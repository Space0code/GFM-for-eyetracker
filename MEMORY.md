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

## Open Plans
- Next priority: run the quick Table-6 arousal comparison (`Random`, `Majority`, `GNN_v1`, `GNN_v2`, `LightGBM`) on a small subset, inspect metrics/runtime/logs, and only then decide whether to tune v2 or broaden to ablations.

## Experiment Context
- For questions about recent experiments, trainings, models, or data, check the latest git commit(s) and explicitly state the assumption that the user likely means the most recently modified experiment context.

## Archived Notes
- None yet.
