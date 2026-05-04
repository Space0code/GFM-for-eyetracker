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

## Locked Decisions
- Use the `gfm` conda environment for Python work.
- Ignore files under `archive/`.
- Keep scripts configurable with sensible defaults and log final arguments at startup.
- Normalize confusion-matrix rows to per-class percentages and use a fixed color scale from 0.0 to 1.0.
- Use the `Blues` color scheme for heatmaps.

## Recent Notes
- 2026-05-04: Created this memory file and added repo instructions to read it at the start of each conversation and update it when important project context changes.

## Open Plans
- Keep this section updated with active implementation or experiment plans from future conversations.

## Experiment Context
- For questions about recent experiments, trainings, models, or data, check the latest git commit(s) and explicitly state the assumption that the user likely means the most recently modified experiment context.

## Archived Notes
- None yet.
