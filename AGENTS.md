# AGENTS.md

Canonical source for stable assistant instructions in this repository. Keep
current state in `MEMORY.md`, dataset facts in `MAHNOB_dataset_report.md`,
thesis synthesis in `diploma_knowledge_base.md`, and detailed run history in
`docs/experiment_log.md`. Non-central thesis references live in
`docs/diploma_reference_archive.md`.

## General
- You are datascience coding assistant
- At the beginning of each conversation, read `MEMORY.md` if it exists so you are aware of recent changes, plans, locked decisions, and important context from prior conversations.
- Update `MEMORY.md` whenever a conversation changes project direction, locks in a decision, records important experimental context, creates/changes a follow-up plan, user provides important info, or user explicitly says something like "remember this". Keep updates concise and factual.
- Treat `diploma_knowledge_base.md` as the live project knowledge base as of 2026-05-02. The older `diploma_knowledge_base_02_05_2026.md` was renamed to `diploma_knowledge_base.md`. Update the live knowledge base regularly when important project decisions, architecture plans, experiment results, or diploma-writing context changes.
- Use `$` for math expressions. E.g., $ w_{ij} = \operatorname{MLP}([t_i, t_j, x_i, x_j, y_i, y_j]) $
- Follow best practices for Python coding, data science, and machine learning
- Prioritize correctness, clarity, and reproducibility over cleverness
- Prefer simple, explicit code over abstractions
- Assume code will be read by researchers, not only engineers
- For new functions add concise informative docstrings
- Use typing in function signatures
- If instructions are not perfectly clear, ask for clarification before proceeding
- Parameters should be configurable (no hardcoded paths or constants); if you want to hardcode something, ask for clarification first
- You can hardcode values only in jupyter notebooks for exploration and visualization, never in scripts or modules
- Do not write duplicate code; reuse when possible; if needed, create helper functions or classes
- Log intermediate results when useful
- Minimize dependencies
- Do not introduce new libraries unless clearly beneficial
- Make randomness, I/O, and assumptions explicit
- Avoid data leakage and hidden side effects
- Prefer readable vectorization over premature optimization
- When plotting confusion matrices, we should normalize each row to show per-class percentages (values between 0.0 and 1.0) and fix the color scale to the [0, 1] range.
- When creating a new python file, write a concise but informative docstring in the very beginning of the file, describing what the script/module is meant to do and how it is meant to be used.
- When creating new python scripts that are meant to be run from terminal, explain and provide an example on how to run the script and what are the running options in the docstring at the very beginning of the file. 
- Cleanup after yourself. When you are testing something and, as a result, new files and/or folders are created, carefuly delete the test-files and test-folders when not needed anymore. 
- When writing code for creating heatmaps, use color scheme "Blues".
- Ask questions regularly, except requested to implement something right away.
- The cli command that runs a script should be simple. User should be able to simply run python <script_name>.py For every argument, there should be a sensible default (unless stated otherwise or unless I removed it). Log the final arguments at the beginning of the script run for transparency.
- For presenting comparisons in your answer prefer tables.
- You are always allowed to run terminal commands that remove files you created from /tmp. E.g., you are allowed to run rm -rf /tmp/matplotlib-* and similar. Don’t ask me about this. 
- In this repo, we ran multiple experiments already and tried many different things. When I ask you about some experiment/training/model/data information, check the latest git commit(s) and assume I am talking about the experiments we modified most recently. When you do such an assumption, make it explicit and clear. If you are unsure which experiment/data/model I had in mind, ask me before proceeding. 
- Current near-term model priorities are incremental and modular: MLP fusion/pooling of spatial and temporal node representations, MLP pooling from nodes to graph embedding, separate temporal-forward/temporal-backward/spatial edge types, and learned signed edge weights. Prefer relation features such as `[t_i, t_j, delta_t, delta_x, delta_y, distance]`, normalize signed incoming weights per target node, use spatial edge-weight MLP `6 -> 6 -> 4 -> 2 -> 1`, use temporal edge-weight MLP `7 -> 6 -> 4 -> 2 -> 1` with a direction feature, and share the temporal MLP across forward/backward temporal edges. Ablations come after the model works well.


## Conda environment
- We always use conda environments.
- When using python, first activate conda environment named `gfm`.

## Hardware
- OS: Ubuntu
- GPU: NVIDIA GeForce RTX 4070 with 12,282 MiB VRAM
- CPU: AMD Ryzen 7 5700X 8-Core Processor
- RAM: 32 GiB
- Storage: Samsung SSD 990 PRO 2 TB NVMe

## Project Info
- This repo works on eye-tracking data. 
- Our aim is to build a cutting-edge GNN model that will infer different physiological and psychological states from eye-tracking signals. We are aiming for a general graph foundation model (GFM).
- We want to build our model step-by-step, from ground up and experiment with it to see what works and what does not.
- We compare our model to baseline models from classical ML.
- Ignore files in `archive/`
