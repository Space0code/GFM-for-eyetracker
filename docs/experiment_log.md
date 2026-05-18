# Experiment Log

This file keeps detailed experiment and runner history that is too operational for
`MEMORY.md` or `diploma_knowledge_base.md`. Use `MEMORY.md` for the current state,
`diploma_knowledge_base.md` for thesis-facing synthesis, and this file for
traceable run/config details.

## 2026-05 Quick V1/V2 Development And Table-6 Runs

### 2026-05-05: GNN v2 architecture implementation

Implemented GNN v2 architecture in four commits:

- frozen `SpatioTemporalHeteroGNNV1`;
- v2 split temporal forward/backward graph schema;
- relation concat+MLP fusion;
- attention graph pooling;
- learned signed normalized edge weights.

Added quick Table-6 arousal comparison runner:

```bash
python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py
```

Current edge-weight details:

- relation features such as `[t_i, t_j, delta_t, delta_x, delta_y, distance]`;
- signed incoming edge weights normalized per target node by signed-score magnitude;
- spatial MLP `6 -> 6 -> 4 -> 2 -> 1`;
- temporal MLP `7 -> 6 -> 4 -> 2 -> 1` with a direction feature;
- temporal forward/backward share the same temporal MLP.

### 2026-05-07: Quick runner baseline/config updates

The quick v1/v2 comparison now includes `Random` and `Majority` multiclass
baselines by default. Comparison plots use fixed display order:

`Random`, `Majority`, `GNN_v1`, `GNN_v2`, `MLP`, then remaining trained models
alphabetically.

The runner is YAML-first: subjects, recordings, CV settings, graph settings,
cache settings, and epochs should be controlled in the suite YAML by default.
CLI flags only apply optional explicit overrides.

All requested baseline models are grouped into one suite/trainer invocation so
baselines share one dataset load and one CV split construction. `GNN_v1` and
`GNN_v2` remain separate invocations because their architecture configs differ.

### 2026-05-07: Torch compile issue and LOSO arousal run

A quick Table-6 arousal run failed during `loss.backward()` with a PyTorch
Inductor `torch.compile` backward-graph assertion on dynamic PyG graph batches.
This is treated as a compiler/runtime issue, not evidence of an invalid model
objective. The quick v1/v2 runner disables `use_torch_compile` by default,
offers `--use-torch-compile` for explicit re-enable, and mirrors stdout/stderr to
`quick_v1_v2_comparison.log` inside each timestamped results folder.

Completed LOSO Table-6 arousal comparison:

```bash
python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py \
  --models GNN_v1,GNN_v2,LightGBM,random,majority \
  --num-epochs 30 \
  --cv-strategy subject_loo
```

Results folder:

`results/quick_v1_v2_comparison/2026-05-07_14-54-32`

Aggregated standard metrics:

| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | AUC |
|---|---:|---:|---:|---:|---:|
| `GNN_v2` | 0.5562 | 0.4740 | 0.4462 | 0.5233 | 0.7113 |
| `GNN_v1` | 0.5416 | 0.4603 | 0.4297 | 0.5038 | 0.6950 |
| `LightGBM` | 0.4707 | 0.3854 | 0.3505 | 0.4288 | 0.6041 |
| `Majority` | 0.4771 | 0.3333 | 0.2139 | 0.3125 | 0.5000 |
| `Random` | 0.3031 | 0.3068 | 0.2895 | 0.3201 | 0.4812 |

The command-level ranking plot and combined confusion matrices include `Random`
and `Majority`.

### 2026-05-11: Multi-strategy CV and class-balance diagnostics

The quick v1/v2 comparison runner supports multiple comma-separated CV
strategies in one run, for example:

```bash
python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py \
  --cv-strategy subject_loo,recording_loo
```

or:

```bash
python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py \
  --cv-strategy subject_kfold,recording_kfold \
  --n-splits 3
```

The summary CSV stores one row per `(cv_strategy, model)`, ranking plots show one
panel per strategy, and multi-strategy confusion matrices are saved as
`figures/confusion_matrices_<strategy>.png`.

The runner also writes class-balance diagnostics for completed runs:

- `tables/label_distribution_by_fold.csv`;
- `tables/label_distribution_aggregate.csv`;
- `plots/label_distribution_counts.png`;
- `plots/label_distribution_proportions.png`.

### 2026-05-11: Dataset exclusion and minimum window samples

Added explicit `dataset.exclude_subjects` support across snapshot building,
graph loading, tabular sample building, and cache keys. Default HCI configs now
conservatively exclude `P9`, `P12`, and `P15` to align with the MAHNOB paper's
excluded participants under current uncertainty about the reduced local ET-only
copy. Added root report `MAHNOB_dataset_report.md` with verified local counts and
open uncertainties.

`dataset.min_samples_per_window` now controls both baseline window filtering and
GNN graph-window filtering. Previously GNN still hardcoded the minimum to
`max(kt, ks) + 1`; current HCI/Table-6 suite defaults were raised to `60`.

### 2026-05-12: Pooling/config ownership updates

Added explicit dual pooling controls for GNN v2:

- `graph_pooling` controls node-to-graph pooling (`mean`, `mean_max`, or
  `attention`);
- `relation_pooling` controls per-layer spatial/temporal relation fusion
  (`attention` or `mlp`);
- `pooling` is kept as a backward-compatible alias for graph pooling.

Updated pooling defaults per active v2 work:

- graph/head pooling defaults to attention;
- relation pooling defaults to concat+MLP (`relation_pooling: mlp`);
- optional `head_pooling` config alias maps to graph-level pooling.

Consolidated quick-comparison config ownership under:

`src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/`

`run_quick_v1_v2_comparison.py` now defaults to:

`src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`

That wrapper and `run_hci_experiment_suite_small.yaml` both point to local quick
trainer configs for the task types they run:

- `quick_v1_v2_train_binary_hci_tagging.yaml`;
- `quick_v1_v2_train_multiclass_hci_tagging.yaml`.

Archived unused legacy config:

`archive/src/emotions/multiclass/configs/train_multiclass_hci_tagging_emotion_id_legacy.yaml`

Python variant overrides now only select variant identity:

- `model_version`;
- `graph_version`;
- edge-weight mode;
- baseline/GNN enable flags.

Shared architecture settings, especially `gnn.model.conv_type`, live in:

`src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`

For fair `GNN_v1`/`GNN_v2` convolution comparisons, change
`global_overrides.gnn.model.conv_type` in that wrapper YAML to either `GCNConv`
or `GATConv`. `GATConv` ignores scalar edge weights in both v1 and v2 model code.

The active quick v1/v2 Table-6 comparison wrapper was prepared for a GAT run by
setting `global_overrides.gnn.model.conv_type: GATConv`.

### 2026-05-12: Training-progress and task-selection outputs

Quick v1/v2 comparison now records training-progress artifacts for multiclass
runs:

- GNN folds save `gnn_training_history.csv`;
- MLP baseline folds save sklearn `loss_curve_` as `mlp_training_history.csv`
  when available;
- the quick runner aggregates histories to `tables/training_history.csv`;
- plots are written under `plots/` for loss curves, validation balanced
  accuracy/macro-F1 curves, and best-epoch distributions.

Quick v1/v2 Table-6 task selection is controlled by `quick_comparison.table6_tasks`
in:

`src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`

Valid values include `[arousal]`, `[valence]`, and `[arousal, valence]`.
Table-6 arousal class 1 display name now uses the MAHNOB paper wording
`Medium aroused`.

Fixed held-out test loss visibility for multiclass quick comparisons:

- multiclass summaries now include `loss`;
- quick comparison outputs can save `plots/test_loss_by_model.png`;
- training-history/test-loss plotting is covered by tests;
- quick Table-6 task parsing handles normalized experiment IDs.

### 2026-05-12/13: Focused Table-6 valence v2 matrix

Ran focused Table-6 valence subject-kfold checks with conservative loading:

- `num_workers=0`;
- `persistent_workers=false`;
- `pin_memory=false`;
- no torch compile;
- LightGBM `n_jobs=4`.

Script:

`src/emotions/gnn_improvement_experiments/run_table6_valence_v2_checks.py`

Results:

- original: `results/table6_valence_v2_checks/2026-05-12_16-09-19`;
- mirrored copy: `results/quick_v1_v2_comparison/2026-05-12_16-09-19_table6_valence_v2_checks`.

All 16 variants succeeded. Summary table:

`matrix_summary.csv`

Aggregated training plots:

- `plots/training_progress_loss.png`;
- `plots/training_progress_validation_metrics.png`;
- `plots/best_epoch_distribution.png`;
- `plots/test_loss_by_model.png`.

Key standard balanced accuracy findings:

| Comparison | Result |
|---|---:|
| `GNN_v1` convergence comparison | 0.5237 |
| current 10-layer weighted-GCN `GNN_v2` | 0.4983 |
| `LightGBM` | 0.4358 |
| `MLP` | 0.4175 |
| best v2 depth: 3 layers | 0.5285 |
| v2 depth: 1 layer | 0.5180 |
| v2 depth: 5 layers | 0.5175 |
| v2 depth: 10 layers | 0.5120 |
| weighted `GCNConv` architecture | 0.5107 |
| unweighted `GATConv` architecture | 0.4967 |
| unweighted `GCNConv` architecture | 0.4960 |

Fixed-epoch no-early-stopping sweep:

| Epochs | Balanced accuracy |
|---:|---:|
| 50 | 0.5320 |
| 10 | 0.5233 |
| 200 | 0.5203 |
| 5 | 0.5071 |
| 30 | 0.4986 |
| 1 | 0.4943 |

Convergence interpretation:

- GNN training losses generally decrease;
- long no-early-stopping runs overfit;
- mean validation loss at final epoch rose to about `2.05` for 30 epochs,
  `2.24` for 50 epochs, and `6.05` for 200 epochs;
- mean best epochs were early: `4.0`, `14.7`, and `9.3`, respectively;
- use validation-loss early stopping for long GNN v2 runs;
- prioritize shallower v2 depths, especially 3 layers.

Regenerated focused valence v2 matrix plots in:

`results/quick_v1_v2_comparison/2026-05-12_16-09-19_table6_valence_v2_checks/plots`

from saved factual inputs:

- `matrix_summary.csv`;
- `tables/training_history.csv`.

Plot readability updates:

- training-progress legends moved to separate legend images;
- crowded x-ticks rotated;
- old versions backed up under `plots/previous_unreadable_versions/`;
- redundant `matrix_summary_partial.csv` deleted after confirming it matched
  final summary;
- `matrix_summary.csv` now places metric columns near the front for readability.

### 2026-05-12/13: Low-vs-high Table-6 checks

Ran low-vs-high only Table-6 subject-kfold comparison by dropping the medium
class from Table-6 mappings.

Config:

`results/quick_v1_v2_comparison/generated_low_high_configs/table6_low_high_arousal_valence_subject_kfold.yaml`

Results:

`results/quick_v1_v2_comparison/2026-05-12_18-14-03`

The run confirmed raw labels `{0,2}` only in the model label mapping.

Standard balanced accuracy:

| Task | `GNN_v1` | `GNN_v2` | `LightGBM` | `MLP` |
|---|---:|---:|---:|---:|
| Arousal | 0.5211 | 0.5000 | 0.5046 | 0.5144 |
| Valence | 0.6507 | 0.6234 | 0.6052 | 0.5890 |

Test-loss plot, training-history plots, label-distribution plots, and
confusion-matrix figures were saved in the result directory.

Added optional Table-6 multiclass pre-CV class downsampling in the trainer so
GNN and tabular baselines can use balanced mapped-label populations.

Ran arousal low-vs-high subject-kfold with low arousal randomly downsampled to
match the total high-arousal window count.

Config:

`results/quick_v1_v2_comparison/generated_low_high_configs/table6_low_high_arousal_downsample_low_subject_kfold.yaml`

Results:

`results/quick_v1_v2_comparison/2026-05-13_08-48-31`

Class mapping and counts:

- raw `{0,2}` -> encoded `{0,1}`;
- counts before downsampling `{0: 2360, 2: 928}`;
- counts after downsampling `{0: 928, 2: 928}`.

Standard balanced accuracy:

| Model | Balanced accuracy |
|---|---:|
| `GNN_v1` | 0.5472 |
| `MLP` | 0.5424 |
| `GNN_v2` | 0.5348 |
| `LightGBM` | 0.5278 |

### 2026-05-13: Default config update after valence depth results

Updated Table-6 GNN v2 default config after valence depth results:

- set `num_layers: 3`;
- set `early_stopping_patience: 20`.

Updated files:

- `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`;
- `src/emotions/suite/configs/run_hci_experiment_suite_table6_3class.yaml`.

Current GNN training code saves `best_model.pt` whenever validation loss improves
and loads that checkpoint before test evaluation.

### 2026-05-13: Low/high valence 7-fold subject-kfold

Low/high Table-6 valence 7-fold subject-kfold run completed with current GNN v2
defaults.

Config:

`results/quick_v1_v2_comparison/generated_low_high_configs/table6_low_high_valence_subject_kfold_7fold_conservative.yaml`

Results:

`results/quick_v1_v2_comparison/2026-05-13_09-36-39`

Standard balanced accuracy:

| Model | Balanced accuracy |
|---|---:|
| `GNN_v2` | 0.6646 |
| `GNN_v1` | 0.6473 |
| `LightGBM` | 0.6104 |
| `MLP` | 0.6003 |

Validation-loss stability comparison with the existing 3-fold low/high valence
run:

| Model | 3-fold best-val-loss SD | 7-fold best-val-loss SD | 3-fold SE | 7-fold SE |
|---|---:|---:|---:|---:|
| `GNN_v2` | 0.0466 | 0.0386 | 0.0269 | 0.0146 |
| `GNN_v1` | 0.0407 | 0.0728 | 0.0235 | 0.0275 |

### 2026-05-13: Table-6 label-noise proxy analysis

Label-noise proxy analysis completed in:

`results/label_noise_analysis/2026-05-13_table6_self_report_alignment`

The analysis compares the emotion-id-derived Table-6 target against participant
self-report rating buckets on the 1-9 scale.

Section-level mismatch rates:

| Target | Mismatch rate |
|---|---:|
| Arousal 3-class | 48.6% |
| Arousal low/high binary | 29.7% |
| Valence 3-class | 27.5% |
| Valence low/high binary | 3.3% |

Interpretation: this strongly supports treating low/high valence as a cleaner
target than arousal in the current ET-only setup.

### 2026-05-13: Journal synthesis

Updated `docs/journal.md` with a concise English report for the 2026-05-12/13
Table-6 experiment set.

Main conclusions:

- current GNNs learn some signal but validation-loss convergence remains noisy;
- 3-layer v2 is preferred over deeper v2 variants;
- learned weighted GCN slightly beats unweighted GCN/GAT;
- long fixed training without early stopping is unstable;
- low-vs-high valence is much more learnable than arousal;
- downsampled arousal improves only modestly.

### 2026-05-18: GazeMAE frozen-backbone baseline integration

Implemented `GazeMAE_MLP` as a frozen pretrained GazeMAE position+velocity
encoder plus trainable PyTorch MLP head. The baseline uses the same suite
snapshots, windowing, CV splits, labels, metrics, plots, and quick-comparison
summary flow as other multiclass baselines.

Locked preprocessing decision: clip MAHNOB coordinates to the actual screen
resolution `1280x800`, with no scaling and no mean normalization. Do not scale
vertical coordinates to 1024, because this would alter position and velocity
magnitudes.

Smoke checks:

- synthetic 10s window -> five 2s chunks for position and velocity, final
  embedding shape `512`;
- quick dry run with `--models GazeMAE_MLP` generated a baseline-only wrapper;
- tiny subject-kfold smoke run on `P1`, `P5`, `P8` with `GazeMAE_MLP,MLP`
  created summary CSVs, label-distribution tables, ranking/test-loss plots,
  training-history plots, and confusion matrices under `/tmp/gazemae_quick_smoke`.

Follow-up cache improvement: strengthened embedding reuse across quick/suite
runs by keying GazeMAE cache entries from the stable suite `snapshot_hash` and
`snapshot_cache_key` when a snapshot manifest is available, instead of the
timestamped snapshot CSV path. The cache key also includes checkpoint file
hashes and preprocessing settings, so reuse remains tied to identical data,
model, and preprocessing identities.

Additional checks:

- two different manifest paths with the same `snapshot_hash` produced the same
  GazeMAE cache key;
- changing `snapshot_hash` changed the cache key;
- `--models GazeMAE_MLP --dry-run` still generated a baseline-only wrapper with
  `run_experiments.gnn: false`.
- recheck smoke on a small subject-kfold valence subset with `GazeMAE_MLP,MLP`
  completed end-to-end, including summary CSVs, loss plots, label-distribution
  tables, and confusion matrices;
- rerunning the same small setup with `GazeMAE_MLP` loaded the cached GazeMAE
  embeddings directly, confirming practical cache reuse across timestamped
  quick-comparison output directories.

Follow-up self-contained runtime cleanup: replaced the cross-repo runtime
dependency on `/home/ppg/eyetracking/gazemae` with a minimal local GazeMAE
encoder implementation in `src/emotions/gazemae_model.py` and converted
encoder-only checkpoints in `models/gazemae/`. The converted checkpoints keep
only encoder and bottleneck state dicts:

- `models/gazemae/pos-i3738-encoder-state.pt`;
- `models/gazemae/vel-i8528-encoder-state.pt`.

Parity and smoke checks:

- local position encoder matched the original external checkpoint output with
  `max_abs_diff=0.0` on random `[batch, 2, 1000]` chunks;
- local velocity encoder matched the original external checkpoint output with
  `max_abs_diff=0.0` on random `[batch, 2, 1000]` chunks;
- local embedder smoke produced five 2s chunks and a finite 512-dimensional
  pooled embedding;
- small subject-kfold quick run with `GazeMAE_MLP,MLP` completed end-to-end
  using only local model assets;
- rerunning the same small setup loaded cached GazeMAE embeddings directly.

### 2026-05-18: Full 3-class valence quick comparison with GazeMAE_MLP

Ran the quick comparison script end-to-end as a user-facing check of the new
self-contained `GazeMAE_MLP` baseline:

```bash
conda run -n gfm python src/emotions/gnn_improvement_experiments/run_quick_v1_v2_comparison.py \
  --models Random,Majority,GNN_v2,GazeMAE_MLP
```

Config/run context:

- base wrapper:
  `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/run_hci_experiment_suite_table6_3class.yaml`;
- selected task from wrapper: Table-6 valence 3-class;
- CV: `subject_kfold`, `n_splits=5`, `val_size=1`, seed `42`;
- effective models: `Random`, `Majority`, `GazeMAE_MLP`, `GNN_v2`;
- output directory: `results/quick_v1_v2_comparison/2026-05-18_12-06-36`;
- implementation note: the first attempt completed but used separate baseline
  and GNN suite invocations, which produced different subject folds. The quick
  runner and multiclass trainer were fixed so this final run executes all four
  requested models in one suite invocation with aligned baseline/GNN fold group
  identities.

Standard aggregated metrics from `quick_comparison_summary.csv`:

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | AUC | Loss |
|---|---:|---:|---:|---:|---:|---:|
| GazeMAE_MLP | 0.5125 | 0.5060 | 0.5016 | 0.5044 | 0.6849 | 1.2684 |
| GNN_v2 | 0.5047 | 0.5026 | 0.4930 | 0.4951 | 0.6931 | 1.0358 |
| Random | 0.3363 | 0.3355 | 0.3335 | 0.3384 | 0.5016 | 23.9214 |
| Majority | 0.3516 | 0.3333 | 0.1731 | 0.1839 | 0.5000 | 23.3715 |

The final aligned run completed without further errors. The GazeMAE path loaded
5,034 cached embedding samples, and the GNN path built 5,034 graph samples.
GNN_v2 completed all five folds with early stopping; best epochs were 21, 7,
39, 61, and 11. Command-level outputs include the ranking plot, test-loss plot,
label-distribution tables/plots, training-history CSV/plots, per-fold loss
plots for GNN_v2 and GazeMAE_MLP, and combined confusion matrices.
