# RETAIN Suite Summary (2026-03-05 13:04:55)

This run is configuration-equivalent to `results/suite_FULL/2026-03-05_13-25-29` except for subject filtering.

## Config Equivalence vs `suite_FULL`

| config field | this run (`RETAIN_2026-03-05_13-04-55`) | `suite_FULL/2026-03-05_13-25-29` |
|---|---|---|
| `suite.results_dir` | `results/suite` | `results/suite_FULL` |
| `global_overrides.dataset.filter_subjects` | `P1,P8,P5,P4,P28,P2,P27` | `null` (all subjects) |
| Other wrapper settings | same | same |

Interpretation: same pipeline/model settings; this run differs by using a curated subject subset only.

## Performance Delta vs `suite_FULL` (subset - full, classification, recording_loo)

| model | delta accuracy | delta balanced_accuracy | delta f1 | delta auc |
|---|---:|---:|---:|---:|
| GNN | +0.0153 | +0.0392 | +0.0241 | +0.0507 |
| SVM | +0.0103 | +0.0232 | +0.0226 | +0.0376 |
| LightGBM | +0.0244 | +0.0231 | +0.0264 | +0.0369 |
| MLP | +0.0212 | +0.0519 | +0.0231 | +0.0566 |

This confirms the subset run reports better classification numbers overall, consistent with selecting subjects that are easier/more stable for this benchmark.

## Why This Folder Is Kept
- We keep this run as a reference for which subjects produce stronger results under the same configuration.
- The key analysis artifact is:
  - `results/suite/RETAIN_2026-03-05_13-04-55/suite_classification_gnn_comparison.md`
- Use this run for subset-vs-full sensitivity analysis, not as the primary full-population benchmark.
