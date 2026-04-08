# RETAIN_2026-04-02_17-56-07 Brief Report

## Scope
- Suite run root: `results/suite/RETAIN_2026-04-02_17-56-07`
- Experiments executed: 2 (both `multiclass`, Table-6 3-class targets)
- CV strategy: `subject_kfold` (`n_splits=5`)
- Models compared: `Mean`, `SVM`, `LightGBM`, `MLP`, `GNN`
- Both experiments completed successfully (`suite_experiment_registry.csv`)

## Data Snapshot Notes
- Cleaned snapshot rows: `2,855,527`
- Subjects: `24`
- Recordings: `20`
- One tiny window was skipped in each run due to `kt=2`, `ks=2` constraints (`subject_P30_recording_earworm_f.avi.csv`).

### Label balance
- `table6-arousal-3class`: class proportions `0.4699 / 0.3542 / 0.1760`, entropy `1.4834`.
  - EDA warning: severe class imbalance (minority class < 0.2).
- `table6-valence-3class`: class proportions `0.3720 / 0.2739 / 0.3542`, entropy `1.5728`.
  - EDA warning: none.

## Main Results (Aggregated/Standard)
`accuracy` here means raw, unbalanced accuracy.

| Experiment | Best Baseline Raw Acc | GNN Raw Acc | Delta Raw Acc (pp) | Best Baseline (f1) | GNN f1 | Delta f1 | Baseline AUC | GNN AUC | Delta AUC | Baseline BalAcc | GNN BalAcc | Delta BalAcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Table6 Arousal 3-class | 0.4816 (SVM) | 0.5384 | +5.68 | 0.3471 (MLP) | 0.4387 | +0.0916 | 0.5815 | 0.6761 | +0.0945 | 0.3658 | 0.4478 | +0.0820 |
| Table6 Valence 3-class | 0.4610 (SVM) | 0.5382 | +7.73 | 0.4259 (SVM) | 0.5205 | +0.0946 | 0.6468 | 0.7155 | +0.0686 | 0.4486 | 0.5322 | +0.0836 |

## Comparison to Dataset Paper (Table 7)
Paper eye-gaze benchmarks (Table 7) documented in `docs/journal.md`:
- Arousal: accuracy `63.5%`, F1 `0.60`
- Valence: accuracy `68.8%`, F1 `0.68`

Using our aggregated GNN results from this run:

| Task | Paper eye-gaze accuracy | Our GNN accuracy | Delta (pp) | Paper eye-gaze F1 | Our GNN macro-F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Table6 Arousal 3-class | 0.6350 | 0.5384 | -9.66 | 0.60 | 0.4387 | -0.1613 |
| Table6 Valence 3-class | 0.6880 | 0.5382 | -14.98 | 0.68 | 0.5205 | -0.1595 |

For clarity: paper "accuracy" and our "raw accuracy" are both standard (unbalanced) accuracy.

Important comparability caveat:
- The paper reports an ET handcrafted-feature pipeline with RBF SVM and participant-independent LOO protocol.  
- Our run uses graph windows + GNN under `subject_kfold` with different preprocessing/model assumptions.  
- So this is a directional benchmark comparison, not a strict apples-to-apples reproduction.

## Per-Pair View (from training logs)
- Arousal 3-class:
  - GNN: `accuracy=0.5942`, `macro_f1=0.4450`, `macro_auc_ovr=0.7867`.
- Valence 3-class:
  - GNN: `accuracy=0.6241`, `macro_f1=0.6089`, `macro_auc_ovr=0.8067`.

## Training Behavior (GNN)
- Arousal run (5 folds): mean fold train time `119.47s`, best epoch range `9..30`.
- Valence run (5 folds): mean fold train time `111.87s`, best epoch range `2..30`.
- Early stopping appears active and effective (best epochs vary by fold).

## Takeaways
- The new Table-6 multiclass pipeline is working end-to-end (EDA, training, plotting, suite aggregation).
- GNN is clearly strongest across both tasks and key classification metrics.
- Valence 3-class appears easier than arousal 3-class, consistent with better class balance and higher entropy.
- For arousal, imbalance mitigation could be the next lever (e.g., class weighting, sampling strategy, focal loss experiments).
