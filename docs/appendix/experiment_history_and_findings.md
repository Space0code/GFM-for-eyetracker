# Experiment History and Findings

This appendix consolidates the chronology from `docs/journal.md` and result artifacts.

## 1. Timeline summary

| Phase | Evidence anchor | Main focus | Outcome |
|---|---|---|---|
| Early `gnext` phase | `archive/src/gnext/*` + `docs/journal.md` | Next-point gaze prediction (GraphSAGE) | Useful trend-learning proof of concept |
| eSEEd regression phase | `docs/journal.md` (eSEEd regression sections) | Multi-target emotion regression | Partial signal; poor cross-split robustness |
| eSEEd binary + cleaning | `docs/journal.md` (binary + cleaning sections) | Binary emotion tasks on eSEEd | GNN collapse discovered; majority often hard to beat |
| Collapse debugging | `docs/journal.md` (collapse diagnostics) | Embedding/variance diagnostics | Early representation collapse + low-logit-variance regime identified |
| Transition to HCI tagging | `docs/journal.md` ("Transitioning to MAHNOB-HCI-TAGGING") | Cleaner data + richer labels | Clear improvement in stability and learnability |
| HCI full suite run | 2026-03-05 13:04:55 | Unified binary/multiclass/regression/tagging benchmark | Best complete reference run |
| GNN ablation suite | 2026-03-05 | One-factor variants on valence/arousal subset | Depth/early-stopping had strongest effects |
| “Optimal” rerun | 2026-03-05 16:11:26 | Attempted improved global config | Partial run, failed at `subject_loo` boundary |

## 2. What worked vs what did not

## 2.1 Worked (relative to earlier phases)

- Moving from eSEEd to HCI reduced severe data pathologies.
- Binary emotion tasks on HCI became non-collapsed and competitive.
- On the complete suite (`RETAIN_2026-03-05_13-04-55`), GNN was strongest on emotion-control (`balanced_accuracy=0.6813`).
- Deeper GNN (`num_layers=10`) improved valence in ablations (+4.35 pp vs ablation baseline).

## 2.2 Weak or inconsistent

- Multiclass remained weak:
  - `emotion-id` balanced accuracy: `0.1803` (GNN, `recording_loo`, aggregated)
  - VA quadrant balanced accuracy: `0.2866` (GNN, `recording_loo`, aggregated)
- Regression remained weak (best non-Mean CCC in this run: `0.0932`).
- Some “promising” configs did not transfer reliably across targets/splits.
- Edge-weight toggling and target aggregation (`mean` vs `last`) showed near-zero effect in tested subset.

## 3. Complete suite benchmark snapshot

Reference run: `results/suite/RETAIN_2026-03-05_13-04-55`

Binary emotion tasks (`recording_loo`, aggregated):

| Task | Best model by balanced accuracy | Balanced accuracy | GNN balanced accuracy |
|---|---|---:|---:|
| valence | LightGBM | 0.5598 | 0.5502 |
| arousal | LightGBM | 0.5220 | 0.4958 |
| control | GNN | 0.6813 | 0.6813 |
| predictability | MLP | 0.5005 | 0.4891 |

Regression tasks (`recording_loo`, aggregated):

| Task | Best non-mean model | MAE | CCC | Spearman |
|---|---|---:|---:|---:|
| emotion-arousal | MLP | 1.9273 | 0.0932 | 0.1119 |
| emotion-valence | MLP (CCC/Spearman) | 2.1682 | 0.0574 | 0.1173 |

## 4. Ablation findings (focused valence/arousal subset)

Source: historical focused ablation run summarized in `docs/journal.md` and `docs/experiment_log.md`.

Baseline variant (`baseline_default`):
- valence balanced accuracy: 0.5139
- arousal balanced accuracy: 0.5223

Best observed improvements:
- valence: `num_layers_10` -> 0.5574 (+4.35 pp)
- arousal: `early_stopping_on` -> 0.5424 (+2.01 pp)

Family-level interpretation:

| Family | Main observation |
|---|---|
| `num_layers` | Most impactful for valence in tested setup; highly sensitive (some depths hurt) |
| `early_stopping` | Helped arousal, hurt valence in tested setup |
| `kt/ks` grid | Small effects only |
| `edge_weights` | No measurable gain in tested subset |
| `target_aggregation` | No measurable gain (`mean` vs `last`) |
| `pooling`/`conv_type` | Mixed, no consistent global win |

## 5. Known failure in latest "optimal" run

Run: `results/suite/RETAIN_2026-03-05_16-11-26`

- Both experiments marked failed due to `IndexError` at fold transition to `subject_loo`.
- `recording_loo` artifacts exist and were reconstructed in that run README.
- Because it is partial/incomplete, use `RETAIN_2026-03-05_13-04-55` as canonical complete benchmark.

## 6. Historical collapse diagnosis (eSEEd period)

From `docs/journal.md`:
- Large variance drop already at early message-passing (and/or preprocess stage).
- Head logits had very low spread, producing near-constant predictions.
- Practical effect: frequent collapse to one class in binary experiments.

These findings motivated:
- stronger normalization decisions
- careful target/scaling protocol
- migration to cleaner HCI pipeline
- systematic ablations

## 7. Practical takeaways for next iteration

- Keep HCI as primary benchmark while preserving eSEEd for stress testing.
- Prioritize robust depth/regularization schedules over edge-weight complexity.
- Focus on multiclass/regression representation quality (current weakest link).
- Treat valence/arousal as partially coupled but not fully transferable optimization targets.
