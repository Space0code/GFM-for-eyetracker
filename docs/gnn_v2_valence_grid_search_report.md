# GNN v2 Table-6 Valence Grid Search Report

Source run: `results/gnn_v2_valence_grid_search/2026-05-27_16-07-35`

Task: 3-class Table-6 valence, subject k-fold (`k=5`, `val_size=1`), GNN v2 only. Winner selection used accuracy first, macro F1 second, and lower loss as tie-breaker.

## Main Result

| Selection | Variant | Layers | Hidden | kt | ks | kf | Accuracy | Macro F1 | Balanced accuracy | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Best phase 1 | `layers4_hidden256` | 4 | 256 | 2 | 2 | 3 | 0.5251 | 0.5093 | 0.5191 | 1.0673 |
| Best phase 2 | `kt3_ks2_kf1` | 4 | 256 | 3 | 2 | 1 | 0.5260 | 0.5104 | 0.5200 | 1.0690 |

Phase 2 improved only slightly over the best phase 1 architecture: +0.0009 accuracy and +0.0012 macro F1. The tuning signal is therefore real but small.

## Phase 1: Architecture Capacity

| Rank | Variant | Layers | Hidden | Loader | Runtime (s) | Accuracy | Macro F1 | Balanced accuracy | Loss |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | `layers4_hidden256` | 4 | 256 | `oom_batch128` | 1892.8 | 0.5251 | 0.5093 | 0.5191 | 1.0673 |
| 2 | `layers2_hidden128` | 2 | 128 | `fast` | 1138.0 | 0.5164 | 0.5040 | 0.5143 | 1.0274 |
| 3 | `layers2_hidden64` | 2 | 64 | `fast` | 981.4 | 0.5161 | 0.5055 | 0.5135 | 1.0056 |
| 4 | `layers3_hidden256` | 3 | 256 | `oom_batch128` | 1759.1 | 0.5154 | 0.5015 | 0.5095 | 1.0579 |
| 5 | `layers4_hidden64` | 4 | 64 | `fast` | 1256.7 | 0.5140 | 0.4989 | 0.5087 | 1.0250 |
| 6 | `layers3_hidden64` | 3 | 64 | `fast` | 1160.3 | 0.5128 | 0.5021 | 0.5094 | 1.0207 |
| 7 | `layers2_hidden256` | 2 | 256 | `oom_batch128` | 1591.4 | 0.5097 | 0.4953 | 0.5054 | 1.0819 |
| 8 | `layers3_hidden128` | 3 | 128 | `fast` | 1167.4 | 0.5094 | 0.4995 | 0.5093 | 1.0280 |
| 9 | `layers4_hidden128` | 4 | 128 | `fast` | 1439.2 | 0.5036 | 0.4909 | 0.5046 | 1.0328 |

The largest model won by accuracy, but the margin over the smallest model is modest: `4x256` scored 0.5251 accuracy, while `2x64` scored 0.5161. The smaller model also had the best phase 1 loss and ran without OOM retry. This supports running the phase 2 graph-density search again for `2x64` as a better complexity/performance trade-off candidate.

## Phase 2: Best Variants

| Rank | Variant | kt | ks | kf | Loader | Runtime (s) | Accuracy | Macro F1 | Balanced accuracy | Loss |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | `kt3_ks2_kf1` | 3 | 2 | 1 | `oom_batch128` | 1799.4 | 0.5260 | 0.5104 | 0.5200 | 1.0690 |
| 2 | `kt3_ks1_kf2` | 3 | 1 | 2 | `oom_batch128` | 1842.4 | 0.5256 | 0.5100 | 0.5196 | 1.0673 |
| 3 | `kt2_ks3_kf3` | 2 | 3 | 3 | `oom_batch128` | 1976.6 | 0.5255 | 0.5098 | 0.5196 | 1.0683 |
| 4 | `kt2_ks3_kf2` | 2 | 3 | 2 | `oom_batch128` | 1854.4 | 0.5254 | 0.5099 | 0.5196 | 1.0679 |
| 5 | `kt3_ks1_kf3` | 3 | 1 | 3 | `oom_batch128` | 1975.9 | 0.5252 | 0.5095 | 0.5192 | 1.0677 |
| 6 | `kt2_ks2_kf3` | 2 | 2 | 3 | `oom_batch128` | 1881.0 | 0.5251 | 0.5093 | 0.5191 | 1.0673 |
| 7 | `kt2_ks2_kf2` | 2 | 2 | 2 | `oom_batch128` | 1766.5 | 0.5251 | 0.5092 | 0.5191 | 1.0670 |
| 8 | `kt3_ks2_kf2` | 3 | 2 | 2 | `oom_batch128` | 1916.2 | 0.5242 | 0.5088 | 0.5185 | 1.0704 |
| 9 | `kt1_ks3_kf3` | 1 | 3 | 3 | `oom_batch128` | 1905.1 | 0.5238 | 0.5075 | 0.5214 | 1.0951 |
| 10 | `kt1_ks1_kf3` | 1 | 1 | 3 | `oom_batch128` | 1714.9 | 0.5237 | 0.5073 | 0.5210 | 1.0917 |

## Phase 2: Clear Losers

| Variant | kt | ks | kf | Loader | Runtime (s) | Accuracy | Macro F1 | Balanced accuracy | Loss |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `kt2_ks3_kf5` | 2 | 3 | 5 | `oom_batch64` | 2151.6 | 0.4933 | 0.4709 | 0.4872 | 1.1489 |
| `kt1_ks2_kf5` | 1 | 2 | 5 | `oom_batch64` | 1873.8 | 0.4937 | 0.4711 | 0.4871 | 1.1381 |
| `kt2_ks1_kf5` | 2 | 1 | 5 | `oom_batch64` | 1926.1 | 0.4938 | 0.4710 | 0.4875 | 1.1448 |
| `kt3_ks1_kf5` | 3 | 1 | 5 | `oom_batch64` | 2092.5 | 0.4941 | 0.4715 | 0.4878 | 1.1480 |
| `kt2_ks2_kf5` | 2 | 2 | 5 | `oom_batch64` | 2024.4 | 0.4943 | 0.4717 | 0.4880 | 1.1477 |
| `kt1_ks3_kf5` | 1 | 3 | 5 | `oom_batch64` | 1970.1 | 0.4945 | 0.4718 | 0.4878 | 1.1390 |

## Patterns

| Factor | Mean accuracy | Best accuracy | Mean macro F1 | Pattern |
|---|---:|---:|---:|---|
| `kt=1` | 0.5162 | 0.5238 | 0.4990 | Best mean, but not best individual result. |
| `kt=2` | 0.5158 | 0.5255 | 0.4983 | Very similar to `kt=1`; strong top variants. |
| `kt=3` | 0.5143 | 0.5260 | 0.4978 | Best individual result, but lower mean because dense combinations degrade. |
| `ks=1` | 0.5163 | 0.5256 | 0.4993 | Best mean; spatial neighborhoods do not need to be large. |
| `ks=2` | 0.5156 | 0.5260 | 0.4984 | Contains the best variant. |
| `ks=3` | 0.5145 | 0.5255 | 0.4975 | Slightly lower mean and slower. |
| `kf=1` | 0.5213 | 0.5260 | 0.5055 | Strong and relatively cheaper. |
| `kf=2` | 0.5235 | 0.5256 | 0.5076 | Best average setting. |
| `kf=3` | 0.5195 | 0.5255 | 0.5034 | Still competitive in selected combinations. |
| `kf=5` | 0.4974 | 0.5057 | 0.4771 | Consistently harmful and slower. |

Overall, moderate graph density helps, but excessive fixation density hurts. The strongest variants usually combine moderate temporal/spatial density with `kf` in `{1,2,3}`. `kf=5` is the clearest negative pattern: it increases runtime, causes `batch_size=64` retries, and consistently reduces accuracy and macro F1.

## Interpretation

The model seems sensitive to graph density more than to any single direction of temporal or spatial expansion. Larger `kt` or `ks` can help in selected combinations, but the mean results do not improve monotonically. The best variant uses `kt=3, ks=2, kf=1`, suggesting that a somewhat wider temporal context and moderate spatial context are useful when fixation edges are kept sparse.

The architecture search does not give a simple "bigger is always better" conclusion. The largest model won, but `2x64` was close, faster, had no OOM retry, and had the lowest phase 1 loss. For diploma experiments, `2x64` is therefore a defensible compact-model candidate if its own phase 2 graph-density search remains competitive.

## Next Run

Run the same phase 2 graph-density grid with the compact architecture:

```bash
conda run -n gfm python src/emotions/gnn_improvement_experiments/run_gnn_v2_valence_grid_search.py \
  --only-phase phase2 \
  --phase2-num-layers 2 \
  --phase2-hidden-channels 64 \
  --output-root results/gnn_v2_valence_grid_search_2x64_phase2
```

I intentionally do not drop or add phase 2 variants for this run. Keeping the same `kt × ks × kf` grid isolates the architecture change and makes the trade-off comparison interpretable.
