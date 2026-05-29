# Compact GNN v2 Table-6 Valence Phase-2 Grid Search Report

Source run: `results/gnn_v2_valence_grid_search_2x64_phase2/2026-05-28_16-54-08`

Task: 3-class Table-6 valence, subject k-fold (`k=5`, `val_size=1`), GNN v2 only. This run reused the compact architecture from phase 1 (`num_layers=2`, `hidden_channels=64`) and searched only graph-density parameters: `kt in {1,2,3}`, `ks in {1,2,3}`, and `fixation_dilation_k in {1,2,3,5}`. Winner selection used accuracy first, macro F1 second, and lower loss as tie-breaker.

The run completed successfully after resume: `36/36` phase-2 variants finished with `status=success`. The interruption was caused by a full disk during checkpoint writing, not by a model/training failure; generated dataset cache files were cleared and the run resumed from the existing summaries.

## Main Result

| Selection | Variant | Layers | Hidden | kt | ks | kf | Accuracy | Macro F1 | Balanced accuracy | Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Best compact phase 2 | `kt3_ks1_kf5` | 2 | 64 | 3 | 1 | 5 | 0.5221 | 0.5100 | 0.5176 | 1.0014 |
| Compact phase 1 baseline | `layers2_hidden64` | 2 | 64 | 2 | 2 | 3 | 0.5161 | 0.5055 | 0.5135 | 1.0056 |
| Best large phase 2 reference | `kt3_ks2_kf1` | 4 | 256 | 3 | 2 | 1 | 0.5260 | 0.5104 | 0.5200 | 1.0690 |

The compact search improves the compact phase-1 baseline by about `+0.0060` accuracy and `+0.0045` macro F1. It remains slightly below the best large-model phase-2 run in accuracy (`0.5221` vs. `0.5260`), but it is very close in macro F1 (`0.5100` vs. `0.5104`) and runs without OOM batch-size fallback.

## Best Variants

| Rank | Variant | kt | ks | kf | Runtime (s) | Accuracy | Macro F1 | Balanced accuracy | Loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `kt3_ks1_kf5` | 3 | 1 | 5 | 1420.3 | 0.5221 | 0.5100 | 0.5176 | 1.0014 |
| 2 | `kt1_ks1_kf5` | 1 | 1 | 5 | 951.4 | 0.5179 | 0.5073 | 0.5150 | 1.0024 |
| 3 | `kt1_ks1_kf3` | 1 | 1 | 3 | 852.9 | 0.5176 | 0.5070 | 0.5148 | 1.0023 |
| 4 | `kt1_ks3_kf5` | 1 | 3 | 5 | 1065.7 | 0.5174 | 0.5068 | 0.5147 | 1.0034 |
| 5 | `kt1_ks2_kf3` | 1 | 2 | 3 | 907.9 | 0.5173 | 0.5066 | 0.5146 | 1.0031 |

## Weakest Variants

| Variant | kt | ks | kf | Runtime (s) | Accuracy | Macro F1 | Balanced accuracy | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `kt2_ks1_kf2` | 2 | 1 | 2 | 775.8 | 0.5069 | 0.4953 | 0.5046 | 1.0017 |
| `kt3_ks2_kf3` | 3 | 2 | 3 | 1322.0 | 0.5081 | 0.4951 | 0.5047 | 1.0382 |
| `kt3_ks2_kf1` | 3 | 2 | 1 | 1140.9 | 0.5083 | 0.4957 | 0.5049 | 1.0376 |
| `kt3_ks2_kf2` | 3 | 2 | 2 | 1225.0 | 0.5083 | 0.4953 | 0.5049 | 1.0380 |
| `kt3_ks3_kf1` | 3 | 3 | 1 | 1292.4 | 0.5085 | 0.4958 | 0.5053 | 1.0382 |

## Parameter Patterns

| Factor | Mean accuracy | Mean macro F1 | Mean loss | Mean runtime (s) | Pattern |
|---|---:|---:|---:|---:|---|
| `kt=1` | 0.5168 | 0.5060 | 1.0033 | 881.3 | Best average and fastest; compact model does not generally need wide temporal neighborhoods. |
| `kt=2` | 0.5148 | 0.5037 | 1.0077 | 976.0 | Middle ground, but not best on average. |
| `kt=3` | 0.5112 | 0.4984 | 1.0298 | 1300.7 | Contains the best single variant, but often hurts average performance and runtime. |
| `ks=1` | 0.5150 | 0.5038 | 1.0070 | 940.8 | Best mean; compact model benefits from keeping spatial edges modest. |
| `ks=2` | 0.5134 | 0.5016 | 1.0189 | 1085.1 | Weakest mean. |
| `ks=3` | 0.5143 | 0.5027 | 1.0149 | 1132.2 | Slightly better than `ks=2`, but slower than `ks=1`. |
| `kf=1` | 0.5132 | 0.5019 | 1.0129 | 909.0 | Not as strong for compact model as it was for the large model. |
| `kf=2` | 0.5132 | 0.5015 | 1.0160 | 1015.8 | Similar to `kf=1`. |
| `kf=3` | 0.5144 | 0.5028 | 1.0164 | 1105.3 | Slightly stronger. |
| `kf=5` | 0.5161 | 0.5047 | 1.0091 | 1180.6 | Best compact-model average, despite being slower. |

## Interpretation

The compact `2x64` model behaves differently from the larger `4x256` model. In the large-model grid, high fixation density (`kf=5`) was the clearest negative pattern and caused memory pressure. In this compact run, `kf=5` is the best average fixation setting and the winner also uses `kf=5`. This suggests that the smaller model can use denser fixation connectivity without the same over-capacity or memory penalty seen in the larger architecture.

The broadest temporal setting (`kt=3`) is not generally helpful for the compact model, even though the single best configuration uses it. Most strong compact variants use `kt=1`, and the best mean combination by `kt,ks` is `kt=1, ks=3`, followed closely by `kt=1, ks=2` and `kt=1, ks=1`. This points to a conservative graph-density recipe for the compact model: keep temporal connectivity small and use fixation density as the main extra communication path.

## Recommendation

For diploma-scale final runs, the compact model is a defensible trade-off candidate:

- `2x64 + kt3_ks1_kf5` is close to the best large model in macro F1 and only about `0.004` lower in accuracy.
- It avoids OOM retry and is simpler to describe and run.
- If minimizing runtime and graph density matters, `kt1_ks1_kf5` or `kt1_ks1_kf3` are attractive alternatives with only a small accuracy drop from the compact winner.

For strict maximum accuracy, the previous large-model winner (`4x256 + kt3_ks2_kf1`) remains best. For a cleaner complexity/performance trade-off, use compact `2x64 + kt3_ks1_kf5`.
