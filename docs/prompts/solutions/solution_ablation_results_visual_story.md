# Solution: Ablation Results Visual Story

## RANKED_TABLE

| Variant | Valence_delta_pp | Arousal_delta_pp | Note |
|---|---:|---:|---|
| `num_layers_10` (pass2 best) | `+4.35` | `-0.99` | Strongest valence gain vs `baseline_default` |
| `early_stopping_on` | `-1.32` | `+2.01` | Strongest arousal gain; valence tradeoff observed |
| `kt/ks` variants | `[-0.19, -0.01]` | `[-0.14, +0.23]` | Low-impact family in tested setup |
| `edge_weights_on` | `+0.00` | `+0.00` | No measurable change vs baseline in this suite |
| `target_aggregation_last` | `+0.00` | `+0.00` | No measurable change vs `mean` in this suite |
| `other depth choices` | `+3.28` to `-4.36` | `+0.62` to `-3.29` | Depth is sensitive; several settings degrade strongly |

## CHART_SPEC_VALENCE
- Chart type: horizontal bar chart
- X-axis: `Valence_delta_pp` (vs baseline `0.5139`)
- Y-axis: variant names
- Sort: descending by `Valence_delta_pp`
- Visual emphasis: highlight `num_layers_10`
- Add vertical reference line at `0.0`
- Annotation: `Depth is the dominant positive lever for valence`

## CHART_SPEC_AROUSAL
- Chart type: horizontal bar chart
- X-axis: `Arousal_delta_pp` (vs baseline `0.5223`)
- Y-axis: variant names
- Sort: descending by `Arousal_delta_pp`
- Visual emphasis: highlight `early_stopping_on`
- Add vertical reference line at `0.0`
- Annotation: `Arousal improves with early stopping, with tradeoff`

## NARRATIVE
- Valence and arousal respond differently to configuration changes.
- Depth produced the clearest valence improvement in provided results.
- Early stopping helped arousal most, but introduced a valence tradeoff.
- Minor structural toggles (`kt/ks`, edge weights, target aggregation) were low leverage.

## DECISION
- Recommended default: `num_layers_10` (best observed valence improvement).
- Alternative A: `early_stopping_on` for arousal-focused objective.
- Alternative B: baseline/default config as stable control reference.
- Rationale: prioritize high-effect levers first; defer low-impact toggles until depth/optimization is exhausted.
