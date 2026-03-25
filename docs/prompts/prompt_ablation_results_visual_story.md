# Intention
Generate a **visual story of ablation results** with clear takeaways.

# Prompt
Build a decision-oriented ablation summary from these facts:
- Baseline (balanced accuracy): valence `0.5139`, arousal `0.5223`.
- Best valence: `num_layers_10` = `0.5574` (`+4.35 pp`).
- Best arousal: `early_stopping_on` = `0.5424` (`+2.01 pp`), with valence drop.
- `kt/ks` changes: small effects.
- Edge weights on/off: near no change.
- Target aggregation mean vs last: near no change.
- Depth sensitivity is high (some depths degrade strongly).

Rules:
- Use percentage points (`pp`) for deltas.
- If a value is not given, write `not_reported` (do not guess).
- Keep it concise and slide-friendly.

Output format (strict):
1. `RANKED_TABLE` with columns:
   `Variant | Valence_delta_pp | Arousal_delta_pp | Note`
2. `CHART_SPEC_VALENCE` (Markdown bullet spec for one chart)
3. `CHART_SPEC_AROUSAL` (Markdown bullet spec for one chart)
4. `NARRATIVE` (4 bullets)
5. `DECISION` (recommended config + 2 alternatives + rationale)
