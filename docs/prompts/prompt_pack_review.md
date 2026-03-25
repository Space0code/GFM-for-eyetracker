# Prompt Pack Review (Sub-agent QA)

This file summarizes what was flagged during parallel sub-agent review and what was fixed.

| Prompt file | Main issue(s) found | Fix applied |
|---|---|---|
| `prompt_gnn_high_level_overview_image.md` | Ambiguous abstraction level, mixed output formats, weak anti-hallucination guard | Enforced medium abstraction, Graphviz-only output, strict sectioned output, explicit unknown-handling, omitted training-loop details |
| `prompt_gnn_detailed_dataflow_schema.md` | Split-point ambiguity, mixed task artifacts, unclear inferred vs explicit details | Added explicit split/threshold/scaler rules, classification vs regression artifact separation, inferred-detail marking, strict output structure |
| `prompt_experiment_history_timeline_visual.md` | Hallucination risk for missing outcomes, unclear diagram type/length | Enforced exact milestone list/order, Mermaid timeline format, missing-detail policy (`unknown`), strict output sections and bullet limits |
| `prompt_ablation_results_visual_story.md` | Incomplete ranking context, unclear delta semantics, unspecified output schema | Defined pp deltas, `not_reported` policy, strict output blocks (`RANKED_TABLE`, chart specs, narrative, decision) |
| `prompt_next_steps_brainstorm.md` | Vague runtime tiers, potential invented knobs/results, weak actionability constraints | Defined S/M/L runtime tiers, required `unknown` for uncertain knobs, one-factor bias, strict experiment table schema, concise two-week plan format |
| `prompt_presentation_slide_outline.md` | Hallucination risk, inconsistent structure, unclear backup-slide behavior | Added `[TBD_metric]` policy, strict markdown table columns, exact slide-count rule, backup-slide condition, final decision block |

## Result

All six prompt packs were tightened to be:
- more deterministic,
- more resistant to hallucination,
- easier to compare across repeated runs,
- better aligned to presentation-ready outputs.
