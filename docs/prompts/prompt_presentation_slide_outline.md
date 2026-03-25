# Intention
Generate a **full slide-deck outline** for project presentation.

# Prompt
Create a 12-15 slide outline for presenting this eye-tracking GNN project to a mixed ML/research audience.

Required sections (must appear):
1. Problem and motivation
2. Data and preprocessing
3. GNN architecture
4. Baseline models
5. Experimental protocol
6. Main benchmark results
7. Ablation findings
8. Failures and lessons learned
9. Proposed next steps

Rules:
- Do not invent metrics, dataset sizes, or claims.
- If specific values are missing, use placeholders like `[TBD_metric]`.
- Keep tone rigorous and honest.

Output format (strict):
- Return a Markdown table with columns:
  `Slide # | Title | Key message | Visual recommendation | Speaker notes | Backup slide (if relevant)`
- Use exactly 13 slides unless clearly justified otherwise.
- Speaker notes: 2-4 bullets per slide.
- Add backup slide only when a realistic follow-up question is likely.

Finish with:
- `TOP_3_DECISIONS_FOR_NEXT_ITERATION` (3 bullets)
