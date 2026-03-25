# Intention
Generate a **single high-level visual** of the GNN for presentation use.

# Prompt
You are helping me design one presentation figure for my eye-tracking GNN project.

Create a **slide-ready architecture diagram** at medium abstraction (high-level, not code internals) showing the end-to-end inference path.

Use these exact facts:
- Input is eye-tracking windows.
- Each window becomes a graph with:
  - nodes = time samples within that window,
  - temporal edges = nearby timesteps (kt neighborhood),
  - spatial edges = kNN in screen-space (ks neighborhood).
- Node features: `x-avg`, `y-avg`, `pupil-size-left-avg`, `pupil-size-right-avg`.
- Model: optional preprocess MLP -> multi-layer heterogeneous GNN (temporal + spatial relations) -> residual + layernorm -> graph pooling (mean or mean+max) -> head MLP.
- Output task can be binary, multiclass, or regression.

Constraints:
- Do **not** include training-loop details (optimizer, scheduler, losses, CV splits).
- Use one node type and two edge types unless explicitly stated otherwise.
- If anything is unknown, mark it as `unknown` instead of guessing.

Styling goals:
- Clean, modern, academic look.
- 16:9 slide-friendly layout.
- Strong visual hierarchy with minimal clutter.
- Include a legend for edge types.
- Include 2 short callouts: `time dynamics`, `screen-space patterns`.

Output format (strict):
1. `VARIANT_A_TECHNICAL` (complete diagram spec, Graphviz DOT)
2. `VARIANT_B_EXECUTIVE` (complete diagram spec, Graphviz DOT)
3. `CAPTION_A` (2-3 sentences)
4. `CAPTION_B` (2-3 sentences)

Keep text in each diagram node concise (roughly <= 10 words where possible).
