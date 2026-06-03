# Master prompt for Paper Banana: thesis diagrams

Use Paper Banana's scientific diagram / methodology diagram workflow to create
publication-ready thesis figures. Create three diagram specifications for a
bachelor's thesis about a graph neural network for emotion recognition from
eye-tracking data. Diagram 2 has two requested variants, so the expected output
is four separate standalone SVG files:

1. `diagram1_cevovod_priprave_podatkov.svg`
2. `diagram2_predlagani_gnn_arhitektura_inferenca.svg`
3. `diagram2_predlagani_gnn_arhitektura_ucenje.svg`
4. `diagram3_ena_heterogena_gnn_plast.svg`

Do not combine these into one multi-panel figure. Each SVG must be usable as an
independent figure in a LaTeX thesis.

## Recommended Paper Banana settings

These are generation settings, not figure labels:

- use the Scientific Diagram Maker / Methodology Diagram workflow;
- use 2K resolution for drafts and 4K for final camera-ready export if available;
- use the default multi-iteration refinement unless the interface requires a
  specific number; 3 iterations is a good default;
- after generation, review the diagrams for scientific accuracy and label
  readability before using them in the thesis.

The diagrams should be coherent with one another, but each diagram should use
the layout that best communicates its own content. Follow the specific
composition and readability constraints given under each diagram.

## Global visual constraint: use only palette B

Use this color palette consistently across all three diagrams:

- dark gray for text, outlines, arrows, and neutral structure: `#2F3437`
- pastel blue: `#8ECAE6`
- pastel teal: `#A8DADC`
- pastel purple: `#B8A1D9`
- pastel orange: `#F4A261`
- pastel pink: `#E9A6A6`
- light background: `#F7F7F2`

You may use transparency, lighter tints, or white space if needed, but avoid
introducing unrelated saturated colors. If a darker variant is needed for
contrast, prefer dark gray `#2F3437` rather than inventing a new hue.

All visible explanatory text in the diagrams must be in Slovenian. Use correct
Slovenian technical terminology. Keep text concise and readable. Code-like model
and layer names may remain unchanged where they are standard labels in the
thesis: `GNN`, `GCNConv`, `MLP`, `LayerNorm`, `Dropout`, `GazeMAE`, `SVM`,
`LightGBM`, and `HeteroData`.

Generate editable vector graphics as SVG. Use real SVG text elements where
possible so labels can be edited later in a vector editor. Do not rasterize text.
The diagrams should be suitable for inclusion in a LaTeX thesis.

Do not render Markdown syntax as visible text in the figures. Bullets, section
headings, file names, and instructions from this prompt are for you, not labels
to copy literally into the diagram.

Use small panel labels or compact figure titles only if they help. Do not create
large decorative titles or subtitles that consume vertical space; the thesis
will provide captions outside the figure.

## Global relation color mapping

Use this mapping consistently when relation types are visually distinguished:

- temporal forward / `časovne naprej`: pastel blue `#8ECAE6`
- temporal backward / `časovne nazaj`: pastel teal `#A8DADC`
- spatial / `prostorske`: pastel orange `#F4A261`
- fixation / `fiksacijske`: pastel purple `#B8A1D9`
- prediction/loss/output accents: pastel pink `#E9A6A6`

If a diagram does not need separate relation colors, keep the palette subtle and
use dark gray for arrows and outlines.

## Global quality checklist

Before final output, verify every SVG against these criteria:

- the scientific content matches the requested architecture and data pipeline;
- every visible label is in Slovenian;
- no text is clipped or too close to a border;
- no arrowhead overlaps a box label;
- the figure remains readable when scaled to thesis text width;
- the layout is visually simple enough to understand quickly;
- no unnecessary decorative effects, 3D objects, gradients, or drop shadows;
  keep the style flat, clean, and academic.

---

# Diagram 1: Visokonivojski cevovod priprave podatkov

## Purpose

Create a high-level data-processing diagram showing how eye-tracking data become
model inputs. The diagram should be understandable as a standalone methodology
figure in a thesis. It should explain the main path from MAHNOB-HCI
eye-tracking measurements to graph samples for the GNN, while also showing the
two non-graph model branches.

This is a high-level diagram, not the full implementation-level pipeline. It
should be clear, compact, and visually convincing.

The generated diagrams must be readable at thesis text width. Diagram 1 contains
more lists than the architecture diagrams, so the main risk is overcrowding.
Use fewer, larger blocks rather than many small boxes with tiny text.

## Canvas and readability constraints

Use a fixed landscape SVG canvas with all content inside the viewBox:

- recommended canvas: `1800 × 1000` or `1900 × 1050`;
- no element may extend outside the canvas;
- no browser scrollbars should appear when the SVG is opened;
- leave at least 35 px of outer margin on every side;
- minimum body text size: 20 px;
- minimum small annotation text size: 18 px;
- optional small panel title size: 24--28 px;
- do not use text smaller than 18 px;
- use compact arrowheads that do not overlap boxes;
- leave visible whitespace between arrowheads and target boxes.

Every text label must fit inside its box. If a label is long, split it into
2--3 short lines. Do not allow text to touch box borders or overlap arrows.

## Required composition

Use a compact two-level composition:

- upper/main band: the main graph-data path;
- lower/secondary band: the two non-graph branches.

The main graph-data path should be visually dominant and should occupy most of
the diagram width:

`MAHNOB-HCI meritve sledilnika pogleda` → `Čiščenje in filtriranje` →
`10-sekundna okna` → `Grafovski učni vzorec`

The two non-graph branches should start from `10-sekundna okna` and sit below
the main flow. They must be smaller and visually secondary. Use short vertical
or diagonal arrows from the window block to these branches. Do not route long
curved arrows across the whole diagram.

Do not make the diagram a single long horizontal strip of equally sized blocks.
The graph sample block may be wider/taller than the earlier blocks because it
contains the most important output representation.

## Main message

The main message is:

> From MAHNOB-HCI eye-tracking signals, we clean and filter the data, split it
> into 10-second windows, and convert each usable window either into a graph for
> the GNN or into alternative window-level representations for non-graph models.

## Required main flow

Show the following main flow from left to right:

1. `MAHNOB-HCI meritve sledilnika pogleda`
2. `Čiščenje in filtriranje`
3. `10-sekundna okna`
4. `Grafovski učni vzorec`

Do not add a separate fifth block called `vhod v GNN`. The graph sample is the
processed data output for the GNN.

Keep this as the dominant visual path. Use thick solid arrows for this path and
thinner arrows for secondary branches.

## Block 1: MAHNOB-HCI measurements

This first block should remain fairly detailed. Show a table-like or structured
view of the available raw eye-tracking signals.

Include these signals:

- `x-koordinata pogleda`
- `y-koordinata pogleda`
- `velikost leve zenice`
- `velikost desne zenice`
- `čas meritve`
- `razdalja do sledilnika`
- `indeks fiksacije`
- `trajanje fiksacije`

Important: show `razdalja do sledilnika` as one signal only. Do not split it
into left-eye and right-eye distance in this diagram.

It is fine to include small example traces or values for each signal, but they
do not need to be numerically exact.

Do not show all eight signals as eight separate large boxes. Prefer one
structured table-like block with two columns or four compact rows. The block must
remain readable and should not become taller than the whole diagram.

## Block 2: cleaning and filtering

Show the key cleaning/filtering operations:

- `veljavnost obeh oči`
- `interpolacija kratkih vrzeli`
- `glajenje zenic`
- `izključitev neveljavnih oznak`
- `filter osamelcev q01-q99`

Use concise labels. The details are explained in the thesis text, so this block
only needs to communicate the major processing ideas.

Keep this block compact. It is acceptable to show the operations as short
checklist items inside one box. Do not draw a separate pipeline stage for every
cleaning operation.

## Block 3: 10-second windows

Show the segmentation into non-overlapping windows.

Include these facts:

- `T = 10 s`
- `brez prekrivanja`
- `najmanj 60 meritev na okno`
- `ena oznaka se prenese na celotno okno`

The diagram should make it clear that each usable window becomes one training
sample.

Draw this block visually as segmented windows or a short timeline split into
several 10-second chunks. Keep the four facts above as concise labels, not as a
paragraph.

## Block 4: graph training sample

This block should show the processed graph data for the GNN. It should not show
the internal GNN architecture.

Show a graph where:

- nodes represent eye-tracking measurements inside one 10-second window;
- edges represent relationships between measurements;
- the graph is the final data representation for the GNN.

Include a concise node-feature list:

- `x, y`
- `leva/desna zenica`
- `normalizirani čas`
- `razdalja do sledilnika`
- `trajanje fiksacije`

Show three edge families and label them:

- `časovne povezave naprej/nazaj, k_t = 1`
- `prostorske povezave najbližjih sosedov, k_s = 1`
- `razširjene povezave znotraj fiksacije, k_f = 3`

At the bottom or in a clearly visible place in this block, state the output:

- `izhod: heterogeni graf / HeteroData`
- `ciljna oznaka na ravni okna/grafa: valenca ali vzburjenost`

Draw the graph in a way that looks like a real eye-tracking graph rather than a
generic network icon. The graph does not need to be mathematically exact, but it
should visually suggest gaze positions connected by temporal, spatial, and
fixation relations.

Make the graph sample block the most visually important output block. However,
do not cram every node feature and every edge family into the graph drawing
itself. Use one mini graph plus two compact side notes:

- `značilke vozlišč`;
- `tipi povezav`.

Use relation colors consistently with the global palette and with Diagrams 2 and
3. If the relation labels become too long, use shortened labels in the diagram:

- `časovne naprej/nazaj`;
- `prostorske`;
- `fiksacijske`.

## Non-graph branches

From the `10-sekundna okna` block, show two secondary branches below the main
flow:

1. `SVM / LightGBM / MLP`
   - label: `agregirane statistike istih signalov`

2. `GazeMAE + MLP`
   - label: `samo (x, y), pet 2-sekundnih kosov, 500 Hz`

These branches should be visually secondary. They are important, but the main
focus of this diagram is the transformation into the graph sample.

Put these two branches side by side in the lower band. Use smaller boxes than
the main graph path, but keep their text readable. Do not let these branches
compete visually with `Grafovski učni vzorec`.

## Final self-check before output

Before finalizing the SVG, check:

- no text is clipped or touches a box border;
- no arrowhead overlaps text;
- the main flow is understandable in less than five seconds;
- the non-graph branches are clearly secondary;
- the graph sample block is the most important output;
- the figure remains readable when scaled to thesis text width;
- all parameter values match the selected architecture: `k_t = 1`, `k_s = 1`,
  `k_f = 3`.

## Avoid

- Do not draw stacked neural network layers in this diagram.
- Do not show GNN message passing, pooling, attention, or classifier heads here.
- Do not split the tracker distance into left and right distance.
- Do not make the non-graph branches visually dominate the graph path.
- Do not create a very long one-row pipeline that requires horizontal scrolling.
- Do not use tiny table text for the raw signals.
- Do not route long sweeping arrows across the whole figure.
- Do not make all four main blocks equal if the graph sample block needs more
  room.
- Do not use colors outside palette B unless absolutely necessary for contrast.

---

# Diagram 2: Arhitektura predlaganega GNN

## Purpose

Create a clean architecture diagram of the proposed heterogeneous GNN used in
the thesis. The diagram should show the end-to-end model architecture after the
graph sample has already been constructed. It should be suitable for the model
chapter of a bachelor's thesis.

This is not a data-preparation diagram. Do not repeat the full raw-data cleaning
pipeline from Diagram 1. Start from the heterogeneous graph sample and show how
the proposed GNN maps it to a class prediction.

The first generated attempt made the architecture too wide and hard to read.
Avoid a single long chain of many small blocks. Use a compact two-band layout:
the main node/graph representation path on the upper band and the edge-weight
path below it. The two bands should meet at the GNN block.

Create two variants of this diagram:

1. `diagram2_predlagani_gnn_arhitektura_inferenca.svg`
   - inference-only version;
   - no loss block and no backpropagation arrow.

2. `diagram2_predlagani_gnn_arhitektura_ucenje.svg`
   - training version;
   - includes a subtle dashed training path with the loss.

The two variants should be visually almost identical, except for the additional
loss/training path in the second variant.

## Canvas and readability constraints

Use a fixed landscape SVG canvas with all content inside the viewBox:

- recommended canvas: `1800 × 950` or `1900 × 1000`;
- no element may extend outside the canvas;
- no browser scrollbars should appear when the SVG is opened;
- leave at least 35 px of outer margin on every side;
- minimum body text size: 20 px;
- minimum small annotation text size: 18 px;
- optional small panel title size: 24--28 px;
- do not use text smaller than 18 px;
- use arrowheads that are small enough not to overlap boxes;
- leave visible whitespace between arrowheads and target boxes.

Every text label must fit inside its box. If a label is long, split it into
2--3 short lines. Do not allow text to touch box borders or overlap arrows.

## Main message

The main message is:

> The proposed model maps a 10-second heterogeneous graph of eye-tracking
> measurements into node representations, combines relation-specific message
> passing through two heterogeneous GNN layers, aggregates node representations
> with attention, and predicts the valence or arousal class.

## Fixed architecture values

Use these values explicitly in the diagram:

- `število plasti: L = 2`
- `širina predstavitev: H = 64`
- `časovni sosedi: k_t = 1`
- `prostorski sosedi: k_s = 1`
- `fiksacijska razširitev: k_f = 3`

Do not use generic placeholders such as `L ×` or `H` without also showing the
fixed values above. The diagram should make clear that this is the selected
architecture for the thesis figure.

Show these fixed values as a compact row of small chips near the top of the
figure, not as one long sentence. Use for example:

- `L = 2`
- `H = 64`
- `k_t = 1`
- `k_s = 1`
- `k_f = 3`

## Required forward flow

Show the following forward flow, but do not draw it as one long uninterrupted
row. Use the required two-band structure described below.

1. `heterogeni graf`
2. `značilke vozlišč X`
3. `vhodni MLP`
4. `začetne predstavitve h^(0)`
5. `2 × heterogena GNN plast`
6. `predstavitve vozlišč h_i`
7. `pozornostno združevanje`
8. `predstavitev grafa h_G`
9. `napovedna glava MLP`
10. `napoved razreda`

Keep this main path visually dominant and easy to follow.

## Required composition

Use this composition:

- left column: input graph and extracted feature blocks;
- middle column: node preprocessing, edge-weight computation, and the GNN block;
- right column: graph-level readout and prediction.

Upper band:

`heterogeni graf` → `značilke vozlišč X` → `vhodni MLP` → `h^(0), H = 64`
→ `2 × heterogena GNN plast`

Lower band:

`značilke povezav R` → `MLP za uteži povezav` → `tanh` →
`normalizacija po ciljnih vozliščih` → `w_ij^(r)` → feeds upward into the
`2 × heterogena GNN plast` block.

Right side after the GNN block:

`predstavitve vozlišč h_i` → `pozornostno združevanje` →
`predstavitev grafa h_G` → `napovedna glava MLP` → `napoved razreda`

The lower edge-weight band should be directly below the node path, not far away
at the bottom of the canvas. Use short orthogonal or gently curved arrows. Do
not draw a long sweeping curve from the graph input to the GNN block.

## Heterogeneous graph input

The first block should show a compact graph icon or mini graph, not just a
generic rectangle. The graph should suggest eye-tracking measurements in a
10-second window.

Inside or next to this input block, show the relation types in this exact order:

- `časovne povezave naprej`
- `časovne povezave nazaj`
- `prostorske povezave`
- `fiksacijske povezave`

Also include a compact note:

- `ciljna oznaka: valenca ali vzburjenost`

Keep this block smaller than the GNN block. The mini graph is illustrative; the
architecture flow should remain the main visual focus.

## Node-feature branch

Show the node-feature path clearly:

- `značilke vozlišč X`
- `x, y`
- `leva/desna zenica`
- `normalizirani čas`
- `razdalja do sledilnika`
- `trajanje fiksacije`

The feature list may be inside the `značilke vozlišč X` block or in a small
attached note. Keep it readable. Do not make the feature list tiny.

The node-feature path should go through:

- `vhodni MLP`
- `h^(0), H = 64`

## Edge-weight branch

Show a separate edge-weight branch that feeds into the heterogeneous GNN layers.
This branch should be visually secondary but still clearly present.

Use this flow:

1. `značilke povezav R`
2. `MLP za uteži povezav`
3. `tanh`
4. `normalizacija po ciljnih vozliščih`
5. `predznačene uteži w_ij^(r)`

For `značilke povezav R`, include a compact list:

- `t_i, t_j, Δt`
- `Δx, Δy, razdalja`
- `Δd`
- `smer ρ pri časovnih povezavah`

The diagram should communicate that learned signed scalar edge weights are used
by the relation-specific GCN convolutions.

Important layout requirement: the arrow from `predznačene uteži w_ij^(r)` to the
GNN block should be short and should enter the bottom or lower-left side of the
GNN block. Do not route it as a long curved arrow across the figure.

## Central GNN block

The central block should be large enough to be legible and should be labeled:

- `2 × heterogena GNN plast`

Inside the central block or immediately below it, include:

- `GCNConv po relacijah`
- `GELU`
- `fuzija relacij z MLP`
- `rezidualna povezava`
- `LayerNorm + Dropout`

Do not draw the full internal structure of one layer here; that is Diagram 3.
This block should summarize the repeated layer stack.

The GNN block should be the visual anchor of the diagram. It should be bigger
than the preprocessing and readout blocks, but not so large that the rest of the
diagram is pushed to the margins.

## Graph-level readout and classifier

After the GNN block, show:

- `predstavitve vozlišč h_i`
- `pozornostno združevanje`
- `predstavitev grafa h_G`
- `napovedna glava MLP`
- `napoved razreda`

In the `napoved razreda` block, show that the same architecture is trained
separately for the two tasks:

- `razred valence`
- `razred vzburjenosti`

Do not imply that one model predicts both targets jointly. The thesis uses
separate classification tasks.

## Training variant only

For `diagram2_predlagani_gnn_arhitektura_ucenje.svg`, add a subtle training path:

- a block labeled `navzkrižna entropija`;
- a dashed arrow from `napoved razreda` to `navzkrižna entropija`;
- a dashed arrow from `navzkrižna entropija` back toward the learnable model
  blocks;
- label the dashed feedback path `učenje parametrov`.

This training path must not dominate the diagram. It should be visually
secondary and should not make the main forward path harder to read.

The dashed training path must be local and tidy:

- place `navzkrižna entropija` directly below `napoved razreda`;
- use a short vertical dashed arrow from `napoved razreda` down to the loss;
- use one short dashed return arrow from the loss to a small bracket labeled
  `učljivi parametri`;
- the bracket should point to the learnable blocks as a group, not draw a long
  curve across the entire diagram;
- do not route the dashed return arrow through the middle of the diagram.

For `diagram2_predlagani_gnn_arhitektura_inferenca.svg`, omit this entire loss
and training path.

## Readability requirements

This diagram must be especially readable:

- Use a wide but compact landscape layout.
- Prefer 8--10 major boxes total in the main path. If needed, merge small
  consecutive operations into a single box with a two-line label.
- Keep line crossings to a minimum.
- Use large text and generous spacing.
- Avoid long paragraphs inside boxes.
- Use no more than 2 lines of text in most blocks.
- Use small attached notes only where needed.
- Make sure all labels remain readable when the SVG is scaled to thesis text
  width.
- Before finalizing the SVG, check that no text is clipped, no arrowhead touches
  text, and no box label is partly hidden behind an arrow.

## Avoid

- Do not show the raw MAHNOB-HCI cleaning pipeline again.
- Do not draw a generic black-box neural network without the GNN-specific
  components.
- Do not collapse the four relation types into a single undifferentiated edge
  type.
- Do not use the old architecture with only temporal and spatial relations.
- Do not create a very long one-row pipeline that requires horizontal scrolling.
- Do not draw long sweeping arrows across the whole diagram.
- Do not put the edge-weight branch far below the main path.
- Do not use English labels.
- Do not use colors outside palette B unless absolutely necessary for contrast.

---

# Diagram 3: Ena heterogena GNN plast

## Purpose

Create a detailed but readable diagram of one layer of the proposed heterogeneous
GNN. This diagram should explain what happens inside the central `heterogena GNN
plast` block from Diagram 2.

The diagram should focus only on one GNN layer. Do not include graph-level
attention pooling, the graph representation, the prediction head, or the loss.

The first generated attempt was mostly correct but too wide, with a huge
diagonal residual arrow and a squeezed output on the right. Avoid that. Use a
compact grid layout and route the residual connection as a clean local top lane
that enters the residual-sum block directly.

## Canvas and readability constraints

Use a fixed landscape SVG canvas with all content inside the viewBox:

- recommended canvas: `1800 × 950` or `1900 × 1000`;
- no element may extend outside the canvas;
- no browser scrollbars should appear when the SVG is opened;
- leave at least 35 px of outer margin on every side;
- minimum body text size: 20 px;
- minimum small annotation text size: 18 px;
- optional small panel title size: 24--28 px;
- do not use text smaller than 18 px;
- use compact arrowheads that do not overlap boxes;
- leave clear spacing between arrows and labels.

Every text label must fit inside its box. If a label is long, split it into
2--3 short lines. Do not allow text to touch box borders or overlap arrows.

## Main message

The main message is:

> One heterogeneous GNN layer performs relation-specific message passing for
> temporal-forward, temporal-backward, spatial, and fixation relations, fuses the
> resulting relation-specific node representations with an MLP, and stabilizes
> the update with a residual connection, LayerNorm, and Dropout.

## Fixed architecture context

Show the layer as part of the selected architecture:

- `plast ℓ od 2`
- `širina predstavitev H = 64`

The layer diagram should not imply an arbitrary number of layers. It should make
clear that this layer is repeated twice in the selected architecture.

## Required layout

Use a left-to-right grid layout with four parallel relation-specific paths in
the middle. The diagram should have four visual zones:

1. inputs;
2. relation-specific message passing;
3. relation fusion;
4. residual update and stabilization.

The relation paths should be vertically aligned and evenly spaced. The fusion
and stabilization blocks should not be pushed all the way to the right edge.

Inputs on the left:

1. `predstavitve vozlišč h_i^(ℓ)`
2. `uteži povezav w_ij^(r)`

The `h_i^(ℓ)` input should feed all four relation-specific paths. The
`w_ij^(r)` input should also feed the relation-specific paths, but it can be
drawn as a thinner secondary set of arrows.

Use two separate input buses:

- a solid bus for `h_i^(ℓ)`;
- a dashed bus for `w_ij^(r)`.

Keep the buses visually separated. Do not let the dashed edge-weight bus overlap
the solid node-representation bus.

## Relation-specific message passing

In the center, show four parallel blocks in this exact order from top to bottom:

1. `GCNConv`
   `časovne naprej`

2. `GCNConv`
   `časovne nazaj`

3. `GCNConv`
   `prostorske`

4. `GCNConv`
   `fiksacijske`

Each block should clearly represent relation-specific message passing.

Each relation path should output one relation-specific node representation:

- from `časovne naprej`: `h_i^(t+)`
- from `časovne nazaj`: `h_i^(t-)`
- from `prostorske`: `h_i^(s)`
- from `fiksacijske`: `h_i^(f)`

Use the palette consistently and use subtle relation accents if useful:

- temporal-forward and temporal-backward paths may use pastel blue/teal accents;
- spatial path may use pastel orange accents;
- fixation path may use pastel purple or pastel pink accents.

Do not invent a fourth saturated color.

Put `GCNConv + GELU` on the first line of each relation block and the relation
name on the second line. Keep each relation block the same size.

## Relation fusion

After the four relation-specific outputs, show:

1. `konkatenacija`
2. `MLP fuzija relacij`

Include a compact formula near this fusion block:

`[h_i^(t+) || h_i^(t-) || h_i^(s) || h_i^(f)] → h_i^fuz`

The formula is optional. If it would become smaller than 18 px or visually
clutter the diagram, omit the formula and keep only the `konkatenacija` and
`MLP fuzija relacij` blocks. Readability is more important than including every
symbol.

## Residual and stabilization path

Show a residual arrow from the input `h_i^(ℓ)` around the relation-specific paths
and fusion block into:

- `rezidualni seštevek`

Then show a compact stabilization sequence:

1. `LayerNorm`
2. `Dropout`

Finally output:

- `predstavitve vozlišč h_i^(ℓ+1)`

The residual arrow must be visually clear but local and tidy:

- route it as a top lane above the relation paths;
- use horizontal and vertical segments rather than a large diagonal curve;
- enter the `rezidualni seštevek` block directly;
- do not point the residual arrow at a title or at the general stabilization
  area;
- do not let it cross through relation outputs or the fusion block.

Show `GELU` inside the relation-specific/fusion part of the layer, not as a
separate operation after the residual sum. The intended order is relation-specific
convolution, GELU, relation fusion, residual sum, LayerNorm, and Dropout.

To preserve space, it is acceptable to combine `LayerNorm` and `Dropout` into
one block labeled `LayerNorm + Dropout`, as long as the residual sum remains a
separate block.

## Edge weights

Make clear that the relation-specific convolutions use learned signed scalar
edge weights:

- label the secondary input as `predznačene uteži povezav w_ij^(r)`;
- connect it to all four `GCNConv` blocks;
- include a small note: `normalizirane po ciljnih vozliščih`.

Do not repeat the full edge-weight MLP pipeline in this diagram. That pipeline
belongs to Diagram 2.

The edge-weight arrows should be thin and dashed. They should enter the relation
blocks from the left or lower-left. Do not draw them as large decorative curves.

## Readability requirements

This diagram must be very legible:

- Use a wide landscape composition.
- Use a grid-like layout with aligned blocks.
- Keep the four relation paths visually parallel.
- Keep arrow routing clean and avoid arrow crossings.
- Use large labels.
- Do not use dense equations.
- Do not put too much text inside the relation blocks.
- Make sure the diagram remains readable at thesis text width.
- Avoid a long chain that stretches to the edge of the canvas. If the right side
  becomes cramped, combine `LayerNorm` and `Dropout` into one stabilization
  block.
- Before finalizing the SVG, check that no text is clipped, no arrowhead touches
  text, and the output block is not squeezed against the right margin.

## Avoid

- Do not show graph-level pooling or classification.
- Do not show raw data preprocessing.
- Do not show only two relations; the current model has four relation types.
- Do not merge temporal-forward and temporal-backward into one block.
- Do not draw a generic neural-network stack.
- Do not draw a huge diagonal residual arrow across the whole figure.
- Do not let the residual arrow end anywhere except the `rezidualni seštevek`
  block.
- Do not use tiny formulas that become unreadable at thesis width.
- Do not use English labels.
- Do not use colors outside palette B unless absolutely necessary for contrast.
