# Master prompt for Paper Banana: thesis diagrams

You will create three diagrams for a bachelor's thesis about a graph neural
network for emotion recognition from eye-tracking data.

The diagrams should be coherent with one another, but do not force them into a
rigid template. Choose the composition, spacing, iconography, and level of
visual abstraction that best communicates each diagram. The only strict visual
constraint is the color palette below.

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

All visible text in the diagrams must be in Slovenian. Use correct Slovenian
technical terminology. Keep text concise and readable.

If possible, generate the diagrams as editable vector graphics, preferably SVG.
The diagrams should be suitable for inclusion in a LaTeX thesis and later manual
editing in a vector editor.

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

## Block 2: cleaning and filtering

Show the key cleaning/filtering operations:

- `veljavnost obeh oči`
- `interpolacija kratkih vrzeli`
- `glajenje zenic`
- `izključitev neveljavnih oznak`
- `filter osamelcev q01-q99`

Use concise labels. The details are explained in the thesis text, so this block
only needs to communicate the major processing ideas.

## Block 3: 10-second windows

Show the segmentation into non-overlapping windows.

Include these facts:

- `T = 10 s`
- `brez prekrivanja`
- `najmanj 60 meritev na okno`
- `ena oznaka se prenese na celotno okno`

The diagram should make it clear that each usable window becomes one training
sample.

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

- `časovne povezave naprej/nazaj, k_t = 2`
- `prostorske povezave najbližjih sosedov, k_s = 2`
- `razširjene povezave znotraj fiksacije, k_f = 3`

At the bottom or in a clearly visible place in this block, state the output:

- `izhod: heterogeni graf / HeteroData`
- `ciljna oznaka na ravni okna/grafa: valenca ali vzburjenost`

If possible, draw the graph in a way that looks like a real eye-tracking graph
rather than a generic network icon. The graph does not need to be mathematically
exact, but it should visually suggest gaze positions connected by temporal,
spatial, and fixation relations.

## Non-graph branches

From the `10-sekundna okna` block, show two secondary branches below the main
flow:

1. `SVM / LightGBM / MLP`
   - label: `agregirane statistike istih signalov`

2. `GazeMAE + MLP`
   - label: `samo (x, y), pet 2-sekundnih kosov, 500 Hz`

These branches should be visually secondary. They are important, but the main
focus of this diagram is the transformation into the graph sample.

## Avoid

- Do not draw stacked neural network layers in this diagram.
- Do not show GNN message passing, pooling, attention, or classifier heads here.
- Do not split the tracker distance into left and right distance.
- Do not make the non-graph branches visually dominate the graph path.
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

Create two variants of this diagram:

1. `diagram2_predlagani_gnn_arhitektura_inferenca.svg`
   - inference-only version;
   - no loss block and no backpropagation arrow.

2. `diagram2_predlagani_gnn_arhitektura_ucenje.svg`
   - training version;
   - includes a subtle dashed training path with the loss.

The two variants should be visually almost identical, except for the additional
loss/training path in the second variant.

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

## Required forward flow

Show the following left-to-right flow:

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

For `diagram2_predlagani_gnn_arhitektura_inferenca.svg`, omit this entire loss
and training path.

## Readability requirements

This diagram must be especially readable:

- Use a wide horizontal layout.
- Keep line crossings to a minimum.
- Use large text and generous spacing.
- Avoid long paragraphs inside boxes.
- Use no more than 2 lines of text in most blocks.
- Use small attached notes only where needed.
- Make sure all labels remain readable when the SVG is scaled to thesis text
  width.

## Avoid

- Do not show the raw MAHNOB-HCI cleaning pipeline again.
- Do not draw a generic black-box neural network without the GNN-specific
  components.
- Do not collapse the four relation types into a single undifferentiated edge
  type.
- Do not use the old architecture with only temporal and spatial relations.
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

Use a left-to-right layout with four parallel relation-specific paths in the
middle.

Inputs on the left:

1. `predstavitve vozlišč h_i^(ℓ)`
2. `uteži povezav w_ij^(r)`

The `h_i^(ℓ)` input should feed all four relation-specific paths. The
`w_ij^(r)` input should also feed the relation-specific paths, but it can be
drawn as a thinner secondary set of arrows.

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

## Relation fusion

After the four relation-specific outputs, show:

1. `konkatenacija`
2. `MLP fuzija relacij`

Include a compact formula near this fusion block:

`[h_i^(t+) || h_i^(t-) || h_i^(s) || h_i^(f)] → h_i^fuz`

If the formula would become too small, use the words only and omit the formula.
Readability is more important than including every symbol.

## Residual and stabilization path

Show a residual arrow from the input `h_i^(ℓ)` around the relation-specific paths
and fusion block into:

- `rezidualni seštevek`

Then show:

1. `LayerNorm`
2. `Dropout`

Finally output:

- `predstavitve vozlišč h_i^(ℓ+1)`

The residual arrow should be visually clear but should not cross through the
middle of the diagram. Route it around the main blocks.

Show `GELU` inside the relation-specific/fusion part of the layer, not as a
separate operation after the residual sum. The intended order is relation-specific
convolution, GELU, relation fusion, residual sum, LayerNorm, and Dropout.

## Edge weights

Make clear that the relation-specific convolutions use learned signed scalar
edge weights:

- label the secondary input as `predznačene uteži povezav w_ij^(r)`;
- connect it to all four `GCNConv` blocks;
- include a small note: `normalizirane po ciljnih vozliščih`.

Do not repeat the full edge-weight MLP pipeline in this diagram. That pipeline
belongs to Diagram 2.

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

## Avoid

- Do not show graph-level pooling or classification.
- Do not show raw data preprocessing.
- Do not show only two relations; the current model has four relation types.
- Do not merge temporal-forward and temporal-backward into one block.
- Do not draw a generic neural-network stack.
- Do not use English labels.
- Do not use colors outside palette B unless absolutely necessary for contrast.
