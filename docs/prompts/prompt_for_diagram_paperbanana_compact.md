# Compact Paper Banana Prompt: Thesis Diagrams

Use Paper Banana Scientific Diagram Maker / Methodology Diagram workflow. Create four separate standalone editable PNG files, not one multi-panel figure:

1. `diagram1_cevovod_priprave_podatkov.png`
2. `diagram2_predlagani_gnn_arhitektura_inferenca.png`
3. `diagram2_predlagani_gnn_arhitektura_ucenje.png`
4. `diagram3_ena_heterogena_gnn_plast.png`

Recommended generation settings if available: Scientific Diagram Maker / Methodology Diagram workflow, 1K, default multi-iteration refinement; 3 iterations is a good default. These are settings, not visible labels.

These figures are for a LaTeX bachelor's thesis about a GNN for emotion recognition from eye-tracking data. Use flat, clean, academic vector style. Use real editable text; do not rasterize text. Do not render Markdown syntax, filenames, bullets, or prompt instructions as visible figure text. Avoid decorative 3D, gradients, drop shadows, and large titles; thesis captions will be outside the figures. If a small figure title/panel label helps, keep it 24--28 px.

Global readability: all content inside the SVG viewBox, no browser scrollbars, 35 px outer margin, body text >=20 px, small annotations >=18 px, compact arrowheads, whitespace between arrows and boxes. All visible explanatory text in Slovenian. Standard code/model names may remain as written: `GNN`, `GCNConv`, `MLP`, `LayerNorm`, `Dropout`, `GazeMAE`, `SVM`, `LightGBM`, `HeteroData`. Split long labels into 2--3 lines. No clipped text, no text touching borders, no arrowhead overlapping labels. Figure must remain readable at thesis text width.

Palette B only:
- dark gray text/outlines/arrows `#2F3437`
- pastel blue `#8ECAE6`
- pastel teal `#A8DADC`
- pastel purple `#B8A1D9`
- pastel orange `#F4A261`
- pastel pink `#E9A6A6`
- light background `#F7F7F2`

You may use lighter tints/transparency of these colors, but do not introduce other saturated colors. If darker contrast is needed, use dark gray `#2F3437`.

Relation color mapping when relevant:
- `časovne naprej`: blue `#8ECAE6`
- `časovne nazaj`: teal `#A8DADC`
- `prostorske`: orange `#F4A261`
- `fiksacijske`: purple `#B8A1D9`
- prediction/loss/output accents: pink `#E9A6A6`

Before final output, check scientific faithfulness, simplicity, readability, Slovenian labels, no visual clutter, no long sweeping arrows, and no accidental extra components.

---

# Diagram 1: Visokonivojski cevovod priprave podatkov

Goal: high-level data-processing diagram. It explains how MAHNOB-HCI eye-tracking data become graph samples for the GNN and also shows two secondary non-graph branches. It is not the GNN architecture.

Composition: compact two-level layout. Upper/main band is dominant graph-data path:

`MAHNOB-HCI meritve sledilnika pogleda` -> `Čiščenje in filtriranje` -> `10-sekundna okna` -> `Grafovski učni vzorec`

Lower/secondary band: two smaller branches from `10-sekundna okna`, side by side. Use thinner arrows. Do not make all four main blocks equal; `Grafovski učni vzorec` may be larger because it is the key output. No separate `vhod v GNN` block.

Block 1, `MAHNOB-HCI meritve sledilnika pogleda`: show one readable table-like block, not 8 separate boxes. Include:
- `x-koordinata pogleda`
- `y-koordinata pogleda`
- `velikost leve zenice`
- `velikost desne zenice`
- `čas meritve`
- `razdalja do sledilnika`
- `indeks fiksacije`
- `trajanje fiksacije`

Show `razdalja do sledilnika` as one signal only; do not split left/right distance. Optional tiny traces/values are fine but not required.

Block 2, `Čiščenje in filtriranje`: one compact checklist-style block with:
- `veljavnost obeh oči`
- `interpolacija kratkih vrzeli`
- `glajenje zenic`
- `izključitev neveljavnih oznak`
- `filter osamelcev q01-q99`

Block 3, `10-sekundna okna`: draw as segmented timeline/windows. Include concise labels:
- `T = 10 s`
- `brez prekrivanja`
- `najmanj 60 meritev na okno`
- `ena oznaka se prenese na celotno okno`

Make clear: one usable window = one training sample.

Block 4, `Grafovski učni vzorec`: processed graph data for GNN, not internal GNN. Make it the most important output block. Show a mini graph resembling eye-tracking gaze positions, not a generic network icon. Nodes = measurements in one 10 s window; edges = temporal/spatial/fixation relations. Use one mini graph plus two compact side notes:

`značilke vozlišč`:
- `x, y`
- `leva/desna zenica`
- `normalizirani čas`
- `razdalja do sledilnika`
- `trajanje fiksacije`

`tipi povezav`:
- `časovne naprej/nazaj, k_t = 1`
- `prostorske, k_s = 1`
- `fiksacijske, k_f = 3`

Also show:
- `izhod: heterogeni graf / HeteroData`
- `ciljna oznaka na ravni okna/grafa: valenca ali vzburjenost`

Lower branches from `10-sekundna okna`:
1. `SVM / LightGBM / MLP`, label `agregirane statistike istih signalov`
2. `GazeMAE + MLP`, label `samo (x, y), pet 2-sekundnih kosov, 500 Hz`

Avoid: GNN message passing, pooling, attention, classifier heads, stacked neural network layers, long one-row pipeline, tiny table text, long sweeping arrows, non-graph branches dominating.

---

# Diagram 2: Arhitektura predlaganega GNN

Goal: model architecture after graph construction. Start from heterogeneous graph sample and show how the proposed GNN maps it to class prediction. Do not repeat raw-data cleaning. Create two variants:
- inference PNG: no loss/training path.
- training PNG: same layout plus subtle local loss/training path.

Canvas: landscape `1800 x 950` or `1900 x 1000`.

The first test was too wide; use compact two-band layout. Upper band = main node/graph path. Lower band = edge-weight path. They meet at the GNN block. Use short orthogonal/gently curved arrows, no long sweeping curves.

Fixed architecture values, shown as compact chips near top, not one long sentence:
- `L = 2`
- `H = 64`
- `k_t = 1`
- `k_s = 1`
- `k_f = 3`

Required flow, but not as one long row:
1. `heterogeni graf`
2. `značilke vozlišč X`
3. `vhodni MLP`
4. `h^(0), H = 64`
5. `2 × heterogena GNN plast`
6. `predstavitve vozlišč h_i`
7. `pozornostno združevanje`
8. `predstavitev grafa h_G`
9. `napovedna glava MLP`
10. `napoved razreda`

Composition:
- left column: input graph and extracted feature blocks
- middle: node preprocessing, edge-weight computation, GNN block
- right: readout and prediction

Upper band:
`heterogeni graf` -> `značilke vozlišč X` -> `vhodni MLP` -> `h^(0), H = 64` -> `2 × heterogena GNN plast`

Lower band:
`značilke povezav R` -> `MLP za uteži povezav` -> `tanh` -> `normalizacija po ciljnih vozliščih` -> `predznačene uteži w_ij^(r)` -> short arrow upward into bottom/lower-left of GNN block.

This lower band represents learned signed scalar edge weights used by relation-specific `GCNConv` layers.

Right side:
`predstavitve vozlišč h_i` -> `pozornostno združevanje` -> `predstavitev grafa h_G` -> `napovedna glava MLP` -> `napoved razreda`

Input graph block: compact mini graph, smaller than GNN block, suggesting 10 s eye-tracking graph. Include relation types in this order:
- `časovne povezave naprej`
- `časovne povezave nazaj`
- `prostorske povezave`
- `fiksacijske povezave`

Also compact note: `ciljna oznaka: valenca ali vzburjenost`.

Node feature block `značilke vozlišč X` includes:
- `x, y`
- `leva/desna zenica`
- `normalizirani čas`
- `razdalja do sledilnika`
- `trajanje fiksacije`

Edge feature block `značilke povezav R` includes:
- `t_i, t_j, Δt`
- `Δx, Δy, razdalja`
- `Δd`
- `smer ρ pri časovnih povezavah`

GNN block is visual anchor, bigger than preprocessing/readout blocks but not oversized. Label:
`2 × heterogena GNN plast`

Inside or just below it, concise items:
- `GCNConv po relacijah`
- `GELU`
- `fuzija relacij z MLP`
- `rezidualna povezava`
- `LayerNorm + Dropout`

Do not draw full layer internals here; Diagram 3 does that.

Prediction block: show same architecture is trained separately for:
- `razred valence`
- `razred vzburjenosti`

Do not imply joint multi-task prediction.

The diagram should communicate architecture, not implementation minutiae. Do not show optimizer, epochs, batching, train/test split, or data cleaning here.

Training variant only: add local tidy dashed path:
- `napoved razreda` -> short vertical dashed arrow -> `navzkrižna entropija`
- one short dashed return arrow from loss to small bracket `učljivi parametri`
- bracket points to learnable blocks as a group.

Do not route dashed return arrow through middle or across whole diagram. In inference variant, omit loss and training path entirely.

Avoid: raw MAHNOB cleaning, generic black-box NN, collapsing four relations, old temporal+spatial-only architecture, long one-row pipeline, edge-weight branch far below main path, English explanatory labels.

---

# Diagram 3: Ena heterogena GNN plast

Goal: readable internal diagram of one `heterogena GNN plast` from Diagram 2. Show only one layer. Do not include graph-level pooling, graph representation, classifier head, loss, or raw preprocessing.

Canvas: landscape `1800 x 950` or `1900 x 1000`.

The first test was too wide, with huge diagonal residual arrow and squeezed output. Use compact grid layout. Residual connection must be a local top lane entering `rezidualni seštevek` directly.

Context labels:
- `plast ℓ od 2`
- `širina predstavitev H = 64`

Make clear that this layer is repeated twice in the selected architecture, but draw only one layer.

Layout zones left-to-right:
1. inputs
2. relation-specific message passing
3. relation fusion
4. residual update and stabilization

Inputs left:
- `predstavitve vozlišč h_i^(ℓ)`
- `predznačene uteži povezav w_ij^(r)`
- note for weights: `normalizirane po ciljnih vozliščih`

Use two visually separate input buses:
- solid bus from `h_i^(ℓ)` to all relation blocks
- thin dashed bus from `w_ij^(r)` to all relation blocks

Do not let dashed weight bus overlap the solid node bus.

Four parallel relation blocks, same size, aligned, top-to-bottom exactly:
1. `GCNConv + GELU` / `časovne naprej` -> `h_i^(t+)`
2. `GCNConv + GELU` / `časovne nazaj` -> `h_i^(t-)`
3. `GCNConv + GELU` / `prostorske` -> `h_i^(s)`
4. `GCNConv + GELU` / `fiksacijske` -> `h_i^(f)`

Use relation colors from global mapping. Keep labels large.

After relation outputs:
`konkatenacija` -> `MLP fuzija relacij`

Optional formula near fusion only if >=18 px and not cluttered:
`[h_i^(t+) || h_i^(t-) || h_i^(s) || h_i^(f)] → h_i^fuz`
If too small, omit formula.

Residual/stabilization:
- residual arrow from input `h_i^(ℓ)` as clean top lane above relation paths;
- use horizontal/vertical segments, not big diagonal curve;
- residual arrow enters `rezidualni seštevek` directly;
- after `MLP fuzija relacij`, arrow also enters `rezidualni seštevek`;
- then `LayerNorm + Dropout` (or separate `LayerNorm`, `Dropout` if space);
- output `predstavitve vozlišč h_i^(ℓ+1)`.

Intended order: relation-specific convolution, GELU, relation fusion, residual sum, LayerNorm, Dropout.

Avoid: graph-level pooling/classification, raw preprocessing, only two relations, merging temporal forward/backward, generic NN stack, huge diagonal residual arrow, residual ending anywhere except `rezidualni seštevek`, tiny formulas, English explanatory labels.
