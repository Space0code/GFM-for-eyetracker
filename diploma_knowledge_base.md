# Diploma knowledge base — GNN za prepoznavo čustev iz sledilnika pogleda

**Generated:** 2026-05-02  
**Purpose:** compressed project source for future ChatGPT conversations and diploma writing.  
**Scope:** combines uploaded `.md` notes, attachment-image content converted to text, and the available `journal.md` context. Later notes are weighted more strongly than early brainstorming.

---

## 0. Current thesis focus

### Working thesis title

**Grafovska nevronska mreža za prepoznavo čustev iz sledilnika pogleda**

### Practical framing

The thesis should not be framed as “building a general eye-tracking foundation model.” That remains a longer-term research direction. The diploma should be framed more narrowly:

> Develop and evaluate a graph neural network for emotion recognition from eye-tracking data, using MAHNOB-HCI-TAGGING as the main dataset.

The central object of study is the **GNN architecture and graph representation**, not emotion science itself and not eye-tracking hardware. Eye-tracking and emotion recognition are the validation setting.

### Required core question

The thesis must explicitly answer:

> **What exactly are we classifying, and why?**

The answer should connect the classification target to a practical reason, not only to architecture. Example framing:

> We classify emotional state labels derived from MAHNOB-HCI self-reports because this provides a supervised downstream task for testing whether spatio-temporal graph representations of gaze and pupil signals encode affect-relevant information.

### Current recommended thesis scope

Use one main dataset and a minimal but rigorous experiment set:

1. define graph construction from eye-tracking windows,
2. compare GNN against non-graph baselines,
3. test selected GNN design choices by ablation,
4. report complexity and practical feasibility,
5. discuss limitations and future extension toward broader graph-based eye-tracking representation learning.

---

## 1. Timeline and project decisions

### 2025-06-01 — early mentor meeting

Main instructions:

- test graph variants shown in the presentation plus alternatives suggested by Lovro;
- compare:
  - predictive performance,
  - model performance/runtime,
  - model and implementation complexity,
  - feasibility of running on a phone;
- start writing.

This was an early broad-scope stage.

### 2025-07 — meeting with Gašper

Initial plan:

- baseline task: next point prediction `(x, y)`;
- baseline models:
  - 1D CNN for features + LSTM for temporal dependency,
  - simple GraphSAGE-like model;
- think about publication venues:
  - LOG,
  - ETRA,
  - ECAI,
  - UbiComp;
- separate related work on `eyetracklib` and possible software paper.

This is now mostly historical context. The next-point task was later found too easy/trivial because the next gaze point is often close to the current gaze point.

### 2025-07 to 2025-09 — broad GFM planning

Early roadmap:

1. literature review,
2. collect datasets:
   - MAHNOB-HCI / HCI Tagging,
   - SEED / eSEED,
   - PUPILLARY_DATA,
   - cognitive load datasets from IJS/USI,
3. unify data format,
4. train simple GNN and standard baselines,
5. design a broader GFM-like model,
6. target publication.

Possible downstream tasks discussed:

- next frame / scanpath prediction,
- clustering gaze behavior,
- fixation/saccade/blink/smooth pursuit detection,
- cognitive load,
- valence/arousal,
- stress,
- emotion recognition,
- AOI/TOI classification,
- corrupted-signal reconstruction.

This remains useful for future paper planning but is too broad for the diploma.

### 2025-09-12 — possible tasks and graph/time-series notes

Candidate tasks ranked by likely difficulty:

1. next saccade prediction:
   - easier in reading,
   - harder in cognitive load data,
   - medium in movie watching;
2. blink prediction;
3. stress regression;
4. emotion prediction;
5. AOI classification;
6. TOI classification;
7. high-level behavior inference: reading, movie watching, programming, scrolling, office work.

Graph/time-series modeling options:

- **series-as-graph:** time points/events become nodes;
- **series-as-node:** each time series/sensor becomes a node;
- possible node definitions:
  - data point,
  - metric/signal channel such as `x`, `y`, pupil,
  - subject or recording;
- possible use of graph transformers for longer-range dependencies.

For the diploma, the current strongest direction is **series-as-graph over eye-tracking windows**, where samples/events become nodes and spatial/temporal relations become edges.

---

## 2. Dataset trajectory

## 2.1 eSEEd_v2: initial dataset and lessons

### Dataset setup

Used first because it contains eye-tracking data labeled with emotion intensities.

- 48 subjects.
- 10 short movie recordings, approximately 1 minute each.
- Signals:
  - timestamp,
  - gaze coordinates `(x, y)`,
  - left and right pupil size,
  - other signals not initially used.
- Labels:
  - four emotion intensities on a 0–10 scale:
    - Anger,
    - Tenderness,
    - Sadness,
    - Disgust.

### Initial regression task

Output: four scalar emotion intensities.

Initial baselines on the whole dataset:

| Model | MSE | MAE | SD error | R² | Pearson r |
|---|---:|---:|---:|---:|---:|
| MeanEstimator | 12.4251 | 3.0708 | 3.5243 | 0.0634 | 0.2526 |
| SVM | 15.2372 | 2.7305 | 3.5730 | -0.1486 | 0.2135 |
| GaussianNB | 20.3462 | 3.8823 | 4.2364 | -0.5337 | -0.0329 |
| LightGBM | 10.8878 | 2.7743 | 3.2984 | 0.1793 | 0.4243 |
| Spatio-temporal hetero GCN | 11.8035 | 2.9173 | 3.4356 | 0.1128 | 0.3359 |

Interpretation: GNN was competitive but not clearly better than strong tabular baselines.

### Cross-validation regimes

Three evaluation settings were used:

1. **Subject LOO** — one subject left out.
2. **Recording LOO** — one recording/video left out.
3. **Combined LOO** — subject-recording matrix split; leave out row `i` and column `j`, test on cell `(i, j)`.

### Random parameter search

Parameters searched:

- window length: `5, 10, 30, 60` seconds,
- temporal neighbors `k_t ∈ [1, 30]`,
- spatial neighbors `k_s ∈ [1, 6]`,
- hidden channels: `32, 64, 128, 256, 512`,
- preprocess MLP: true/false,
- self-loops: true/false.

Observed trends:

1. Larger `k_t` improved Pearson r.
2. Larger `k_s` helped in RecordingLOO, but this may be dataset-specific because all subjects watched the same clips in the same order.
3. Smaller windows, especially 5–10 seconds, performed better.
4. Preprocess MLP did not initially show a clear benefit.
5. Self-loops did not initially show a clear benefit.

### Important metric warning

Pearson correlation was initially calculated on flattened vectors. This was too optimistic and not aligned with the real question. Per-emotion Pearson r was negative for at least 3 of 4 emotions in some analyses.

Better evaluation:

- graph/window-level absolute errors: MAE/RMSE;
- recording-level aggregated predictions: Spearman ρ, concordance correlation coefficient (CCC), macro-F1/AUC for classification tasks;
- aggregation should not always be mean due to imbalance; median or majority may be better.

### eSEEd_v2 EDA

Key findings:

- Around 1.32% NaNs overall.
- Subjects 3 and 15 had especially high missingness.
- `(x, y)` and pupil sizes roughly normal:
  - `x, y` centered around 0.5 with std ≈ 0.1,
  - pupil size around 4 with std ≈ 0.8.
- Emotion labels were not normally distributed:
  - zero was dominant,
  - anger had more high values,
  - tenderness had few high values,
  - sadness and disgust looked noisy/random.
- Recordings 4 and 5 were mismatched between reported and intended emotion.
- 48/480 subject-recording pairs were fully neutral: all emotions equal 0.
- Several subject-recording pairs had implausible left/right pupil differences.
- Subject outliers:
  - subject 41: very high emotions except tenderness,
  - subjects 48, 39, 26, 5: extremely low emotions.

### eSEEd_v2 cleaning

Pupil-size problems were substantial:

- left and right pupil sometimes differed by several millimeters at the same time point;
- points with large left/right pupil discrepancy were removed;
- subjects with remaining high discrepancy were removed;
- points with `x` or `y` outside `[0,1]` were removed;
- subjects with high missingness were removed.

Remaining subjects after cleaning:

`1, 2, 4, 9, 10, 11, 12, 14, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 35, 38, 41, 42, 43, 44, 45, 46, 47`

### Binary classification on cleaned eSEEd_v2

Task: for one emotion, classify `0` vs `>0`.

Result:

- majority baseline was rarely beaten;
- GNN often collapsed to predicting only the global majority class;
- no classifier showed strong evidence of robust pattern recognition.

Conclusion:

> eSEEd_v2 is not a good main thesis dataset. It is useful as negative evidence and as a motivation for switching to MAHNOB-HCI, but it should not dominate the diploma.

---

## 2.2 GNN collapse debugging

### Observed problem

The GNN collapsed toward near-constant predictions.

Embedding diagnostics showed:

- raw input had substantial diversity;
- first message-passing layer caused a large variance drop;
- with preprocess MLP enabled, collapse shifted partly into the preprocess MLP;
- graph-level embeddings retained some variance;
- prediction head produced low-variance logits and near-constant probabilities.

Interpretation:

1. oversmoothing starts early, especially in GCN-style aggregation;
2. preprocessing can compress useful variability if not normalized/residualized;
3. the prediction head can still collapse even when pooled embeddings retain variance.

### Mitigations tried

1. z-score normalization of features;
2. removed time from node features;
3. added `dt` as edge weights:
   \[
   w = \exp(-\Delta t / \tau)
   \]
   with \(\tau = 0.05\);
4. added LayerNorm to preprocessing MLP:
   `Linear → GELU → LayerNorm → Dropout → Linear → LayerNorm`.

LayerNorm was the largest improvement in representation behavior: preprocess MLP no longer collapsed immediately. However, final predictions still often behaved like majority-class predictions.

### Lesson for thesis

This can support an architectural motivation:

> Naive GCN-style aggregation can oversmooth dense spatio-temporal gaze graphs. Separate spatial/temporal processing, normalization, residual connections, and careful pooling are therefore not cosmetic choices but necessary design decisions.

---

## 2.3 MAHNOB-HCI-TAGGING: current main dataset

### Reason for switching

eSEEd_v2 did not show reliable learnable signal after cleaning and classification simplification. MAHNOB-HCI has:

- more citations and use in affective computing;
- multimodal physiological and eye-tracking data;
- clearer experimental protocol;
- cleaner signal quality in preliminary EDA;
- more possible classification tasks.

### Verified dataset facts

MAHNOB-HCI contains two experiments:

1. emotion recognition from responses to emotional videos,
2. implicit tagging: reaction to correct/incorrect tags.

Emotion experiment facts:

- 30 recruited subjects;
- 27 used in paper analysis because of technical issues with P9, P12, P15;
- 20 emotional video clips per subject;
- neutral clip before each emotional clip;
- self-reports:
  - emotion keyword,
  - arousal,
  - valence,
  - dominance,
  - predictability;
- arousal/valence/dominance/predictability on a 9-point scale;
- emotion keywords:
  - neutral,
  - anxiety,
  - amusement,
  - sadness,
  - joy,
  - disgust,
  - anger,
  - surprise,
  - fear;
- eye tracking with Tobii X120, sampled at 60 Hz in the analyzed setup;
- other modalities include ECG, GSR, respiration, skin temperature, EEG, and face videos;
- neutral 15-second segment stored separately.

### Project EDA on MAHNOB-HCI

Preliminary processed data:

- 942 sections,
- 24 subjects,
- 3,797,165 rows,
- 3,214,000 labeled rows,
- 583,165 unlabeled baseline/neutral rows.

Label coverage:

- 9 emotion classes,
- largest classes:
  - Neutral: 726,659 rows,
  - Amusement: 669,438 rows.

Signal quality:

- outlier removal in sampled plots:
  - `x`: about 1.0%,
  - `y`: about 3.08%,
  - pupil sizes: about 1.6%;
- left/right pupil consistency:
  - correlation ≈ 0.902;
  - moderate subject-specific asymmetry, largest mean absolute difference around `P28 = 0.590`.

Conclusion:

> MAHNOB-HCI is cleaner and more useful than eSEEd_v2 for the diploma.

### MAHNOB paper baseline

From the dataset paper:

- participant-independent leave-one-participant-out CV;
- per-participant min-max normalization to `[0,1]`;
- one-way ANOVA feature selection on train fold;
- libSVM with RBF kernel;
- targets:
  - 3-class arousal,
  - 3-class valence,
  derived from keyword feedback mapping.

Paper's eye-gaze modality used 38 handcrafted features, not only raw `(x,y)`:

| Eye-gaze channel | # features | Captures |
|---|---:|---|
| Pupil diameter | 6 | mean, std, spectral power in low-frequency pupil oscillation bands |
| Gaze distance | 4 | approach/avoidance dynamics |
| Eye blinking | 4 | blink depth/rate/duration/closed-time |
| Gaze coordinates | 24 | distribution stats, fixation/scan behavior, spectral descriptors, dispersion stats |

Reported paper results:

| Modality | Arousal accuracy | Valence accuracy | Arousal F1 | Valence F1 |
|---|---:|---:|---:|---:|
| Peripheral physiology | 46.2% | 45.5% | 0.38 | 0.39 |
| EEG | 52.4% | 57.0% | 0.42 | 0.56 |
| Eye gaze | 63.5% | 68.8% | 0.60 | 0.68 |

Implication:

> The dataset paper shows that gaze contains emotion-relevant information, but their method relies on handcrafted feature engineering. The thesis can instead test whether graph construction and GNN message passing can learn useful representations from less hand-engineered spatio-temporal gaze windows.

---

## 3. What exactly to infer

Several target formulations were considered.

### Option A — 4 continuous intensities

For eSEEd_v2:

- label: \(y \in \mathbb{R}^4\),
- predicts raw 0–10 emotion intensities,
- preserves mixed emotions,
- suffers from subject rating-scale bias and noisy supervision.

### Option B — discretized per-emotion levels

For each emotion:

- neutral / low / medium / high,
- preferably ordinal classification rather than plain softmax.

### Option C — single dominant emotion + neutral

- one class from dominant emotion,
- neutral if all intensities below threshold \(\tau\),
- simple but discards mixed states and is sensitive to small differences.

### Option D — two-stage neutral detection + emotion estimation

1. neutral vs emotional,
2. regress or classify emotion among non-neutral samples.

Pros:

- handles neutral dominance;
- clearer decision logic.

Cons:

- error propagation;
- more complexity.

### Option E — multi-label emotion presence

For each emotion:

\[
y_e = \mathbb{1}[\text{intensity}_e \ge \tau_e]
\]

Supports co-occurring emotions but depends on thresholds.

### Option F — within-subject labels

Normalize or rank labels per subject.

Pros:

- reduces subject rating-scale bias;
- useful for LOSO.

Cons:

- loses absolute intensity meaning.

Recommended robust normalization:

- median + IQR rather than z-score,
- because emotional ratings are not symmetric and have outliers.

Important rule:

> If labels are transformed, the transformation must be fitted on the train set only.

### Option G — valence/arousal

Map emotion keywords to valence/arousal targets. This is common in literature and useful for comparison.

### Option H — soft distribution over emotions

Convert intensity vector into a soft target distribution; train with KL divergence or cross-entropy.

### Option I — ordinal regression

For ordered emotion levels:

- predict \(P(y \ge k)\) thresholds;
- loss: cumulative binary cross-entropy / CORAL-like ordinal loss;
- useful because labels are subjective and ordered.

### Current practical recommendation

For the diploma, choose a small number of clear tasks on MAHNOB-HCI:

1. **main task:** emotion/affective-state classification from MAHNOB-HCI labels;
2. optionally:
   - binary emotional vs neutral,
   - 3-class valence,
   - 3-class arousal.

Do not over-expand into all possible label formulations. The thesis should evaluate GNN behavior, not solve every emotion-labeling variant.

---

## 4. Graph construction and model architecture

## 4.1 Current graph representation

Current working graph idea:

- each window of eye-tracking data becomes one graph;
- each node represents a sample or event in the window;
- node features can include:
  - normalized time,
  - gaze coordinate `x`,
  - gaze coordinate `y`,
  - left pupil size,
  - right pupil size,
  - optional missingness indicators;
- edge types:
  - temporal forward edges,
  - temporal backward edges,
  - spatial edges based on nearest neighbors in gaze space.

Earlier implementation used:

- `k_t` nearest temporal neighbors,
- `k_s` nearest spatial neighbors,
- mostly undirected/unweighted links,
- HeteroConv with GCNConv or GATConv,
- optional preprocess MLP,
- 2-layer MLP prediction head.

### Whiteboard concept converted to text

A node can be interpreted as containing:

\[
\mathbf{x}_v = [x, y, p]
\]

where \(p\) denotes pupil size or pupil-related features.

Two broad edge types:

1. **Temporal edge**
   - connects samples/events across time;
   - can include \(\Delta t\) or learned temporal relation;
   - possible embedding: time/order/saccade-like direction.

2. **Spatial edge**
   - connects spatially close gaze points;
   - can include \(\Delta x, \Delta y\) or spatial distance.

Possible embeddings listed on the board:

1. `(x, y)` embedding,
2. pupil embedding,
3. event embedding, possibly 3D or multi-type:
   - fixation,
   - saccade,
   - blink.

This supports a heterogeneous graph design.

## 4.2 Mentor meeting 2025-12-02: architecture constraints

Key decisions:

- Start simple:
  - first GCN,
  - then GAT;
- Graph transformer can make the graph effectively fully connected and should be delayed until simpler GNNs are understood.
- GraphSAGE may not be ideal:
  - it samples/permutates neighbors;
  - this can lose time-series structure;
  - it is more useful for dense graphs, whereas the current graph is sparse.
- GIN is probably not practical here.
- LSTM is probably not the right first choice.
- Add edges in both temporal directions.
- For temporal order, use temporal encoding or a separate order-aware embedding model.
- Prefer node features over edge features unless edge features are necessary.
- If edge features are central, consider a line graph transformation where edges become nodes.
- Preprocess MLP can map heterogeneous feature units into a common latent space before message passing.

Proposed pipeline:

\[
\text{preprocess MLP} \rightarrow \text{GNN} \rightarrow \text{graph/window embedding} \rightarrow \text{MLP head}
\]

## 4.3 Mentor meeting 2026-04-08: current architectural direction

### Keep diploma narrow

Diploma:

- one specific dataset: MAHNOB-HCI-TAGGING,
- one main task family: emotion/affective-state recognition,
- focus on GNN architecture and graph representation.

Future paper:

- broader eye-tracking GNN/GFM,
- cross-dataset validation,
- larger experimental sweep.

### Separate spatial and temporal information

The model should explicitly distinguish:

- spatial connections,
- temporal connections.

Two separate temporal matrices for forward/backward edges may not be necessary if a learned MLP aggregation handles spatial and temporal information.

### Replace naive mean aggregation

Recommended aggregation strategy:

1. compute spatial and temporal representations for each node;
2. concatenate them at node level;
3. pass through an MLP;
4. only then aggregate to graph/window level.

Alternative:

1. aggregate spatial and temporal streams separately to graph level;
2. concatenate graph-level embeddings;
3. classify with MLP.

Node-level version is probably more expressive.

### Learn temporal edge weights

Instead of manually defining:

\[
w_{ij} = e^{-\Delta t}
\]

try learning the temporal edge weight:

\[
w_{ij} = \operatorname{MLP}([t_i, t_j])
\]

Suggested small MLP:

\[
2 \rightarrow 2 \rightarrow 2 \rightarrow 1
\]

This is cheap and avoids committing to a hand-designed temporal decay.

### Architecture priorities from 2026-05-05

Near-term GNN upgrades should stay incremental, modular, and easy to ablate. Current priority order:

1. separate edge types into `temporal_forward`, `temporal_backward`, and `spatial`;
2. compute relation-specific node representations and combine them with concat + MLP pooling/fusion;
3. replace simple node-to-graph pooling with a small MLP-based graph pooling module;
4. replace the handcrafted temporal decay \(w_{ij}=e^{-\Delta t}\) with learned edge weights:

$
w_{ij} =
\operatorname{MLP}([t_i, t_j, \Delta t, \Delta x, \Delta y, d_{ij}])
$

with layer sizes for spatial edges:

$
6 \rightarrow 6 \rightarrow 4 \rightarrow 2 \rightarrow 1
$

where \(d_{ij}\) is spatial distance. Temporal edges should add a direction feature, giving temporal edge-weight MLP layer sizes:

$
7 \rightarrow 6 \rightarrow 4 \rightarrow 2 \rightarrow 1
$

Edge weights may be signed, but signed incoming weights should be normalized per target node to keep message scales stable, for example by dividing by the sum of incoming absolute signed scores plus \(\epsilon\).

For the fastest path to a stronger working model, use separate edge-weight MLPs for spatial and temporal edges immediately. Temporal forward and backward edges should share the same temporal weight MLP, with direction encoded as an input feature or equivalent relation indicator. Ablations should come after the model works well enough to justify careful comparisons.

### Count and report scale

The thesis should include scale estimates:

- number of windows,
- number of nodes per window,
- number of edges,
- number of subjects,
- number of recordings,
- number of parameters,
- approximate computational cost,
- train/inference runtime if possible.

This supports the design argument and prevents the architecture section from being purely qualitative.

---

## 4.4 Spatio-temporal GNN block variants

### Variant 1 — current heterograph message passing

Use a heterogeneous GNN with edge types:

- `temporal_forward`,
- `temporal_backward`,
- `spatial`.

For each relation, compute relation-specific messages. Then combine:

\[
\mathbf{h}_v =
\operatorname{MLP}
\left(
\mathbf{h}^{spatial}_v
\Vert
\mathbf{h}^{temporal\_forward}_v
\Vert
\mathbf{h}^{temporal\_backward}_v
\right)
\]

Then graph-level pooling:

\[
\mathbf{h}_G = \operatorname{Pool}_{v \in V}(\mathbf{h}_v)
\]

where `Pool` can be mean, sum, max, attention pooling, or median-like robust pooling if implemented.

### Variant 2 — spatial first, temporal second

Converted from the ST-GNN image:

For each time step \(t\), run a spatial GNN only:

\[
H_t^{(S)} = \operatorname{GNN}_{spatial}(X_t, A_t)
\quad \text{for } t=1,\ldots,T
\]

This captures spatial correlations: which gaze points/events are close in space.

For each node \(i\), the model then has a temporal sequence:

\[
\left(
H_1^{(S)}[i,:],
H_2^{(S)}[i,:],
\ldots,
H_T^{(S)}[i,:]
\right)
\]

Feed it along the time axis into a temporal model:

- 1D CNN / TCN,
- GRU/LSTM,
- temporal self-attention / Transformer.

Formally:

\[
Z_i =
\operatorname{TemporalNet}
\left(
H_{1:T}^{(S)}[i,:]
\right)
\]

This is a “spatial GNN → temporal network” architecture.

Possible stacking:

\[
\text{Temporal} \rightarrow \text{Spatial} \rightarrow \text{Temporal} \rightarrow \text{Spatial} \rightarrow \cdots
\]

Diploma note: this is conceptually useful but may be too complex if the main heterograph model is not yet stable.

### Variant 3 — graph transformer

Graph transformers are useful for long-range dependencies but should be treated carefully.

Attention formula:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Differences from standard transformer:

| Aspect | Standard Transformer | Graph Transformer |
|---|---|---|
| Data structure | ordered sequences | graphs |
| Attention scope | usually fully connected among tokens | can be local or graph-constrained |
| Positional encoding | absolute sinusoidal or learned | graph-aware |
| Edge awareness | no explicit edge modeling by default | can include node/edge features |
| Complexity | \(O(N^2)\) | can be reduced via neighborhood constraints |

Differences from classical GNNs:

| Aspect | GNN | Graph Transformer |
|---|---|---|
| Information flow | sequential message passing among connected nodes | attention can aggregate flexibly |
| Long-range dependencies | can struggle due to limited hops | better if attention is sufficiently global |
| Oversmoothing | common in deep GNNs | often less severe |
| Oversquashing | possible when many messages compress into small vectors | attention can mitigate |
| Heterogeneous variants | RGCN, HAN, HGT | HGT, Graphormer, HEAT |

For diploma: graph transformers can be discussed as future work or one baseline if time allows, but should not derail the core GNN experiments.

---

## 5. Positional and structural encodings

Graph transformer notes distinguish positional encoding and structural encoding.

### Positional encoding (PE)

Answers:

> Where am I?

| Type | Level | Meaning | Examples |
|---|---|---|---|
| Local PE | node | position relative to a local cluster/substructure | local random-walk features, distance to cluster centroid |
| Global PE | node | position in the whole graph | Laplacian eigenvectors, adjacency eigenvectors, distance to graph centroid, component ID |
| Relative PE | edge | relation between two nodes | shortest-path distance, random-walk distance, heat-kernel distance, geodesic distance |

### Structural encoding (SE)

Answers:

> What does my neighborhood look like?

| Type | Level | Meaning | Examples |
|---|---|---|---|
| Local SE | node | local substructure around node | degree, random-walk diagonal, triangle/ring counts, Ricci curvature |
| Global SE | graph | global graph structure | eigenvalues, diameter, girth, connected components, node/edge counts |
| Relative SE | edge | difference between two local structures | pairwise distance, gradient of local SE, same-cluster indicator |

For eye-tracking graphs, practical encodings could be:

- node time index or normalized timestamp;
- gaze-space coordinates;
- local degree;
- temporal distance \(\Delta t\);
- spatial distance \(\sqrt{\Delta x^2+\Delta y^2}\);
- same fixation/cluster indicator if available;
- missingness/blink indicators.

---

## 6. Training objectives and losses

### Supervised classification

For binary or multi-class tasks:

- binary cross entropy,
- weighted BCE if imbalance is high,
- cross entropy for multi-class,
- macro-F1 and AUC for evaluation.

### Regression

For intensity labels:

- MAE,
- MSE/RMSE,
- Huber loss.

Huber loss was suggested because MSE can encourage mean-like predictions under noisy labels.

### Ordinal regression

For ordered emotion levels:

- use cumulative binary thresholds \(P(y \ge k)\),
- train with cumulative BCE / CORAL-like loss,
- useful when labels are ordinal but not truly interval-scaled.

### Multi-task learning idea

Potential design:

- head 1: main regression/classification on raw labels;
- head 2: auxiliary ordinal or subject-normalized target.

Example:

\[
\mathcal{L}
=
\lambda_1 \mathcal{L}_{main}
+
\lambda_2 \mathcal{L}_{aux}
\]

Suggested initial weights:

- \(\lambda_1 = 1\),
- \(\lambda_2 \in [0.1, 0.5]\).

Tune on main-task validation performance only.

Alternative:

- uncertainty-based weighting following Kendall-style multi-task loss weighting.

### Self-supervised + supervised mixing

From mentor meeting:

During training, with probability \(p\):

- use self-supervised objective,
- e.g. masked node/attribute reconstruction or edge prediction;

with probability \(1-p\):

- use supervised classification on labeled data.

Masked objectives:

- mask attributes and reconstruct them;
- remove edges and predict them;
- reconstruct corrupted gaze/pupil segments.

For diploma: this can be future work unless supervised results are already stable.

### Next-point prediction issue

Next gaze point prediction was tried conceptually, but it is often too easy:

> next gaze ≈ current gaze.

Potential fixes:

- weight loss by movement distance:
  \[
  w = 1 + \alpha \min(1, d/r)
  \]
  or
  \[
  w = \alpha \cdot \sigma(\beta(d-r))
  \]
- oversample saccades or large movements;
- predict multiple future steps;
- train a preprocessing model to distinguish fixation vs saccade;
- use multi-step rollout loss.

This is currently lower priority for the diploma.

---

## 7. Baselines

The diploma should include both graph and non-graph baselines.

### Non-graph baselines

Recommended:

- Majority / Mean estimator,
- SVM,
- LightGBM,
- MLP.

Less useful:

- GaussianNB, because it performed poorly and is not a strong baseline for these continuous/noisy features.

Baselines should use the same underlying windows/data as GNN where possible.

### Graph baselines

Recommended:

1. simple GCN,
2. GAT,
3. current/final hetero ST-GNN,
4. ablated versions:
   - no spatial edges,
   - no temporal edges,
   - no pupil features,
   - no learned temporal weights,
   - mean aggregation vs concat+MLP.

Potential but optional:

- graph transformer,
- 1D CNN/TCN,
- sequence transformer.

### Fairness concerns

- Same train/test splits.
- Normalize using train fold only.
- Avoid leakage from subject-specific normalization if subject is in test fold.
- If per-subject normalization is used, explain whether it is realistic at inference time.

---

## 8. Evaluation protocol

### Recommended splits

For MAHNOB-HCI:

1. **Leave-one-subject-out** or participant-independent split:
   - closest to paper baseline,
   - tests generalization to unseen subjects.

2. Optionally leave-one-recording/video-out:
   - tests generalization to unseen stimuli,
   - can be hard and may answer a different question.

### Metrics

For classification:

- accuracy,
- balanced accuracy,
- macro-F1,
- AUC for binary or one-vs-rest if appropriate,
- confusion matrix.

For regression:

- MAE,
- RMSE,
- Spearman ρ,
- CCC if calibration matters.

For aggregated predictions:

- aggregate predictions by recording/section;
- use median or majority rather than mean when class imbalance is severe.

### Report uncertainty

Use:

- mean ± std across folds,
- confidence intervals if feasible.

### Scale reporting

Report:

- number of subjects,
- number of recordings/sections,
- number of windows,
- window length,
- sampling rate,
- average nodes/window,
- average edges/window by edge type,
- model parameter count,
- train time,
- inference time/window if feasible.

---

## 9. Literature and related work

Reliability labels:

- **verified/useful:** reliable source found or primary source exists.
- **candidate:** likely relevant, needs full reading.
- **needs verification:** appears in notes but not sufficiently checked.
- **background only:** useful for context, not central.

## 9.1 Main dataset literature

### MAHNOB-HCI database — verified/useful

**Soleymani et al. (2012), “A Multimodal Database for Affect Recognition and Implicit Tagging.”**

Use for:

- dataset description,
- participant-independent CV comparison,
- modality baseline,
- motivation that gaze signals carry emotion-relevant information.

Important facts:

- 27 participants used in the paper analysis;
- 20 emotional videos;
- self-reported emotion keyword, arousal, valence, dominance, predictability;
- includes synchronized gaze, EEG, physiological, face-video, and audio data;
- eye-gaze features reached 63.5% arousal accuracy and 68.8% valence accuracy in their setup.

Suggested citation target:

- IEEE Transactions on Affective Computing, 2012.
- DOI: `10.1109/T-AFFC.2011.25`.

## 9.2 Eye-tracking representation models

### GazeMAE — verified/useful

**GazeMAE: General Representations of Eye Movements using a Micro-Macro Autoencoder.**

Useful as related work for gaze-native self-supervised representation learning.

Notes:

- self-supervised;
- gaze-specific;
- uses position and velocity signals;
- learns micro- and macro-scale representations;
- not graph-based.

Status:

- relevant baseline/context for future representation-learning article;
- not necessarily a diploma baseline unless time allows.

### CLRGaze — candidate

Contrastive self-supervised gaze sequence model.

Notes:

- relevant to gaze embeddings;
- needs full paper verification before use in thesis.

### U’n’Eye — verified/useful

Eye-movement event detector.

Use:

- event classification context;
- possible future use of penultimate layer embeddings;
- not central to current emotion classification task.

### Deep EM Classifier / OEMC — candidate

Event classifiers for fixation/saccade/pursuit.

Use:

- context for event-level gaze modeling;
- not central unless the thesis uses detected events as node types.

### eye2vec — needs verification

Emerging gaze representation work. Track for future, do not rely on it without checking.

## 9.3 Graph-based eye-tracking models

### EyeGraph — verified/useful

**EyeGraph: Modularity-aware Spatio Temporal Graph Clustering for Continuous Event-based Eye Tracking.**

Important:

- NeurIPS 2024 Datasets and Benchmarks Track;
- event-camera near-eye dataset;
- dynamic spatio-temporal graph;
- unsupervised modularity-aware graph clustering;
- tracks pupil movement;
- not a foundation model and not emotion recognition.

Use:

- strongest direct evidence that graph-based eye-tracking is an active direction;
- useful contrast: event-based pupil tracking vs current gaze/pupil emotion recognition.

### GazeGNN — candidate

Graph model integrating radiologists’ eye-tracking data with X-ray image content.

Use:

- example of graph-based gaze in medical image analysis;
- needs exact citation and paper check before thesis use.

### Hartley ETRA 2024 — needs verification

Reported as using GNNs for task classification from fixation graphs.

Use only after checking exact source.

### I-MPN / mobile eye-tracking object interaction — candidate

Inductive Message Passing Network for mobile eye-tracking object recognition.

Use:

- example of graph message passing on mobile eye-tracker data;
- verify exact citation before thesis use.

### Gaze gestures with GCN — candidate

GCN for gaze gesture classification.

Use:

- graph-based eye movement classification context;
- verify exact citation.

## 9.4 Time-series foundation models

### MOMENT — verified/useful

General time-series foundation model.

Use:

- comparison point for “foundation models used with gaze but not graph-based”;
- relevant because Gaze-READ/GazeMTM uses MOMENT-like models.

### Gaze-READ / GazeMTM — verified/useful

Framework for unsupervised gaze anomaly detection.

Important:

- leverages foundation model for gaze anomaly detection;
- adapts MOMENT via masked modeling for gaze representations;
- not graph-based;
- not a gaze-specific graph foundation model.

Use:

- supports claim that foundation models are entering gaze analysis, but graph-native gaze FMs are not established.

## 9.5 Graph foundation models

### Position: Graph Foundation Models Are Already Here — verified/useful

**Mao et al., ICML 2024.**

Use:

- explains GFM direction and “graph vocabulary” perspective;
- argues GFMs are emerging and already partially present;
- useful for future-work framing.

Key ideas from notes:

- task-specific GNNs often lack transferability;
- GFMs aim at cross-domain/cross-task transfer;
- local structural proximity and global structural proximity matter;
- homophily and heterophily patterns must both be handled;
- feature-centric models may use transformers and masked-node modeling.

### Graph Foundation Models: A Comprehensive Survey — candidate/verified depending on version

Notes mention `arxiv.org/html/2402.02216v2`, but web search also found a later survey with arXiv `2505.15116`.

Use carefully:

- the topic is relevant;
- verify exact paper metadata before final thesis citation;
- safe high-level points:
  - GFMs are still early;
  - challenges include feature heterogeneity, structure heterogeneity, task heterogeneity;
  - open problems include scalability, data availability, evaluation, adaptation, and theory.

### BrainGFM / ECG GFM / ECGBERT / ETHOS — candidate

Useful background for physiological/biomedical foundation models.

For diploma:

- probably not central;
- can appear in future-work or broader related-work paragraph only after checking.

## 9.6 Graph transformers and efficient attention

### Graph Transformer overview — background only

Useful concepts:

- graph-aware positional/structural encodings;
- edge-aware attention;
- graph transformers mitigate long-range dependency limitations;
- can be more expensive and more complex.

### Efficient attention ideas — background only

Potential future work for large gaze graphs:

- global anchor nodes,
- GOAT / LargeGT-like designs,
- low-rank approximations such as Linformer/Performer,
- routing transformers,
- cluster-GCN adaptation,
- layer-wise neighbor sampling.

Not needed for first diploma implementation unless graph size becomes prohibitive.

---

## 10. Converted attachment images

This section records content that was previously stored only in attachment images.

### 10.1 Variational graph representation equation

\[
p(\mathbf{h}_v \mid \mathcal{G}) =
\int
p(\mathbf{h}_v \mid \mathbf{z}_v, \mathcal{G})
\cdot
p(\mathbf{z}_v \mid \mathcal{G})
\, d\mathbf{z}_v
\]

Interpretation:

- node representation \(\mathbf{h}_v\) is marginalized over latent variable \(\mathbf{z}_v\);
- relevant to probabilistic graph representation learning;
- likely not central to the diploma unless uncertainty/probabilistic embeddings are used.

### 10.2 CLIP-style graph-language alignment loss

\[
\mathcal{L}_{clip}
=
-
\sum_{(v_i,v_j)\in\mathcal{P}}
\log
\frac{
\exp(
\operatorname{sim}(
\mathbf{h}^{GNN}_{v_i},
\mathbf{h}^{LLM}_{v_j}
)/\tau)
}{
\sum_{w\in\mathcal{V}}
\exp(
\operatorname{sim}(
\mathbf{h}^{GNN}_{v_i},
\mathbf{h}^{LLM}_{w}
)/\tau)
}
\]

Interpretation:

- aligns GNN node embeddings with LLM/text embeddings;
- relevant to graph-language foundation models;
- not needed for current diploma;
- keep for future GFM paper.

### 10.3 Graph notation table

| Symbol | Description |
|---|---|
| \(\mathcal{G}\) | graph |
| \(\mathcal{V}, \mathcal{E}\) | node and edge sets |
| \(N, M\) | number of nodes and edges |
| \(v_i \in \mathcal{V}\) | node in graph |
| \(e_{ij} \in \mathcal{E}\) | edge in graph |
| \(\mathbf{X} \in \mathbb{R}^{N \times D}\) | node attribute matrix |
| \(\mathbf{x}_i \in \mathbb{R}^{D}\) | feature vector for node \(v_i\) |
| \(\mathbf{E} \in \mathbb{R}^{M \times D}\) | edge attribute matrix |
| \(\mathbf{e}_{ij} \in \mathbb{R}^{D}\) | feature vector for edge \(e_{ij}\) |
| \(\mathbf{A} \in \{0,1\}^{N \times N}\) | adjacency matrix |
| \(\mathbf{D}\) | textual information on graphs |
| \(\mathbf{d}_{v_i}\) | text description associated with node \(v_i\) |
| \(\mathbf{d}_{e_{ij}}\) | text description associated with edge \(e_{ij}\) |
| \(\mathbf{d}_{\mathcal{G}}\) | textual description associated with the whole graph |
| \(\mathbf{Z} \in \mathbb{R}^{N \times D'}\) | learned node representations |
| \(\mathbf{z}_i \in \mathbb{R}^{D'}\) | learned representation of node \(v_i\) |
| \(\mathcal{N}_v\) | neighborhood of node \(v\) |
| \(\mathcal{T}\) | set of augmentation functions |
| \(\mathbf{W}, \Theta, w, \theta\) | learnable parameters |
| \(t \sim \mathcal{T}\) | augmentation sampled from \(\mathcal{T}\) |
| \(|\cdot|\) | set cardinality |
| \(\Vert\) | concatenation |
| \(\operatorname{GNN}(\cdot)\) | graph neural network encoder |
| \(\operatorname{LLM}(\cdot)\) | large language model encoder |

### 10.4 Attention formula

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Use in background section only if graph transformers are discussed.

### 10.5 Time-series graph methods table

Converted summary from image.

| Approach | Year | Venue | Task | Conversion | Spatial module | Temporal module | Missing values | Input graph | Learned relations | Graph heuristics |
|---|---:|---|---|---|---|---|---|---|---|---|
| MTPool | 2021 | NN | M | - | Spatial GNN | T-C | No | NR | S | - |
| Time2Graph+ | 2021 | IEEE TKDE | U | Series-as-Graph | Spatial GNN | - | No | R | - | PS |
| RainDrop | 2022 | ICLR | M | - | Spatial GNN | T-A | Yes | NR | S | - |
| SimTSC | 2022 | SDM | U+M | Series-as-Node | Spatial GNN | T-C | No | R | - | PS |
| LB-SimTSC | 2023 | arXiv | U+M | Series-as-Node | Spatial GNN | T-C | No | R | - | PS |
| TodyNet | 2023 | arXiv | M | - | Spatial GNN | T-C | No | NR | D | - |
| EC-GCN | 2023 | Comput. Netw. | U | Series-as-Graph | Spatial GNN | T-C | No | R | D | PS |
| MTS2Graph | 2024 | Pattern Recognit. | M | Series-as-Graph | Spatial GNN | T-C | No | NR | - | - |

Legend inferred from table:
- `U`: univariate,
- `M`: multivariate,
- `T-C`: temporal convolution,
- `T-A`: temporal attention,
- `R`: required graph,
- `NR`: graph not required or not explicitly required,
- `S`: static learned relations,
- `D`: dynamic learned relations,
- `PS`: predefined/heuristic graph structure.

Use as background for graph-based time-series classification. Verify exact paper names before formal citation.

### 10.6 Graph foundation model timeline

Converted description:

Graph-learning development can be summarized as increasing task-solving capacity:

1. **Pre-2010s: statistical methods**
   - spectral methods,
   - graph kernels,
   - feature engineering,
   - heuristic-driven,
   - assist with specific graph tasks.

2. **Around 2010: graph embeddings**
   - DeepWalk,
   - matrix factorization,
   - shallow graph embeddings,
   - n-grams on random walks,
   - solve structure-aware tasks.

3. **Around 2016: graph neural networks**
   - GCN,
   - GAT,
   - graph transformer,
   - message passing,
   - end-to-end training,
   - solve semantic-aware graph tasks.

4. **Around 2023: graph foundation models**
   - OFA,
   - GFT,
   - UniGraph,
   - pretrain + adaptation,
   - cross-domain and cross-task generalization,
   - goal: solve various graph tasks more universally.

Use as conceptual background only. Do not include the image unless needed for presentation.

---

## 11. What not to overemphasize in the diploma

Avoid letting the thesis become too broad.

Do not overemphasize:

- full graph foundation model claims;
- LLM integration;
- promptable graph models;
- cross-dataset pretraining;
- physiological foundation models in general;
- every possible eye-tracking task;
- every possible emotion-label transformation.

These are valuable for future work and a later article, but the diploma needs a controlled experimental story.

---

## 12. Suggested diploma structure

### 1. Introduction

- Why eye-tracking signals can contain affective information.
- Why graph representations are plausible for gaze/pupil time series.
- Research question:
  - Can a spatio-temporal GNN classify affective states from eye-tracking windows better than non-graph baselines?
- Contributions:
  1. graph construction pipeline for eye-tracking windows,
  2. GNN architecture separating spatial and temporal relations,
  3. empirical comparison with baselines on MAHNOB-HCI,
  4. ablation study of graph/model components.

### 2. Background and related work

- Eye tracking and affect recognition.
- MAHNOB-HCI dataset.
- GNNs and spatio-temporal graphs.
- Graph/time-series models for gaze data.
- Short paragraph on broader GFM context.

### 3. Data and preprocessing

- MAHNOB-HCI structure.
- Signals used.
- Labels selected.
- Cleaning:
  - outliers,
  - missing values,
  - pupil consistency,
  - normalization.
- Windowing.

### 4. Graph construction

- Node definition.
- Node features.
- Edge types:
  - spatial,
  - temporal forward,
  - temporal backward.
- Edge weights:
  - none,
  - handcrafted \(e^{-\Delta t}\),
  - learned MLP weights.
- Graph statistics.

### 5. Models

- Baselines:
  - Majority,
  - SVM,
  - LightGBM,
  - MLP.
- GNN:
  - preprocess MLP,
  - relation-specific message passing,
  - concat+MLP aggregation,
  - graph-level pooling,
  - prediction head.
- Ablations.

### 6. Experiments

- Splits.
- Metrics.
- Hyperparameters.
- Parameter count and runtime.
- Results.

### 7. Discussion

- What improved performance?
- What failed?
- Oversmoothing / collapse analysis.
- Dataset limitations.
- Generalization limitations.
- Why this is a first step toward broader eye-tracking representation learning.

### 8. Conclusion

- Short summary.
- Future work:
  - graph transformer,
  - self-supervised pretraining,
  - cross-dataset validation,
  - foundation-model direction.

---

## 13. Immediate action list

Most important next steps:

1. finalize the exact MAHNOB-HCI classification target;
2. fix train/test-safe preprocessing;
3. define final graph construction with three edge types: temporal forward, temporal backward, and spatial;
4. implement 2–4 ablations only, prioritizing relation-specific fusion, MLP graph pooling, and learned edge weights;
5. compute graph/model scale statistics;
6. start writing chapters 1–4 immediately;
7. treat GFM as future work, not as the diploma’s central claim.

---

## 14. Notes on source reliability and removed/weak items

### Keep as reliable or useful

- MAHNOB-HCI paper.
- EyeGraph NeurIPS 2024.
- MOMENT time-series foundation model.
- Gaze-READ / GazeMTM.
- Mao et al. “Graph Foundation Models Are Already Here.”
- GazeMAE.
- General graph transformer / PE / SE concepts.

### Keep only after verification

- Hartley ETRA 2024 GNN fixation/task classification.
- GazeGNN exact WACV reference.
- I-MPN exact Scientific Reports reference.
- H2G2-Net details.
- A-CensNet exact performance claims.
- CLRGaze details.
- eye2vec details.
- BrainGFM / ECG GFM / ECGBERT / ETHOS details.

### Remove from main thesis context

- Apple Notes internal links.
- vague “LLM tells what subject is doing” idea except as future speculation.
- broad physiological foundation-model plans unless needed for future-work paragraph.
- publication venue planning except personal project planning.

---

## 15. Images that still need to be kept as sources

No attachment image is strictly required as a source if this Markdown is used. The important content has been converted to text, equations, or tables.

Optional to keep outside sources:

1. the whiteboard architecture photo, only for personal memory of the original discussion;
2. original paper figures/tables if you later want to cite or reproduce them formally.

For thesis writing, prefer citing the original papers rather than the screenshot images.
