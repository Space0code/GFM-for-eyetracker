# GNN v2 Component Decomposition Notes

**Last updated:** 2026-05-14

This document is a thesis-writing aid for explaining the current GNN v2
architecture. It focuses only on the decomposition of the model into components,
the function of each component, and how those components can be justified in the
diploma.

**Note for LLM assistants:** before using this document as current project
truth, ask the user whether any architectural changes have happened since the
last update of this documentation and whether those changes should be described.

---

## 1. Purpose of the GNN v2 Architecture

GNN v2 is a spatio-temporal heterogeneous graph neural network for classifying
affective states from eye-tracking windows. Its main purpose is to represent a
short gaze sequence not only as a time series, but as a graph with several
meaningful relation types:

- temporal progression through the signal;
- spatial proximity between gaze samples;
- continuity inside detected fixations.

The model is designed around one central assumption:

> Eye-tracking windows contain affect-relevant patterns in local temporal
> dynamics, spatial scan behavior, pupil response, distance-to-tracker changes,
> and fixation structure. These sources of information should be represented as
> separate but jointly learnable components.

For thesis writing, GNN v2 should be presented as a modular architecture. Each
component corresponds to a modelling decision that can be motivated separately
and, where useful, tested by ablation.

---

## 2. Data Window as a Graph

### 2.1 Input Window

The model operates on fixed-length eye-tracking windows. In the current MAHNOB-HCI
experiments, the default window length is 10 seconds, although the architecture
itself is not tied to exactly this duration.

For one window, let there be $N$ valid eye-tracking samples. Each sample becomes
one graph node:

$ v_i,\quad i = 1,\ldots,N $

The graph-level label is the affective class assigned to that window, for
example valence or arousal class.

### 2.2 Why Use a Graph?

A raw eye-tracking signal is naturally sequential, but the sequence also has a
spatial structure. Two gaze samples may be far apart in time but close on the
screen, or close in time but represent a rapid movement across screen regions.
A graph can encode both:

- temporal neighborhood;
- spatial neighborhood;
- event-level membership, such as belonging to the same fixation.

This is the first major modelling decision of GNN v2: instead of summarizing a
window only into handcrafted aggregate statistics, the model preserves local
sample-level structure and lets message passing learn from it.

---

## 3. Node Features

Each node represents one eye-tracking sample. The current GNN v2 node feature
vector can contain the following features:

$ x_i =
[
x_i^{gaze},
y_i^{gaze},
p_i^{left},
p_i^{right},
d_i^{avg},
f_i^{dur}
] $

where:

- $x_i^{gaze}$ and $y_i^{gaze}$ are average gaze coordinates;
- $p_i^{left}$ and $p_i^{right}$ are left and right pupil sizes;
- $d_i^{avg}$ is the average eye-tracker-to-eyes distance;
- $f_i^{dur}$ is fixation duration for the fixation containing the sample.

The first four features are the core node features. The distance and fixation
duration features are optional and controlled through configuration flags:

- `use_distance_avg`;
- `use_fixation_duration`.

### 3.1 Gaze Coordinates

Gaze coordinates encode where the participant is looking on the screen. They
are essential for spatial graph construction because spatial edges are computed
from nearest neighbors in the gaze-coordinate plane.

In the thesis, these features can be described as the geometric basis of the
graph.

### 3.2 Pupil Size

Pupil size is included because it can reflect physiological responses related
to arousal, cognitive load, and visual stimulation. In this project, left and
right pupil sizes are kept as separate node features rather than immediately
averaged. This allows the model to learn whether asymmetries or eye-specific
differences are useful.

### 3.3 Eye-Tracker-to-Eyes Distance

`distance-avg` is derived from `distance-left` and `distance-right`:

$ d_i^{avg} =
\operatorname{mean}(d_i^{left}, d_i^{right}) $

using the available eye distance values when one side is missing.

This feature is conceptually different from the spatial edge distance between
two gaze points. It describes the participant's physical distance from the eye
tracker, not the distance between screen coordinates. It may capture posture,
approach/avoidance movement, or signal-quality-relevant variation.

For writing, keep the terminology explicit:

- **gaze spatial distance:** distance between two gaze points on the screen;
- **eye-tracker distance:** distance between eyes and the tracking device.

### 3.4 Fixation Duration

`fixation-duration` is an event-level signal associated with detected fixations.
It is used as a continuous node feature when enabled. Non-fixation samples or
missing values are encoded with duration $0$, so non-fixation samples are not
dropped merely because they do not belong to a fixation.

This feature gives the model a direct cue about local gaze stability. Longer
fixations may indicate more sustained attention to a screen region, whereas
shorter fixation duration or zero duration may correspond to transitions or
non-fixation samples.

### 3.5 Why Fixation ID Is Not a Node Feature

The raw `fixation-index` is not used as a numerical node feature. Its absolute
value has no stable physical meaning across recordings. For example, fixation
ID 10 in one recording is not semantically larger or more important than
fixation ID 4 in another recording.

Instead, `fixation-index` is used only as a grouping variable for constructing
fixation edges.

This is an important methodological point for the thesis: identifier-like
variables should not be treated as continuous measurement features.

---

## 4. Edge Types

GNN v2 uses a heterogeneous graph with several relation types. Each relation has
a different interpretation and therefore receives its own message-passing
operation.

The current relation set is:

| Relation | Meaning | Construction |
|---|---|---|
| `temporal_forward` | local influence from earlier to later samples | connect temporal neighbors where destination index is later than source |
| `temporal_backward` | local influence from later to earlier samples | connect temporal neighbors where destination index is earlier than source |
| `spatial` | proximity in gaze-coordinate space | k-nearest neighbors in $(x,y)$ gaze space, bidirectional |
| `fixation` | continuity inside the same fixation | bidirectional edges between consecutive samples with the same fixation ID |

### 4.1 Temporal Forward Edges

Temporal forward edges connect each sample to nearby later samples within a
configured temporal neighborhood. If the temporal neighborhood size is $k_t$,
then node $i$ can connect to nodes $i+1,\ldots,i+k_t$, where those nodes exist.

These edges represent the usual forward direction of time. They allow the model
to propagate information from earlier gaze states to later gaze states.

### 4.2 Temporal Backward Edges

Temporal backward edges connect each sample to nearby earlier samples. They
allow a node representation to be informed by local context on both sides of
the sample.

The separation between forward and backward temporal edges is useful because
the direction of time has semantic meaning in eye movement. A transition from
point A to point B is not necessarily equivalent to the reverse transition.

### 4.3 Spatial Edges

Spatial edges connect gaze samples that are close in screen-coordinate space.
These edges can connect samples that are not necessarily adjacent in time.

The intuition is that revisits to nearby screen regions, clusters of gaze
positions, and spatial dispersion patterns may carry information about
attention and affective processing.

### 4.4 Fixation Edges

Fixation edges are built from `fixation-index`. Consecutive samples with the
same non-missing fixation ID are connected in both directions:

$ v_i \leftrightarrow v_{i+1}
\quad \text{if} \quad
fixation\_id_i = fixation\_id_{i+1} $

The implementation deliberately uses consecutive same-fixation edges rather
than a full clique among all samples in the fixation. This keeps edge counts
small and preserves the local temporal continuity of the fixation.

Fixation edges are optional and controlled by:

- `use_fixation_edges`.

The thesis interpretation is that fixation edges add an event-level relation
without introducing separate fixation meta-nodes. Full fixation meta-nodes are
left for future work.

---

## 5. Edge Features and Learned Edge Weights

GNN v2 can use learned signed edge weights. Instead of assigning every edge the
same importance, the model computes a scalar weight from features describing the
relation between the source and target nodes.

### 5.1 Base Edge Feature Vector

For an edge from source node $i$ to target node $j$, the base relation feature
vector is:

$ e_{ij}^{base} =
[
t_i,
t_j,
\Delta t_{ij},
\Delta x_{ij},
\Delta y_{ij},
s_{ij}
] $

where:

- $t_i$ and $t_j$ are timestamps;
- $\Delta t_{ij} = t_j - t_i$;
- $\Delta x_{ij} = x_j - x_i$;
- $\Delta y_{ij} = y_j - y_i$;
- $s_{ij} = \sqrt{(\Delta x_{ij})^2 + (\Delta y_{ij})^2}$ is gaze spatial distance.

For spatial and fixation edges, this base 6-dimensional vector is sufficient
when `delta_distance` is disabled.

### 5.2 Temporal Direction Feature

Temporal edges additionally include a direction indicator:

$ r_{ij} =
\begin{cases}
1, & \text{temporal forward edge} \\
-1, & \text{temporal backward edge}
\end{cases} $

The temporal edge vector without `delta_distance` is therefore:

$ e_{ij}^{temporal} =
[
t_i,
t_j,
\Delta t_{ij},
\Delta x_{ij},
\Delta y_{ij},
s_{ij},
r_{ij}
] $

This gives the temporal edge-weight MLP explicit information about direction.

### 5.3 Delta Distance Feature

When `use_delta_distance_edge_feature` is enabled, the edge feature vector also
contains:

$ \Delta d_{ij}^{avg} = d_j^{avg} - d_i^{avg} $

This captures whether the participant moved closer to or farther from the eye
tracker between two connected samples.

With this feature enabled:

- spatial edge attributes have 7 features;
- fixation edge attributes have 7 features;
- temporal edge attributes have 8 features.

Without it:

- spatial edge attributes have 6 features;
- fixation edge attributes have 6 features;
- temporal edge attributes have 7 features.

### 5.4 Edge-Weight MLPs

Each relation family uses a small MLP to map edge features to one raw scalar
score:

$ a_{ij} = \operatorname{MLP}_{rel}(e_{ij}) $

The model uses separate MLPs for:

- spatial edges;
- temporal edges, shared by forward and backward temporal relations;
- fixation edges.

The default hidden shape follows the earlier lightweight edge-weight design:

$ input \rightarrow 6 \rightarrow 4 \rightarrow 2 \rightarrow 1 $

where the input dimension is determined by whether `delta_distance` is enabled.

### 5.5 Signed Weight Normalization

The raw edge scores are transformed with $\tanh$:

$ \tilde{a}_{ij} = \tanh(a_{ij}) $

This allows both positive and negative edge contributions. The scores are then
normalized per target node by the sum of absolute incoming scores:

$ w_{ij} =
\frac{\tilde{a}_{ij}}
{\sum_{k \in \mathcal{N}(j)} |\tilde{a}_{kj}| + \epsilon} $

This normalization keeps the total incoming message scale stable while
preserving the sign of each edge.

In the thesis, this can be motivated as a compromise between expressive learned
relation weighting and numerical stability.

---

## 6. Preprocessing MLP

Before graph message passing, GNN v2 can apply a preprocessing MLP to node
features:

$ h_i^{(0)} = \operatorname{MLP}_{pre}(x_i) $

This maps heterogeneous input units into a shared latent space. Gaze
coordinates, pupil sizes, eye-tracker distance, and fixation duration are
measured in different units and ranges; the preprocessing MLP gives the model a
learned feature embedding before neighborhood aggregation.

If disabled, the raw node feature vector is passed directly to the first graph
convolution.

The preprocessing MLP is controlled by:

- `use_preprocess_mlp`.

---

## 7. Relation-Specific Message Passing

For each GNN layer, the model applies a separate graph convolution for each
enabled relation. For node $i$ and relation $r$, this produces:

$ h_{i,r}^{(\ell)} =
\operatorname{GNNConv}_r(
h^{(\ell-1)}, E_r, W_r
) $

where:

- $h^{(\ell-1)}$ are node representations from the previous layer;
- $E_r$ is the edge set for relation $r$;
- $W_r$ are learned or provided edge weights for relation $r$.

The current relation outputs may include:

$ h_{i,spatial}^{(\ell)},\quad
h_{i,forward}^{(\ell)},\quad
h_{i,backward}^{(\ell)},\quad
h_{i,fixation}^{(\ell)} $

if fixation edges are enabled.

If a relation exists in the model but a particular graph contains no edges of
that relation, its relation output is treated as a zero vector. This avoids
creating artificial messages from empty edge sets.

### Why Relation-Specific Convolutions Matter

The model should not treat all neighbors as equivalent. A spatial neighbor, a
future temporal neighbor, a past temporal neighbor, and a same-fixation neighbor
represent different hypotheses about how gaze behavior should be aggregated.

Relation-specific message passing lets each relation type learn its own
transformation while still contributing to a shared node representation.

---

## 8. Relation Fusion MLP

After relation-specific message passing, GNN v2 combines relation outputs at the
node level. With all current relations enabled:

$ h_i^{(\ell),concat} =
[
h_{i,spatial}^{(\ell)}
\Vert
h_{i,forward}^{(\ell)}
\Vert
h_{i,backward}^{(\ell)}
\Vert
h_{i,fixation}^{(\ell)}
] $

The concatenated vector is passed through an MLP:

$ \hat{h}_i^{(\ell)} =
\operatorname{MLP}_{fusion}^{(\ell)}
(h_i^{(\ell),concat}) $

This is the main pooling/fusion operation across relation-specific node
representations.

### Thesis Interpretation

The relation fusion MLP answers the question:

> Given spatial, forward-temporal, backward-temporal, and fixation-based
> context for the same sample, how should these signals be combined into one
> updated node representation?

This is more expressive than simply averaging relation outputs, because the
model can learn nonlinear interactions between relation types.

---

## 9. Residual Connection, Normalization, and Dropout

After relation fusion, each layer applies a residual connection and layer
normalization:

$ h_i^{(\ell)} =
\operatorname{LayerNorm}
(
\hat{h}_i^{(\ell)} + h_i^{(\ell-1)}
) $

For the first layer, if the input feature dimension differs from hidden
dimension, the residual branch is projected to the hidden dimension.

Dropout is applied to reduce overfitting.

### Why This Matters

Eye-tracking datasets are relatively small compared with typical deep learning
datasets. The model also operates on many correlated samples from the same
subject and recording. Residual connections and normalization help stabilize
training, while dropout helps reduce overfitting.

This component is also useful for explaining why very deep GNNs may not be
necessary. Current experiments suggest that shallower models, especially around
3 layers, can be more stable than deeper 10-layer variants.

---

## 10. Graph-Level Attention Pooling

After the final GNN layer, the model has one representation per node:

$ h_i^{(L)} $

These node representations must be aggregated into one graph/window embedding.
The current GNN v2 default is attention pooling:

$ \alpha_i =
\operatorname{softmax}_{graph}
(
\operatorname{MLP}_{att}(h_i^{(L)})
) $

$ h_G =
\sum_{i=1}^{N} \alpha_i h_i^{(L)} $

where the softmax is computed separately within each graph in the batch.

### Thesis Interpretation

Attention pooling is motivated by the fact that not every sample in a window is
equally informative. Some moments may contain stronger affect-relevant changes,
such as pupil response, rapid gaze movement, sustained fixation, or distance
change.

The attention weights should be described as learned model weights, not as a
causal explanation of emotion. They can support interpretability, but they do
not prove that a specific gaze sample caused the label.

---

## 11. Classification Head

The graph embedding $h_G$ is passed to a final MLP head:

$ \hat{y} = \operatorname{MLP}_{head}(h_G) $

For classification, the output dimension corresponds to the number of target
classes. The head is standard supervised machinery; the more important thesis
contribution is the graph representation and the relation-aware message passing
that produces $h_G$.

---

## 12. Configuration Flags

The newest version is designed so optional distance and fixation components can
be switched on or off in config files.

| Flag | Component controlled |
|---|---|
| `use_distance_avg` | appends `distance-avg` to node features |
| `use_fixation_duration` | appends `fixation-duration` to node features |
| `use_delta_distance_edge_feature` | appends $\Delta d^{avg}$ to learned edge features |
| `use_fixation_edges` | adds the `fixation` relation |
| `use_edge_weights` | enables edge weights in supported GNN convolutions |
| `edge_weight_mode` | chooses handcrafted or learned signed edge weights |
| `relation_pooling` | chooses relation fusion mode, currently MLP by default |
| `graph_pooling` / `head_pooling` | chooses graph-level pooling, attention by default |
| `num_layers` | number of relation-message-passing layers |

The default configs enable the new distance/fixation flags. Experiments that
need the older core-feature model should explicitly disable the relevant flags
in the config. This makes the latest GNN v2 architecture the default while still
keeping ablations easy to run.

---

## 13. Suggested Thesis Explanation Order

A clear thesis section could explain GNN v2 in this order:

1. **Window graph construction**
   - samples as nodes;
   - graph label at window level.

2. **Node features**
   - gaze coordinates;
   - pupil size;
   - optional eye-tracker distance;
   - optional fixation duration.

3. **Edge types**
   - temporal forward;
   - temporal backward;
   - spatial;
   - fixation.

4. **Edge features and learned weights**
   - relation feature vector;
   - temporal direction;
   - optional `delta_distance`;
   - signed normalization.

5. **Relation-specific GNN layers**
   - one convolution per relation;
   - relation outputs as separate hypotheses.

6. **MLP relation fusion**
   - combine relation-specific node representations.

7. **Residual, normalization, dropout**
   - training stability and regularization.

8. **Attention graph pooling**
   - node-to-window aggregation.

9. **Classification head**
   - final supervised prediction.

This order moves from data representation to model computation to prediction,
which should be easier for diploma readers than starting from implementation
details.

---

## 14. Component-to-Ablation Mapping

The component decomposition can also guide ablations. The purpose is not to
prove that every component always improves performance, but to test which parts
of the graph hypothesis are empirically useful.

| Component | Possible ablation | Question answered |
|---|---|---|
| Graph representation | GNN vs LightGBM/MLP baselines | Does sample-level graph structure help? |
| Temporal direction split | merged temporal relation vs forward/backward relations | Does temporal direction matter? |
| Spatial relation | remove spatial edges | Does screen-space proximity add useful context? |
| Fixation relation | disable `use_fixation_edges` | Does fixation continuity help? |
| Distance node feature | disable `use_distance_avg` | Does eye-tracker distance improve classification? |
| Fixation duration node feature | disable `use_fixation_duration` | Does fixation-duration information help? |
| Delta distance edge feature | disable `use_delta_distance_edge_feature` | Does local approach/retreat information help edge weighting? |
| Learned edge weights | learned signed vs unweighted or handcrafted | Should relation strength be learned? |
| Relation fusion | MLP fusion vs simpler fusion | Is nonlinear relation mixing useful? |
| Graph pooling | attention vs mean/mean-max pooling | Are some samples more informative than others? |
| Depth | 1/3/5/10 layers | How much message-passing depth is useful before overfitting? |

For the final diploma narrative, these ablations should be prioritized based on
time and result stability. The most important comparisons are the ones that
support the main architectural claims: graph structure, relation separation,
learned edge weighting, and distance/fixation extensions.

---

## 15. Limitations to Mention

Several limitations should be stated clearly:

- `distance-avg` and `fixation-duration` may not be available in all
  eye-tracking datasets, so they improve MAHNOB-specific completeness but may
  reduce cross-dataset generality.
- `fixation-index` depends on the eye tracker or preprocessing pipeline's
  fixation detection. Different detectors may produce different fixation
  groupings.
- Attention weights and learned edge weights are useful for inspection, but
  they are not causal explanations.
- The model remains supervised and task-specific; it is not yet a general
  graph foundation model for eye tracking.
- Fixation meta-nodes are not included in this version. The current fixation
  relation is a lightweight event-continuity approximation.
