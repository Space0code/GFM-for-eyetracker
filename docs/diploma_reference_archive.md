# Diploma Reference Archive

This file stores useful but non-central reference material that was moved out of
`diploma_knowledge_base.md` to keep the main knowledge base focused on thesis
framing, architecture, experiments, and writing decisions.

## Converted Attachment Images

This section records content that was previously stored only in attachment
images.

### Variational Graph Representation Equation

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

### CLIP-Style Graph-Language Alignment Loss

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

### Graph Notation Table

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

### Attention Formula

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

Use in background section only if graph transformers are discussed.

### Time-Series Graph Methods Table

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

- `U`: univariate;
- `M`: multivariate;
- `T-C`: temporal convolution;
- `T-A`: temporal attention;
- `R`: required graph;
- `NR`: graph not required or not explicitly required;
- `S`: static learned relations;
- `D`: dynamic learned relations;
- `PS`: predefined/heuristic graph structure.

Use as background for graph-based time-series classification. Verify exact paper
names before formal citation.

### Graph Foundation Model Timeline

Converted description:

Graph-learning development can be summarized as increasing task-solving capacity:

1. **Pre-2010s: statistical methods**
   - spectral methods;
   - graph kernels;
   - feature engineering;
   - heuristic-driven;
   - assist with specific graph tasks.

2. **Around 2010: graph embeddings**
   - DeepWalk;
   - matrix factorization;
   - shallow graph embeddings;
   - n-grams on random walks;
   - solve structure-aware tasks.

3. **Around 2016: graph neural networks**
   - GCN;
   - GAT;
   - graph transformer;
   - message passing;
   - end-to-end training;
   - solve semantic-aware graph tasks.

4. **Around 2023: graph foundation models**
   - OFA;
   - GFT;
   - UniGraph;
   - pretrain + adaptation;
   - cross-domain and cross-task generalization;
   - goal: solve various graph tasks more universally.

Use as conceptual background only. Do not include the image unless needed for
presentation.

