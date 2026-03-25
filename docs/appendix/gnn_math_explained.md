# How the GNN Works: Math + Intuition

This appendix explains the current `SpatioTemporalHeteroGNN` using rendered LaTeX math and plain-language intuition.

## 1. Objects and notation

For one window-graph \(g\):

- Nodes: \(V_g = \{1,\dots,n_g\}\), where each node is a time sample in the window.
- Node feature matrix: \(X_g \in \mathbb{R}^{n_g \times d_{\text{in}}}\).
- Temporal edge set: \(E_t\).
- Spatial edge set: \(E_s\).
- Optional edge weight:
  \[
  w_{ij} = \exp\!\left(-\frac{|t_i - t_j|}{\tau}\right).
  \]
- Graph label: \(y_g\) (binary, multiclass index, or scalar target).

## 2. Graph construction math

### 2.1 Temporal edges

With temporal horizon \(k_t\), directed edges connect offsets \(\pm 1,\dots,\pm k_t\) whenever valid.

For sufficiently large \(n\), the directed temporal edge count is:
\[
|E_t| = 2 \sum_{d=1}^{k_t}(n-d)
= 2k_t n - k_t(k_t+1).
\]

### 2.2 Spatial edges

Using \(k_s\)-nearest neighbors in gaze coordinates \((x,y)\):

1. Find \(N_s(i)\) via kNN in 2D coordinates.
2. Add bidirectional edges \((i,j)\) and \((j,i)\).
3. Deduplicate.

This yields a sparse geometric graph over the same node set.

## 3. Layer computation

Let \(h_i^{(0)}\) be the initial node embedding:
\[
h^{(0)} =
\begin{cases}
\mathrm{MLP}(X), & \text{if preprocessing MLP is enabled},\\
X, & \text{otherwise}.
\end{cases}
\]

For each layer \(\ell = 1,\dots,L\):

1. Relation-specific convolution:
   \[
   m_{i,t}^{(\ell)} = \mathrm{Conv}_t\!\big(h^{(\ell-1)}, E_t, w_t\big),\quad
   m_{i,s}^{(\ell)} = \mathrm{Conv}_s\!\big(h^{(\ell-1)}, E_s, w_s\big).
   \]
2. Heterogeneous merge:
   \[
   m_i^{(\ell)} = \mathrm{AggRel}\!\left(m_{i,t}^{(\ell)}, m_{i,s}^{(\ell)}\right).
   \]
3. Nonlinearity:
   \[
   u_i^{(\ell)} = \mathrm{GELU}\!\left(m_i^{(\ell)}\right).
   \]
4. Residual and normalization:
   \[
   r_i^{(\ell)} =
   \begin{cases}
   R\!\left(h_i^{(0)}\right), & \ell=1,\\
   h_i^{(\ell-1)}, & \ell>1,
   \end{cases}
   \]
   \[
   h_i^{(\ell)} = \mathrm{LayerNorm}\!\left(u_i^{(\ell)} + r_i^{(\ell)}\right).
   \]
5. Apply dropout to \(h_i^{(\ell)}\).

## 4. Graph readout and prediction

Given final node states \(H_g = \{h_i^{(L)}\}_{i=1}^{n_g}\):

- Mean pooling:
  \[
  z_g = \frac{1}{n_g}\sum_{i=1}^{n_g} h_i^{(L)}.
  \]
- Mean+max pooling:
  \[
  z_g = \left[\mathrm{mean}_i\ h_i^{(L)}\ \Vert\ \mathrm{max}_i\ h_i^{(L)}\right].
  \]

Prediction head:
\[
\hat{y}_g = \texttt{output\_scale} \cdot \mathrm{HeadMLP}(z_g).
\]

## 5. Task losses

- Binary classification (\(\hat{y}_g \in \mathbb{R}\)):
  \[
  \mathcal{L} = \mathrm{BCEWithLogits}\!\left(\hat{y}_g, y_g\right).
  \]
- Multiclass classification (\(\hat{y}_g \in \mathbb{R}^C\)):
  \[
  \mathcal{L} = \mathrm{CrossEntropy}\!\left(\hat{y}_g, y_g\right).
  \]
- Regression:
  \[
  \mathcal{L} = \mathrm{MSE}\!\left(\hat{y}_g, y_g\right).
  \]

## 6. Why this helps for eye-tracking windows

- Temporal edges encode short-range dynamics (how gaze evolves).
- Spatial edges encode geometric locality (where gaze clusters on screen).
- Combining both captures both motion and screen-position behavior.
- Graph pooling maps variable-length windows to fixed-size embeddings for prediction.

## 7. Plain-language explanation

Each window becomes a mini-network of gaze moments:

- One edge type means "close in time."
- Another edge type means "close on screen."

At each GNN layer, each moment updates using both neighborhoods. After several layers, the model compresses the whole window into one vector and predicts the target.

## 8. Practical interpretation of sensitive knobs

- `num_layers`: number of neighborhood-mixing rounds.
- `kt`, `ks`: temporal and spatial neighborhood sizes.
- `pooling`: which statistics are preserved in the graph summary.
- `use_edge_weights`: whether time gaps modulate message strength.

In recent ablations, depth and early stopping showed the strongest effects, while `kt/ks` and edge-weight toggles were smaller under the tested settings.
