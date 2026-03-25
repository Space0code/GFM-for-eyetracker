# GNN Architecture and Training (Detailed)

This appendix explains the current production GNN stack in `src/emotions/*` and how training is orchestrated.

## 1. Main GNN class

Primary model: `SpatioTemporalHeteroGNN` in `src/emotions/model.py`.

### 1.1 Forward graph

```mermaid
flowchart TD
    A["Node features: x, y, pupilL, pupilR"] --> B{"use preprocess mlp"}
    B -->|yes| C["Linear then GELU then LN then Dropout then Linear then LN"]
    B -->|no| D["Raw node features"]
    C --> E["HeteroConv layer 1"]
    D --> E
    E --> F["GELU"]
    F --> G["Residual add"]
    G --> H["LayerNorm and Dropout"]
    H --> I["HeteroConv layers 2 to N with same pattern"]
    I --> J{"Pooling"}
    J -->|mean| K["global mean pool"]
    J -->|mean max| L["concat global mean pool and global max pool"]
    K --> M["Head MLP"]
    L --> M
    M --> N["Output logits or value"]
```

### 1.2 Per-layer internals

- Relation types:
  - `("node","temporal","node")`
  - `("node","spatial","node")`
- Convolution type is configurable:
  - `GCNConv`
  - `GATConv`
- Relation outputs are merged via `HeteroConv(..., aggr=<aggr>)`.
- Residual path:
  - layer 1 uses `input_residual_proj`
  - deeper layers use identity residual (`x_dict["node"]`)
- Norm/regularization:
  - `LayerNorm` after residual add
  - dropout after each layer

## 2. Task-specific wrappers

- Binary classification: `src/emotions/binary/model_binary.py`
  - `out_channels=1`, `output_scale=1.0`
  - trained with `BCEWithLogits`
- Multiclass classification: `src/emotions/multiclass/model_multiclass.py`
  - `out_channels=num_classes`, `output_scale=1.0`
  - trained with `CrossEntropy`
- Regression: direct `SpatioTemporalHeteroGNN` usage in `src/emotions/regression/train_regression.py`
  - `out_channels=1`
  - trained with `MSE`

## 3. Graph input expected by the model

Built by `SpacioTemporalDataset` (`src/data/data.py`) as `HeteroData` with:

- `data["node"].x`: `[num_nodes, in_channels]`
- `data[("node","temporal","node")].edge_index`: `[2, E_t]`
- `data[("node","spatial","node")].edge_index`: `[2, E_s]`
- optional `edge_attr` for both relations when `use_edge_weights=true`
- `data.y`: graph-level target (scalar or vector)

## 4. Training loop behavior (important details)

## 4.1 Binary (`src/emotions/binary/train_binary.py`)

- Per fold:
  - compute threshold from **train split only** (`mean`/`median`/fixed)
  - binarize targets: `y=1 if target > threshold else 0`
- Standardization (optional):
  - graph node features and baseline tabular features both scaled from train split
- Split alignment safety:
  - baseline and GNN fold identities/signatures are compared
  - run aborts if splits differ (prevents unfair comparisons)
- Model selection:
  - best checkpoint by validation loss
  - optional early stopping controls available

## 4.2 Multiclass (`src/emotions/multiclass/train_multiclass.py`)

Two modes:
- `emotion-id`
- `va-quadrant` (threshold valence/arousal to LL/LH/HL/HH; thresholds from train fold)

Per fold:
- build class mapping (`raw_label -> encoded_index`)
- convert graph/tabular labels with identical fold context
- train GNN and baselines with same split strategy

## 4.3 Regression (`src/emotions/regression/train_regression.py`)

- Single-target regression (`target_column`)
- Same split/standardization design as above
- GNN trained via MSE; baselines via sklearn/lightgbm wrappers

## 5. Default/typical configuration knobs

| Group | Typical value (HCI configs) | Where |
|---|---|---|
| `window_length` | `10` sec | `dataset.*` YAML |
| `window_overlap` | `0` | `dataset.*` YAML |
| `kt`,`ks` | `2`,`2` | `dataset.*` YAML |
| `feature_columns` | `x-avg,y-avg,pupil-left,pupil-right` | `dataset.*` YAML |
| `hidden_channels` | `128` | `gnn.model` |
| `num_layers` | often tuned (`2`, `10`) | `gnn.model` |
| `pooling` | `mean` or `mean_max` | `gnn.model` |
| `conv_type` | mostly `GCNConv` | `gnn.model` |
| `batch_size` | `64` | `gnn.training` |
| `lr` | `1e-3` | `gnn.training` |

## 6. Historical predecessor (kept for continuity)

`src/gnext/*` is the predecessor stack:
- task was next-point gaze prediction
- primary model: `NextPointGNN` (GraphSAGE)
- this is historical context, not the current emotion benchmark pipeline

## 7. Why this architecture fits your data

- Temporal relation captures motion continuity.
- Spatial relation captures gaze-position similarity (screen-space neighborhoods).
- Window-level graph prediction matches your label granularity (emotion/tag values per segment).
- Shared backbone across binary/multiclass/regression lets you compare tasks under a common representation.
