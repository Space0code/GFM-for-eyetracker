# BasicGCN TODO

Purpose: implement the architecture-level graph baseline for the diploma
comparison.

This experiment should answer:

> What do we get from the graph representation itself, before adding the full heterogeneous GNN architecture?

Naming:

- diploma-facing name: `Osnovni GCN`;
- code/model name: `BasicGCN`;
- `GNN_v1` remains only an internal research reference and should not be
  presented in the diploma.

Planned baseline:

- use the same v2 graph source data as the proposed heterogeneous GNN;
- use the same signals, windows, splits, labels, standardization, node features,
  and preprocessing decisions as the proposed heterogeneous GNN;
- keep gaze, pupil, fixation, and screen-distance information available;
- simplify the architecture instead of removing signals;
- collapse `temporal_forward`, `temporal_backward`, `spatial`, and `fixation`
  relations into one homogeneous graph;
- remove duplicate homogeneous edges after relation collapse;
- do not use edge features;
- do not use learned scalar edge weights;
- use `GCNConv` with one shared message-passing path;
- use the same prediction head as the proposed heterogeneous GNN where
  practical;
- use the same YAML-controlled training parameters as the proposed
  heterogeneous GNN, including `num_layers`, `hidden_channels`, dropout,
  optimizer settings, early stopping, metrics, and seed handling;
- expose a separate `BasicGCN` readout parameter. The current default should be
  `attention`; `mean` can be added later as a simpler variant;
- support both 2-class low/high and 3-class classification tasks;
- report it as an architectural baseline, not as a signal ablation.

Evaluation plan:

- main diploma experiments use 3-class valence/arousal and subject LOO;
- low/high results are additional main-text analyses;
- additional diploma results use recording LOO where relevant;
- k-fold is not the primary diploma setting for `BasicGCN`, except when it is
  needed for fast development or ablation-related checks.

Open decisions:

- exact code layout: separate script, config-only variant, or a small model class; -> a small model class.
- whether to include a simple `GATConv` comparison here or keep it separate; -> only GCNConv.
- final experiment name used in code artifacts, tables, and plots should be
  `BasicGCN`; diploma text should use `Osnovni GCN`.
