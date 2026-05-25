# Basic GNN TODO

Purpose: implement the architecture-level graph baseline for the diploma
comparison.

This experiment should answer:

> What do we get from the graph representation itself, before adding the full heterogeneous GNN architecture?

Planned baseline:

- use the same source data, windows, splits, node features, and core signals as
  the final GNN;
- keep gaze, pupil, fixation, and screen-distance information available;
- simplify the architecture instead of removing signals;
- collapse graph relations into a homogeneous graph, or otherwise use a single
  shared message-passing path;
- start with a simple `GCNConv` baseline;
- avoid learned scalar edge weights in the first version;
- keep the hidden size, number of layers, optimizer, early stopping, and metrics
  as comparable to the final GNN as practical;
- report it as an architectural baseline, not as a signal ablation.

Open decisions:

- exact code layout: separate script, config-only variant, or a small model class; -> a small model class.
- whether to include a simple `GATConv` comparison here or keep it separate; -> only GCNConv.
- final experiment name used in tables and plots.
