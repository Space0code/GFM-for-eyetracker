# Ablations TODO

Purpose: implement signal/component ablations for the final GNN architecture.

This experiment should answer:

> Which input signals or graph components contribute to the final model?

Planned principle:

- keep the final GNN architecture as unchanged as possible;
- remove or disable one information source at a time;
- compare each ablation against the final architecture under the same ablation
  evaluation protocol, not against `BasicGCN`;
- use the same labels, training protocol, and metrics as the final reported
  runs, with the ablation-specific subject k-fold splits described below.

Candidate ablations:

- without temporal information: remove temporal node features, temporal edge
  features, and `temporal_forward`/`temporal_backward` edges;
- without spatial/gaze information: remove gaze-position node features,
  gaze-derived edge features from all relations, and `spatial` edges;
- without fixation information: remove fixation node features,
  fixation-derived edge features, and `fixation` edges;
- without pupil-size information: remove pupil node features and any
  pupil-derived edge features or graph edges if such features are added;
- without screen-distance information: remove screen-distance node features and
  any screen-distance-derived edge features or graph edges.

When an information source is removed, remove all information derived from that
source: node features, edge features, and graph edges constructed from that
signal.

Evaluation plan:

- run ablations with subject k-fold for time efficiency, currently with `k=5`;
- use the same 3-class valence/arousal task definitions as the main experiment
  unless a later ablation run explicitly targets low/high as a follow-up;
- in the diploma, explain that subject k-fold and subject LOO are comparable
  enough for the purpose of the ablation analysis;
- keep subject LOO as the main result setting for final model comparisons;
- include recording LOO as an additional final-result setting where relevant;
- mention k-fold in the diploma only for the ablation study.

Important distinction:

- `BasicGCN` keeps the input information and simplifies the architecture;
- `ablations` keep the final architecture and remove information sources or
  components.

Open decisions:

- whether ablations should live as config variants only or as a dedicated runner; -> config variants if we can implement this cleanly. otherwise dedicated runner.
- naming convention for result tables and plots.
