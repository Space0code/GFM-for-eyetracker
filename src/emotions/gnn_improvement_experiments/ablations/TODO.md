# Ablations TODO

Purpose: implement signal/component ablations for the final GNN architecture.

This experiment should answer:

> Which input signals or graph components contribute to the final model?

Planned principle:

- keep the final GNN architecture as unchanged as possible;
- remove or disable one information source at a time;
- compare each ablation against the final model, not against the basic GNN;
- use the same data splits, labels, training protocol, and metrics as the final reported runs.

Candidate ablations:

- without temporal information (edges and features);
- without gaze-position information (spatial edges and gaze features);
- withou pupil-size information (pupil features);
- without screen-distance information (screen-distance features);

So, when we remove some information, we want to remove all information regarding that signal - i.e., edges created based on the information, node features and edge features gotten from the signal.

Important distinction:

- `basic_gnn` keeps the input information and simplifies the architecture;
- `ablations` keep the final architecture and remove information sources or
  components.

Open decisions:

- whether ablations should live as config variants only or as a dedicated runner; -> config variants if we can implement this cleanly. otherwise dedicated runner.
- naming convention for result tables and plots.
