# Fixation Edge Construction Analysis

Date: 2026-05-20

This note records the local MAHNOB-HCI/TAGGING analysis used to decide how to
construct same-fixation edges in GNN v2. The final terminology is:

- English: **dilated intra-fixation edges**
- Slovenian: **razširjene povezave znotraj fiksacije**
- Suggested config name: `fixation_edge_mode: dilated`
- Suggested density parameter: `fixation_dilation_k: 3`

## Context

The previous GNN v2 implementation creates `fixation` edges only between
consecutive samples with the same valid `fixation-index`, in both directions.
With temporal edges enabled and `kt >= 1`, these edges are topologically a subset
of temporal edges. They therefore mostly add a separate relation label and
parameter path, not new graph reachability.

We compared this current construction with denser same-fixation alternatives:

- current sequential directed same-fixation edges;
- full directed same-fixation clique, with group size $m$ contributing
  $m(m - 1)$ edges;
- expected sparse clique with keep probability $0.1$;
- an earlier non-cyclic same-fixation jump estimate with $k_f = 10$;
- the final cyclic/permutation-based dilated intra-fixation construction with
  $k_f = 10$.

## Data And Settings

The analysis used the latest cleaned quick/Table-6 snapshot from the recent
GNN/GazeMAE run:

`results/quick_v1_v2_comparison/2026-05-18_12-06-36/model_runs/2026-05-18_12-06-38/experiments/multiclass_table6_valence_3class_emotion-elicitation/snapshot.csv`

The corresponding manifest is:

`results/quick_v1_v2_comparison/2026-05-18_12-06-36/model_runs/2026-05-18_12-06-38/experiments/multiclass_table6_valence_3class_emotion-elicitation/snapshot_manifest.yaml`

Key settings matched the quick/Table-6 GNN v2 configuration:

| Setting | Value |
|---|---:|
| Window length | 10 s |
| `min_samples_per_window` | 60 |
| Temporal horizon `kt` | 2 |
| Spatial neighbors `ks` | 2 |
| Excluded subjects | `P9`, `P12`, `P15` |
| Experiment type | `emotion-elicitation` |
| Label quality | `ok` |
| Candidate source CSV files in manifest | 942 |
| Eligible source CSV files after subject exclusions | 873 |
| Analyzed subjects | 22 |
| Subject-recording groups | 436 |
| Unique recording names | 20 |
| Usable 10 s windows | 5,034 |

Analyzed subjects:

`P1`, `P2`, `P4`, `P5`, `P7`, `P8`, `P10`, `P13`, `P14`, `P16`, `P17`,
`P18`, `P19`, `P20`, `P21`, `P22`, `P23`, `P24`, `P27`, `P28`, `P29`, `P30`.

## Main Aggregates

| Metric per 10 s graph | Mean | Median | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Nodes | 505.9 | 560 | 600 | 601 | 602 | 617 |
| Nodes with valid `fixation-index` | 492.4 | 540 | 599 | 600 | 601 | 601 |
| Fixation groups | 23.4 | 24 | 33 | 35 | 40 | 56 |
| Fixation group size | 21.1 | 16 | 39 | 54 | 101 | 601 |
| Current sequential fixation edges | 936 | 1,030 | 1,148 | 1,158 | 1,176 | 1,200 |
| Full directed clique edges | 19,525 | 15,590 | 34,158 | 45,038 | 110,100 | 360,600 |
| Expected clique edges with keep probability 0.1 | 1,953 | 1,559 | 3,416 | 4,504 | 11,010 | 36,060 |
| Earlier non-cyclic `kf=10` jump estimate | 4,363 | 4,804 | 5,340 | 5,382 | 5,406 | 5,414 |
| Final cyclic `kf=2` dilated fixation edges | 1,956 | 2,152 | 2,388 | 2,400 | 2,404 | 2,404 |
| Final cyclic `kf=3` dilated fixation edges | 2,908 | 3,204 | 3,558 | 3,590 | 3,606 | 3,606 |
| Final cyclic `kf=5` dilated fixation edges | 4,163 | 4,547 | 5,330 | 5,517 | 5,771 | 6,010 |
| Final cyclic `kf=10` dilated fixation edges | 7,273 | 7,892 | 9,856 | 10,313 | 11,175 | 12,020 |
| Temporal edges, `kt=2` | 2,018 | 2,234 | 2,394 | 2,398 | 2,402 | 2,462 |
| Spatial edges, `ks=2` | 1,408 | 1,546 | 1,671 | 1,691 | 1,746 | 1,908 |
| Base total: temporal + spatial + current fixation | 4,362 | 4,806 | 5,199 | 5,225 | 5,282 | 5,496 |

## Share Of Total Edges

| Alternative | Mean share | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| Full clique / temporal + spatial + full clique | 80.2% | 81.0% | 92.0% | 96.5% | 98.9% |
| Keep 0.1 clique / temporal + spatial + keep | 31.6% | 29.9% | 53.5% | 73.3% | 89.9% |
| Earlier non-cyclic `kf=10` jumps / temporal + spatial + jumps | 55.8% | 56.1% | 57.1% | 57.3% | 57.6% |
| Final cyclic `kf=2` dilated / temporal + spatial + dilated | 36.2% | 36.4% | 37.3% | 37.4% | 37.7% |
| Final cyclic `kf=3` dilated / temporal + spatial + dilated | 45.7% | 46.0% | 47.0% | 47.2% | 47.5% |
| Final cyclic `kf=5` dilated / temporal + spatial + dilated | 54.4% | 54.8% | 57.9% | 58.9% | 60.1% |
| Final cyclic `kf=10` dilated / temporal + spatial + dilated | 67.1% | 67.9% | 72.1% | 73.5% | 74.7% |

## Topological Overlap

| Metric | Result |
|---|---:|
| Temporal edges connecting nodes with the same valid `fixation-index` | mean 89.9%, median 90.5%, p95 95.8%, max 100% |
| Current sequential fixation edges already present as temporal edges | 100% |

The current sequential fixation relation therefore adds relation-specific
processing, but it does not add new graph reachability when `kt >= 1`.

## Worst-Case Windows

The worst cases are driven by unusually long fixation groups. The most extreme
window was:

| Subject | Recording | Window | Nodes | Fixation groups | Full clique edges | Keep 0.1 expected edges |
|---|---|---:|---:|---:|---:|---:|
| `P16` | `newyork_f.avi` | 7 | 601 | 1 | 360,600 | 36,060 |

In this case, a full clique would make fixation edges 98.9% of all edges under
the temporal + spatial + full-clique construction. Even a 0.1 sampled clique
would still contribute an expected 36,060 fixation edges and dominate the graph.

## Per-Subject Signal

| Subject | Windows | Mean full clique | Mean keep 0.1 | Mean sequential fixation | Mean fixation groups |
|---|---:|---:|---:|---:|---:|
| `P16` | 193 | 24,538 | 2,454 | 1,035 | 21.5 |
| `P19` | 229 | 25,589 | 2,559 | 975 | 20.7 |
| `P29` | 216 | 25,809 | 2,581 | 943 | 22.9 |
| `P24` | 231 | 12,158 | 1,216 | 722 | 20.0 |
| `P28` | 212 | 13,359 | 1,336 | 843 | 23.1 |
| Overall | 5,034 | 19,525 | 1,953 | 936 | 23.4 |

## Final Construction

We will use **dilated intra-fixation edges** / **razširjene povezave znotraj
fiksacije**.

For a fixation group, use the ordered contiguous run of nodes that share the same
valid `fixation-index`. Let the run contain $F$ nodes with local ranks
$0, 1, \ldots, F - 1$. For density parameter $k_f$, define the dilation step as

$s = \max(1, \lfloor F / k_f + 0.5 \rfloor)$.

This is deterministic nearest-integer half-up rounding. In implementation, use
`floor(F / k_f + 0.5)` rather than Python's banker-style `round`.

The candidate offsets are

$D = \{(1 + q s) \bmod F \mid q = 0, 1, \ldots, k_f - 1\}$.

Remove offset $0$ if it occurs, deduplicate offsets, and then create the
undirected edge set

$E_f = \{\{v_i, v_{(i + d) \bmod F}\} \mid i \in \{0, \ldots, F - 1\}, d \in D, i \ne (i + d) \bmod F\}$.

For PyG message passing, each undirected pair is stored as two directed edges.
There must be no duplicate directed edge pairs within the fixation relation.

Example: for a fixation of length $F = 30$ and $k_f = 10$, the dilation step is
$s = 3$. For local node $0$, the target ranks are:

$1, 4, 7, 10, 13, 16, 19, 22, 25, 28$.

The offset $1$ keeps local same-fixation communication, while the later offsets
spread information across the fixation. The modulo makes the construction
permutation-like: each offset defines a cyclic permutation over the nodes in the
fixation. Duplicate undirected pairs are collapsed before writing the final
bidirectional `edge_index`.

## Rationale

This construction was chosen for the following reasons:

- It adds new long-range same-fixation reachability, unlike the current purely
  sequential fixation edges.
- It keeps the number of edges per fixation tightly controlled.
- It avoids the p99/max edge explosion of a full clique and even of a 0.1 sampled
  full clique in very long fixations.
- It also avoids an edge-count implosion where too few fixation edges remain.
- It distributes information approximately uniformly across a fixation.
- Nearby temporal communication is already represented by temporal edges, so
  the fixation relation can focus on spreading information within the fixation
  event rather than duplicating the temporal relation.

## Implementation Notes

Suggested configuration:

```yaml
fixation_edge_mode: dilated
fixation_dilation_k: 3
```

No backward-compatibility mode is required inside GNN v2. GNN v1 can be used as
the stable older graph construction when an old baseline is needed.

Do not use uncapped full same-fixation cliques as the default graph
construction. If random sparse clique sampling is tested later, it should include
an explicit cap or per-node bound and should be treated as a separate ablation.
