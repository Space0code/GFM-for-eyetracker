# Solution: GNN High-Level Overview Image

## VARIANT_A_TECHNICAL
```mermaid
flowchart TB
  subgraph input["Input"]
    win["Eye-tracking window<br/>(time samples)"]
    feat["Node features<br/>x-avg, y-avg, pupil L/R"]
    win --> feat
  end

  subgraph graph_build["Graph Construction"]
    nodes["Nodes = time samples"]
    temporal["Temporal edges<br/>kt-neighborhood"]
    spatial["Spatial edges<br/>ks kNN in screen-space"]
    nodes --> temporal
    nodes --> spatial
  end

  subgraph model["Model"]
    pre["Optional preprocess MLP"]
    gnn["Multi-layer heterogeneous GNN<br/>(temporal + spatial relations)"]
    res["Residual + LayerNorm"]
    pool["Graph pooling<br/>mean or mean+max"]
    head["Head MLP"]
    pre --> gnn --> res --> pool --> head
  end

  out["Output task<br/>binary / multiclass / regression"]
  call1["time dynamics"]
  call2["screen-space patterns"]
  legend_t["Blue = temporal relation"]
  legend_s["Green = spatial relation"]

  feat --> nodes
  temporal --> pre
  spatial --> pre
  head --> out
  temporal -.-> call1
  spatial -.-> call2

  classDef base fill:#F8FAFC,stroke:#64748B,color:#0F172A;
  classDef temporal fill:#E8F1FF,stroke:#3B82F6,color:#0F172A;
  classDef spatial fill:#EAFBF0,stroke:#16A34A,color:#0F172A;
  classDef out fill:#FFF7ED,stroke:#EA580C,color:#0F172A;
  classDef note fill:#FFFBEB,stroke:#D97706,color:#0F172A;

  class win,feat,nodes,pre,gnn,res,pool,head,legend_t,legend_s base;
  class temporal,legend_t temporal;
  class spatial,legend_s spatial;
  class out out;
  class call1,call2 note;
```

## VARIANT_B_EXECUTIVE
```mermaid
flowchart TB
  A["Eye-tracking window"]
  B["Build graph<br/>(samples + relationships)"]
  C["GNN encoder<br/>(mixes time + space)"]
  D["Pool window signal"]
  E["Predict state"]

  T["time dynamics"]
  S["screen-space patterns"]
  L1["Blue = temporal links"]
  L2["Green = spatial links"]

  A --> B --> C --> D --> E
  C -.-> T
  C -.-> S

  classDef base fill:#F8FAFC,stroke:#64748B,color:#0F172A;
  classDef temporal fill:#E8F1FF,stroke:#3B82F6,color:#0F172A;
  classDef spatial fill:#EAFBF0,stroke:#16A34A,color:#0F172A;

  class A,B,C,D,E base;
  class T,L1 temporal;
  class S,L2 spatial;
```

## CAPTION_A
Each time window is converted into a graph whose nodes are time samples and whose edges encode temporal and spatial relationships. A heterogeneous GNN repeatedly mixes both relation types, then pools window information into one embedding for prediction. The same backbone supports binary, multiclass, and regression tasks.

## CAPTION_B
The model treats each short gaze segment as a connected structure, not just a flat feature vector. It combines how gaze changes over time with where gaze clusters on screen, then predicts the target state from the summarized window. This captures both motion and spatial behavior in one pipeline.
