# Intention
Generate a **detailed end-to-end dataflow schema** from raw files to predictions for both GNN and baselines.

# Prompt
Create a technical dataflow of my eye-tracking ML repository.

Repository anchors:
- HCI conversion: `src/data/data_conversion/hci_tagging_conversion.py`
- eSEEd conversion: `archive/src/data/data_conversion/eSEED_v2_conversion.py`
- Preprocessing: `src/data/data_preprocess.py`
- Snapshot builder: `src/emotions/suite/data_snapshot.py`
- Graph builder: `src/data/data.py` (`SpacioTemporalDataset`)
- Tabular builder: `src/emotions/train_baseline.py` (`build_tabular_samples`)
- Trainers:
  - `src/emotions/binary/train_binary.py`
  - `src/emotions/multiclass/train_multiclass.py`
  - `src/emotions/regression/train_regression.py`
- Suite compare/plots:
  - `src/emotions/suite/compare_suite_results.py`

Required content:
- Show **two input branches** (HCI and eSEEd) merging into canonical processing.
- Show cleaning/filtering details: confidence masking, trimming, interpolation, dropna.
- Show windowing stage clearly before branch split.
- Show separate branches:
  - GNN graph branch (nodes/features + temporal/spatial edges),
  - baseline tabular aggregation branch.
- Show CV splitting point.
- Show fold-safe operations explicitly:
  - thresholds fit on train fold only,
  - scalers fit on train fold only.
- Show outputs/artifacts and clearly separate:
  - classification artifacts (confusion matrices etc.),
  - regression artifacts (no confusion matrices).

Rules:
- Use current repository behavior as source of truth.
- If a detail is inferred, mark it as `(inferred)`.
- Do not invent files/modules that are not listed.

Output format (strict):
1. `DETAILED_MERMAID` (single Mermaid diagram)
2. `SIMPLIFIED_MERMAID` (single Mermaid diagram for slides)
3. `STAGE_TABLE` with columns: `Stage | Input | Transform | Output`
