# GazeMAE + MLP Baseline Implementation

This document describes how the repository implements the `GazeMAE_MLP`
baseline used for comparison with the proposed GNN models. The description is
based on the current code, mainly:

- `src/emotions/gazemae_model.py`
- `src/emotions/gazemae_baseline.py`
- `src/emotions/multiclass/baseline_model_multiclass.py`
- `src/emotions/multiclass/train_multiclass.py`
- `src/emotions/gnn_improvement_experiments/configs/quick_v1_v2/quick_v1_v2_train_multiclass_hci_tagging.yaml`
- `models/gazemae/README.md`

The baseline should be understood as a frozen pretrained gaze-representation
transfer baseline. It is not a full reimplementation or retraining of the
original GazeMAE method. In this project, GazeMAE is used because it is the
closest freely available and practically reusable representative of
self-supervised eye-movement representation learning. The repository loads
pretrained GazeMAE encoders, converts each MAHNOB-HCI eye-tracking window into a
fixed-length embedding, and trains only a small supervised MLP classifier on top
of these frozen embeddings.

## High-Level Pipeline

For each window used in the emotion-classification experiments, the baseline
performs the following steps:

1. Build the same window-level samples as the other tabular baselines, using the
   same dataset filtering, target labels, cross-validation splits, and minimum
   sample constraints.
2. Extract only the gaze coordinate signal required by GazeMAE:
   `time-rel-seconds`, `x-avg`, and `y-avg`.
3. Clip gaze coordinates to the MAHNOB-HCI screen bounds used in this project:
   width `1280`, height `800`.
4. Resample each 10-second eye-tracking window to a fixed 500 Hz sequence.
5. Split the fixed sequence into 2-second chunks.
6. Build a velocity signal from the resampled position signal.
7. Encode position chunks with a frozen pretrained GazeMAE position encoder.
8. Encode velocity chunks with a frozen pretrained GazeMAE velocity encoder.
9. Concatenate position and velocity chunk embeddings.
10. Pool chunk embeddings with mean and standard deviation across chunks.
11. Train a PyTorch MLP head on the resulting 512-dimensional window embedding.

The resulting model name in the experiment suite is `GazeMAE_MLP`.

## Local GazeMAE Assets

The GazeMAE runtime is self-contained in this repository. The project does not
import the original external GazeMAE code during experiments. Instead,
`src/emotions/gazemae_model.py` implements the minimal inference-time encoder
needed to load converted encoder-only checkpoints.

The converted checkpoints are stored in `models/gazemae/`:

| File | Signal | Source checkpoint name | Purpose |
|---|---|---|---|
| `pos-i3738-encoder-state.pt` | Position | `pos-i3738` | Frozen encoder for raw gaze position chunks |
| `vel-i8528-encoder-state.pt` | Velocity | `vel-i8528` | Frozen encoder for derived gaze velocity chunks |

The files keep only the encoder and bottleneck state dictionaries. Decoder
weights, optimizer state, and original training history are intentionally
excluded. The local checkpoint format is checked by
`GAZEMAE_ENCODER_CHECKPOINT_VERSION = "gfm-gazemae-encoder-state-v1"` before a
checkpoint is accepted.

## Encoder Architecture Reconstructed in the Repo

The GazeMAE encoder implementation is in `src/emotions/gazemae_model.py`.
The checkpoint metadata stores the architecture parameters, and
`GazeMAEEncoderConfig.from_mapping(...)` reconstructs a typed config from that
metadata.

Each encoder is a temporal convolutional network (TCN) built from four residual
blocks. A residual block contains:

- a ReLU before the first convolution;
- a first 1D convolution with dilation-specific padding;
- batch normalization;
- a second 1D convolution;
- a 1x1 skip convolution from input channels to output channels;
- ReLU and batch normalization after adding the residual branch;
- optional max-pooling downsampling, although the current checkpoints use no
  downsampling.

The residual block logic is implemented by `GazeMAEResidualBlock`. The TCN body
is implemented by `_GazeMAETCNEncoder`.

The actual metadata in the local checkpoints is:

| Encoder | Filters | Dilations per block | Downsamples | Kernel | Input channels | Latent size | Hierarchical | Multiscale | Causal |
|---|---:|---|---|---:|---:|---:|---|---|---|
| Position | `[128, 128, 128, 128]` | `[(1, 1), (2, 4), (8, 16), (32, 64)]` | `[0, 0, 0, 0]` | `3` | `2` | `64` | true | false | false |
| Velocity | `[256, 256, 256, 256]` | `[(1, 1), (2, 4), (8, 16), (32, 64)]` | `[0, 0, 0, 0]` | `3` | `2` | `64` | true | true | false |

Both local checkpoints are hierarchical. In the current `encode(...)` path, the
TCN computes two pooled outputs:

- `out_1`: the mean over time after blocks 0 and 1;
- `out_2`: the mean over time after blocks 2 and 3.

Each output is passed through its own bottleneck:

$$
z_l = \operatorname{BN}(\operatorname{ReLU}(W_l h_l + b_l)).
$$

Each bottleneck has latent size 64. For hierarchical encoders,
`GazeMAEEncoder.encode(..., cat_output=True)` concatenates the deep and shallow
bottleneck embeddings as `[z_2, z_1]`, giving a 128-dimensional embedding per
chunk for the position encoder and a 128-dimensional embedding per chunk for the
velocity encoder.

When a checkpoint is loaded, `load_gazemae_encoder(...)` moves the encoder to
the selected device, switches it to evaluation mode, and sets
`requires_grad_(False)` for every parameter. Therefore, the GazeMAE backbone is
strictly frozen during baseline training.

## Window Preprocessing

The window embedder is implemented by `GazeMAEWindowEmbedder` in
`src/emotions/gazemae_baseline.py`.

The default GazeMAE configuration is:

| Setting | Default |
|---|---:|
| Position checkpoint | `models/gazemae/pos-i3738-encoder-state.pt` |
| Velocity checkpoint | `models/gazemae/vel-i8528-encoder-state.pt` |
| Screen width | `1280` |
| Screen height | `800` |
| Clip coordinates to screen | `true` |
| Target sampling rate | `500` Hz |
| Chunk length | `2.0` s |
| Chunk pooling | `mean_std` |
| Encoder batch size | `256` |
| Device | `auto` |
| Cache embeddings | `true` |

The active quick multiclass configuration uses 10-second windows with no
overlap, `min_samples_per_window = 60`, and `standardize_features = true`.
The GazeMAE-specific settings are defined under the `gazemae:` block in
`quick_v1_v2_train_multiclass_hci_tagging.yaml`.

### Coordinate Handling

For each window, the embedder keeps the columns:

- `time-rel-seconds`
- `x-avg`
- `y-avg`

These columns are converted to numeric values, rows with missing required
values are dropped, and the remaining rows are sorted by `time-rel-seconds`.
If `clip_to_screen` is enabled, the coordinates are clipped to:

$$
x \in [0, 1280], \qquad y \in [0, 800].
$$

No coordinate scaling is applied. In particular, the implementation does not
scale the vertical coordinate from 800 to 1024 pixels. This is an intentional
project decision because such scaling would distort both absolute position and
derived velocity magnitudes.

### Resampling

For a 10-second window and a 500 Hz target rate, the target length is:

$$
T = 500 \cdot 10 = 5000.
$$

The implementation shifts timestamps so that the first valid sample in the
window is time zero. It then uses the unique shifted timestamps as the old time
axis and linearly interpolates `x-avg` and `y-avg` onto a fixed grid:

$$
\{0, 1/500, 2/500, \ldots, 4999/500\}.
$$

If the timestamp axis is degenerate, the code falls back to interpolation over
sample indices. The output position signal has shape `[2, 5000]`, where the two
channels are `x` and `y`.

### Velocity Signal

The velocity input is derived from the resampled position signal:

$$
v_t = \frac{|p_{t+1} - p_t|}{1000 / f_s},
$$

where $f_s = 500$ Hz and `1000 / f_s` is the number of milliseconds per sample.
The implementation computes the absolute first difference along time for both
coordinate channels, divides by milliseconds per sample, and pads one zero
sample at the end so the velocity sequence has the same length as the position
sequence.

This means the velocity encoder receives a two-channel signal with the same
shape as the position encoder input.

### Chunking

The default chunk duration is 2 seconds. At 500 Hz this gives:

$$
C = 500 \cdot 2 = 1000
$$

samples per chunk. A 10-second window therefore gives five full chunks.
The code trims any incomplete trailing chunk and reshapes the signal from
`[2, T]` to `[num_chunks, 2, chunk_len]`. For the default configuration this is:

$$
[2, 5000] \rightarrow [5, 2, 1000].
$$

Both the position signal and the velocity signal are chunked in the same way.

## Window Embedding Construction

For each 2-second chunk:

- the position chunk is passed to the frozen position encoder;
- the velocity chunk is passed to the frozen velocity encoder;
- each encoder returns a 128-dimensional chunk representation;
- the two representations are concatenated into a 256-dimensional chunk vector.

For a default 10-second window, this produces a chunk matrix:

$$
E \in \mathbb{R}^{5 \times 256}.
$$

The window-level representation is obtained by concatenating the mean and
standard deviation over chunks:

$$
g =
\left[
\operatorname{mean}_{c=1}^{5}(E_c),
\operatorname{std}_{c=1}^{5}(E_c)
\right].
$$

The final feature vector has:

$$
256 + 256 = 512
$$

dimensions. The feature columns are named `gazemae_z_000` through
`gazemae_z_511`.

This 512-dimensional vector is stored in a `TabularWindowSample`, so downstream
training can reuse the same baseline split, label, metric, and artifact logic as
other tabular models.

## Embedding Cache

GazeMAE embedding extraction is cached by `build_gazemae_tabular_samples(...)`.
The default cache directory is:

```text
data/cache/gazemae_embeddings
```

Each cache file is a Joblib file named from a SHA-256 cache key. The cache key
includes:

- the GazeMAE cache version;
- the dataset identity;
- dataset filtering and windowing settings;
- target columns;
- original tabular feature columns and drop-NA columns;
- `min_samples_per_window`;
- position and velocity checkpoint identities, including file hashes when the
  files exist;
- screen size, clipping, target sampling rate, chunk length, and chunk pooling.

When the experiment suite provides a snapshot manifest, the cache uses the
stable `snapshot_hash` and `snapshot_cache_key` instead of a timestamped CSV
path. This allows cached GazeMAE embeddings to be reused across timestamped
suite output directories while still preventing reuse when the data,
preprocessing settings, or checkpoint files change.

## MLP Classifier Head

The supervised classifier is implemented by `GazeMAEMLPMulticlassBaseline` in
`src/emotions/multiclass/baseline_model_multiclass.py`. It trains only on the
frozen 512-dimensional GazeMAE window embeddings.

The MLP architecture is:

```text
Linear(512, hidden_size)
ReLU
Dropout(dropout)
Linear(hidden_size, hidden_size)
ReLU
Dropout(dropout)
Linear(hidden_size, n_classes)
```

The default hyperparameters in the quick multiclass config are:

| Hyperparameter | Default |
|---|---:|
| Hidden size | `128` |
| Epochs | `100` |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Dropout | `0.2` |
| Early-stopping patience | `15` |
| Batch size | `128` |
| Random seed | `42` |
| Device | `auto` |

Training uses:

- `AdamW`;
- `CrossEntropyLoss`;
- mini-batch training with a seeded PyTorch `DataLoader`;
- validation loss for early stopping when a validation split is available;
- restoration of the best validation-loss state at the end of training.

The head stores a per-epoch training history with `epoch`, `train_loss`, and
`val_loss`. During experiment execution, this history is saved as
`mlp_training_history.csv` for both the normal `MLP` baseline and
`GazeMAE_MLP`.

If a training fold contains only one class, the implementation falls back to a
constant-class predictor rather than fitting the neural network.

## Feature Standardization and Cross-Validation

In the multiclass trainer, `GazeMAE_MLP` is treated as a baseline model but uses
its own feature matrix:

- normal tabular baselines use the configured tabular feature columns, such as
  `x-avg`, `y-avg`, and pupil features;
- `GazeMAE_MLP` uses only `GAZEMAE_FEATURE_COLUMNS`, i.e. the 512 frozen
  GazeMAE embedding columns.

When `dataset.standardize_features` is true, a `StandardScaler` is fitted only
on the training split and then applied to the train, validation, and test splits.
This is done separately for each feature kind, so the GazeMAE embeddings are
standardized with a scaler fitted only on the GazeMAE training embeddings for
that fold. The fitted scaler is saved as `feature_scaler.pkl` inside the
corresponding model artifact directory.

The baseline uses the same cross-validation indices as the other baselines and,
in the current quick comparison path, the same suite invocation as the GNN
models. This is important because it aligns the subject or recording folds used
for model comparison.

The quick comparison runner recognizes several aliases for the model name,
including `gazemae`, `gazemae_mlp`, `gazemae-mlp`, `gaze_mae`, and
`gaze_mae_mlp`, all mapped to `GazeMAE_MLP`.

## Relation to the Proposed GNN

The GNN and `GazeMAE_MLP` operate on the same windowed MAHNOB-HCI experiment
samples and the same target definitions, but they represent the signal very
differently:

| Aspect | GazeMAE_MLP | Proposed GNN |
|---|---|---|
| Input unit | Fixed 10-second window represented as a resampled sequence | 10-second window represented as a spatio-temporal graph |
| Main signal used | Gaze coordinates `x-avg`, `y-avg` | Node features such as gaze, pupil, distance, fixation-derived features, depending on config |
| Pretrained component | Frozen GazeMAE position and velocity encoders | None in the current supervised GNN experiments |
| Trainable component | MLP classifier head only | Full GNN model |
| Temporal modeling | Encoded inside pretrained TCN chunks, then pooled over chunks | Explicit graph edges and message passing across temporal and spatial relations |
| Window embedding | Mean and standard deviation of chunk embeddings | Learned graph-level pooling over node representations |
| Role in thesis | Transfer baseline representing accessible self-supervised gaze SOTA | Main proposed model family |

Because the GazeMAE encoders are frozen and were not trained on this MAHNOB-HCI
emotion-classification task, the baseline mainly tests whether pretrained
generic gaze dynamics are useful for the downstream affect-recognition labels.
The GNN, by contrast, learns its representation directly for the target task
from the graph-structured window representation.

## Practical Interpretation for Reporting

For the thesis, the model can be described as:

> A frozen GazeMAE transfer baseline with separate pretrained position and
> velocity TCN encoders. Each 10-second MAHNOB-HCI gaze window is clipped to the
> actual screen bounds, resampled to 500 Hz, split into five 2-second chunks,
> encoded with the frozen GazeMAE encoders, pooled with chunk-wise mean and
> standard deviation into a 512-dimensional vector, and classified with a
> supervised MLP head trained inside the same cross-validation protocol as the
> other baselines.

Important limitations to state:

- The original GazeMAE decoder and pretraining objective are not run in this
  repository.
- The position and velocity encoders are frozen.
- The downstream model is a small task-specific MLP head.
- The baseline uses only gaze coordinates for the GazeMAE embedding stage, not
  the richer graph feature set used by the GNN.
- The implementation is intended as a practical and reproducible transfer
  baseline, not as a claim that the original GazeMAE model is fully reproduced.

