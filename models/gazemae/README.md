# Local GazeMAE Encoder Checkpoints

This directory stores the local pretrained GazeMAE encoder weights used by the
`GazeMAE_MLP` transfer baseline.

The files are encoder-only `state_dict` checkpoints converted from the local
GazeMAE pretrained checkpoints:

- `pos-i3738-encoder-state.pt`: position encoder from `pos-i3738`
- `vel-i8528-encoder-state.pt`: velocity encoder from `vel-i8528`

Only the inference-time encoder and bottleneck weights are kept. Decoder,
optimizer, and training-history state are intentionally excluded so this
repository does not need to import the external GazeMAE code at runtime.

Original model reference:

GazeMAE: General Representations of Eye Movements using a Micro-Macro
Autoencoder, ICPR 2020.
