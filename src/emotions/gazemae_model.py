"""Minimal GazeMAE encoder used for frozen embedding baselines.

This module contains only the inference-time TCN encoder and bottleneck layers
needed to load local encoder-only GazeMAE state dictionaries. It intentionally
does not depend on the original external GazeMAE repository or its decoder,
training, and dataset code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


GAZEMAE_ENCODER_CHECKPOINT_VERSION = "gfm-gazemae-encoder-state-v1"


@dataclass(frozen=True)
class GazeMAEEncoderConfig:
    """Architecture settings needed to reconstruct a pretrained GazeMAE encoder."""

    filters: tuple[int, ...]
    dilations: tuple[tuple[int, int], ...]
    downsamples: tuple[int, ...]
    kernel_size: int = 3
    in_channels: int = 2
    latent_size: int = 64
    hierarchical: bool = True
    multiscale: bool = False
    causal: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GazeMAEEncoderConfig":
        """Build a typed config from checkpoint metadata."""
        return cls(
            filters=tuple(int(value) for value in raw["filters"]),
            dilations=tuple(tuple(int(part) for part in pair) for pair in raw["dilations"]),
            downsamples=tuple(int(value) for value in raw["downsamples"]),
            kernel_size=int(raw.get("kernel_size", 3)),
            in_channels=int(raw.get("in_channels", 2)),
            latent_size=int(raw.get("latent_size", 64)),
            hierarchical=bool(raw.get("hierarchical", True)),
            multiscale=bool(raw.get("multiscale", False)),
            causal=bool(raw.get("causal", False)),
        )

    def as_metadata(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation for converted checkpoints."""
        return {
            "filters": list(self.filters),
            "dilations": [list(pair) for pair in self.dilations],
            "downsamples": list(self.downsamples),
            "kernel_size": int(self.kernel_size),
            "in_channels": int(self.in_channels),
            "latent_size": int(self.latent_size),
            "hierarchical": bool(self.hierarchical),
            "multiscale": bool(self.multiscale),
            "causal": bool(self.causal),
        }


class GazeMAEResidualBlock(nn.Module):
    """Residual TCN block matching the pretrained GazeMAE encoder layout."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dilations: tuple[int, int],
        kernel_size: int,
        causal: bool,
        downsample: int = 0,
    ) -> None:
        super().__init__()
        self.causal = bool(causal)
        self.kernel_size = int(kernel_size)

        self.relu = nn.ReLU()
        self.conv1 = self._build_conv_layer(in_channels, mid_channels, dilations[0])
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = self._build_conv_layer(mid_channels, out_channels, dilations[1])
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.downsample = nn.MaxPool1d(downsample, downsample) if downsample > 0 else None

    def _build_conv_layer(self, in_channels: int, out_channels: int, dilation: int) -> nn.Module:
        if dilation >= 1:
            padding = int(dilation) * (self.kernel_size - 1)
            pad_sides: int | tuple[int, int] = (padding, 0) if self.causal else int(padding / 2)
            return nn.Sequential(
                nn.ConstantPad1d(pad_sides, 0),
                nn.Conv1d(in_channels, out_channels, self.kernel_size, dilation=dilation),
            )
        return nn.Conv1d(in_channels, out_channels, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.relu(self.conv1(x)))
        out = self.conv2(out)
        out = out + self.skip_conv(x)
        out = self.relu(out)
        out = self.bn2(out)
        if self.downsample is not None:
            out = self.downsample(out)
        return out


class _GazeMAETCNEncoder(nn.Module):
    """TCN encoder body matching the original GazeMAE module names."""

    def __init__(self, config: GazeMAEEncoderConfig) -> None:
        super().__init__()
        if len(config.filters) != len(config.dilations) or len(config.filters) != len(config.downsamples):
            raise ValueError("GazeMAE encoder filters, dilations, and downsamples must have the same length.")
        if config.hierarchical and len(config.filters) < 4:
            raise ValueError("Hierarchical GazeMAE encoders require four TCN blocks.")

        self.hierarchical = bool(config.hierarchical)
        self.multiscale = bool(config.multiscale)
        self.causal = bool(config.causal)
        self.in_channels = int(config.in_channels)
        self.kernel_size = int(config.kernel_size)
        self.filters = list(config.filters)
        self.dilations = list(config.dilations)
        self.downsamples = list(config.downsamples)

        if self.hierarchical:
            self.out_dim = int(self.filters[1])
            self.out_dim2 = int(self.filters[3])
        elif self.multiscale:
            self.out_dim = int(sum(self.filters[1::2]))
            self.out_dim2 = None
        else:
            self.out_dim = int(self.filters[-1])
            self.out_dim2 = None

        blocks = []
        for block_num, (filters, dilations, downsample) in enumerate(
            zip(self.filters, self.dilations, self.downsamples)
        ):
            blocks.append(
                GazeMAEResidualBlock(
                    in_channels=self.in_channels if block_num == 0 else self.filters[block_num - 1],
                    mid_channels=filters,
                    out_channels=filters,
                    dilations=dilations,
                    kernel_size=self.kernel_size,
                    causal=self.causal,
                    downsample=downsample,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.hierarchical:
            out_1 = self.blocks[1](self.blocks[0](x))
            out_2 = self.blocks[3](self.blocks[2](out_1))
            return out_1.mean(-1), out_2.mean(-1)

        if not self.multiscale:
            return self.blocks(x).mean(-1)

        block_features = []
        for block_num, block in enumerate(self.blocks):
            x = block(x)
            if block_num % 2 == 1:
                block_features.append(x.mean(-1))
        return torch.cat(block_features, dim=-1)


class GazeMAEEncoder(nn.Module):
    """Frozen-compatible GazeMAE encoder with original state-dict names."""

    def __init__(self, config: GazeMAEEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_size = int(config.latent_size)
        self.hierarchical = bool(config.hierarchical)
        self.encoder = _GazeMAETCNEncoder(config)
        self.bottleneck_fns = nn.ModuleDict({"1": self._build_bottleneck(self.encoder.out_dim)})
        if self.hierarchical:
            if self.encoder.out_dim2 is None:
                raise ValueError("Hierarchical encoder is missing second output dimension.")
            self.bottleneck_fns.update({"2": self._build_bottleneck(self.encoder.out_dim2)})

    def _build_bottleneck(self, in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, self.latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_size),
        )

    def bottleneck(self, x: torch.Tensor, level: str = "1") -> tuple[torch.Tensor, None, None]:
        """Apply one GazeMAE bottleneck level."""
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)
        z = self.bottleneck_fns[level](x)
        return z, None, None

    def encode(
        self,
        x: torch.Tensor,
        cat_output: bool = True,
    ) -> tuple[torch.Tensor | list[torch.Tensor], None | list[None], None | list[None]]:
        """Encode `[batch, 2, time]` chunks using the pretrained GazeMAE encoder."""
        if self.hierarchical:
            enc_out1, enc_out2 = self.encoder(x)
            z1, mean1, logvar1 = self.bottleneck(enc_out1, level="1")
            z2, mean2, logvar2 = self.bottleneck(enc_out2, level="2")
            if not cat_output:
                return [z2, z1], [mean2, mean1], [logvar2, logvar1]
            return torch.cat([z2, z1], dim=-1), None, None
        return self.bottleneck(self.encoder(x), level="1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the concatenated latent representation."""
        encoded, _, _ = self.encode(x, cat_output=True)
        if not isinstance(encoded, torch.Tensor):
            raise RuntimeError("Expected concatenated tensor output from GazeMAEEncoder.")
        return encoded


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_gazemae_encoder(checkpoint_path: Path | str, device: torch.device) -> GazeMAEEncoder:
    """Load a local encoder-only GazeMAE checkpoint for inference."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing local GazeMAE encoder checkpoint: {path}")

    checkpoint = _torch_load(path, device=device)
    version = checkpoint.get("format_version")
    if version != GAZEMAE_ENCODER_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported GazeMAE encoder checkpoint format {version!r}; "
            f"expected {GAZEMAE_ENCODER_CHECKPOINT_VERSION!r}."
        )

    config = GazeMAEEncoderConfig.from_mapping(checkpoint["model_config"])
    model = GazeMAEEncoder(config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_encoder_checkpoint_payload(
    *,
    source_checkpoint_name: str,
    signal_type: str,
    config: GazeMAEEncoderConfig,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Build the serializable local checkpoint payload used by conversion tests."""
    return {
        "format_version": GAZEMAE_ENCODER_CHECKPOINT_VERSION,
        "source_checkpoint_name": str(source_checkpoint_name),
        "signal_type": str(signal_type),
        "model_config": config.as_metadata(),
        "model_state_dict": dict(state_dict),
    }
