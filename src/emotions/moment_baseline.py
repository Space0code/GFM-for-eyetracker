"""Build frozen MOMENT window embeddings for emotion-classification baselines.

This module adapts MOMENT in embedding mode to the MAHNOB-HCI windowed
training pipeline. The MOMENT backbone is always frozen; downstream trainers
receive one fixed-size feature vector per eye-tracking window and train only an
MLP head.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from data.hci_signals import DISTANCE_AVG_COLUMN, FIXATION_DURATION_COLUMN
from emotions.common.dataset_config import build_tabular_samples_kwargs
from emotions.gazemae_baseline import (
    GAZEMAE_FEATURE_COLUMNS,
    _compute_file_sha256,
)
from emotions.train_baseline import TabularWindowSample, build_tabular_samples


MOMENT_FEATURE_PREFIX = "moment_z_"
MOMENT_FEATURE_DIM = 768
MOMENT_FEATURE_COLUMNS = [f"{MOMENT_FEATURE_PREFIX}{idx:03d}" for idx in range(MOMENT_FEATURE_DIM)]
MOMENT_CACHE_VERSION = "moment-cache-v1"
MOMENT_DEFAULT_MODEL_NAME = "AutonLab/MOMENT-1-base"
MOMENT_DEFAULT_PARAMETER_COUNT = 125_000_000

MOMENT_GAZE_MODEL_NAME = "MOMENT_gaze"
MOMENT_PUPIL_MODEL_NAME = "MOMENT_pupil"
MOMENT_GAZE_PUPIL_MODEL_NAME = "MOMENT_gaze_pupil"
MOMENT_ALL_SIGNALS_MODEL_NAME = "MOMENT_all_signals"
MOMENT_GAZEMAE_GAZE_PUPIL_MODEL_NAME = "MOMENT_GazeMAE_gaze_pupil"
MOMENT_GAZEMAE_ALL_SIGNALS_MODEL_NAME = "MOMENT_GazeMAE_all_signals"

MOMENT_MODEL_NAMES = {
    MOMENT_GAZE_MODEL_NAME,
    MOMENT_PUPIL_MODEL_NAME,
    MOMENT_GAZE_PUPIL_MODEL_NAME,
    MOMENT_ALL_SIGNALS_MODEL_NAME,
    MOMENT_GAZEMAE_GAZE_PUPIL_MODEL_NAME,
    MOMENT_GAZEMAE_ALL_SIGNALS_MODEL_NAME,
}
MOMENT_FUSION_MODEL_NAMES = {
    MOMENT_GAZEMAE_GAZE_PUPIL_MODEL_NAME,
    MOMENT_GAZEMAE_ALL_SIGNALS_MODEL_NAME,
}
MOMENT_MODEL_TO_SIGNAL_SUBSET = {
    MOMENT_GAZE_MODEL_NAME: "gaze",
    MOMENT_PUPIL_MODEL_NAME: "pupil",
    MOMENT_GAZE_PUPIL_MODEL_NAME: "gaze_pupil",
    MOMENT_ALL_SIGNALS_MODEL_NAME: "all_signals",
    MOMENT_GAZEMAE_GAZE_PUPIL_MODEL_NAME: "gaze_pupil",
    MOMENT_GAZEMAE_ALL_SIGNALS_MODEL_NAME: "all_signals",
}
MOMENT_SIGNAL_SUBSETS = {
    "gaze": ["x-avg", "y-avg"],
    "pupil": ["pupil-size-left-avg", "pupil-size-right-avg"],
    "gaze_pupil": ["x-avg", "y-avg", "pupil-size-left-avg", "pupil-size-right-avg"],
    "all_signals": [
        "x-avg",
        "y-avg",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
        DISTANCE_AVG_COLUMN,
        FIXATION_DURATION_COLUMN,
    ],
}


@dataclass(frozen=True)
class MomentConfig:
    """Resolved MOMENT embedding settings."""

    model_name: str = MOMENT_DEFAULT_MODEL_NAME
    sequence_length: int = 512
    batch_size: int = 64
    device: str = "auto"
    cache_embeddings: bool = True
    cache_dir: Path = Path("data/cache/moment_embeddings")
    local_files_only: bool = False


def resolve_moment_signal_subset(model_name_or_subset: str) -> str:
    """Return the MOMENT signal subset for a configured model or subset name."""
    key = str(model_name_or_subset).strip()
    subset = MOMENT_MODEL_TO_SIGNAL_SUBSET.get(key, key)
    if subset not in MOMENT_SIGNAL_SUBSETS:
        raise ValueError(
            f"Unknown MOMENT signal subset '{model_name_or_subset}'. "
            f"Allowed: {sorted(MOMENT_SIGNAL_SUBSETS)}"
        )
    return subset


def resolve_moment_feature_columns(model_name_or_subset: str) -> List[str]:
    """Return raw signal columns used by one MOMENT subset."""
    return list(MOMENT_SIGNAL_SUBSETS[resolve_moment_signal_subset(model_name_or_subset)])


def resolve_moment_config(raw_cfg: Mapping[str, Any] | None, dataset_cfg: Mapping[str, Any]) -> MomentConfig:
    """Resolve config values for frozen MOMENT embedding extraction."""
    raw_cfg = raw_cfg or {}
    cache_dir = raw_cfg.get("cache_dir")
    if cache_dir is None:
        cache_dir = Path(str(dataset_cfg.get("cache_dir", "data/cache"))) / "moment_embeddings"
    return MomentConfig(
        model_name=str(raw_cfg.get("model_name", MOMENT_DEFAULT_MODEL_NAME)),
        sequence_length=int(raw_cfg.get("sequence_length", 512)),
        batch_size=int(raw_cfg.get("batch_size", 64)),
        device=str(raw_cfg.get("device", "auto")),
        cache_embeddings=bool(raw_cfg.get("cache_embeddings", True)),
        cache_dir=Path(str(cache_dir)),
        local_files_only=bool(raw_cfg.get("local_files_only", False)),
    )


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _interp_1d(old_x: np.ndarray, old_y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    if len(old_x) < 2:
        return np.full_like(new_x, fill_value=float(old_y[0]), dtype=np.float32)
    return np.interp(new_x, old_x, old_y).astype(np.float32)


def _resample_channels_to_fixed_len(
    values: np.ndarray,
    timestamps_seconds: np.ndarray,
    target_len: int,
    window_seconds: float,
) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected signal array shape [channels, time], got {values.shape}.")
    if target_len <= 0:
        raise ValueError(f"Invalid MOMENT sequence_length={target_len}.")

    shifted = timestamps_seconds.astype(np.float64) - float(timestamps_seconds[0])
    unique_axis, unique_indices = np.unique(shifted, return_index=True)
    if len(unique_axis) >= 2 and unique_axis[-1] > 0:
        old_axis = unique_axis
        values = values[:, unique_indices]
        new_axis = np.linspace(0.0, float(window_seconds), num=target_len, endpoint=False, dtype=np.float64)
    else:
        old_axis = np.arange(values.shape[1], dtype=np.float64)
        new_axis = np.linspace(0.0, float(values.shape[1] - 1), num=target_len, dtype=np.float64)

    out = np.zeros((values.shape[0], target_len), dtype=np.float32)
    for channel_idx in range(values.shape[0]):
        out[channel_idx] = _interp_1d(old_axis, values[channel_idx], new_axis)
    return out


def _load_moment_pipeline(config: MomentConfig, device: torch.device) -> torch.nn.Module:
    try:
        from momentfm import MOMENTPipeline
    except ImportError as exc:
        raise ImportError(
            "MOMENT baselines require the optional 'momentfm' package. "
            "Install it in the active gfm environment with: pip install momentfm"
        ) from exc

    model = MOMENTPipeline.from_pretrained(
        config.model_name,
        model_kwargs={"task_name": "embedding"},
        local_files_only=bool(config.local_files_only),
    )
    model.init()
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class MomentWindowEmbedder:
    """Frozen MOMENT encoder for one multichannel eye-tracking window."""

    def __init__(self, config: MomentConfig, signal_subset: str, window_seconds: float) -> None:
        self.config = config
        self.signal_subset = resolve_moment_signal_subset(signal_subset)
        self.signal_columns = resolve_moment_feature_columns(self.signal_subset)
        self.window_seconds = float(window_seconds)
        self.device = _resolve_device(config.device)
        self.model = _load_moment_pipeline(config, device=self.device)

    def _prepare_window(self, window_df: pd.DataFrame) -> np.ndarray:
        required = ["time-rel-seconds", *self.signal_columns]
        missing = [column for column in required if column not in window_df.columns]
        if missing:
            raise ValueError(f"Missing required MOMENT columns for subset {self.signal_subset}: {missing}")

        frame = window_df.loc[:, required].copy()
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=required).sort_values("time-rel-seconds").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Cannot build MOMENT embedding for an empty/NaN-only {self.signal_subset} window.")

        values = frame.loc[:, self.signal_columns].to_numpy(dtype=np.float32).T
        timestamps = frame["time-rel-seconds"].to_numpy(dtype=np.float64)
        return _resample_channels_to_fixed_len(
            values=values,
            timestamps_seconds=timestamps,
            target_len=int(self.config.sequence_length),
            window_seconds=self.window_seconds,
        )

    def embed_window(self, window_df: pd.DataFrame) -> np.ndarray:
        """Return one frozen MOMENT embedding for one window."""
        signal = self._prepare_window(window_df)
        with torch.no_grad():
            x_enc = torch.tensor(signal[None, :, :], dtype=torch.float32, device=self.device)
            output = self.model(x_enc=x_enc)
            embeddings = getattr(output, "embeddings", None)
            if embeddings is None:
                raise RuntimeError("MOMENT output did not contain an 'embeddings' tensor.")
            embedding = embeddings.detach().cpu().numpy()
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(f"Expected MOMENT embedding shape [1, D], got {embedding.shape}.")
        embedding = embedding[0].astype(np.float32)
        if embedding.shape[0] != MOMENT_FEATURE_DIM:
            raise ValueError(
                f"Expected {MOMENT_FEATURE_DIM} MOMENT features for {self.config.model_name}, "
                f"got {embedding.shape[0]}."
            )
        return embedding

    def embed_window_as_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        embedding = self.embed_window(window_df)
        return {name: float(value) for name, value in zip(MOMENT_FEATURE_COLUMNS, embedding)}


def _load_snapshot_manifest(dataset_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_candidates: List[Path] = []
    manifest_path = dataset_cfg.get("snapshot_manifest_path")
    if isinstance(manifest_path, str) and manifest_path.strip():
        manifest_candidates.append(Path(manifest_path))

    data_filepath = dataset_cfg.get("data_filepath")
    if isinstance(data_filepath, str) and data_filepath.strip():
        data_path = Path(data_filepath)
        manifest_candidates.append(data_path.with_name("snapshot_manifest.yaml"))

    for path in manifest_candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if isinstance(payload, dict):
            return payload
    return {}


def _resolve_dataset_identity(dataset_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return stable data identity for MOMENT cache reuse across suite runs."""
    manifest = _load_snapshot_manifest(dataset_cfg)
    snapshot_hash = dataset_cfg.get("snapshot_hash") or manifest.get("snapshot_hash")
    cache_payload = manifest.get("cache") if isinstance(manifest.get("cache"), dict) else {}
    snapshot_cache_key = dataset_cfg.get("snapshot_cache_key") or cache_payload.get("cache_key")
    if snapshot_hash:
        return {
            "kind": "suite_snapshot",
            "snapshot_hash": str(snapshot_hash),
            "snapshot_cache_key": str(snapshot_cache_key) if snapshot_cache_key else None,
            "row_count": int(manifest["row_count"]) if "row_count" in manifest else None,
            "column_count": int(manifest["column_count"]) if "column_count" in manifest else None,
        }

    data_filepath = dataset_cfg.get("data_filepath")
    if isinstance(data_filepath, str) and Path(data_filepath).exists():
        path = Path(data_filepath)
        return {
            "kind": "file_sha256",
            "path_name": path.name,
            "sha256": _compute_file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    return {
        "kind": "path_fallback",
        "data_filepath": str(data_filepath),
        "data_dir": str(dataset_cfg.get("data_dir")),
    }


def _cache_key(
    *,
    config: MomentConfig,
    signal_subset: str,
    dataset_cfg: Mapping[str, Any],
    target_columns: List[str],
    feature_columns: List[str],
    dropna_columns: List[str],
    min_samples_per_window: int,
) -> str:
    payload = {
        "version": MOMENT_CACHE_VERSION,
        "dataset_identity": _resolve_dataset_identity(dataset_cfg),
        "dataset": {
            key: dataset_cfg.get(key)
            for key in [
                "data_dir",
                "filter_subjects",
                "filter_recordings",
                "exclude_subjects",
                "window_length",
                "window_overlap",
                "target_aggregation",
                "allowed_experiment_types",
                "label_quality_column",
                "allowed_label_quality_values",
            ]
        },
        "target_columns": target_columns,
        "feature_columns": feature_columns,
        "dropna_columns": dropna_columns,
        "min_samples_per_window": int(min_samples_per_window),
        "moment": {
            "model_name": config.model_name,
            "signal_subset": resolve_moment_signal_subset(signal_subset),
            "signal_columns": resolve_moment_feature_columns(signal_subset),
            "sequence_length": int(config.sequence_length),
        },
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_moment_tabular_samples(
    *,
    dataset_cfg: Mapping[str, Any],
    target_columns: List[str],
    feature_columns: List[str],
    dropna_columns: List[str],
    min_samples_per_window: int,
    moment_cfg: Mapping[str, Any] | None,
    signal_subset: str,
) -> List[TabularWindowSample]:
    """Build TabularWindowSample objects whose features are frozen MOMENT embeddings."""
    resolved = resolve_moment_config(moment_cfg, dataset_cfg=dataset_cfg)
    subset = resolve_moment_signal_subset(signal_subset)
    moment_feature_columns = resolve_moment_feature_columns(subset)
    cache_key = _cache_key(
        config=resolved,
        signal_subset=subset,
        dataset_cfg=dataset_cfg,
        target_columns=target_columns,
        feature_columns=moment_feature_columns,
        dropna_columns=dropna_columns,
        min_samples_per_window=min_samples_per_window,
    )
    cache_path = resolved.cache_dir / f"{cache_key}.joblib"
    if resolved.cache_embeddings and cache_path.exists():
        return joblib.load(cache_path)

    embedder = MomentWindowEmbedder(
        config=resolved,
        signal_subset=subset,
        window_seconds=float(dataset_cfg.get("window_length", 10)),
    )
    samples = build_tabular_samples(
        **build_tabular_samples_kwargs(
            dataset_cfg=dataset_cfg,
            target_columns=target_columns,
            feature_columns=moment_feature_columns,
            dropna_columns=dropna_columns,
            min_samples_per_window=min_samples_per_window,
        ),
        window_feature_builder=embedder.embed_window_as_features,
    )
    if resolved.cache_embeddings:
        resolved.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(samples, cache_path)
    return samples


def build_moment_gazemae_fusion_samples(
    *,
    moment_samples: List[TabularWindowSample],
    gazemae_samples: List[TabularWindowSample],
) -> List[TabularWindowSample]:
    """Concatenate MOMENT and GazeMAE frozen embeddings for matching windows."""
    if len(moment_samples) != len(gazemae_samples):
        raise ValueError(
            "MOMENT/GazeMAE sample counts differ for fusion: "
            f"{len(moment_samples)} vs {len(gazemae_samples)}."
        )

    fused: List[TabularWindowSample] = []
    for idx, (moment_sample, gazemae_sample) in enumerate(zip(moment_samples, gazemae_samples)):
        if (moment_sample.subject, moment_sample.recording) != (gazemae_sample.subject, gazemae_sample.recording):
            raise ValueError(
                "MOMENT/GazeMAE sample metadata mismatch at index "
                f"{idx}: {(moment_sample.subject, moment_sample.recording)} vs "
                f"{(gazemae_sample.subject, gazemae_sample.recording)}."
            )
        features = {
            **{name: moment_sample.features[name] for name in MOMENT_FEATURE_COLUMNS},
            **{name: gazemae_sample.features[name] for name in GAZEMAE_FEATURE_COLUMNS},
        }
        fused.append(
            TabularWindowSample(
                features=features,
                targets=dict(moment_sample.targets),
                subject=moment_sample.subject,
                recording=moment_sample.recording,
            )
        )
    return fused


def moment_frozen_parameter_count(moment_cfg: Mapping[str, Any] | None = None) -> int:
    """Return the expected frozen parameter count for MOMENT-base reporting."""
    raw_cfg = moment_cfg or {}
    if str(raw_cfg.get("model_name", MOMENT_DEFAULT_MODEL_NAME)) == MOMENT_DEFAULT_MODEL_NAME:
        return MOMENT_DEFAULT_PARAMETER_COUNT
    return int(raw_cfg.get("frozen_parameter_count", MOMENT_DEFAULT_PARAMETER_COUNT))
