"""Build frozen GazeMAE window embeddings for emotion-classification baselines.

This module adapts pretrained GazeMAE position and velocity encoders to the
MAHNOB-HCI windowed training pipeline. It keeps the backbone frozen and returns
one tabular sample per eye-tracking window so the existing cross-validation,
label mapping, and baseline-metric code can train only a downstream head.
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

from emotions.common.dataset_config import build_tabular_samples_kwargs
from emotions.gazemae_model import load_gazemae_encoder
from emotions.train_baseline import TabularWindowSample, build_tabular_samples


GAZEMAE_MODEL_NAME = "GazeMAE_MLP"
GAZEMAE_FEATURE_PREFIX = "gazemae_z_"
GAZEMAE_FEATURE_DIM = 512
GAZEMAE_FEATURE_COLUMNS = [f"{GAZEMAE_FEATURE_PREFIX}{idx:03d}" for idx in range(GAZEMAE_FEATURE_DIM)]
GAZEMAE_CACHE_VERSION = "gazemae-cache-v3"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAZEMAE_MODEL_DIR = PROJECT_ROOT / "models" / "gazemae"
DEFAULT_GAZEMAE_POS_MODEL = DEFAULT_GAZEMAE_MODEL_DIR / "pos-i3738-encoder-state.pt"
DEFAULT_GAZEMAE_VEL_MODEL = DEFAULT_GAZEMAE_MODEL_DIR / "vel-i8528-encoder-state.pt"


@dataclass(frozen=True)
class GazeMAEConfig:
    """Resolved GazeMAE embedding settings."""

    model_pos: Path = DEFAULT_GAZEMAE_POS_MODEL
    model_vel: Path = DEFAULT_GAZEMAE_VEL_MODEL
    screen_width: float = 1280.0
    screen_height: float = 800.0
    clip_to_screen: bool = True
    target_hz: int = 500
    chunk_seconds: float = 2.0
    chunk_pooling: str = "mean_std"
    batch_size: int = 256
    device: str = "auto"
    cache_embeddings: bool = True
    cache_dir: Path = Path("data/cache/gazemae_embeddings")


def _resolve_repo_path(raw_path: Any, default_path: Path) -> Path:
    """Resolve optional config paths relative to the current repository root."""
    if raw_path is None:
        return default_path
    path = Path(str(raw_path))
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_gazemae_config(raw_cfg: Mapping[str, Any] | None, dataset_cfg: Mapping[str, Any]) -> GazeMAEConfig:
    """Resolve config values with thesis-locked MAHNOB defaults."""
    raw_cfg = raw_cfg or {}
    cache_dir = raw_cfg.get("cache_dir")
    if cache_dir is None:
        cache_dir = Path(str(dataset_cfg.get("cache_dir", "data/cache"))) / "gazemae_embeddings"
    return GazeMAEConfig(
        model_pos=_resolve_repo_path(raw_cfg.get("model_pos"), DEFAULT_GAZEMAE_POS_MODEL),
        model_vel=_resolve_repo_path(raw_cfg.get("model_vel"), DEFAULT_GAZEMAE_VEL_MODEL),
        screen_width=float(raw_cfg.get("screen_width", 1280)),
        screen_height=float(raw_cfg.get("screen_height", 800)),
        clip_to_screen=bool(raw_cfg.get("clip_to_screen", True)),
        target_hz=int(raw_cfg.get("target_hz", 500)),
        chunk_seconds=float(raw_cfg.get("chunk_seconds", 2.0)),
        chunk_pooling=str(raw_cfg.get("chunk_pooling", "mean_std")),
        batch_size=int(raw_cfg.get("batch_size", 256)),
        device=str(raw_cfg.get("device", "auto")),
        cache_embeddings=bool(raw_cfg.get("cache_embeddings", True)),
        cache_dir=Path(str(cache_dir)),
    )


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _interp_1d(old_x: np.ndarray, old_y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    if len(old_x) < 2:
        return np.full_like(new_x, fill_value=float(old_y[0]), dtype=np.float32)
    return np.interp(new_x, old_x, old_y).astype(np.float32)


def _resample_xy_to_fixed_len(
    xy: np.ndarray,
    timestamps_seconds: np.ndarray,
    target_len: int,
    window_seconds: float,
) -> np.ndarray:
    if xy.ndim != 2 or xy.shape[0] != 2:
        raise ValueError(f"Expected XY shape [2, T], got {xy.shape}.")
    if target_len <= 0:
        raise ValueError(f"Invalid target_len={target_len}.")

    shifted = timestamps_seconds.astype(np.float64) - float(timestamps_seconds[0])
    unique_axis, unique_indices = np.unique(shifted, return_index=True)
    if len(unique_axis) >= 2 and unique_axis[-1] > 0:
        old_axis = unique_axis
        xy = xy[:, unique_indices]
        new_axis = np.linspace(0.0, float(window_seconds), num=target_len, endpoint=False, dtype=np.float64)
    else:
        old_axis = np.arange(xy.shape[1], dtype=np.float64)
        new_axis = np.linspace(0.0, float(xy.shape[1] - 1), num=target_len, dtype=np.float64)

    out = np.zeros((2, target_len), dtype=np.float32)
    out[0] = _interp_1d(old_axis, xy[0], new_axis)
    out[1] = _interp_1d(old_axis, xy[1], new_axis)
    return out


def _build_velocity_from_position(pos_signal: np.ndarray, target_hz: int) -> np.ndarray:
    ms_per_sample = 1000.0 / float(target_hz)
    velocity = np.abs(np.diff(pos_signal, axis=1)) / ms_per_sample
    velocity = np.pad(velocity, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
    return velocity.astype(np.float32)


def _signal_to_chunks(signal: np.ndarray, chunk_len: int) -> np.ndarray:
    full_chunks = int(signal.shape[1] // chunk_len)
    if full_chunks <= 0:
        raise ValueError(
            f"GazeMAE signal is shorter than one chunk: signal_len={signal.shape[1]}, chunk_len={chunk_len}."
        )
    trimmed = signal[:, : full_chunks * chunk_len]
    return trimmed.reshape(2, full_chunks, chunk_len).transpose(1, 0, 2).astype(np.float32)


class GazeMAEWindowEmbedder:
    """Frozen GazeMAE position/velocity encoder for one window at a time."""

    def __init__(self, config: GazeMAEConfig, window_seconds: float) -> None:
        if config.chunk_pooling != "mean_std":
            raise ValueError("Only gazemae.chunk_pooling='mean_std' is currently supported.")
        self.config = config
        self.window_seconds = float(window_seconds)
        self.device = _resolve_device(config.device)
        self.target_len = int(round(float(config.target_hz) * self.window_seconds))
        self.chunk_len = int(round(float(config.target_hz) * float(config.chunk_seconds)))
        self.network_pos = load_gazemae_encoder(config.model_pos, device=self.device)
        self.network_vel = load_gazemae_encoder(config.model_vel, device=self.device)

    def _prepare_chunks(self, window_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        required = ["time-rel-seconds", "x-avg", "y-avg"]
        missing = [column for column in required if column not in window_df.columns]
        if missing:
            raise ValueError(f"Missing required GazeMAE columns: {missing}")

        frame = window_df.loc[:, required].copy()
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=required).sort_values("time-rel-seconds").reset_index(drop=True)
        if frame.empty:
            raise ValueError("Cannot build GazeMAE embedding for an empty/NaN-only window.")

        if self.config.clip_to_screen:
            frame["x-avg"] = frame["x-avg"].clip(lower=0.0, upper=float(self.config.screen_width))
            frame["y-avg"] = frame["y-avg"].clip(lower=0.0, upper=float(self.config.screen_height))

        xy = frame.loc[:, ["x-avg", "y-avg"]].to_numpy(dtype=np.float32).T
        timestamps = frame["time-rel-seconds"].to_numpy(dtype=np.float64)
        pos_signal = _resample_xy_to_fixed_len(
            xy=xy,
            timestamps_seconds=timestamps,
            target_len=self.target_len,
            window_seconds=self.window_seconds,
        )
        vel_signal = _build_velocity_from_position(pos_signal, target_hz=self.config.target_hz)
        return _signal_to_chunks(pos_signal, self.chunk_len), _signal_to_chunks(vel_signal, self.chunk_len)

    def _encode_chunks(self, network: torch.nn.Module, chunks: np.ndarray) -> np.ndarray:
        outputs: List[np.ndarray] = []
        for start in range(0, len(chunks), max(1, int(self.config.batch_size))):
            batch_np = chunks[start : start + int(self.config.batch_size)]
            with torch.no_grad():
                batch = torch.tensor(batch_np, dtype=torch.float32, device=self.device)
                encoded = network.encode(batch)[0]
                outputs.append(encoded.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def embed_window(self, window_df: pd.DataFrame) -> np.ndarray:
        """Return pooled `[pos, vel] mean/std features for one window."""
        pos_chunks, vel_chunks = self._prepare_chunks(window_df)
        pos_embeddings = self._encode_chunks(self.network_pos, pos_chunks)
        vel_embeddings = self._encode_chunks(self.network_vel, vel_chunks)
        chunk_embeddings = np.concatenate([pos_embeddings, vel_embeddings], axis=1)
        pooled = np.concatenate(
            [
                np.mean(chunk_embeddings, axis=0),
                np.std(chunk_embeddings, axis=0),
            ],
            axis=0,
        )
        if pooled.shape[0] != GAZEMAE_FEATURE_DIM:
            raise ValueError(f"Expected {GAZEMAE_FEATURE_DIM} GazeMAE features, got {pooled.shape[0]}.")
        return pooled.astype(np.float32)

    def embed_window_as_features(self, window_df: pd.DataFrame) -> Dict[str, float]:
        embedding = self.embed_window(window_df)
        return {name: float(value) for name, value in zip(GAZEMAE_FEATURE_COLUMNS, embedding)}


def _cache_key(
    *,
    config: GazeMAEConfig,
    dataset_cfg: Mapping[str, Any],
    target_columns: List[str],
    feature_columns: List[str],
    dropna_columns: List[str],
    min_samples_per_window: int,
) -> str:
    data_identity = _resolve_dataset_identity(dataset_cfg)

    payload = {
        "version": GAZEMAE_CACHE_VERSION,
        "dataset_identity": data_identity,
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
        "gazemae": {
            "model_pos": _model_file_identity(config.model_pos),
            "model_vel": _model_file_identity(config.model_vel),
            "screen_width": config.screen_width,
            "screen_height": config.screen_height,
            "clip_to_screen": config.clip_to_screen,
            "target_hz": config.target_hz,
            "chunk_seconds": config.chunk_seconds,
            "chunk_pooling": config.chunk_pooling,
        },
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_file_identity(path: Path) -> Dict[str, Any]:
    if path.exists():
        stat = path.stat()
        return {
            "path_name": path.name,
            "size_bytes": int(stat.st_size),
            "sha256": _compute_file_sha256(path),
        }
    return {"path": str(path), "missing": True}


def _compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
    """Return stable data identity for GazeMAE cache reuse across suite runs."""
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


def build_gazemae_tabular_samples(
    *,
    dataset_cfg: Mapping[str, Any],
    target_columns: List[str],
    feature_columns: List[str],
    dropna_columns: List[str],
    min_samples_per_window: int,
    gazemae_cfg: Mapping[str, Any] | None,
) -> List[TabularWindowSample]:
    """Build TabularWindowSample objects whose features are frozen GazeMAE embeddings."""
    resolved = resolve_gazemae_config(gazemae_cfg, dataset_cfg=dataset_cfg)
    key = _cache_key(
        config=resolved,
        dataset_cfg=dataset_cfg,
        target_columns=target_columns,
        feature_columns=feature_columns,
        dropna_columns=dropna_columns,
        min_samples_per_window=min_samples_per_window,
    )
    cache_path = resolved.cache_dir / f"{key}.joblib"
    if resolved.cache_embeddings and cache_path.exists():
        return joblib.load(cache_path)

    embedder = GazeMAEWindowEmbedder(
        config=resolved,
        window_seconds=float(dataset_cfg.get("window_length", 10)),
    )
    samples = build_tabular_samples(
        **build_tabular_samples_kwargs(
            dataset_cfg=dataset_cfg,
            target_columns=target_columns,
            feature_columns=feature_columns,
            dropna_columns=dropna_columns,
            min_samples_per_window=min_samples_per_window,
        ),
        window_feature_builder=embedder.embed_window_as_features,
    )
    if resolved.cache_embeddings:
        resolved.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(samples, cache_path)
    return samples
