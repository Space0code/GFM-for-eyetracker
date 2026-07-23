"""Measure full MOMENT and MOMENT+GazeMAE inference for thesis Table 8.2.

The benchmark loads the frozen encoders and the saved MLP heads from every
subject-k-fold split of the retained quick comparison. It measures the model
forward pass from already prepared tensors to class probabilities, matching
the scope of the existing GazeMAE encoder+pooling+head benchmark.

Run from the repository root with the ``gfm`` environment active:

    python scripts/benchmark_moment_full_inference.py

Use ``--help`` to override the retained-run path, timing repetitions, device,
model paths, or output path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotions.common.model_benchmarking import measure_repeated_wall_time
from emotions.gazemae_baseline import (
    DEFAULT_GAZEMAE_POS_MODEL,
    DEFAULT_GAZEMAE_VEL_MODEL,
)
from emotions.gazemae_model import load_gazemae_encoder
from emotions.moment_baseline import MomentConfig, MomentWindowEmbedder
from emotions.multiclass.baseline_model_multiclass import _TorchMLPHead


DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "quick_v1_v2_comparison"
    / "RETAIN_2026-06-12_16-29-08"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure encoder+pooling+MLP inference for MOMENT_pupil and "
            "MOMENT_GazeMAE_gaze_pupil using saved fold heads."
        )
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--windows-per-run", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--moment-model", default="AutonLab/MOMENT-1-base")
    parser.add_argument("--moment-sequence-length", type=int, default=512)
    parser.add_argument("--moment-local-files-only", action="store_true")
    parser.add_argument("--gazemae-pos-model", type=Path, default=DEFAULT_GAZEMAE_POS_MODEL)
    parser.add_argument("--gazemae-vel-model", type=Path, default=DEFAULT_GAZEMAE_VEL_MODEL)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--gazemae-target-hz", type=int, default=500)
    parser.add_argument("--gazemae-chunk-seconds", type=float, default=2.0)
    parser.add_argument(
        "--only-gazemae",
        action="store_true",
        help="Recheck only GazeMAE encoder+pooling+head inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <run-root>/tables/missing_full_inference_benchmark.json.",
    )
    return parser.parse_args()


def _resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def _load_head(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    # These are trusted checkpoints created by this repository. ``weights_only``
    # cannot load the stored NumPy class-index array with current PyTorch.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    head = _TorchMLPHead(
        in_features=int(state_dict["net.0.weight"].shape[1]),
        hidden_size=int(checkpoint["hidden_layer_size"]),
        n_classes=int(state_dict["net.6.weight"].shape[0]),
        dropout=float(checkpoint["dropout"]),
    )
    head.load_state_dict(state_dict)
    head.to(device)
    head.eval()
    return head


def _find_heads(run_root: Path, signal_set: str, model_name: str) -> list[Path]:
    paths = sorted(
        (run_root / "model_runs" / signal_set).glob(
            f"**/subject_kfold/*/baselines/{model_name}/model.pt"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"No saved heads found for {model_name} under "
            f"{run_root / 'model_runs' / signal_set}."
        )
    return paths


def _time_heads(
    *,
    model_name: str,
    head_paths: list[Path],
    build_run_once: Callable[[torch.nn.Module], Callable[[], None]],
    n_windows: int,
    warmup_runs: int,
    timed_runs: int,
    device: torch.device,
) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    for head_path in head_paths:
        head = _load_head(head_path, device=device)
        timing = measure_repeated_wall_time(
            build_run_once(head),
            n_items=n_windows,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            device=device,
        )
        fold_result = {
            "fold": head_path.parents[2].name,
            "head_path": str(head_path.relative_to(PROJECT_ROOT)),
            **timing,
        }
        fold_results.append(fold_result)
        print(
            f"{model_name} {fold_result['fold']}: "
            f"{timing['inference_ms_per_window']:.6f} ms/window"
        )

    total_seconds = sum(float(row["inference_total_seconds"]) for row in fold_results)
    total_predictions = sum(
        int(row["inference_total_item_predictions"]) for row in fold_results
    )
    aggregate_ms = 1000.0 * total_seconds / max(1, total_predictions)
    print(f"{model_name} aggregate: {aggregate_ms:.6f} ms/window")
    return {
        "model": model_name,
        "inference_scope": "prepared_tensor_encoder_pooling_head_softmax",
        "fold_count": len(fold_results),
        "inference_total_seconds": total_seconds,
        "inference_total_window_predictions": total_predictions,
        "inference_ms_per_window": aggregate_ms,
        "fold_results": fold_results,
    }


def _prepare_gazemae_benchmark(
    *,
    pos_model_path: Path,
    vel_model_path: Path,
    device: torch.device,
    n_windows: int,
    window_seconds: float,
    target_hz: int,
    chunk_seconds: float,
) -> dict[str, Any]:
    """Load GazeMAE encoders and create the prepared synthetic model inputs."""
    pos_encoder = load_gazemae_encoder(pos_model_path.resolve(), device=device)
    vel_encoder = load_gazemae_encoder(vel_model_path.resolve(), device=device)
    chunk_len = int(round(float(target_hz) * float(chunk_seconds)))
    chunks_per_window = max(
        1,
        int(round(float(target_hz) * float(window_seconds)) // max(1, chunk_len)),
    )
    n_chunks = n_windows * chunks_per_window
    pos_chunks = torch.zeros(
        (n_chunks, 2, chunk_len), dtype=torch.float32, device=device
    )
    return {
        "pos_encoder": pos_encoder,
        "vel_encoder": vel_encoder,
        "pos_chunks": pos_chunks,
        "vel_chunks": torch.zeros_like(pos_chunks),
        "chunks_per_window": chunks_per_window,
    }


def _encode_gazemae(
    components: dict[str, Any], *, n_windows: int
) -> torch.Tensor:
    """Encode and pool prepared GazeMAE position and velocity chunks."""
    pos_embeddings = components["pos_encoder"].encode(components["pos_chunks"])[0]
    vel_embeddings = components["vel_encoder"].encode(components["vel_chunks"])[0]
    chunk_embeddings = torch.cat([pos_embeddings, vel_embeddings], dim=1).reshape(
        n_windows, int(components["chunks_per_window"]), -1
    )
    return torch.cat(
        [
            chunk_embeddings.mean(dim=1),
            chunk_embeddings.std(dim=1, unbiased=False),
        ],
        dim=1,
    )


def _save_results(
    *,
    output_path: Path,
    resolved_args: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Write benchmark arguments and results to one JSON artifact."""
    payload = {"arguments": resolved_args, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved benchmark: {output_path}")


def main() -> None:
    args = _parse_args()
    if args.windows_per_run < 1:
        raise ValueError("--windows-per-run must be at least 1.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.timed_runs < 1:
        raise ValueError("--timed-runs must be at least 1.")

    run_root = args.run_root.resolve()
    default_output_name = (
        "gazemae_full_inference_recheck.json"
        if args.only_gazemae
        else "missing_full_inference_benchmark.json"
    )
    output_path = args.output.resolve() if args.output is not None else run_root / "tables" / default_output_name
    device = _resolve_device(args.device)
    n_windows = int(args.windows_per_run)

    resolved_args = {
        "run_root": str(run_root),
        "output": str(output_path),
        "device": str(device),
        "windows_per_run": n_windows,
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "moment_model": args.moment_model,
        "moment_sequence_length": int(args.moment_sequence_length),
        "moment_local_files_only": bool(args.moment_local_files_only),
        "gazemae_pos_model": str(args.gazemae_pos_model.resolve()),
        "gazemae_vel_model": str(args.gazemae_vel_model.resolve()),
        "window_seconds": float(args.window_seconds),
        "gazemae_target_hz": int(args.gazemae_target_hz),
        "gazemae_chunk_seconds": float(args.gazemae_chunk_seconds),
        "only_gazemae": bool(args.only_gazemae),
    }
    print(json.dumps(resolved_args, indent=2))

    if args.only_gazemae:
        gazemae = _prepare_gazemae_benchmark(
            pos_model_path=args.gazemae_pos_model,
            vel_model_path=args.gazemae_vel_model,
            device=device,
            n_windows=n_windows,
            window_seconds=float(args.window_seconds),
            target_hz=int(args.gazemae_target_hz),
            chunk_seconds=float(args.gazemae_chunk_seconds),
        )

        def build_gazemae_run(head: torch.nn.Module) -> Callable[[], None]:
            def run_once() -> None:
                with torch.inference_mode():
                    embeddings = _encode_gazemae(gazemae, n_windows=n_windows)
                    probabilities = torch.softmax(head(embeddings), dim=1)
                    _ = probabilities

            return run_once

        gazemae_result = _time_heads(
            model_name="GazeMAE_MLP",
            head_paths=_find_heads(run_root, "gaze_only", "GazeMAE_MLP"),
            build_run_once=build_gazemae_run,
            n_windows=n_windows,
            warmup_runs=int(args.warmup_runs),
            timed_runs=int(args.timed_runs),
            device=device,
        )
        _save_results(
            output_path=output_path,
            resolved_args=resolved_args,
            results=[gazemae_result],
        )
        return

    moment_config = MomentConfig(
        model_name=str(args.moment_model),
        sequence_length=int(args.moment_sequence_length),
        batch_size=n_windows,
        device=str(device),
        cache_embeddings=False,
        local_files_only=bool(args.moment_local_files_only),
    )
    moment_pupil = MomentWindowEmbedder(
        config=moment_config,
        signal_subset="pupil",
        window_seconds=float(args.window_seconds),
    ).model
    moment_pupil.eval()
    pupil_input = torch.zeros(
        (n_windows, 2, int(args.moment_sequence_length)),
        dtype=torch.float32,
        device=device,
    )

    def build_moment_run(head: torch.nn.Module) -> Callable[[], None]:
        def run_once() -> None:
            with torch.inference_mode():
                embeddings = moment_pupil(x_enc=pupil_input).embeddings
                probabilities = torch.softmax(head(embeddings), dim=1)
                _ = probabilities

        return run_once

    moment_result = _time_heads(
        model_name="MOMENT_pupil",
        head_paths=_find_heads(run_root, "pupil_only", "MOMENT_pupil"),
        build_run_once=build_moment_run,
        n_windows=n_windows,
        warmup_runs=int(args.warmup_runs),
        timed_runs=int(args.timed_runs),
        device=device,
    )

    moment_fusion = moment_pupil
    fusion_input = torch.zeros(
        (n_windows, 4, int(args.moment_sequence_length)),
        dtype=torch.float32,
        device=device,
    )
    gazemae = _prepare_gazemae_benchmark(
        pos_model_path=args.gazemae_pos_model,
        vel_model_path=args.gazemae_vel_model,
        device=device,
        n_windows=n_windows,
        window_seconds=float(args.window_seconds),
        target_hz=int(args.gazemae_target_hz),
        chunk_seconds=float(args.gazemae_chunk_seconds),
    )

    def build_fusion_run(head: torch.nn.Module) -> Callable[[], None]:
        def run_once() -> None:
            with torch.inference_mode():
                moment_embeddings = moment_fusion(x_enc=fusion_input).embeddings
                gazemae_embeddings = _encode_gazemae(
                    gazemae, n_windows=n_windows
                )
                fused_embeddings = torch.cat(
                    [moment_embeddings, gazemae_embeddings], dim=1
                )
                probabilities = torch.softmax(head(fused_embeddings), dim=1)
                _ = probabilities

        return run_once

    fusion_result = _time_heads(
        model_name="MOMENT_GazeMAE_gaze_pupil",
        head_paths=_find_heads(
            run_root, "gaze_pupil", "MOMENT_GazeMAE_gaze_pupil"
        ),
        build_run_once=build_fusion_run,
        n_windows=n_windows,
        warmup_runs=int(args.warmup_runs),
        timed_runs=int(args.timed_runs),
        device=device,
    )

    _save_results(
        output_path=output_path,
        resolved_args=resolved_args,
        results=[moment_result, fusion_result],
    )


if __name__ == "__main__":
    main()
