"""Create readable thesis figures of small MAHNOB-HCI graph windows.

The script samples real MAHNOB-HCI eye-tracking measurements, constructs the
same relation families used by the proposed GNN, renders 50 candidate graph
figures, keeps the clearest 3 "far" examples and 3 "close" examples, and deletes
the temporary candidate figures.

Example:
    conda run -n gfm python scripts/create_thesis_gnn_graph_examples.py

Useful options:
    python scripts/create_thesis_gnn_graph_examples.py \
        --num-candidates 50 \
        --output-dir ../diploma-latex/slike/konstrukcija_grafa \
        --formats png pdf
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import KDTree, distance_matrix


plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

PALETTE = {
    "dark": "#2F3437",
    "blue": "#8ECAE6",
    "teal": "#A8DADC",
    "purple": "#B8A1D9",
    "orange": "#F4A261",
    "pink": "#E9A6A6",
    "background": "#F7F7F2",
}

RELATION_STYLES = {
    "temporal_forward": {
        "label": "časovne naprej",
        "color": PALETTE["orange"],
        "rad": 0.18,
        "alpha": 0.80,
        "linewidth": 1.8,
    },
    "temporal_backward": {
        "label": "časovne nazaj",
        "color": PALETTE["pink"],
        "rad": 0.18,
        "alpha": 0.72,
        "linewidth": 1.8,
    },
    "spatial": {
        "label": "prostorske kNN",
        "color": PALETTE["blue"],
        "rad": 0.25,
        "alpha": 0.70,
        "linewidth": 1.6,
    },
    "fixation": {
        "label": "znotraj fiksacije",
        "color": PALETTE["purple"],
        "rad": -0.28,
        "alpha": 0.68,
        "linewidth": 1.8,
    },
}

NODE_FILL_COLOR = "#CCAAE0"
NODE_HIGHLIGHT_COLOR = "#C298DA"

FIGURE_TITLE = "Majhen primer konstrukcije grafa iz podatkov sledilnika pogleda"
PANEL_FILE_STEMS = {
    "all": "vse_povezave",
    "all_plain": "vse_povezave_brez_oznak",
    "temporal": "casovne",
    "spatial": "prostorske",
    "fixation": "fiksacijske",
}

REQUIRED_COLUMNS = [
    "time-rel-seconds",
    "x-avg",
    "y-avg",
    "pupil-size-left-avg",
    "pupil-size-right-avg",
    "distance-left",
    "distance-right",
    "distance-avg",
    "fixation-index",
    "fixation-duration",
    "fixation",
    "subject",
    "recording",
    "is-stimulus",
]


@dataclass(frozen=True)
class CandidateGraph:
    """Small rendered graph candidate with metadata used for ranking."""

    candidate_id: int
    kind: str
    source_path: Path
    subject: str
    recording: str
    nodes: pd.DataFrame
    edges: dict[str, list[tuple[int, int]]]
    score: float
    fixation_count: int
    spread: float
    min_pairwise_distance: float
    fixation_cluster_quality: float
    total_edges: int
    kt: int
    ks: int
    fixation_dilation_k: int
    temporary_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/hci-tagging/emotion-elicitation"),
        help="Directory with processed MAHNOB-HCI emotion-elicitation CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../diploma-latex/slike/konstrukcija_grafa"),
        help="Directory where the selected graph figures are saved.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help="Number of random graph figures to generate before selection.",
    )
    parser.add_argument("--keep-per-kind", type=int, default=3, help="Selected figures per view type.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--kt", type=int, default=1, help="Temporal neighborhood size.")
    parser.add_argument("--ks", type=int, default=1, help="Spatial kNN neighborhood size.")
    parser.add_argument("--screen-width", type=float, default=1280.0, help="Maximum valid gaze x-coordinate.")
    parser.add_argument("--screen-height", type=float, default=800.0, help="Maximum valid gaze y-coordinate.")
    parser.add_argument(
        "--fixation-dilation-k",
        type=int,
        default=3,
        help="Dilated intra-fixation offsets per displayed fixation run.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["svg", "png", "pdf"],
        help="Output formats for selected figures.",
    )
    parser.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=["P9", "P12", "P15"],
        help="Subjects excluded from sampling.",
    )
    return parser.parse_args()


def _log_args(args: argparse.Namespace) -> None:
    print("Final arguments:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")


def _read_candidate_file(
    path: Path,
    exclude_subjects: set[str],
    screen_width: float,
    screen_height: float,
) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in REQUIRED_COLUMNS if column in available]
    df = pd.read_csv(path, usecols=usecols)
    if "distance-avg" not in df.columns and {"distance-left", "distance-right"}.issubset(df.columns):
        df["distance-avg"] = df[["distance-left", "distance-right"]].mean(axis=1, skipna=True)

    if "subject" in df.columns:
        df = df[~df["subject"].isin(exclude_subjects)]
    if "is-stimulus" in df.columns:
        df = df[df["is-stimulus"].astype("boolean").fillna(False)]

    numeric_columns = [
        "time-rel-seconds",
        "x-avg",
        "y-avg",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
        "distance-avg",
        "fixation-index",
        "fixation-duration",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    feature_columns = [
        "time-rel-seconds",
        "x-avg",
        "y-avg",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
        "distance-avg",
        "fixation-index",
        "fixation-duration",
    ]
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        return pd.DataFrame()
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    df = df[
        df["x-avg"].between(0.0, screen_width)
        & df["y-avg"].between(0.0, screen_height)
    ].reset_index(drop=True)
    if len(df) < 4:
        return pd.DataFrame()
    if "fixation" in df.columns:
        fixation_mask = df["fixation"].astype("boolean").fillna(False)
        df = df[fixation_mask].reset_index(drop=True)
    return df


def _find_fixation_runs(df: pd.DataFrame) -> list[tuple[int, int, float]]:
    runs: list[tuple[int, int, float]] = []
    if df.empty:
        return runs
    start = 0
    current_id = float(df.loc[0, "fixation-index"])
    for idx in range(1, len(df)):
        next_id = float(df.loc[idx, "fixation-index"])
        if not math.isclose(next_id, current_id):
            if idx - start >= 2:
                runs.append((start, idx, current_id))
            start = idx
            current_id = next_id
    if len(df) - start >= 2:
        runs.append((start, len(df), current_id))
    return runs


def _evenly_spaced_indices(start: int, stop: int, count: int) -> list[int]:
    if stop <= start:
        return []
    count = min(count, stop - start)
    return sorted({int(round(value)) for value in np.linspace(start, stop - 1, count)})


def _sample_nodes_from_runs(
    df: pd.DataFrame,
    runs: list[tuple[int, int, float]],
    target_fixation_count: int,
    rng: np.random.Generator,
) -> pd.DataFrame | None:
    if len(runs) < target_fixation_count:
        return None
    run_start_idx = int(rng.integers(0, len(runs) - target_fixation_count + 1))
    chosen_runs = runs[run_start_idx : run_start_idx + target_fixation_count]
    samples_per_run = 5 if target_fixation_count == 1 else 4 if target_fixation_count == 2 else 3
    selected: list[int] = []
    for start, stop, _ in chosen_runs:
        selected.extend(_evenly_spaced_indices(start, stop, samples_per_run))
    selected = sorted(dict.fromkeys(selected))
    if not 4 <= len(selected) <= 14:
        return None
    nodes = df.loc[selected].copy().reset_index(drop=True)
    t0 = float(nodes["time-rel-seconds"].iloc[0])
    t1 = float(nodes["time-rel-seconds"].iloc[-1])
    duration = max(t1 - t0, 1e-9)
    nodes["time-window-normalized"] = (nodes["time-rel-seconds"] - t0) / duration
    nodes["local_index"] = np.arange(len(nodes))
    return nodes


def _dedupe_edges(edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(set((int(src), int(dst)) for src, dst in edges if int(src) != int(dst)))


def _build_temporal_edges(num_nodes: int, kt: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    forward: list[tuple[int, int]] = []
    backward: list[tuple[int, int]] = []
    for src in range(num_nodes):
        for offset in range(1, kt + 1):
            dst = src + offset
            if dst < num_nodes:
                forward.append((src, dst))
                backward.append((dst, src))
    return _dedupe_edges(forward), _dedupe_edges(backward)


def _build_spatial_edges(nodes: pd.DataFrame, ks: int) -> list[tuple[int, int]]:
    xy = nodes[["x-avg", "y-avg"]].to_numpy(dtype=float)
    if len(xy) <= 1:
        return []
    tree = KDTree(xy)
    _, indices = tree.query(xy, k=min(ks + 1, len(xy)))
    if indices.ndim == 1:
        indices = indices[:, None]
    edges: list[tuple[int, int]] = []
    for src, neighbors in enumerate(indices):
        for dst in neighbors:
            if int(dst) != src:
                edges.append((src, int(dst)))
                edges.append((int(dst), src))
    return _dedupe_edges(edges)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _fixation_offsets(run_length: int, dilation_k: int) -> tuple[int, ...]:
    if run_length <= 1:
        return ()
    step = max(1, _round_half_up(run_length / float(dilation_k)))
    offsets = []
    for q in range(dilation_k):
        offset = (1 + q * step) % run_length
        if offset != 0:
            offsets.append(offset)
    return tuple(dict.fromkeys(offsets))


def _build_fixation_edges(nodes: pd.DataFrame, dilation_k: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    fixation_ids = nodes["fixation-index"].to_numpy(dtype=float)
    start = 0
    while start < len(fixation_ids):
        stop = start + 1
        while stop < len(fixation_ids) and math.isclose(fixation_ids[stop], fixation_ids[start]):
            stop += 1
        run_length = stop - start
        for offset in _fixation_offsets(run_length, dilation_k):
            for local_idx in range(run_length):
                src = start + local_idx
                dst = start + ((local_idx + offset) % run_length)
                edges.append((src, dst))
                edges.append((dst, src))
        start = stop
    return _dedupe_edges(edges)


def _fixation_cluster_quality(nodes: pd.DataFrame) -> float:
    xy = nodes[["x-avg", "y-avg"]].to_numpy(dtype=float)
    fixation_ids = nodes["fixation-index"].to_numpy(dtype=float)
    centroids: list[np.ndarray] = []
    radii: list[float] = []
    weights: list[int] = []

    for fixation_id in dict.fromkeys(fixation_ids.tolist()):
        cluster_xy = xy[np.isclose(fixation_ids, fixation_id)]
        if len(cluster_xy) == 0:
            continue
        centroid = np.mean(cluster_xy, axis=0)
        centroids.append(centroid)
        weights.append(len(cluster_xy))
        if len(cluster_xy) == 1:
            radii.append(0.0)
        else:
            radii.append(float(np.mean(np.linalg.norm(cluster_xy - centroid, axis=1))))

    if not centroids:
        return 0.0

    mean_radius = float(np.average(radii, weights=weights))
    compactness = 1.0 / (1.0 + mean_radius / 18.0)
    if len(centroids) == 1:
        return compactness

    centroid_xy = np.vstack(centroids)
    centroid_distances = distance_matrix(centroid_xy, centroid_xy)
    centroid_distances[centroid_distances == 0.0] = np.inf
    min_centroid_distance = float(np.min(centroid_distances))
    max_radius = max(float(np.max(radii)), 1.0)
    separation = min_centroid_distance / (min_centroid_distance + 2.0 * max_radius)
    return 0.55 * compactness + 0.45 * separation


def _score_candidate(kind: str, nodes: pd.DataFrame, edges: dict[str, list[tuple[int, int]]]) -> tuple[float, float, float, float, int]:
    xy = nodes[["x-avg", "y-avg"]].to_numpy(dtype=float)
    spread = float(np.linalg.norm(np.ptp(xy, axis=0)))
    if len(xy) > 1:
        distances = distance_matrix(xy, xy)
        distances[distances == 0.0] = np.inf
        min_pairwise = float(np.min(distances))
    else:
        min_pairwise = 0.0
    total_edges = sum(len(relation_edges) for relation_edges in edges.values())
    relation_bonus = sum(1.0 for relation_edges in edges.values() if relation_edges)
    edge_penalty = max(0.0, (total_edges - 86) / 20.0)
    node_penalty = abs(len(nodes) - (10 if kind == "far" else 7)) * 0.16
    if kind == "far":
        spread_term = min(spread / 170.0, 1.8)
        distance_term = min(min_pairwise / 22.0, 1.2)
    else:
        spread_term = 1.3 - min(abs(spread - 60.0) / 80.0, 1.3)
        distance_term = min(min_pairwise / 18.0, 1.0)
    cluster_quality = _fixation_cluster_quality(nodes)
    cluster_bonus = 0.8 * cluster_quality
    score = relation_bonus + spread_term + distance_term + cluster_bonus - edge_penalty - node_penalty
    return score, spread, min_pairwise, cluster_quality, total_edges


def _build_candidate(
    candidate_id: int,
    kind: str,
    csv_files: list[Path],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> CandidateGraph | None:
    target_fixation_count = int(rng.integers(3, 5)) if kind == "far" else int(rng.integers(1, 3))
    path = csv_files[int(rng.integers(0, len(csv_files)))]
    df = _read_candidate_file(path, set(args.exclude_subjects), args.screen_width, args.screen_height)
    if df.empty:
        return None
    runs = _find_fixation_runs(df)
    nodes = _sample_nodes_from_runs(df, runs, target_fixation_count, rng)
    if nodes is None:
        return None
    temporal_forward, temporal_backward = _build_temporal_edges(len(nodes), args.kt)
    edges = {
        "temporal_forward": temporal_forward,
        "temporal_backward": temporal_backward,
        "spatial": _build_spatial_edges(nodes, args.ks),
        "fixation": _build_fixation_edges(nodes, args.fixation_dilation_k),
    }
    if any(not relation_edges for relation_edges in edges.values()):
        return None
    score, spread, min_pairwise, cluster_quality, total_edges = _score_candidate(kind, nodes, edges)
    subject = str(nodes["subject"].iloc[0]) if "subject" in nodes.columns else "neznan subjekt"
    recording = str(nodes["recording"].iloc[0]) if "recording" in nodes.columns else path.stem
    return CandidateGraph(
        candidate_id=candidate_id,
        kind=kind,
        source_path=path,
        subject=subject,
        recording=recording,
        nodes=nodes,
        edges=edges,
        score=score,
        fixation_count=target_fixation_count,
        spread=spread,
        min_pairwise_distance=min_pairwise,
        fixation_cluster_quality=cluster_quality,
        total_edges=total_edges,
        kt=args.kt,
        ks=args.ks,
        fixation_dilation_k=args.fixation_dilation_k,
    )


def _edge_label(relation: str) -> str:
    return RELATION_STYLES[relation]["label"]


def _draw_edges(
    ax: Axes,
    nodes: pd.DataFrame,
    edges: list[tuple[int, int]],
    relation: str,
    linewidth_scale: float = 1.0,
    arrow_scale: float = 1.0,
) -> None:
    style = RELATION_STYLES[relation]
    xy = nodes[["x-avg", "y-avg"]].to_numpy(dtype=float)
    for src, dst in edges:
        patch = FancyArrowPatch(
            xy[src],
            xy[dst],
            arrowstyle="-|>",
            mutation_scale=8 * arrow_scale,
            color=style["color"],
            alpha=style["alpha"],
            linewidth=style["linewidth"] * linewidth_scale,
            shrinkA=6 * arrow_scale,
            shrinkB=6 * arrow_scale,
            connectionstyle=f"arc3,rad={style['rad']}",
            zorder=1,
        )
        ax.add_patch(patch)


def _node_color_map(nodes: pd.DataFrame) -> dict[float, str]:
    ids = list(dict.fromkeys(float(value) for value in nodes["fixation-index"].tolist()))
    return {fixation_id: NODE_FILL_COLOR for fixation_id in ids}


def _draw_nodes(
    ax: Axes,
    nodes: pd.DataFrame,
    detailed_labels: bool,
    node_size: float = 165.0,
    detailed_node_size: float = 118.0,
    node_fontsize: float = 10.2,
    detailed_node_fontsize: float = 9.2,
    detail_label_fontsize: float = 8.2,
    label_box_pad: float = 0.18,
    label_vertical_span_factor: float = 0.58,
    label_side_offset_factor: float = 0.56,
) -> None:
    colors = _node_color_map(nodes)
    x_values = nodes["x-avg"].to_numpy(dtype=float)
    y_values = nodes["y-avg"].to_numpy(dtype=float)
    x_span = max(float(np.ptp(x_values)), 45.0)
    y_span = max(float(np.ptp(y_values)), 45.0)
    label_positions: dict[int, tuple[float, float]] = {}
    if detailed_labels:
        midpoint = float(np.median(x_values))
        left_indices = [int(row["local_index"]) for _, row in nodes.iterrows() if float(row["x-avg"]) <= midpoint]
        right_indices = [int(row["local_index"]) for _, row in nodes.iterrows() if float(row["x-avg"]) > midpoint]
        if not left_indices or not right_indices:
            ordered = [int(idx) for idx in np.argsort(x_values)]
            split = max(1, len(ordered) // 2)
            left_indices = ordered[:split]
            right_indices = ordered[split:]

        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        label_y_min = y_min - label_vertical_span_factor * y_span
        label_y_max = y_max + label_vertical_span_factor * y_span
        label_x_left = float(np.min(x_values)) - label_side_offset_factor * x_span
        label_x_right = float(np.max(x_values)) + label_side_offset_factor * x_span
        node_y_by_idx = {
            int(row["local_index"]): float(row["y-avg"])
            for _, row in nodes.iterrows()
        }
        for side_indices, label_x in [(left_indices, label_x_left), (right_indices, label_x_right)]:
            ordered_side = sorted(side_indices, key=lambda idx: node_y_by_idx[idx])
            slots = np.linspace(label_y_min, label_y_max, len(ordered_side))
            for idx, label_y in zip(ordered_side, slots):
                label_positions[idx] = (label_x, float(label_y))
    detail_index = int(nodes["local_index"].iloc[len(nodes) // 2]) if detailed_labels else -1
    for _, row in nodes.iterrows():
        idx = int(row["local_index"])
        fixation_id = float(row["fixation-index"])
        x = float(row["x-avg"])
        y = float(row["y-avg"])
        if detailed_labels:
            label = (
                f"x={x:.0f}, y={y:.0f}\n"
                f"zen={row['pupil-size-left-avg']:.2f}/{row['pupil-size-right-avg']:.2f}, "
                f"čas={row['time-window-normalized']:.2f}\n"
                f"d={row['distance-avg'] / 10.0:.1f} cm, fiks={row['fixation-duration']:.0f} ms"
            )
            ax.scatter(
                [x],
                [y],
                s=detailed_node_size,
                color=NODE_HIGHLIGHT_COLOR if idx == detail_index else colors[fixation_id],
                edgecolor=PALETTE["dark"],
                linewidth=0.9,
                zorder=4,
            )
            ax.text(
                x,
                y,
                str(idx),
                ha="center",
                va="center",
                fontsize=detailed_node_fontsize,
                color=PALETTE["dark"],
                zorder=6,
            )
            if idx == detail_index:
                label_x = float(np.min(x_values)) + 0.15 * x_span
                label_y = float(np.min(y_values)) - 0.20 * y_span
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(label_x, label_y),
                    ha="center",
                    va="center",
                    fontsize=detail_label_fontsize,
                    color=PALETTE["dark"],
                    zorder=5,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": PALETTE["dark"],
                        "alpha": 0.55,
                        "linewidth": 0.85,
                        "shrinkA": 5,
                        "shrinkB": 5,
                    },
                    bbox={
                        "boxstyle": f"round,pad={label_box_pad},rounding_size=0.08",
                        "facecolor": "#F7F2FF",
                        "edgecolor": PALETTE["dark"],
                        "linewidth": 0.75,
                        "alpha": 1.0,
                    },
                )
        else:
            ax.scatter(
                [x],
                [y],
                s=node_size,
                color=colors[fixation_id],
                edgecolor=PALETTE["dark"],
                linewidth=0.9,
                zorder=3,
            )
            ax.text(x, y, str(idx), ha="center", va="center", fontsize=node_fontsize, color=PALETTE["dark"], zorder=4)


def _style_axis(
    ax: Axes,
    nodes: pd.DataFrame,
    title: str,
    padding_factor: float = 0.32,
    title_fontsize: float = 14.0,
    axis_label_fontsize: float = 12.0,
    tick_fontsize: float = 10.0,
    title_pad: float = 11.0,
) -> None:
    x_values = nodes["x-avg"].to_numpy(dtype=float)
    y_values = nodes["y-avg"].to_numpy(dtype=float)
    x_span = max(float(np.ptp(x_values)), 45.0)
    y_span = max(float(np.ptp(y_values)), 45.0)
    ax.set_xlim(float(np.min(x_values)) - padding_factor * x_span, float(np.max(x_values)) + padding_factor * x_span)
    ax.set_ylim(float(np.max(y_values)) + padding_factor * y_span, float(np.min(y_values)) - padding_factor * y_span)
    ax.set_title(title, fontsize=title_fontsize, color=PALETTE["dark"], pad=title_pad)
    ax.set_xlabel("x-koordinata pogleda", fontsize=axis_label_fontsize, color=PALETTE["dark"])
    ax.set_ylabel("y-koordinata pogleda", fontsize=axis_label_fontsize, color=PALETTE["dark"])
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors=PALETTE["dark"])
    ax.grid(False)
    ax.set_facecolor(PALETTE["background"])
    for spine in ax.spines.values():
        spine.set_color("#CFD8DC")
        spine.set_linewidth(0.7)


def _panel_title(candidate: CandidateGraph, panel: str) -> str:
    if panel == "all":
        return "Primer časovno-prostorskega grafa"
    if panel == "all_plain":
        return "vse povezave"
    if panel == "temporal":
        return f"Časovne povezave v izseku grafa, $k_t={candidate.kt}$"
    if panel == "spatial":
        return f"Prostorske povezave v izseku grafa, $k_s={candidate.ks}$"
    if panel == "fixation":
        return f"Fiksacijske povezave v izseku grafa, $k_f={candidate.fixation_dilation_k}$"
    raise ValueError(f"Unknown panel: {panel}")


def _panel_relations(panel: str) -> list[str]:
    if panel in {"all", "all_plain"}:
        return ["temporal_forward", "temporal_backward", "spatial", "fixation"]
    if panel == "temporal":
        return ["temporal_forward", "temporal_backward"]
    if panel == "spatial":
        return ["spatial"]
    if panel == "fixation":
        return ["fixation"]
    raise ValueError(f"Unknown panel: {panel}")


def _draw_panel(
    ax: Axes,
    candidate: CandidateGraph,
    panel: str,
    single_panel: bool = False,
) -> None:
    nodes = candidate.nodes
    edge_linewidth_scale = 1.42 if single_panel else 1.0
    edge_arrow_scale = 1.32 if single_panel else 1.0
    for relation in _panel_relations(panel):
        _draw_edges(
            ax,
            nodes,
            candidate.edges[relation],
            relation,
            linewidth_scale=edge_linewidth_scale,
            arrow_scale=edge_arrow_scale,
        )

    if panel == "all":
        _draw_nodes(
            ax,
            nodes,
            detailed_labels=True,
            detailed_node_size=300.0 if single_panel else 150.0,
            detailed_node_fontsize=16.0 if single_panel else 10.5,
            detail_label_fontsize=17.0 if single_panel else 8.8,
            label_box_pad=0.26 if single_panel else 0.18,
            label_vertical_span_factor=0.82 if single_panel else 0.58,
            label_side_offset_factor=0.72 if single_panel else 0.56,
        )
    else:
        _draw_nodes(
            ax,
            nodes,
            detailed_labels=False,
            node_size=420.0 if single_panel else 165.0,
            node_fontsize=15.4 if single_panel else 10.2,
        )

    if single_panel:
        padding_factor = 0.45 if panel == "all" else 0.50 if panel == "all_plain" else 0.43
        _style_axis(
            ax,
            nodes,
            _panel_title(candidate, panel),
            padding_factor=padding_factor,
            title_fontsize=23.0,
            axis_label_fontsize=19.0,
            tick_fontsize=16.0,
            title_pad=17.0,
        )
    else:
        padding_factor = 0.90 if panel == "all" else 0.32
        _style_axis(ax, nodes, _panel_title(candidate, panel), padding_factor=padding_factor)


def _add_legend(fig: plt.Figure, relations: list[str], fontsize: float, y_anchor: float) -> None:
    legend_handles = []
    for relation in relations:
        style = RELATION_STYLES[relation]
        handle = plt.Line2D([0], [0], color=style["color"], lw=3.0, label=_edge_label(relation))
        legend_handles.append(handle)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, y_anchor),
    )


def _plot_candidate(candidate: CandidateGraph, output_path: Path, image_format: str) -> None:
    transparent = image_format == "svg"
    fig = plt.figure(figsize=(16.0, 12.4), facecolor=PALETTE["background"])
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.32,
    )
    axes = {
        "all": fig.add_subplot(grid[0, 0]),
        "temporal": fig.add_subplot(grid[0, 1]),
        "spatial": fig.add_subplot(grid[1, 0]),
        "fixation": fig.add_subplot(grid[1, 1]),
    }

    for panel, ax in axes.items():
        _draw_panel(ax, candidate, panel)

    fig.text(
        0.5,
        0.965,
        FIGURE_TITLE,
        fontsize=18,
        color=PALETTE["dark"],
        ha="center",
        va="top",
    )

    _add_legend(fig, _panel_relations("all"), fontsize=13, y_anchor=0.04)
    fig.tight_layout(rect=[0.035, 0.14, 0.965, 0.79])
    fig.savefig(
        output_path,
        format=image_format,
        dpi=260,
        facecolor="none" if transparent else PALETTE["background"],
        transparent=transparent,
    )
    plt.close(fig)


def _plot_candidate_panel(
    candidate: CandidateGraph,
    panel: str,
    output_path: Path,
    image_format: str,
) -> None:
    transparent = image_format == "svg"
    figure_size = (13.8, 9.2) if panel == "all" else (12.0, 9.0) if panel == "all_plain" else (10.8, 8.4)
    fig, ax = plt.subplots(figsize=figure_size, facecolor=PALETTE["background"])
    _draw_panel(ax, candidate, panel, single_panel=True)
    relations = _panel_relations(panel)
    _add_legend(fig, relations, fontsize=15.5, y_anchor=0.03)
    fig.tight_layout(rect=[0.055, 0.12, 0.97, 0.94])
    fig.savefig(
        output_path,
        format=image_format,
        dpi=300,
        facecolor="none" if transparent else PALETTE["background"],
        transparent=transparent,
    )
    plt.close(fig)


def _filename_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-zčČšŠžŽ_-]+", "_", text)
    return text.strip("_") or "neznano"


def _candidate_file_stem(kind: str, rank: int, candidate: CandidateGraph) -> str:
    nodes = candidate.nodes
    time_start = _filename_slug(f"{float(nodes['time-rel-seconds'].iloc[0]):.2f}")
    time_end = _filename_slug(f"{float(nodes['time-rel-seconds'].iloc[-1]):.2f}")
    subject = _filename_slug(candidate.subject)
    recording = _filename_slug(candidate.recording)
    return (
        f"gnn_graph_{kind}_{rank:02d}_{subject}_{recording}_"
        f"t{time_start}-{time_end}_kt{candidate.kt}_ks{candidate.ks}_"
        f"kf{candidate.fixation_dilation_k}_f{candidate.fixation_count}_n{len(nodes)}"
    )


def _copy_selected_figures(
    selected: dict[str, list[CandidateGraph]],
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    saved: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("gnn_graph_*"):
        if old_path.is_dir():
            shutil.rmtree(old_path)
        else:
            old_path.unlink()

    for kind, candidates in selected.items():
        for rank, candidate in enumerate(candidates, start=1):
            file_stem = _candidate_file_stem(kind, rank, candidate)
            candidate_dir = output_dir / file_stem
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for image_format in formats:
                target = output_dir / f"{file_stem}.{image_format}"
                collage_target = candidate_dir / f"kolaz.{image_format}"
                if candidate.temporary_path is not None and image_format == "svg":
                    shutil.copy2(candidate.temporary_path, target)
                    shutil.copy2(candidate.temporary_path, collage_target)
                else:
                    _plot_candidate(candidate, target, image_format)
                    _plot_candidate(candidate, collage_target, image_format)
                saved.append(target)
                saved.append(collage_target)
                for panel, panel_stem in PANEL_FILE_STEMS.items():
                    panel_target = candidate_dir / f"{panel_stem}.{image_format}"
                    _plot_candidate_panel(candidate, panel, panel_target, image_format)
                    saved.append(panel_target)
    return saved


def main() -> None:
    args = parse_args()
    _log_args(args)
    if args.num_candidates < args.keep_per_kind * 2:
        raise ValueError("--num-candidates must be at least twice --keep-per-kind.")
    csv_files = sorted(args.data_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {args.data_root}.")

    rng = np.random.default_rng(args.seed)
    temporary_dir = args.output_dir / "_gnn_graph_candidates_tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[CandidateGraph] = []
    attempts = 0
    try:
        while len(candidates) < args.num_candidates and attempts < args.num_candidates * 12:
            attempts += 1
            candidate_id = len(candidates) + 1
            kind = "far" if candidate_id % 2 else "close"
            candidate = _build_candidate(candidate_id, kind, csv_files, args, rng)
            if candidate is None:
                continue
            temporary_path = temporary_dir / f"candidate_{candidate_id:02d}_{kind}.svg"
            _plot_candidate(candidate, temporary_path, "svg")
            candidates.append(
                CandidateGraph(
                    **{
                        **candidate.__dict__,
                        "temporary_path": temporary_path,
                    }
                )
            )
            print(
                f"Generated candidate {candidate_id:02d}/{args.num_candidates}: "
                f"{kind}, score={candidate.score:.2f}, nodes={len(candidate.nodes)}, "
                f"fixations={candidate.fixation_count}, cluster={candidate.fixation_cluster_quality:.2f}, "
                f"edges={candidate.total_edges}"
            )

        selected: dict[str, list[CandidateGraph]] = {}
        for kind in ["far", "close"]:
            kind_candidates = [candidate for candidate in candidates if candidate.kind == kind]
            if len(kind_candidates) < args.keep_per_kind:
                raise RuntimeError(
                    f"Only {len(kind_candidates)} valid '{kind}' candidates were generated; "
                    f"need {args.keep_per_kind}."
                )
            selected[kind] = sorted(kind_candidates, key=lambda candidate: candidate.score, reverse=True)[
                : args.keep_per_kind
            ]

        saved = _copy_selected_figures(selected, args.output_dir, list(args.formats))
        print("\nSelected figures:")
        for kind in ["far", "close"]:
            for rank, candidate in enumerate(selected[kind], start=1):
                print(
                    f"  {kind} #{rank}: candidate={candidate.candidate_id}, "
                    f"score={candidate.score:.2f}, nodes={len(candidate.nodes)}, "
                    f"fixations={candidate.fixation_count}, cluster={candidate.fixation_cluster_quality:.2f}, "
                    f"edges={candidate.total_edges}, "
                    f"source={candidate.source_path}"
                )
        print("\nSaved selected files:")
        for path in saved:
            print(f"  {path}")
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
            print(f"\nDeleted temporary candidate figures: {temporary_dir}")


if __name__ == "__main__":
    main()
