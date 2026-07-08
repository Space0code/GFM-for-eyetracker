"""Create a thesis figure for dilated intra-fixation edges.

The figure is a deterministic schematic companion to the fixation-edge
definition in the diploma text. It uses a circular layout for one fixation run,
highlights the selected node, its one-hop neighbors, and the additional nodes
reachable in two hops.

Example:
    python scripts/create_thesis_fixation_dilated_edges_figure.py
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


PALETTE = {
    "dark": "#2F3437",
    "blue": "#6FBDE0",
    "teal": "#91D4D6",
    "purple": "#B8A1D9",
    "orange": "#F4A261",
    "pink": "#E9A6A6",
    "background": "#F7F7F2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../diploma-latex/slike/konstrukcija_grafa"),
        help="Directory where the figure is saved.",
    )
    parser.add_argument("--F", type=int, default=21, help="Fixation length in nodes.")
    parser.add_argument("--kf", type=int, default=3, help="Fixation dilation density parameter.")
    parser.add_argument("--L", type=int, default=2, help="Number of message-passing hops to highlight.")
    parser.add_argument("--v", type=int, default=0, help="Selected local node index.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats.",
    )
    return parser.parse_args()


def round_half_up(value: float) -> int:
    """Round to nearest integer with .5 rounded upward."""
    return int(math.floor(float(value) + 0.5))


def build_dilated_fixation_edges(F: int, kf: int) -> tuple[int, list[int], list[tuple[int, int]], dict[int, set[int]]]:
    """Build deduplicated undirected dilated intra-fixation edges."""
    if F < 2:
        raise ValueError("F must be >= 2 for a visible fixation graph.")
    if kf < 1:
        raise ValueError("kf must be >= 1.")

    step = max(1, round_half_up(F / kf))
    offsets = sorted({(1 + q * step) % F for q in range(kf)} - {0})

    edges: set[tuple[int, int]] = set()
    for source in range(F):
        for offset in offsets:
            target = (source + offset) % F
            if source == target:
                continue
            edges.add(tuple(sorted((source, target))))

    adjacency = {idx: set() for idx in range(F)}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)

    return step, offsets, sorted(edges), adjacency


def hop_depths(adjacency: dict[int, set[int]], source: int, max_depth: int) -> dict[int, int]:
    """Return shortest hop depth from source up to max_depth."""
    depths = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        if depths[node] >= max_depth:
            continue
        for neighbor in sorted(adjacency[node]):
            if neighbor not in depths:
                depths[neighbor] = depths[node] + 1
                queue.append(neighbor)
    return depths


def draw_curved_edge(
    ax: plt.Axes,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float,
    radius: float,
    zorder: int,
) -> None:
    """Draw an undirected curved edge as a line without arrow heads."""
    patch = FancyArrowPatch(
        posA=start_xy,
        posB=end_xy,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={radius}",
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        mutation_scale=1,
        zorder=zorder,
    )
    ax.add_patch(patch)


def make_figure(F: int, kf: int, L: int, v: int, output_dir: Path, formats: list[str]) -> list[Path]:
    step, offsets, edges, adjacency = build_dilated_fixation_edges(F=F, kf=kf)
    v = max(0, min(int(v), F - 1))
    depths = hop_depths(adjacency, source=v, max_depth=L)
    one_hop = {node for node, depth in depths.items() if depth == 1}
    two_hop = {node for node, depth in depths.items() if depth == 2}

    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, F, endpoint=False)
    xy = np.column_stack([np.cos(angles), np.sin(angles)])

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for source, target in edges:
        draw_curved_edge(
            ax=ax,
            start_xy=xy[source],
            end_xy=xy[target],
            color=PALETTE["dark"],
            linewidth=1.0,
            alpha=0.22,
            radius=0.12,
            zorder=0,
        )

    for target in sorted(adjacency[v]):
        draw_curved_edge(
            ax=ax,
            start_xy=xy[v],
            end_xy=xy[target],
            color=PALETTE["blue"],
            linewidth=2.9,
            alpha=0.95,
            radius=0.12,
            zorder=2,
        )

    second_hop_edges = [
        (source, target)
        for source, target in edges
        if (source in one_hop and target in two_hop) or (target in one_hop and source in two_hop)
    ]
    for source, target in second_hop_edges:
        draw_curved_edge(
            ax=ax,
            start_xy=xy[source],
            end_xy=xy[target],
            color=PALETTE["teal"],
            linewidth=2.1,
            alpha=0.72,
            radius=-0.10,
            zorder=2,
        )

    node_colors = []
    node_sizes = []
    for node in range(F):
        if node == v:
            node_colors.append(PALETTE["orange"])
            node_sizes.append(250)
        elif node in one_hop:
            node_colors.append(PALETTE["blue"])
            node_sizes.append(190)
        elif node in two_hop:
            node_colors.append(PALETTE["teal"])
            node_sizes.append(180)
        else:
            node_colors.append("#D9D9D2")
            node_sizes.append(145)

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_sizes,
        c=node_colors,
        edgecolor=PALETTE["dark"],
        linewidth=0.9,
        zorder=3,
    )

    for idx, (x_pos, y_pos) in enumerate(xy):
        ax.text(
            x_pos,
            y_pos,
            str(idx),
            ha="center",
            va="center",
            color=PALETTE["dark"],
            fontsize=9.2,
            fontweight="bold" if idx == v else "normal",
            zorder=4,
        )

    summary = f"$F={F}$, $k_f={kf}$, $L={L}$,\n$s={step}$, $D=\\{{{', '.join(str(o) for o in offsets)}\\}}$"
    ax.text(
        0,
        1.25,
        summary,
        ha="center",
        va="center",
        fontsize=11.5,
        color=PALETTE["dark"],
    )

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["orange"], markeredgecolor=PALETTE["dark"], markersize=9, label="izbrano vozlišče $v=0$"),
        Line2D([0], [0], marker="o", color=PALETTE["blue"], markerfacecolor=PALETTE["blue"], markeredgecolor=PALETTE["dark"], markersize=8, linewidth=2.7, label="1-hop sosedi"),
        Line2D([0], [0], marker="o", color=PALETTE["teal"], markerfacecolor=PALETTE["teal"], markeredgecolor=PALETTE["dark"], markersize=8, linewidth=2.1, label="2-hop sosedi"),
        Line2D([0], [0], marker="o", color=PALETTE["dark"], markerfacecolor="#D9D9D2", markeredgecolor=PALETTE["dark"], markersize=7, linewidth=0.8, alpha=0.55, label="ostale povezave"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.065),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=2.0,
        columnspacing=1.8,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.30, 1.36)
    ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"razsirjene_fiksacijske_povezave_F{F}_kf{kf}_L{L}_v{v}"
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, dpi=300)
        paths.append(path)
    plt.close(fig)

    print("Figure parameters:")
    print(f"  F: {F}")
    print(f"  kf: {kf}")
    print(f"  L: {L}")
    print(f"  v: {v}")
    print(f"  s: {step}")
    print(f"  offsets: {offsets}")
    print(f"  undirected_edges: {len(edges)}")
    print(f"  directed_edges: {2 * len(edges)}")
    print(f"  one_hop_neighbors: {sorted(one_hop)}")
    print(f"  two_hop_neighbors: {sorted(two_hop)}")
    print(f"  reached_within_L: {sorted(depths)}")
    print(f"  unreached_within_L: {sorted(set(range(F)) - set(depths))}")
    print("Saved:")
    for path in paths:
        print(f"  {path}")
    return paths


def main() -> None:
    args = parse_args()
    make_figure(
        F=args.F,
        kf=args.kf,
        L=args.L,
        v=args.v,
        output_dir=args.output_dir,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
