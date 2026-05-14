# -*- coding: utf-8 -*-
"""
Draw Fig. 5 for the real-orchard F-LCA case from JSON/CSV data.
Put this script in the same folder as:
    real_orchard_input.json
    real_orchard_best_path.json
    real_citrus_tree_points_xy.csv
Then run:
    python draw_real_orchard_fig5_local.py
Outputs will be saved to:
    fig5_python_redraw_output/
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import pandas as pd

# Use the folder where this script is located.
BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "fig5_python_redraw_output"
OUT_DIR.mkdir(exist_ok=True)

INPUT_JSON = BASE / "real_orchard_input.json"
PATH_JSON = BASE / "real_orchard_best_path.json"
TREE_CSV = BASE / "real_citrus_tree_points_xy.csv"

with INPUT_JSON.open("r", encoding="utf-8") as f:
    scene = json.load(f)
with PATH_JSON.open("r", encoding="utf-8") as f:
    result = json.load(f)

tree_df = pd.read_csv(TREE_CSV, encoding="utf-8-sig")

width = float(scene["width_m"])
height = float(scene["height_m"])
starts = {int(k): tuple(v) for k, v in result["start_positions"].items()}
paths = {int(k): [tuple(p) for p in v] for k, v in result["vehicle_paths"].items()}

# Low-saturation path colors. Keep the whole figure close to a gray academic style.
COLORS = {
    0: "#4C78A8",  # muted blue
    1: "#B07A3A",  # muted brown-orange
    2: "#5A8A5E",  # muted green
    3: "#8B6A9E",  # muted purple
}
LINESTYLES = {0: "-", 1: "-", 2: "-", 3: "-"}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
})


def build_legend_handles():
    handles = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="#D9D9D9", markeredgecolor="#222222",
            markeredgewidth=0.55, markersize=6.2,
            label="Real tree locations"
        ),
        Line2D(
            [0], [0], marker="s", color="none",
            markerfacecolor="white", markeredgecolor="#111111",
            markeredgewidth=0.75, markersize=6.0,
            label="UGV start positions"
        ),
    ]
    for vid in sorted(paths):
        handles.append(
            Line2D(
                [0], [0], color=COLORS.get(vid, "#555555"), lw=2.1,
                linestyle=LINESTYLES.get(vid, "-"), label=f"UGV {vid + 1}"
            )
        )
    return handles


def draw_orchard(ax):
    # Tree locations.
    ax.scatter(
        tree_df["x_m"], tree_df["y_m"],
        s=34, facecolors="#D9D9D9", edgecolors="#222222",
        linewidths=0.52, zorder=4
    )

    # Start-position label offsets.
    label_pos = {
        0: (starts[0][0] + 0.65, starts[0][1] + 0.75, "left", "bottom"),
        1: (starts[1][0] + 0.55, starts[1][1] + 0.75, "left", "bottom"),
        2: (starts[2][0] + 0.65, starts[2][1] + 0.65, "left", "bottom"),
        3: (starts[3][0] + 0.55, starts[3][1] + 0.65, "left", "bottom"),
    }

    # Vehicle paths.
    for vid in sorted(paths):
        start = starts[vid]
        path = paths[vid]
        full_path = [start] + path
        xs = [p[0] for p in full_path]
        ys = [p[1] for p in full_path]
        color = COLORS.get(vid, "#555555")

        ax.plot(
            xs, ys, color=color, linewidth=1.72,
            linestyle=LINESTYLES.get(vid, "-"), alpha=0.95,
            solid_capstyle="round", solid_joinstyle="round", zorder=2
        )
        ax.plot(
            xs[1:], ys[1:], linestyle="None", marker="o", markersize=2.55,
            markerfacecolor=color, markeredgecolor=color, alpha=0.95, zorder=5
        )
        ax.scatter(
            [start[0]], [start[1]], marker="s", s=50,
            facecolor=color, edgecolor="#111111", linewidth=0.72, zorder=6
        )
        if vid in label_pos:
            x, y, ha, va = label_pos[vid]
            ax.text(
                x, y, f"S{vid + 1}", fontsize=8.0, fontweight="bold",
                color=color, ha=ha, va=va, zorder=7
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.6, width + 2.8)
    ax.set_ylim(-2.4, height + 3.2)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.grid(True, linestyle="--", linewidth=0.42, color="#D0D0D0", alpha=0.85)
    ax.tick_params(direction="out", length=3, width=0.8, pad=2)


def save_all(fig, stem):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(
        OUT_DIR / f"{stem}.tif",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )


def main():
    # Version A: two-panel version, matching manuscript caption: (a) path, (b) legend.
    fig = plt.figure(figsize=(8.2, 5.2), constrained_layout=False)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4.8, 1.2], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    draw_orchard(ax)
    ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.5, fontweight="bold", clip_on=False)

    ax_legend.axis("off")
    ax_legend.text(0.00, 1.02, "(b)", transform=ax_legend.transAxes,
                   ha="left", va="bottom", fontsize=9.5, fontweight="bold", clip_on=False)
    ax_legend.legend(
        handles=build_legend_handles(), loc="center left", frameon=False,
        handlelength=2.25, labelspacing=0.95, borderaxespad=0
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.965, wspace=0.10)
    save_all(fig, "real_orchard_fig5_two_panel")
    plt.close(fig)

    # Version B: single-panel version with legend above the plot.
    fig, ax = plt.subplots(figsize=(7.2, 5.35))
    draw_orchard(ax)
    ax.legend(
        handles=build_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.16),
        ncol=3, frameon=True, framealpha=0.96, borderpad=0.42,
        handlelength=2.7, columnspacing=1.1
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.83)
    save_all(fig, "real_orchard_fig5_single_panel")
    plt.close(fig)

    print(f"Saved figures to: {OUT_DIR}")
    print("Main output: real_orchard_fig5_two_panel.tif")


if __name__ == "__main__":
    main()
