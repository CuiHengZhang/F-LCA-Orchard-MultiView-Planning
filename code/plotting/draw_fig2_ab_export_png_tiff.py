# -*- coding: utf-8 -*-
"""
Fig. 2. Tree-level task representation for multi-view observation
A4-print optimized version.

Main improvements:
    1) Larger in-figure text for A4 printing.
    2) Larger observation points and thicker arrows.
    3) Slightly tighter canvas margins so the 3 x 3 grid occupies more visible area.
    4) Exports both separate subfigures and a combined two-panel figure.

Outputs:
    fig2a_single_point_large_v2.png
    fig2a_single_point_large_v2.tiff
    fig2b_four_view_large_v2.png
    fig2b_four_view_large_v2.tiff
    fig2_ab_combined_large_v2.png
    fig2_ab_combined_large_v2.tiff
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

# =========================================================
# Global style
# =========================================================
AVAILABLE_FONTS = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Times New Roman" if "Times New Roman" in AVAILABLE_FONTS else "DejaVu Serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================================================
# Output settings
# =========================================================
DPI = 600

# Individual subfigure size.
# The relative text size is intentionally larger than the original script.
FIG_W = 4.2
FIG_H = 4.2

# Tighter axis limits make the grid occupy more of the exported image.
XMIN, XMAX = -0.06, 3.06
YMIN, YMAX = -0.06, 3.06

# =========================================================
# Colors
# =========================================================
GRID_EDGE = "#3A3A3A"
TEXT = "#222222"

WHITE_CELL = "#FFFFFF"
NEIGHBOR_CELL = "#E6E6E6"
CENTER_CELL = "#B9B9B9"

# Fig. 2a
VISIT_CELL = "#DCE8F4"
BLUE = "#4B6F97"
ARROW_BLUE = "#5E7FA8"

# Fig. 2b
VIEW_CELL = "#E3EEE0"
GREEN = "#4F7F5A"
ARROW_GREEN = "#7EA07A"

# =========================================================
# Geometry and style parameters
# =========================================================
CELL_LW = 1.35

NUM_OFFSET_X = 0.17
NUM_OFFSET_Y = 0.82

# Enlarged for A4 readability.
TREE_FONTSIZE = 20
NUM_FONTSIZE = 17
DIR_FONTSIZE = 16
SINGLE_LABEL_FONTSIZE = 14.8
SUBCAPTION_FONTSIZE = 12.5

POINT_RADIUS = 0.112

ARROW_LW = 4.0
ARROW_SCALE = 31

# Observation viewpoint centers for Fig. 2b
VIEWPOINTS = {
    "N": (1.50, 2.52),
    "E": (2.48, 1.50),
    "S": (1.50, 0.48),
    "W": (0.52, 1.50),
}

# Direction label positions
LABEL_POSITIONS = {
    "N": (1.74, 2.52),
    "E": (2.80, 1.50),
    "S": (1.50, 0.19),
    "W": (0.22, 1.50),
}

# =========================================================
# Canvas
# =========================================================
def create_canvas():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    # Keep exactly the same output size for Fig. 2a and Fig. 2b.
    # The margins are reduced so that the useful diagram area is larger.
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)

    ax.set_aspect("equal")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.axis("off")

    return fig, ax


def setup_panel_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.axis("off")


# =========================================================
# Basic drawing functions
# =========================================================
def draw_cell(ax, x, y, facecolor):
    ax.add_patch(
        Rectangle(
            (x, y),
            1,
            1,
            facecolor=facecolor,
            edgecolor=GRID_EDGE,
            linewidth=CELL_LW,
            zorder=1,
        )
    )


def put_num_top_left(ax, cell_x, cell_y, text):
    ax.text(
        cell_x + NUM_OFFSET_X,
        cell_y + NUM_OFFSET_Y,
        text,
        ha="left",
        va="center",
        fontsize=NUM_FONTSIZE,
        color=TEXT,
        zorder=6,
    )


def draw_tree(ax):
    ax.text(
        1.50,
        1.56,
        "Tree",
        ha="center",
        va="center",
        fontsize=TREE_FONTSIZE,
        color=TEXT,
        zorder=8,
    )


# =========================================================
# Fig. 2a: conventional single-point representation
# =========================================================
def draw_grid_single_point(ax):
    # Top row: 1 2 3
    draw_cell(ax, 0, 2, WHITE_CELL)
    draw_cell(ax, 1, 2, NEIGHBOR_CELL)
    draw_cell(ax, 2, 2, WHITE_CELL)

    # Middle row: 8 Tree 4
    draw_cell(ax, 0, 1, NEIGHBOR_CELL)
    draw_cell(ax, 1, 1, CENTER_CELL)
    draw_cell(ax, 2, 1, NEIGHBOR_CELL)

    # Bottom row: 7 6 5
    draw_cell(ax, 0, 0, WHITE_CELL)
    draw_cell(ax, 1, 0, VISIT_CELL)
    draw_cell(ax, 2, 0, WHITE_CELL)

    # Numbers
    put_num_top_left(ax, 0, 2, "1")
    put_num_top_left(ax, 1, 2, "2")
    put_num_top_left(ax, 2, 2, "3")

    put_num_top_left(ax, 0, 1, "8")
    put_num_top_left(ax, 2, 1, "4")

    put_num_top_left(ax, 0, 0, "7")
    put_num_top_left(ax, 1, 0, "6")
    put_num_top_left(ax, 2, 0, "5")


def draw_single_point_content(ax):
    # Arrow to the single visiting point
    ax.add_patch(
        FancyArrowPatch(
            (1.50, 1.16),
            (1.50, 0.72),
            arrowstyle="-|>",
            mutation_scale=ARROW_SCALE,
            linewidth=ARROW_LW,
            color=ARROW_BLUE,
            zorder=5,
        )
    )

    # Single visiting point
    ax.add_patch(
        Circle(
            (1.50, 0.58),
            radius=POINT_RADIUS,
            facecolor=BLUE,
            edgecolor="white",
            linewidth=1.15,
            zorder=7,
        )
    )

    ax.text(
        1.50,
        0.215,
        "single\nvisit",
        ha="center",
        va="center",
        fontsize=SINGLE_LABEL_FONTSIZE,
        color=BLUE,
        zorder=7,
        linespacing=0.84,
    )


def draw_fig2a_on_axis(ax):
    draw_grid_single_point(ax)
    draw_single_point_content(ax)
    draw_tree(ax)


def make_fig2a(out_dir):
    plt.close("all")
    fig, ax = create_canvas()
    draw_fig2a_on_axis(ax)
    save_both_formats(fig, "fig2a_single_point_large_v2", out_dir)
    plt.close(fig)


# =========================================================
# Fig. 2b: proposed four-view task representation
# =========================================================
def draw_grid_four_view(ax):
    # Top row: 1 2 3
    draw_cell(ax, 0, 2, WHITE_CELL)
    draw_cell(ax, 1, 2, VIEW_CELL)
    draw_cell(ax, 2, 2, WHITE_CELL)

    # Middle row: 8 Tree 4
    draw_cell(ax, 0, 1, VIEW_CELL)
    draw_cell(ax, 1, 1, CENTER_CELL)
    draw_cell(ax, 2, 1, VIEW_CELL)

    # Bottom row: 7 6 5
    draw_cell(ax, 0, 0, WHITE_CELL)
    draw_cell(ax, 1, 0, VIEW_CELL)
    draw_cell(ax, 2, 0, WHITE_CELL)

    # Numbers
    put_num_top_left(ax, 0, 2, "1")
    put_num_top_left(ax, 1, 2, "2")
    put_num_top_left(ax, 2, 2, "3")

    put_num_top_left(ax, 0, 1, "8")
    put_num_top_left(ax, 2, 1, "4")

    put_num_top_left(ax, 0, 0, "7")
    put_num_top_left(ax, 1, 0, "6")
    put_num_top_left(ax, 2, 0, "5")


def draw_direction_arrow(ax, start, end):
    sx, sy = start
    ex, ey = end

    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)

    ux = dx / length
    uy = dy / length

    start_shift = 0.31
    end_shift = 0.13

    new_start = (sx + ux * start_shift, sy + uy * start_shift)
    new_end = (ex - ux * end_shift, ey - uy * end_shift)

    ax.add_patch(
        FancyArrowPatch(
            new_start,
            new_end,
            arrowstyle="-|>",
            mutation_scale=ARROW_SCALE,
            linewidth=ARROW_LW,
            color=ARROW_GREEN,
            zorder=4,
        )
    )


def draw_four_view_content(ax):
    center = (1.50, 1.50)

    # Direction arrows from tree center to four viewpoints
    for _, viewpoint in VIEWPOINTS.items():
        draw_direction_arrow(ax, center, viewpoint)

    # Blue observation viewpoint dots
    for _, (x, y) in VIEWPOINTS.items():
        ax.add_patch(
            Circle(
                (x, y),
                radius=POINT_RADIUS,
                facecolor=BLUE,
                edgecolor="white",
                linewidth=1.15,
                zorder=7,
            )
        )

    # N / E / S / W direction labels
    for direction, (x, y) in LABEL_POSITIONS.items():
        ax.text(
            x,
            y,
            direction,
            ha="center",
            va="center",
            fontsize=DIR_FONTSIZE,
            fontweight="bold",
            color=GREEN,
            zorder=8,
        )


def draw_fig2b_on_axis(ax):
    draw_grid_four_view(ax)
    draw_four_view_content(ax)
    draw_tree(ax)


def make_fig2b(out_dir):
    plt.close("all")
    fig, ax = create_canvas()
    draw_fig2b_on_axis(ax)
    save_both_formats(fig, "fig2b_four_view_large_v2", out_dir)
    plt.close(fig)


# =========================================================
# Combined two-panel output
# =========================================================
def make_combined_fig2(out_dir):
    plt.close("all")

    # Full-width friendly combined figure for direct insertion into Word.
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.70), dpi=DPI)
    fig.subplots_adjust(left=0.045, right=0.955, bottom=0.18, top=0.985, wspace=0.26)

    for ax in axes:
        setup_panel_axis(ax)

    draw_fig2a_on_axis(axes[0])
    draw_fig2b_on_axis(axes[1])

    fig.text(
        0.255,
        0.070,
        "(a) Conventional single-point representation",
        ha="center",
        va="center",
        fontsize=SUBCAPTION_FONTSIZE,
        color=TEXT,
    )
    fig.text(
        0.745,
        0.070,
        "(b) F-LCA four-view observation unit",
        ha="center",
        va="center",
        fontsize=SUBCAPTION_FONTSIZE,
        color=TEXT,
    )

    save_both_formats(fig, "fig2_ab_combined_large_v2", out_dir, tight=True)
    plt.close(fig)


# =========================================================
# Save
# =========================================================
def save_both_formats(fig, out_stem, out_dir, tight=False):
    png_path = out_dir / f"{out_stem}.png"
    tif_path = out_dir / f"{out_stem}.tiff"

    for path in [png_path, tif_path]:
        if path.exists():
            path.unlink()

    save_kwargs = dict(dpi=DPI, facecolor="white")
    if tight:
        save_kwargs.update(dict(bbox_inches="tight", pad_inches=0.02))

    fig.savefig(png_path, **save_kwargs)
    fig.savefig(
        tif_path,
        **save_kwargs,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    print(f"Saved PNG : {png_path}")
    print(f"Saved TIFF: {tif_path}")


# =========================================================
# Main
# =========================================================
def main():
    out_dir = Path(__file__).resolve().parent

    make_fig2a(out_dir)
    make_fig2b(out_dir)
    make_combined_fig2(out_dir)

    print("\nDone.")
    print("Generated files:")
    print("  fig2a_single_point_large_v2.png")
    print("  fig2a_single_point_large_v2.tiff")
    print("  fig2b_four_view_large_v2.png")
    print("  fig2b_four_view_large_v2.tiff")
    print("  fig2_ab_combined_large_v2.png")
    print("  fig2_ab_combined_large_v2.tiff")


if __name__ == "__main__":
    main()
