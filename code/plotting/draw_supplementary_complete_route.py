import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# =========================================================
# 1. File paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
SCENE_FILE = BASE_DIR / "scene_seed44.json"
PATH_FILE = BASE_DIR / "best_path_seed44.json"

OUTPUT_MAIN = BASE_DIR / "figS_complete_route.png"
OUTPUT_LEGEND = BASE_DIR / "figS_complete_route_style_panel.png"

OUTPUT_DPI = 600
SHOW_FIGURES = False
TIFF_COMPRESSION = "tiff_lzw"

# Main map panels use the same fixed canvas size.
MAP_FIGSIZE = (6.8, 6.8)

# Legend remains a vertical explanatory panel.
LEGEND_FIGSIZE = (3.35, 4.95)


# =========================================================
# 2. Visual design
# =========================================================
ROUTE_COLORS = {
    0: "#2C7FB8",
    1: "#D95F02",
    2: "#1B9E77",
    3: "#B07AA1",
}

ROUTE_STYLES = {
    0: "-",
    1: (0, (4.8, 2.0)),
    2: (0, (5.0, 1.6, 1.3, 1.6)),
    3: (0, (1.2, 1.45)),
}

ROUTE_MAIN_WIDTH = {0: 1.05, 1: 1.18, 2: 1.28, 3: 1.38}
ROUTE_LOCAL_WIDTH = {0: 0.82, 1: 0.92, 2: 1.02, 3: 1.12}
ROUTE_ARROW_WIDTH = {0: 0.72, 1: 0.80, 2: 0.88, 3: 0.96}

FIG_FACE = "#FFFFFF"
AX_FACE = "#FFFFFF"

TREE_FACE = "#5F9E62"
TREE_EDGE = "#5F9E62"
TREE_SIZE = 22
TREE_LW = 0.0
TREE_ALPHA = 0.95

VIEWPOINT_COLOR = "#333333"
VIEWPOINT_SIZE_MAIN = 7.2
VIEWPOINT_ALPHA_MAIN = 0.76
VIEWPOINT_SIZE_LEGEND = 24

START_COLOR = "#4C78A8"
END_COLOR = "#B55D5D"
END_TEXT_COLOR = "#111111"

START_MARKER_SIZE = 140
START_MARKER_LW = 1.45
END_MARKER_SIZE = 46
END_MARKER_LW = 1.15
END_LABEL_SIZE = 10.5
START_LABEL_SIZE = 11

OBSTACLE_FILL = "#F0F0F0"
OBSTACLE_EDGE = "#707070"
OBSTACLE_LW = 1.30
OBSTACLE_ALPHA = 0.75
OBSTACLE_LINESTYLE = (0, (3.0, 1.8))

ZOOM_BOX_EDGE = "#303030"
ZOOM_BOX_LW = 1.00
ZOOM_BOX_LINESTYLE = (0, (6.0, 3.0))
ZOOM_BOX_ALPHA = 0.78

# Important:
# These boxes are made square so that the corresponding local panels
# do not look visually narrow after equal-aspect rendering.
ZOOM_BOXES = [
    {
        "label": "(b)",
        "xlim": (66, 102),
        "ylim": (54, 90),
        "label_pos": (67.5, 88.0),
    },
    {
        "label": "(c)",
        "xlim": (7, 28),
        "ylim": (6, 27),
        "label_pos": (8.0, 25.5),
    },
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# 3. Load data
# =========================================================
def load_data():
    if not SCENE_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {SCENE_FILE}. Put this script in the same folder as scene_seed44.json."
        )
    if not PATH_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {PATH_FILE}. Put this script in the same folder as best_path_seed44.json."
        )

    with open(SCENE_FILE, "r", encoding="utf-8") as f:
        scene = json.load(f)
    with open(PATH_FILE, "r", encoding="utf-8") as f:
        solution = json.load(f)

    return scene, solution


# =========================================================
# 4. Utility functions
# =========================================================
def save_png_and_tiff(fig, png_path, dpi=600, pad_inches=0.02, fixed_canvas=True):
    """
    Save both PNG and TIFF.

    fixed_canvas=True:
      - keeps the exact figure canvas size.
      - avoids size differences caused by bbox_inches="tight".
      - recommended for panels that will be assembled into a 2x2 figure.
    """
    png_path = Path(png_path)
    tiff_path = png_path.with_suffix(".tiff")

    if fixed_canvas:
        fig.savefig(
            png_path,
            dpi=dpi,
            format="png",
            facecolor="white",
        )
        fig.savefig(
            tiff_path,
            dpi=dpi,
            format="tiff",
            facecolor="white",
            pil_kwargs={"compression": TIFF_COMPRESSION},
        )
    else:
        fig.savefig(
            png_path,
            dpi=dpi,
            format="png",
            bbox_inches="tight",
            pad_inches=pad_inches,
            facecolor="white",
        )
        fig.savefig(
            tiff_path,
            dpi=dpi,
            format="tiff",
            bbox_inches="tight",
            pad_inches=pad_inches,
            facecolor="white",
            pil_kwargs={"compression": TIFF_COMPRESSION},
        )


def vehicle_id_key(item):
    return int(item[0])


def add_arrow(ax, p0, p1, color, linestyle="-", lw=1.0, alpha=0.90, mutation_scale=7.0):
    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0
    dist = np.hypot(dx, dy)

    if dist < 6:
        return

    frac_end = 0.60
    frac_start = max(0.44, frac_end - min(0.12, 3.2 / dist))

    sx = x0 + frac_start * dx
    sy = y0 + frac_start * dy
    ex = x0 + frac_end * dx
    ey = y0 + frac_end * dy

    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=9,
    )


def plot_line(ax, x, y, color, linestyle, linewidth, alpha, zorder):
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        dash_capstyle="round",
    )


def start_label_offset(x, y, width, height):
    if x < width / 2 and y < height / 2:
        return 1.6, 1.6, "left", "bottom"
    if x > width / 2 and y < height / 2:
        return -1.6, 1.6, "right", "bottom"
    if x < width / 2 and y > height / 2:
        return 1.6, -0.5, "left", "top"
    if x > width / 2 and y > height / 2:
        return -1.6, -0.5, "right", "top"
    return 1.6, 1.6, "left", "bottom"


def end_label_spec(idx):
    specs = {
        0: (1.2, -2.4, "left", "top"),
        1: (1.2, 1.8, "left", "bottom"),
        2: (1.2, 1.8, "left", "bottom"),
        3: (1.2, 1.8, "left", "bottom"),
    }
    return specs.get(idx, (1.2, 1.8, "left", "bottom"))


# =========================================================
# 5. Shared route drawing functions
# =========================================================
def draw_scene_elements(ax, scene):
    # Obstacles
    for obs in scene["obstacles"]:
        circle = Circle(
            (obs["cx"], obs["cy"]),
            obs["r"],
            facecolor=OBSTACLE_FILL,
            edgecolor=OBSTACLE_EDGE,
            linestyle=OBSTACLE_LINESTYLE,
            linewidth=OBSTACLE_LW,
            alpha=OBSTACLE_ALPHA,
            zorder=1,
        )
        ax.add_patch(circle)

    # Trees
    tree_x = [t["x"] for t in scene["trees"]]
    tree_y = [t["y"] for t in scene["trees"]]
    ax.scatter(
        tree_x,
        tree_y,
        s=TREE_SIZE,
        marker="o",
        facecolor=TREE_FACE,
        edgecolor=TREE_EDGE,
        linewidth=TREE_LW,
        alpha=TREE_ALPHA,
        zorder=5,
    )

    # Observation waypoints
    obs_pts = []
    for pts in scene["tree_observation_points"].values():
        obs_pts.extend(pts)

    if obs_pts:
        obs_pts = np.array(obs_pts, dtype=float)
        ax.scatter(
            obs_pts[:, 0],
            obs_pts[:, 1],
            s=VIEWPOINT_SIZE_MAIN,
            marker="o",
            facecolor=VIEWPOINT_COLOR,
            edgecolor="white",
            linewidth=0.20,
            alpha=VIEWPOINT_ALPHA_MAIN,
            zorder=6,
        )


def draw_vehicle_routes(ax, solution):
    for vid_str, veh in sorted(solution["vehicles"].items(), key=vehicle_id_key):
        vid = int(vid_str)
        color = ROUTE_COLORS[vid]
        style = ROUTE_STYLES[vid]

        pts = np.array(veh["path_points"], dtype=float)
        if len(pts) < 5:
            continue

        start = pts[0]
        grouped = pts[1:]
        n_groups = len(grouped) // 4
        prev = start

        for gi in range(n_groups):
            group = grouped[gi * 4:(gi + 1) * 4]
            if len(group) < 4:
                continue

            # Inter-tree route segment
            plot_line(
                ax,
                [prev[0], group[0, 0]],
                [prev[1], group[0, 1]],
                color=color,
                linestyle=style,
                linewidth=ROUTE_MAIN_WIDTH[vid],
                alpha=0.92,
                zorder=7,
            )

            add_arrow(
                ax,
                prev,
                group[0],
                color=color,
                linestyle=style,
                lw=ROUTE_ARROW_WIDTH[vid],
                alpha=0.90,
                mutation_scale=7.0,
            )

            # Local four-viewpoint path
            plot_line(
                ax,
                group[:, 0],
                group[:, 1],
                color=color,
                linestyle=style,
                linewidth=ROUTE_LOCAL_WIDTH[vid],
                alpha=0.90,
                zorder=8,
            )

            prev = group[-1]


def draw_start_markers(ax, scene):
    width = scene["width"]
    height = scene["height"]
    starts = np.array(scene["start_positions"], dtype=float)

    for i, (x, y) in enumerate(starts):
        ax.scatter(
            x,
            y,
            s=START_MARKER_SIZE,
            marker="^",
            facecolor="white",
            edgecolor=START_COLOR,
            linewidth=START_MARKER_LW,
            zorder=11,
            clip_on=False,
        )

        dx, dy, ha, va = start_label_offset(x, y, width, height)

        ax.text(
            x + dx,
            y + dy,
            f"S{i + 1}",
            fontsize=START_LABEL_SIZE,
            fontweight="semibold",
            color="#111111",
            ha=ha,
            va=va,
            zorder=12,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.18),
        )


def draw_end_markers(ax, solution):
    for vid_str, veh in sorted(solution["vehicles"].items(), key=vehicle_id_key):
        vid = int(vid_str)
        pts = np.array(veh["path_points"], dtype=float)
        end = pts[-1]

        ax.scatter(
            end[0],
            end[1],
            s=END_MARKER_SIZE,
            marker="s",
            facecolor="white",
            edgecolor=END_COLOR,
            linewidth=END_MARKER_LW,
            zorder=11,
        )

        dx, dy, ha, va = end_label_spec(vid)

        ax.text(
            end[0] + dx,
            end[1] + dy,
            f"E{vid + 1}",
            fontsize=END_LABEL_SIZE,
            fontweight="semibold",
            color=END_TEXT_COLOR,
            ha=ha,
            va=va,
            zorder=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.18),
        )


def draw_zoom_boxes(ax):
    for box in ZOOM_BOXES:
        x0, x1 = box["xlim"]
        y0, y1 = box["ylim"]

        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="none",
            edgecolor=ZOOM_BOX_EDGE,
            linewidth=ZOOM_BOX_LW,
            linestyle=ZOOM_BOX_LINESTYLE,
            alpha=ZOOM_BOX_ALPHA,
            zorder=13,
        )
        ax.add_patch(rect)

        ax.text(
            box["label_pos"][0],
            box["label_pos"][1],
            box["label"],
            fontsize=10.5,
            fontweight="semibold",
            color="#111111",
            ha="left",
            va="center",
            zorder=14,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
                pad=0.18,
            ),
        )


# =========================================================
# 6. Complete route figure
# =========================================================
def draw_main_figure(scene, solution):
    width = scene["width"]
    height = scene["height"]

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=OUTPUT_DPI)
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(AX_FACE)

    draw_scene_elements(ax, scene)
    draw_vehicle_routes(ax, solution)
    draw_end_markers(ax, solution)
    draw_start_markers(ax, scene)
    draw_zoom_boxes(ax)

    ax.set_xlim(-1.5, width + 1.5)
    ax.set_ylim(-1.5, height + 1.5)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")
    ax.set_xticks(np.arange(0, width + 1, 20))
    ax.set_yticks(np.arange(0, height + 1, 20))
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#202020")

    # Fixed subplot region keeps the exported map panel size consistent.
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.090, top=0.985)

    save_png_and_tiff(
        fig,
        OUTPUT_MAIN,
        dpi=OUTPUT_DPI,
        fixed_canvas=True,
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# =========================================================
# 7. Legend / style panel
# =========================================================
def draw_legend_only():
    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE, dpi=OUTPUT_DPI)
    fig.patch.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x, y, s, size=7.5, weight="normal", ha="left", va="center"):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontweight=weight,
            ha=ha,
            va=va,
            color="#111111",
        )

    def sample_line(x0, x1, y, color, linestyle="-", lw=1.2, alpha=1.0):
        ax.plot(
            [x0, x1],
            [y, y],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            dash_capstyle="round",
        )

    y = 0.955
    text(0.05, y, "Color and line-style encoding for UGV routes", size=8.5, weight="bold")

    y -= 0.055
    text(0.05, y, "UGV", size=6.8, weight="bold")
    text(0.32, y, "Local\nviewpoint path", size=6.2, weight="bold", ha="center")
    text(0.72, y, "Inter-tree\nroute segment", size=6.2, weight="bold", ha="center")

    y -= 0.078
    row_gap = 0.060

    for vid in range(4):
        color = ROUTE_COLORS[vid]
        style = ROUTE_STYLES[vid]

        text(0.05, y, f"UGV{vid + 1}", size=6.8)

        sample_line(
            0.22, 0.42, y,
            color=color,
            linestyle=style,
            lw=ROUTE_LOCAL_WIDTH[vid],
            alpha=0.90,
        )

        sample_line(
            0.60, 0.84, y,
            color=color,
            linestyle=style,
            lw=ROUTE_MAIN_WIDTH[vid],
            alpha=0.92,
        )

        y -= row_gap

    y -= 0.038
    text(0.05, y, "Scene elements", size=8.5, weight="bold")

    y -= 0.058
    ax.scatter(
        0.09, y,
        s=20,
        marker="o",
        facecolor=TREE_FACE,
        edgecolor=TREE_EDGE,
        linewidth=0.0,
        alpha=TREE_ALPHA,
    )
    text(0.18, y, "Trees", size=6.8)

    y -= 0.055
    obstacle = Circle(
        (0.09, y),
        0.028,
        facecolor=OBSTACLE_FILL,
        edgecolor=OBSTACLE_EDGE,
        linestyle=OBSTACLE_LINESTYLE,
        linewidth=1.05,
        alpha=0.85,
    )
    ax.add_patch(obstacle)
    text(0.18, y, "Obstacles", size=6.8)

    y -= 0.055
    ax.scatter(
        0.09, y,
        s=26,
        marker="^",
        facecolor="white",
        edgecolor=START_COLOR,
        linewidth=1.0,
    )
    text(0.18, y, "Start positions", size=6.8)

    y -= 0.055
    ax.scatter(
        0.09, y,
        s=20,
        marker="s",
        facecolor="white",
        edgecolor=END_COLOR,
        linewidth=0.95,
    )
    text(0.18, y, "End positions", size=6.8)

    y -= 0.083
    text(0.05, y, "Segment types", size=8.5, weight="bold")

    y -= 0.058
    sample_line(
        0.07, 0.32, y,
        color="#666666",
        linestyle="-",
        lw=1.00,
        alpha=0.88,
    )
    text(0.38, y, "Local viewpoint connection", size=6.6)

    y -= 0.058
    sample_line(
        0.07, 0.32, y,
        color="#3A3A3A",
        linestyle="-",
        lw=1.40,
        alpha=0.92,
    )
    text(0.38, y, "Inter-tree route segment", size=6.6)

    y -= 0.058
    ax.scatter(
        0.09,
        y,
        s=VIEWPOINT_SIZE_LEGEND,
        marker="o",
        facecolor=VIEWPOINT_COLOR,
        edgecolor=VIEWPOINT_COLOR,
        linewidth=0.25,
        alpha=1.0,
        zorder=5,
        clip_on=False,
    )
    text(0.18, y, "Observation waypoints", size=6.6)

    fig.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00)

    save_png_and_tiff(
        fig,
        OUTPUT_LEGEND,
        dpi=OUTPUT_DPI,
        fixed_canvas=True,
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# =========================================================
# 8. Main
# =========================================================
def main():
    scene, solution = load_data()

    draw_main_figure(scene, solution)
    draw_legend_only()

    print("Done. Complete route and style panel generated:")
    print(f"  PNG : {OUTPUT_MAIN.name}")
    print(f"  TIFF: {OUTPUT_MAIN.with_suffix('.tiff').name}")
    print(f"  PNG : {OUTPUT_LEGEND.name}")
    print(f"  TIFF: {OUTPUT_LEGEND.with_suffix('.tiff').name}")
    print(f"  dpi = {OUTPUT_DPI}")


if __name__ == "__main__":
    main()