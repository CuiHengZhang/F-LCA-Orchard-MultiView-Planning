from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw

import draw_supplementary_complete_route as base


# =========================================================
# 1. Basic settings
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

OUT_A = BASE_DIR / "fig5a_task_allocation_overview"
OUT_B = BASE_DIR / "fig5b_local_multiview_route_realization"
OUT_C = BASE_DIR / "fig5c_obstacle_aware_local_organization"
OUT_D = BASE_DIR / "fig5d_planning_summary"

OUTPUT_DPI = 600
SHOW_FIGURES = False

MAP_FIGSIZE = base.MAP_FIGSIZE
SUMMARY_FIGSIZE = base.MAP_FIGSIZE

# No internal titles in Fig. 5a–5c. Add subfigure titles in Word/manuscript.
ADD_INTERNAL_TITLES = False

# Panel (b): dense local multi-view route region
PANEL_B_XLIM = (7.0, 28.0)
PANEL_B_YLIM = (6.0, 27.0)

# Panel (c): focused obstacle-aware local route region
PANEL_C_XLIM = (84.0, 104.0)
PANEL_C_YLIM = (54.0, 74.0)

LOCAL_TREE_SIZE = 18
LOCAL_VIEWPOINT_SIZE = 7.5

REGION_ALPHA = 0.075
REGION_EDGE_ALPHA = 0.28

HIGHLIGHT_UNIT_FACE = "#FFF2CC"
HIGHLIGHT_UNIT_EDGE = "#7A5C00"
HIGHLIGHT_UNIT_ALPHA = 0.36

MUTED_ROUTE_COLOR = "#B8B8B8"
MUTED_ROUTE_ALPHA = 0.30

FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans"]

D_TITLE_FS = 15.0
D_SECTION_FS = 13.0
D_HEADER_FS = 11.6
D_BODY_FS = 11.4

# UGV icon sizes
# Main map icon is deliberately small; legend icon remains visible.
UGV_MAP_ZOOM = 0.052
UGV_LEGEND_ZOOM = 0.135

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# 2. Utility functions
# =========================================================
def save_figure_pair(fig, output_stem, dpi=600):
    """
    Save each panel as both PNG and TIFF at 600 dpi.
    """
    output_stem = Path(output_stem)
    png_path = output_stem.with_suffix(".png")
    tiff_path = output_stem.with_suffix(".tiff")

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
        pil_kwargs={"compression": "tiff_lzw"},
    )


def vehicle_id_key(item):
    return int(item[0])


def inside_window(x, y, xlim, ylim, margin=0.0):
    return (
        xlim[0] - margin <= x <= xlim[1] + margin
        and ylim[0] - margin <= y <= ylim[1] + margin
    )


def segment_clip_to_window(p0, p1, xlim, ylim):
    """
    Clip a segment to a rectangular window using Liang-Barsky.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    xmin, xmax = xlim
    ymin, ymax = ylim

    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
        else:
            r = qi / pi
            if pi < 0:
                if r > u2:
                    return None
                if r > u1:
                    u1 = r
            else:
                if r < u1:
                    return None
                if r < u2:
                    u2 = r

    cx0 = x0 + u1 * dx
    cy0 = y0 + u1 * dy
    cx1 = x0 + u2 * dx
    cy1 = y0 + u2 * dy

    if np.hypot(cx1 - cx0, cy1 - cy0) < 1e-6:
        return None

    return np.array([cx0, cy0]), np.array([cx1, cy1])


def get_vehicle_groups(solution):
    """
    Each tree task is rendered as four observation waypoints.
    Group each vehicle's path points into four-view task units.
    """
    groups_by_vid = {}

    for vid_str, veh in sorted(solution["vehicles"].items(), key=vehicle_id_key):
        vid = int(vid_str)
        pts = np.array(veh["path_points"], dtype=float)

        if len(pts) < 5:
            groups_by_vid[vid] = []
            continue

        grouped_pts = pts[1:]
        n_groups = len(grouped_pts) // 4

        groups = []
        for gi in range(n_groups):
            group = grouped_pts[gi * 4:(gi + 1) * 4]
            if len(group) == 4:
                groups.append(group)

        groups_by_vid[vid] = groups

    return groups_by_vid


def group_center(group):
    return np.mean(group, axis=0)


def convex_hull(points):
    """
    Monotonic-chain convex hull. No scipy dependency.
    """
    pts = sorted(set((float(x), float(y)) for x, y in points))

    if len(pts) <= 1:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def configure_map_axes(ax, xlim, ylim, tick_step=20):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")

    x_start = np.ceil(xlim[0] / tick_step) * tick_step
    x_end = np.floor(xlim[1] / tick_step) * tick_step
    y_start = np.ceil(ylim[0] / tick_step) * tick_step
    y_end = np.floor(ylim[1] / tick_step) * tick_step

    ax.set_xticks(np.arange(x_start, x_end + 0.1, tick_step))
    ax.set_yticks(np.arange(y_start, y_end + 0.1, tick_step))
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#202020")


def internal_title(ax, title):
    if ADD_INTERNAL_TITLES:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)


# =========================================================
# 3. UGV icon
# =========================================================
def create_ugv_icon_rgba(size=360):
    """
    Create a transparent RGBA UGV-like red vehicle icon in memory.
    The front of the vehicle points to the right in the unrotated icon.
    """
    scale = 4
    w = size * scale
    h = int(size * 0.62) * scale

    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    def sx(v):
        return int(v * scale)

    def sy(v):
        return int(v * scale)

    def rr(box, radius, fill, outline=None, width=1):
        draw.rounded_rectangle(
            [int(v) for v in box],
            radius=int(radius),
            fill=fill,
            outline=outline,
            width=int(width),
        )

    W = size
    H = int(size * 0.62)

    # Tires: top and bottom sides
    tire_color = (18, 18, 18, 255)
    tire_edge = (0, 0, 0, 255)
    tires = [
        (0.20 * W, 0.03 * H, 0.34 * W, 0.18 * H),
        (0.66 * W, 0.03 * H, 0.80 * W, 0.18 * H),
        (0.20 * W, 0.82 * H, 0.34 * W, 0.97 * H),
        (0.66 * W, 0.82 * H, 0.80 * W, 0.97 * H),
    ]

    for box in tires:
        rr(
            [sx(box[0]), sy(box[1]), sx(box[2]), sy(box[3])],
            sx(0.025 * W),
            fill=tire_color,
            outline=tire_edge,
            width=sx(0.006 * W),
        )

    # Main body
    body_box = (0.12 * W, 0.16 * H, 0.88 * W, 0.84 * H)
    rr(
        [sx(body_box[0]), sy(body_box[1]), sx(body_box[2]), sy(body_box[3])],
        sx(0.055 * W),
        fill=(214, 40, 40, 255),
        outline=(120, 15, 15, 255),
        width=sx(0.012 * W),
    )

    # Slightly brighter front hood on the right
    rr(
        [sx(0.62 * W), sy(0.24 * H), sx(0.86 * W), sy(0.76 * H)],
        sx(0.035 * W),
        fill=(232, 58, 58, 255),
        outline=None,
        width=1,
    )

    # Cabin / main white window
    rr(
        [sx(0.30 * W), sy(0.28 * H), sx(0.58 * W), sy(0.72 * H)],
        sx(0.020 * W),
        fill=(255, 255, 255, 255),
        outline=(200, 200, 200, 255),
        width=sx(0.006 * W),
    )

    # Front white windshield, closer to the right/front
    rr(
        [sx(0.63 * W), sy(0.34 * H), sx(0.78 * W), sy(0.66 * H)],
        sx(0.018 * W),
        fill=(255, 255, 255, 255),
        outline=(200, 200, 200, 255),
        width=sx(0.006 * W),
    )

    # Window detail lines
    line_color = (70, 70, 70, 255)
    draw.line(
        [sx(0.35 * W), sy(0.50 * H), sx(0.53 * W), sy(0.50 * H)],
        fill=line_color,
        width=sx(0.009 * W),
    )
    draw.line(
        [sx(0.66 * W), sy(0.50 * H), sx(0.75 * W), sy(0.50 * H)],
        fill=line_color,
        width=sx(0.009 * W),
    )

    # Front bumper: right side, makes front direction clear
    rr(
        [sx(0.86 * W), sy(0.37 * H), sx(0.92 * W), sy(0.63 * H)],
        sx(0.012 * W),
        fill=(35, 35, 35, 255),
        outline=None,
        width=1,
    )

    # Front headlights: right side
    rr(
        [sx(0.80 * W), sy(0.26 * H), sx(0.86 * W), sy(0.35 * H)],
        sx(0.010 * W),
        fill=(250, 245, 190, 255),
        outline=(185, 180, 130, 255),
        width=sx(0.004 * W),
    )
    rr(
        [sx(0.80 * W), sy(0.65 * H), sx(0.86 * W), sy(0.74 * H)],
        sx(0.010 * W),
        fill=(250, 245, 190, 255),
        outline=(185, 180, 130, 255),
        width=sx(0.004 * W),
    )


    # Roof sensor
    rr(
        [sx(0.42 * W), sy(0.08 * H), sx(0.58 * W), sy(0.15 * H)],
        sx(0.010 * W),
        fill=(45, 45, 45, 255),
        outline=None,
        width=1,
    )

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    img = img.resize((size, int(size * 0.62)), resample=resample)
    return img


_UGV_ICON_CACHE = {}


def get_ugv_icon(angle=0):
    """
    Return a rotated RGBA icon as a numpy array.
    """
    key = int(round(angle))
    if key in _UGV_ICON_CACHE:
        return _UGV_ICON_CACHE[key]

    base_img = create_ugv_icon_rgba(size=360)

    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC

    rotated = base_img.rotate(angle, expand=True, resample=resample)
    arr = np.asarray(rotated)
    _UGV_ICON_CACHE[key] = arr
    return arr


def draw_ugv_icon(ax, x, y, zoom=0.7, angle=0, coord="data", zorder=30):
    """
    Draw an image-based UGV icon.
    """
    icon = get_ugv_icon(angle=angle)
    image = OffsetImage(icon, zoom=zoom)
    xycoords = ax.transData if coord == "data" else ax.transAxes

    ab = AnnotationBbox(
        image,
        (x, y),
        xycoords=xycoords,
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0.0,
        zorder=zorder,
    )
    ax.add_artist(ab)


def start_label_spec_custom(x, y, width, height):
    """
    Put S labels close to the start triangles.
    These labels are intentionally separated from the UGV icons.
    """
    if x < width / 2 and y < height / 2:          # bottom-left
        return 1.4, 1.6, "left", "bottom"
    elif x >= width / 2 and y < height / 2:       # bottom-right
        return -1.4, 1.6, "right", "bottom"
    elif x < width / 2 and y >= height / 2:       # top-left
        return 1.4, -1.0, "left", "top"
    else:                                         # top-right
        return -1.4, -1.0, "right", "top"


def start_icon_spec(x, y, width, height):
    """
    Inward offset and heading angle for each start-corner UGV icon.
    The unrotated icon faces right.
    """
    if x < width / 2 and y < height / 2:          # bottom-left, face upper-right
        return 6.2, 4.4, 28
    elif x >= width / 2 and y < height / 2:       # bottom-right, face upper-left
        return -6.2, 4.4, 152
    elif x < width / 2 and y >= height / 2:       # top-left, face lower-right
        return 6.2, -4.4, -28
    else:                                         # top-right, face lower-left
        return -6.2, -4.4, -152


# =========================================================
# 4. Scene drawing
# =========================================================
def draw_scene_custom(
    ax,
    scene,
    tree_size=22,
    viewpoint_size=6.0,
    draw_viewpoints=True,
    obstacle_alpha=0.65,
):
    for obs in scene["obstacles"]:
        circle = Circle(
            (obs["cx"], obs["cy"]),
            obs["r"],
            facecolor=base.OBSTACLE_FILL,
            edgecolor=base.OBSTACLE_EDGE,
            linestyle=base.OBSTACLE_LINESTYLE,
            linewidth=1.10,
            alpha=obstacle_alpha,
            zorder=1,
        )
        ax.add_patch(circle)

    tree_x = [t["x"] for t in scene["trees"]]
    tree_y = [t["y"] for t in scene["trees"]]
    ax.scatter(
        tree_x,
        tree_y,
        s=tree_size,
        marker="o",
        facecolor=base.TREE_FACE,
        edgecolor=base.TREE_EDGE,
        linewidth=0.0,
        alpha=0.95,
        zorder=5,
    )

    if draw_viewpoints:
        obs_pts = []
        for pts in scene["tree_observation_points"].values():
            obs_pts.extend(pts)

        if obs_pts:
            obs_pts = np.array(obs_pts, dtype=float)
            ax.scatter(
                obs_pts[:, 0],
                obs_pts[:, 1],
                s=viewpoint_size,
                marker="o",
                facecolor=base.VIEWPOINT_COLOR,
                edgecolor="white",
                linewidth=0.18,
                alpha=0.68,
                zorder=6,
            )


# =========================================================
# 5. Route drawing
# =========================================================
def draw_routes(
    ax,
    solution,
    highlight_vid=None,
    muted_others=False,
    xlim=None,
    ylim=None,
    arrows=True,
):
    for vid_str, veh in sorted(solution["vehicles"].items(), key=vehicle_id_key):
        vid = int(vid_str)

        if highlight_vid is None:
            color = base.ROUTE_COLORS[vid]
            style = base.ROUTE_STYLES[vid]
            alpha_main = 0.94
            alpha_local = 0.92
            main_lw = base.ROUTE_MAIN_WIDTH[vid]
            local_lw = base.ROUTE_LOCAL_WIDTH[vid]
            arrow_lw = base.ROUTE_ARROW_WIDTH[vid]
        else:
            if vid == highlight_vid:
                color = base.ROUTE_COLORS[vid]
                style = base.ROUTE_STYLES[vid]
                alpha_main = 0.96
                alpha_local = 0.94
                main_lw = base.ROUTE_MAIN_WIDTH[vid] + 0.10
                local_lw = base.ROUTE_LOCAL_WIDTH[vid] + 0.08
                arrow_lw = base.ROUTE_ARROW_WIDTH[vid]
            else:
                if not muted_others:
                    continue
                color = MUTED_ROUTE_COLOR
                style = "-"
                alpha_main = MUTED_ROUTE_ALPHA
                alpha_local = MUTED_ROUTE_ALPHA
                main_lw = 0.65
                local_lw = 0.50
                arrow_lw = 0.50

        pts = np.array(veh["path_points"], dtype=float)
        if len(pts) < 5:
            continue

        start = pts[0]
        grouped_pts = pts[1:]
        n_groups = len(grouped_pts) // 4
        prev = start

        for gi in range(n_groups):
            group = grouped_pts[gi * 4:(gi + 1) * 4]
            if len(group) < 4:
                continue

            should_draw_group = True
            if xlim is not None and ylim is not None:
                should_draw_group = any(
                    inside_window(x, y, xlim, ylim, margin=0.30)
                    for x, y in group
                )

            if should_draw_group:
                base.plot_line(
                    ax,
                    group[:, 0],
                    group[:, 1],
                    color=color,
                    linestyle=style,
                    linewidth=local_lw,
                    alpha=alpha_local,
                    zorder=8,
                )

            p0, p1 = prev, group[0]

            if xlim is not None and ylim is not None:
                clipped = segment_clip_to_window(p0, p1, xlim, ylim)
                if clipped is None:
                    prev = group[-1]
                    continue
                p0, p1 = clipped

            base.plot_line(
                ax,
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                linestyle=style,
                linewidth=main_lw,
                alpha=alpha_main,
                zorder=7,
            )

            if arrows:
                dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
                if dist >= 4.5:
                    base.add_arrow(
                        ax,
                        p0,
                        p1,
                        color=color,
                        linestyle=style,
                        lw=arrow_lw,
                        alpha=alpha_main,
                        mutation_scale=7.0,
                    )

            prev = group[-1]


def draw_start_end_markers(ax, scene, solution, xlim=None, ylim=None):
    width = scene["width"]
    height = scene["height"]

    starts = np.array(scene["start_positions"], dtype=float)

    for i, (x, y) in enumerate(starts):
        if xlim is not None and ylim is not None:
            if not inside_window(x, y, xlim, ylim, margin=1.5):
                continue

        # Start triangle
        ax.scatter(
            x,
            y,
            s=base.START_MARKER_SIZE * 0.82,
            marker="^",
            facecolor="white",
            edgecolor=base.START_COLOR,
            linewidth=1.25,
            zorder=12,
            clip_on=False if xlim is None else True,
        )

        # Start label, separated from UGV icon
        dx, dy, ha, va = start_label_spec_custom(x, y, width, height)
        ax.text(
            x + dx,
            y + dy,
            f"S{i + 1}",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
            ha=ha,
            va=va,
            zorder=30,
            clip_on=False if xlim is None else True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.12),
        )

        # UGV icon only in full-map panel
        if xlim is None and ylim is None:
            ix, iy, ang = start_icon_spec(x, y, width, height)
            draw_ugv_icon(
                ax,
                x + ix,
                y + iy,
                zoom=UGV_MAP_ZOOM,
                angle=ang,
                coord="data",
                zorder=14,
            )

    # End positions
    for vid_str, veh in sorted(solution["vehicles"].items(), key=vehicle_id_key):
        vid = int(vid_str)
        pts = np.array(veh["path_points"], dtype=float)

        if len(pts) == 0:
            continue

        x, y = pts[-1]

        if xlim is not None and ylim is not None:
            if not inside_window(x, y, xlim, ylim, margin=1.5):
                continue

        ax.scatter(
            x,
            y,
            s=base.END_MARKER_SIZE * 0.82,
            marker="s",
            facecolor="white",
            edgecolor=base.END_COLOR,
            linewidth=1.0,
            zorder=12,
        )

        dx, dy, ha, va = base.end_label_spec(vid)
        ax.text(
            x + dx,
            y + dy,
            f"E{vid + 1}",
            fontsize=9.5,
            fontweight="bold",
            color=base.END_TEXT_COLOR,
            ha=ha,
            va=va,
            zorder=13,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.15),
        )


# =========================================================
# 6. Panel (a): task allocation overview
# =========================================================
def draw_panel_a(scene, solution, groups_by_vid):
    width = scene["width"]
    height = scene["height"]

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=OUTPUT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for vid, groups in groups_by_vid.items():
        centers = np.array([group_center(g) for g in groups], dtype=float)

        if len(centers) >= 3:
            hull = convex_hull(centers)

            if len(hull) >= 3:
                poly = Polygon(
                    hull,
                    closed=True,
                    facecolor=base.ROUTE_COLORS[vid],
                    edgecolor=base.ROUTE_COLORS[vid],
                    linewidth=0.8,
                    alpha=REGION_ALPHA,
                    zorder=0,
                )
                ax.add_patch(poly)

                closed_hull = np.vstack([hull, hull[0]])
                ax.plot(
                    closed_hull[:, 0],
                    closed_hull[:, 1],
                    color=base.ROUTE_COLORS[vid],
                    linewidth=0.8,
                    alpha=REGION_EDGE_ALPHA,
                    zorder=0.5,
                )

    draw_scene_custom(
        ax,
        scene,
        tree_size=20,
        viewpoint_size=5.4,
        draw_viewpoints=True,
        obstacle_alpha=0.66,
    )

    draw_routes(ax, solution, arrows=True)
    draw_start_end_markers(ax, scene, solution)

    boxes = [
        {"xlim": PANEL_B_XLIM, "ylim": PANEL_B_YLIM},
        {"xlim": PANEL_C_XLIM, "ylim": PANEL_C_YLIM},
    ]

    for box in boxes:
        x0, x1 = box["xlim"]
        y0, y1 = box["ylim"]

        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="none",
            edgecolor="#303030",
            linewidth=1.05,
            linestyle=(0, (5.0, 2.4)),
            alpha=0.86,
            zorder=20,
        )
        ax.add_patch(rect)

    configure_map_axes(ax, (-1.5, width + 1.5), (-1.5, height + 1.5), tick_step=20)
    internal_title(ax, "Task allocation overview")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.090, top=0.985)
    save_figure_pair(fig, OUT_A, dpi=OUTPUT_DPI)

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# =========================================================
# 7. Panel (b): local multi-view route realization
# =========================================================
def draw_panel_b(scene, solution, groups_by_vid):
    xlim = PANEL_B_XLIM
    ylim = PANEL_B_YLIM

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=OUTPUT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_scene_custom(
        ax,
        scene,
        tree_size=LOCAL_TREE_SIZE,
        viewpoint_size=LOCAL_VIEWPOINT_SIZE,
        draw_viewpoints=True,
        obstacle_alpha=0.45,
    )

    draw_routes(
        ax,
        solution,
        highlight_vid=0,
        muted_others=True,
        xlim=xlim,
        ylim=ylim,
        arrows=True,
    )

    groups = groups_by_vid.get(0, [])
    selected = []

    # Highlight all complete four-view observation units inside the zoomed region.
    # A unit is highlighted only when all four observation waypoints lie within
    # the local window. This avoids implying that only a few trees have
    # four-view observation units.
    for g in groups:
        if all(inside_window(px, py, xlim, ylim, margin=0.0) for px, py in g):
            selected.append(g)

    for g in selected:
        poly = Polygon(
            g,
            closed=True,
            facecolor=HIGHLIGHT_UNIT_FACE,
            edgecolor=HIGHLIGHT_UNIT_EDGE,
            linewidth=0.70,
            linestyle=(0, (3.0, 2.0)),
            alpha=0.28,
            zorder=6.8,
        )
        ax.add_patch(poly)

    configure_map_axes(ax, xlim, ylim, tick_step=5)
    internal_title(ax, "Local multi-view route realization")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.090, top=0.985)
    save_figure_pair(fig, OUT_B, dpi=OUTPUT_DPI)

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# =========================================================
# 8. Panel (c): obstacle-aware local organization
# =========================================================
def draw_panel_c(scene, solution, groups_by_vid):
    xlim = PANEL_C_XLIM
    ylim = PANEL_C_YLIM

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=OUTPUT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_scene_custom(
        ax,
        scene,
        tree_size=LOCAL_TREE_SIZE,
        viewpoint_size=LOCAL_VIEWPOINT_SIZE,
        draw_viewpoints=True,
        obstacle_alpha=0.62,
    )

    draw_routes(
        ax,
        solution,
        highlight_vid=1,
        muted_others=True,
        xlim=xlim,
        ylim=ylim,
        arrows=True,
    )

    groups = groups_by_vid.get(1, [])
    selected = []

    # Highlight all complete four-view observation units inside the obstacle-aware
    # local window as well, but with a lighter style than panel (b) so the
    # obstacle-route relationship remains the visual focus.
    for g in groups:
        if all(inside_window(px, py, xlim, ylim, margin=0.0) for px, py in g):
            selected.append(g)

    for g in selected:
        poly = Polygon(
            g,
            closed=True,
            facecolor=HIGHLIGHT_UNIT_FACE,
            edgecolor=HIGHLIGHT_UNIT_EDGE,
            linewidth=0.65,
            linestyle=(0, (3.0, 2.0)),
            alpha=0.22,
            zorder=6.7,
        )
        ax.add_patch(poly)

    configure_map_axes(ax, xlim, ylim, tick_step=5)
    internal_title(ax, "Obstacle-aware local organization")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.090, top=0.985)
    save_figure_pair(fig, OUT_C, dpi=OUTPUT_DPI)

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# =========================================================
# 9. Panel (d): planning summary and symbol encoding
# =========================================================
def draw_panel_d(scene):
    fig, ax = plt.subplots(figsize=SUMMARY_FIGSIZE, dpi=OUTPUT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x, y, s, size=D_BODY_FS, weight="normal", ha="left", va="center", color="#111111"):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontweight=weight,
            ha=ha,
            va=va,
            color=color,
            family="sans-serif",
        )

    def sample_line(x0, x1, y, color, linestyle="-", lw=2.2, alpha=1.0):
        ax.plot(
            [x0, x1],
            [y, y],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            dash_capstyle="round",
            transform=ax.transAxes,
        )

    # Main title retained as part of the legend/summary panel
    text(0.05, 0.955, "Planning summary and symbol encoding", size=D_TITLE_FS, weight="bold")

    text(0.05, 0.885, "Representative case", size=D_SECTION_FS, weight="bold")

    # The legend text has been enlarged for manuscript readability.
    # Therefore the representative-case values are arranged with wider spacing
    # and the coverage item is placed on a separate row to avoid overlap.
    y = 0.835
    text(0.07, y, "Seed:", size=D_BODY_FS, weight="semibold")
    text(0.155, y, "44", size=D_BODY_FS)

    text(0.29, y, "UGVs:", size=D_BODY_FS, weight="semibold")
    text(0.390, y, "4", size=D_BODY_FS)

    text(0.52, y, "Trees:", size=D_BODY_FS, weight="semibold")
    text(0.625, y, "80", size=D_BODY_FS)

    y = 0.790
    text(0.07, y, "Required viewpoints:", size=D_BODY_FS, weight="semibold")
    text(0.355, y, "320", size=D_BODY_FS)

    y = 0.745
    text(0.07, y, "Theoretical / effective coverage:", size=D_BODY_FS, weight="semibold")
    text(0.555, y, "100% / 100%", size=D_BODY_FS)

    text(0.05, 0.670, "Color and line-style encoding for UGV routes", size=D_SECTION_FS, weight="bold")

    head_y = 0.610
    text(0.05, head_y, "UGV", size=D_HEADER_FS, weight="bold")
    text(0.30, head_y, "Local viewpoint path", size=D_HEADER_FS, weight="bold", ha="center")
    text(0.72, head_y, "Inter-tree route segment", size=D_HEADER_FS, weight="bold", ha="center")

    row_y = 0.550
    row_gap = 0.063

    for vid in range(4):
        color = base.ROUTE_COLORS[vid]
        style = base.ROUTE_STYLES[vid]

        text(0.05, row_y, f"UGV{vid + 1}", size=D_BODY_FS)

        sample_line(
            0.18, 0.43, row_y,
            color=color,
            linestyle=style,
            lw=max(2.15, base.ROUTE_LOCAL_WIDTH[vid] + 0.75),
            alpha=1.0,
        )

        sample_line(
            0.58, 0.86, row_y,
            color=color,
            linestyle=style,
            lw=max(2.35, base.ROUTE_MAIN_WIDTH[vid] + 0.75),
            alpha=1.0,
        )

        row_y -= row_gap

    text(0.05, 0.285, "Scene elements", size=D_SECTION_FS, weight="bold")

    # Left column
    y = 0.225
    draw_ugv_icon(
        ax,
        0.09,
        y,
        zoom=UGV_LEGEND_ZOOM,
        angle=0,
        coord="axes",
        zorder=20,
    )
    text(0.16, y, "UGV", size=D_BODY_FS)

    y = 0.155
    ax.scatter(
        0.09, y,
        s=70,
        marker="o",
        facecolor=base.TREE_FACE,
        edgecolor=base.TREE_EDGE,
        linewidth=0.0,
        transform=ax.transAxes,
    )
    text(0.16, y, "Trees", size=D_BODY_FS)

    y = 0.085
    obstacle = Circle(
        (0.09, y),
        0.033,
        facecolor=base.OBSTACLE_FILL,
        edgecolor=base.OBSTACLE_EDGE,
        linestyle=base.OBSTACLE_LINESTYLE,
        linewidth=1.15,
        transform=ax.transAxes,
    )
    ax.add_patch(obstacle)
    text(0.16, y, "Obstacles", size=D_BODY_FS)

    # Right column
    y = 0.225
    ax.scatter(
        0.52, y,
        s=38,
        marker="o",
        facecolor=base.VIEWPOINT_COLOR,
        edgecolor="white",
        linewidth=0.45,
        transform=ax.transAxes,
    )
    text(0.59, y, "Observation waypoints", size=D_BODY_FS)

    y = 0.155
    ax.scatter(
        0.52, y,
        s=78,
        marker="^",
        facecolor="white",
        edgecolor=base.START_COLOR,
        linewidth=1.25,
        transform=ax.transAxes,
    )
    text(0.59, y, "Start positions", size=D_BODY_FS)

    y = 0.085
    ax.scatter(
        0.52, y,
        s=70,
        marker="s",
        facecolor="white",
        edgecolor=base.END_COLOR,
        linewidth=1.20,
        transform=ax.transAxes,
    )
    text(0.59, y, "End positions", size=D_BODY_FS)

    fig.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00)
    save_figure_pair(fig, OUT_D, dpi=OUTPUT_DPI)

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# =========================================================
# 10. Main
# =========================================================
def main():
    scene, solution = base.load_data()
    groups_by_vid = get_vehicle_groups(solution)

    draw_panel_a(scene, solution, groups_by_vid)
    draw_panel_b(scene, solution, groups_by_vid)
    draw_panel_c(scene, solution, groups_by_vid)
    draw_panel_d(scene)

    print("Done. Generated 4 panels in both PNG and TIFF formats:")
    for stem in [OUT_A, OUT_B, OUT_C, OUT_D]:
        print(f"  {stem.with_suffix('.png').name}")
        print(f"  {stem.with_suffix('.tiff').name}")
    print(f"dpi = {OUTPUT_DPI}")


if __name__ == "__main__":
    main()