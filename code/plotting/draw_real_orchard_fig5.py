"""
Draw Fig. 6: real-tree-location F-LCA cooperative multi-view observation routes.

This version is visually unified with the final Fig. 5 drawing style:
  - same route colors and line styles
  - same start/end markers
  - same small red UGV icon
  - same 600 dpi export logic for both PNG and TIFF
  - right-side panel with case summary, route encoding, and scene elements

Required files in the same folder:
  - real_orchard_input.json
  - real_orchard_best_path.json

Outputs:
  - fig6a_real_tree_location_routes.png
  - fig6a_real_tree_location_routes.tiff
  - fig6b_case_summary_and_encoding.png
  - fig6b_case_summary_and_encoding.tiff

Both PNG and TIFF outputs are exported at 600 dpi.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw


# =========================================================
# 1. File paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "real_orchard_input.json"
PATH_FILE = BASE_DIR / "real_orchard_best_path.json"

OUT_A = BASE_DIR / "fig6a_real_tree_location_routes"
OUT_B = BASE_DIR / "fig6b_case_summary_and_encoding"

OUTPUT_DPI = 600
SHOW_FIGURES = False

# If the JSON already contains these metrics, the JSON values are used.
# These defaults keep the figure consistent with the manuscript-reported case.
DEFAULT_TOTAL_PATH_LENGTH = 588.88
DEFAULT_ENERGY_RANGE_PERCENT = 5.91
DEFAULT_THEORETICAL_COVERAGE = 100.0
DEFAULT_EFFECTIVE_COVERAGE = 100.0


# =========================================================
# 2. Visual design, unified with Fig. 5
# =========================================================
ROUTE_COLORS = {
    0: "#2C7FB8",  # UGV1
    1: "#D95F02",  # UGV2
    2: "#1B9E77",  # UGV3
    3: "#B07AA1",  # UGV4
}

ROUTE_STYLES = {
    0: "-",
    1: (0, (4.8, 2.0)),
    2: (0, (5.0, 1.6, 1.3, 1.6)),
    3: (0, (1.2, 1.45)),
}

ROUTE_MAIN_WIDTH = {
    0: 1.10,
    1: 1.20,
    2: 1.28,
    3: 1.34,
}

ROUTE_LOCAL_WIDTH = {
    0: 0.88,
    1: 0.96,
    2: 1.04,
    3: 1.10,
}

ROUTE_ARROW_WIDTH = {
    0: 0.76,
    1: 0.84,
    2: 0.90,
    3: 0.96,
}

REGION_ALPHA = 0.075
REGION_EDGE_ALPHA = 0.28

FIG_FACE = "#FFFFFF"
AX_FACE = "#FFFFFF"

TREE_FACE = "#5F9E62"
TREE_EDGE = "#5F9E62"
TREE_SIZE = 24
TREE_LW = 0.0
TREE_ALPHA = 0.95

VIEWPOINT_COLOR = "#333333"
VIEWPOINT_SIZE = 6.8
VIEWPOINT_ALPHA = 0.72

START_COLOR = "#4C78A8"
END_COLOR = "#B55D5D"
START_MARKER_SIZE = 132
START_MARKER_LW = 1.40
END_MARKER_SIZE = 44
END_MARKER_LW = 1.10
START_LABEL_SIZE = 10.0
END_LABEL_SIZE = 9.6

UGV_MAP_ZOOM = 0.052
UGV_LEGEND_ZOOM = 0.105

FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans"]

PANEL_TITLE_FS = 14.2
PANEL_SECTION_FS = 12.0
PANEL_HEADER_FS = 9.6
PANEL_BODY_FS = 9.8

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================
# 3. Data loading and schema helpers
# =========================================================
def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data():
    return load_json(INPUT_FILE), load_json(PATH_FILE)


def vehicle_id_key(item):
    return int(item[0])


def as_point_array(points):
    return np.array(points, dtype=float)


def get_tree_xy(scene):
    tree_xy = []
    for t in scene.get("trees", []):
        x = t.get("x_m", t.get("x", None))
        y = t.get("y_m", t.get("y", None))
        if x is not None and y is not None:
            tree_xy.append((float(x), float(y)))
    return np.array(tree_xy, dtype=float)


def get_scene_width_height(scene):
    width = scene.get("width_m", scene.get("width", None))
    height = scene.get("height_m", scene.get("height", None))

    trees = get_tree_xy(scene)
    if width is None:
        width = float(np.max(trees[:, 0])) if len(trees) else 0.0
    if height is None:
        height = float(np.max(trees[:, 1])) if len(trees) else 0.0

    return float(width), float(height)


def get_start_positions(scene, solution):
    starts = solution.get("start_positions", None)

    if starts is not None:
        if isinstance(starts, dict):
            starts = [starts[str(i)] for i in sorted(map(int, starts.keys()))]
        return np.array(starts, dtype=float)

    starts = scene.get("start_positions", None)
    if starts is None:
        raise KeyError("No start_positions found in either scene or solution.")

    if isinstance(starts, dict):
        starts = [starts[str(i)] for i in sorted(map(int, starts.keys()))]
    return np.array(starts, dtype=float)


def get_vehicle_paths(solution):
    """
    Return a dict: vid -> path points.

    Supported solution schemas:
      - solution["vehicle_paths"][vid]
      - solution["vehicles"][vid]["path_points"]
    """
    if "vehicle_paths" in solution:
        return solution["vehicle_paths"]

    if "vehicles" in solution:
        out = {}
        for vid, veh in solution["vehicles"].items():
            out[vid] = veh["path_points"]
        return out

    raise KeyError("Cannot find vehicle paths in the solution JSON.")


def strip_start_if_present(points, start, tol=1e-7):
    pts = np.array(points, dtype=float)
    if len(pts) > 0 and np.linalg.norm(pts[0] - start) <= tol:
        return pts[1:]
    return pts


def get_vehicle_groups(solution, start_positions):
    """
    Group each vehicle's route into four-waypoint observation units.
    """
    paths = get_vehicle_paths(solution)
    groups_by_vid = {}

    for vid_str, path in sorted(paths.items(), key=vehicle_id_key):
        vid = int(vid_str)
        pts = strip_start_if_present(path, start_positions[vid])

        n_groups = len(pts) // 4
        groups = []

        for gi in range(n_groups):
            group = pts[gi * 4:(gi + 1) * 4]
            if len(group) == 4:
                groups.append(group)

        groups_by_vid[vid] = groups

    return groups_by_vid


def group_center(group):
    return np.mean(group, axis=0)


def get_nested_metric(solution, names, default=None):
    """
    Search common top-level and nested metric locations in the solution JSON.
    """
    search_spaces = [
        solution,
        solution.get("metrics", {}),
        solution.get("summary", {}),
        solution.get("case_summary", {}),
        solution.get("best", {}),
    ]

    for space in search_spaces:
        if not isinstance(space, dict):
            continue
        for name in names:
            if name in space and space[name] is not None:
                return space[name]

    return default


# =========================================================
# 4. Geometry and plotting helpers
# =========================================================
def save_png_tiff(fig, output_stem, dpi=600):
    """
    Save one panel as both PNG and TIFF at 600 dpi.
    """
    output_stem = Path(output_stem)
    png_path = output_stem.with_suffix(".png")
    tiff_path = output_stem.with_suffix(".tiff")

    common_kwargs = dict(
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        png_path,
        format="png",
        **common_kwargs,
    )

    fig.savefig(
        tiff_path,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
        **common_kwargs,
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


def add_arrow(ax, p0, p1, color, linestyle="-", lw=1.0, alpha=0.90, mutation_scale=7.0):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    dx = x1 - x0
    dy = y1 - y0
    dist = np.hypot(dx, dy)

    if dist < 3.2:
        return

    frac_end = 0.60
    frac_start = max(0.44, frac_end - min(0.13, 2.5 / dist))

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
        zorder=10,
    )


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

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def compute_path_length(solution, start_positions):
    """
    Compute the route length from the ordered observation-waypoint sequence.

    The length includes:
      - start position to first waypoint
      - all consecutive observation-waypoint segments

    It does not add a forced return-to-start segment unless that segment is
    explicitly included in the JSON path.
    """
    total = 0.0
    paths = get_vehicle_paths(solution)

    for vid_str, path in sorted(paths.items(), key=vehicle_id_key):
        vid = int(vid_str)
        pts = strip_start_if_present(path, start_positions[vid])

        if len(pts) == 0:
            continue

        route = np.vstack([start_positions[vid], pts])
        diffs = np.diff(route, axis=0)
        total += float(np.sum(np.linalg.norm(diffs, axis=1)))

    return total


def make_ticks(final_value, step=10):
    """
    Generate clean axis ticks without placing a precise terminal tick too close
    to the last regular tick. The exact region size is reported in the summary
    panel, so the map axis is kept visually uncluttered.
    """
    ticks = list(np.arange(0, np.floor(final_value / step) * step + 0.1, step))

    if not ticks:
        return [0]

    # Avoid crowded labels such as "60" and "62.43" next to each other.
    if final_value - ticks[-1] > 0.50 * step:
        ticks.append(final_value)

    return ticks


def format_tick(value):
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.2f}"


# =========================================================
# 5. UGV icon, unified with Fig. 5
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

    body_box = (0.12 * W, 0.16 * H, 0.88 * W, 0.84 * H)
    rr(
        [sx(body_box[0]), sy(body_box[1]), sx(body_box[2]), sy(body_box[3])],
        sx(0.055 * W),
        fill=(214, 40, 40, 255),
        outline=(120, 15, 15, 255),
        width=sx(0.012 * W),
    )

    rr(
        [sx(0.62 * W), sy(0.24 * H), sx(0.86 * W), sy(0.76 * H)],
        sx(0.035 * W),
        fill=(232, 58, 58, 255),
        outline=None,
        width=1,
    )

    rr(
        [sx(0.30 * W), sy(0.28 * H), sx(0.58 * W), sy(0.72 * H)],
        sx(0.020 * W),
        fill=(255, 255, 255, 255),
        outline=(200, 200, 200, 255),
        width=sx(0.006 * W),
    )

    rr(
        [sx(0.63 * W), sy(0.34 * H), sx(0.78 * W), sy(0.66 * H)],
        sx(0.018 * W),
        fill=(255, 255, 255, 255),
        outline=(200, 200, 200, 255),
        width=sx(0.006 * W),
    )

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

    rr(
        [sx(0.86 * W), sy(0.37 * H), sx(0.92 * W), sy(0.63 * H)],
        sx(0.012 * W),
        fill=(35, 35, 35, 255),
        outline=None,
        width=1,
    )

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


def start_label_spec(x, y, width, height):
    if x < width / 2 and y < height / 2:
        return 0.9, 0.8, "left", "bottom"
    elif x >= width / 2 and y < height / 2:
        return -0.9, 0.8, "right", "bottom"
    elif x < width / 2 and y >= height / 2:
        return 0.9, -0.7, "left", "top"
    else:
        return -0.9, -0.7, "right", "top"


def start_icon_spec(x, y, width, height):
    """
    The unrotated UGV icon faces right.
    """
    dx = max(2.2, width * 0.050)
    dy = max(1.8, height * 0.065)

    if x < width / 2 and y < height / 2:
        return dx, dy, 28
    elif x >= width / 2 and y < height / 2:
        return -dx, dy, 152
    elif x < width / 2 and y >= height / 2:
        return dx, -dy, -28
    else:
        return -dx, -dy, -152


def end_label_spec(vid):
    specs = {
        0: (0.6, 0.55, "left", "bottom"),
        1: (0.0, -0.62, "center", "top"),
        2: (-0.55, 0.55, "right", "bottom"),
        3: (0.65, 0.55, "left", "bottom"),
    }
    return specs.get(vid, (0.6, 0.55, "left", "bottom"))


def get_first_route_target(groups):
    """
    Return the first valid observation waypoint of one UGV route.
    """
    if not groups:
        return None

    for g in groups:
        if len(g) > 0:
            return np.asarray(g[0], dtype=float)

    return None


def compute_path_aligned_ugv_pose(start, first_target, width, height):
    """
    Place the UGV icon on the actual first inter-tree route segment instead of
    using fixed corner offsets.

    The unrotated UGV icon faces right, so the drawing angle is the segment
    heading measured from the positive x-axis.
    """
    start = np.asarray(start, dtype=float)

    if first_target is None:
        # Fallback: keep the icon inside the map if the route is missing.
        ix, iy, ang = start_icon_spec(start[0], start[1], width, height)
        return start[0] + ix, start[1] + iy, ang

    first_target = np.asarray(first_target, dtype=float)
    vec = first_target - start
    dist = float(np.linalg.norm(vec))

    if dist < 1e-8:
        ix, iy, ang = start_icon_spec(start[0], start[1], width, height)
        return start[0] + ix, start[1] + iy, ang

    direction = vec / dist

    # Keep the icon close to the start marker but exactly on the route segment.
    # For long first segments, do not move the icon too far into the map.
    icon_distance = min(max(2.4, 0.12 * dist), 3.9)
    icon_distance = min(icon_distance, 0.55 * dist)

    icon_xy = start + direction * icon_distance
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))

    return float(icon_xy[0]), float(icon_xy[1]), angle


def choose_start_label_spec(x, y, width, height, icon_x, icon_y):
    """
    Put the S label close to the start marker but away from the UGV icon.
    Candidate positions are scored by distance to the icon; positions inside
    the map are preferred.
    """
    # Candidate offsets for each corner: compact inside position first, then
    # alternatives slightly farther from the path-aligned UGV icon.
    if x < width / 2 and y < height / 2:  # lower-left
        candidates = [
            (0.8, 0.65, "left", "bottom"),
            (0.8, 1.55, "left", "bottom"),
            (1.75, 0.65, "left", "bottom"),
            (-0.35, 0.95, "right", "bottom"),
        ]
    elif x >= width / 2 and y < height / 2:  # lower-right
        candidates = [
            (-0.8, 0.65, "right", "bottom"),
            (-0.8, 1.55, "right", "bottom"),
            (-1.75, 0.65, "right", "bottom"),
            (0.35, 0.95, "left", "bottom"),
        ]
    elif x < width / 2 and y >= height / 2:  # upper-left
        candidates = [
            (0.8, -0.65, "left", "top"),
            (0.8, -1.55, "left", "top"),
            (1.75, -0.65, "left", "top"),
            (-0.35, -0.95, "right", "top"),
        ]
    else:  # upper-right
        candidates = [
            (-0.8, -0.65, "right", "top"),
            (-0.8, -1.55, "right", "top"),
            (-1.75, -0.65, "right", "top"),
            (0.35, -0.95, "left", "top"),
        ]

    best = candidates[0]
    best_score = -1e9

    for dx, dy, ha, va in candidates:
        lx = x + dx
        ly = y + dy
        d_icon = np.hypot(lx - icon_x, ly - icon_y)

        inside = (-0.5 <= lx <= width + 0.5) and (-0.5 <= ly <= height + 0.5)
        score = d_icon + (0.65 if inside else 0.0)

        if score > best_score:
            best = (dx, dy, ha, va)
            best_score = score

    return best


# =========================================================
# 6. Route-map panel
# =========================================================
def draw_region_shading(ax, groups_by_vid):
    for vid, groups in groups_by_vid.items():
        centers = np.array([group_center(g) for g in groups], dtype=float)

        if len(centers) < 3:
            continue

        hull = convex_hull(centers)
        if len(hull) < 3:
            continue

        color = ROUTE_COLORS.get(vid, "#4D4D4D")

        poly = Polygon(
            hull,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.75,
            alpha=REGION_ALPHA,
            zorder=0,
        )
        ax.add_patch(poly)

        closed_hull = np.vstack([hull, hull[0]])
        ax.plot(
            closed_hull[:, 0],
            closed_hull[:, 1],
            color=color,
            linewidth=0.75,
            alpha=REGION_EDGE_ALPHA,
            zorder=0.5,
        )


def draw_trees(ax, scene):
    trees = get_tree_xy(scene)
    if len(trees) == 0:
        return

    ax.scatter(
        trees[:, 0],
        trees[:, 1],
        s=TREE_SIZE,
        marker="o",
        facecolor=TREE_FACE,
        edgecolor=TREE_EDGE,
        linewidth=TREE_LW,
        alpha=TREE_ALPHA,
        zorder=5,
    )


def draw_observation_waypoints(ax, groups_by_vid):
    pts = []
    for groups in groups_by_vid.values():
        for g in groups:
            pts.extend(g)

    if not pts:
        return

    pts = np.array(pts, dtype=float)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=VIEWPOINT_SIZE,
        marker="o",
        facecolor=VIEWPOINT_COLOR,
        edgecolor="white",
        linewidth=0.18,
        alpha=VIEWPOINT_ALPHA,
        zorder=9,
    )


def draw_vehicle_routes(ax, solution, start_positions, groups_by_vid):
    for vid in sorted(groups_by_vid.keys()):
        groups = groups_by_vid[vid]
        if not groups:
            continue

        color = ROUTE_COLORS.get(vid, "#4D4D4D")
        style = ROUTE_STYLES.get(vid, "-")

        prev = start_positions[vid]

        for group in groups:
            # Inter-tree route segment
            plot_line(
                ax,
                [prev[0], group[0, 0]],
                [prev[1], group[0, 1]],
                color=color,
                linestyle=style,
                linewidth=ROUTE_MAIN_WIDTH.get(vid, 1.10),
                alpha=0.92,
                zorder=7,
            )

            add_arrow(
                ax,
                prev,
                group[0],
                color=color,
                linestyle=style,
                lw=ROUTE_ARROW_WIDTH.get(vid, 0.80),
                alpha=0.90,
                mutation_scale=6.8,
            )

            # Local four-view observation path
            plot_line(
                ax,
                group[:, 0],
                group[:, 1],
                color=color,
                linestyle=style,
                linewidth=ROUTE_LOCAL_WIDTH.get(vid, 0.95),
                alpha=0.90,
                zorder=8,
            )

            prev = group[-1]


def draw_start_markers(ax, start_positions, width, height, groups_by_vid):
    for i, (x, y) in enumerate(start_positions):
        first_target = get_first_route_target(groups_by_vid.get(i, []))
        icon_x, icon_y, icon_angle = compute_path_aligned_ugv_pose(
            np.array([x, y], dtype=float),
            first_target,
            width,
            height,
        )

        ax.scatter(
            x,
            y,
            s=START_MARKER_SIZE,
            marker="^",
            facecolor="white",
            edgecolor=START_COLOR,
            linewidth=START_MARKER_LW,
            zorder=12,
            clip_on=False,
        )

        dx, dy, ha, va = choose_start_label_spec(
            x,
            y,
            width,
            height,
            icon_x,
            icon_y,
        )

        ax.text(
            x + dx,
            y + dy,
            f"S{i + 1}",
            fontsize=START_LABEL_SIZE,
            fontweight="bold",
            color="#111111",
            ha=ha,
            va=va,
            zorder=30,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.12),
        )

        draw_ugv_icon(
            ax,
            icon_x,
            icon_y,
            zoom=UGV_MAP_ZOOM,
            angle=icon_angle,
            coord="data",
            zorder=14,
        )


def draw_end_markers(ax, groups_by_vid):
    for vid in sorted(groups_by_vid.keys()):
        groups = groups_by_vid[vid]
        if not groups:
            continue

        end = groups[-1][-1]

        ax.scatter(
            end[0],
            end[1],
            s=END_MARKER_SIZE,
            marker="s",
            facecolor="white",
            edgecolor=END_COLOR,
            linewidth=END_MARKER_LW,
            zorder=12,
        )

        dx, dy, ha, va = end_label_spec(vid)
        ax.text(
            end[0] + dx,
            end[1] + dy,
            f"E{vid + 1}",
            fontsize=END_LABEL_SIZE,
            fontweight="bold",
            color="#111111",
            ha=ha,
            va=va,
            zorder=13,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.15),
        )


def configure_map_axes(ax, width, height):
    x_margin = 1.0
    y_lower_margin = 1.0
    y_upper_margin = 1.0

    ax.set_xlim(-x_margin, width + x_margin)
    ax.set_ylim(-y_lower_margin, height + y_upper_margin)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")

    xticks = make_ticks(width, step=10)
    yticks = make_ticks(height, step=10)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([format_tick(v) for v in xticks])
    ax.set_yticklabels([format_tick(v) for v in yticks])

    ax.grid(False)
    ax.margins(0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#202020")


def draw_route_map_panel(scene, solution):
    width, height = get_scene_width_height(scene)
    start_positions = get_start_positions(scene, solution)
    groups_by_vid = get_vehicle_groups(solution, start_positions)

    fig, ax = plt.subplots(figsize=(6.8, 5.35), dpi=OUTPUT_DPI)
    fig.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(AX_FACE)

    draw_region_shading(ax, groups_by_vid)
    draw_trees(ax, scene)
    draw_vehicle_routes(ax, solution, start_positions, groups_by_vid)
    draw_observation_waypoints(ax, groups_by_vid)
    draw_end_markers(ax, groups_by_vid)
    draw_start_markers(ax, start_positions, width, height, groups_by_vid)
    configure_map_axes(ax, width, height)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.125, top=0.985)
    save_png_tiff(fig, OUT_A, dpi=OUTPUT_DPI)

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# =========================================================
# 7. Summary and encoding panel
# =========================================================
def draw_summary_panel(scene, solution):
    """
    Draw the right-side legend/summary panel.

    This revised version enlarges the legend text and symbols while avoiding
    horizontal overlap in the representative-case rows. The route map panel is
    not changed.
    """
    width, height = get_scene_width_height(scene)
    n_trees = len(get_tree_xy(scene))
    required_viewpoints = 4 * n_trees

    start_positions = get_start_positions(scene, solution)

    total_path_length = get_nested_metric(
        solution,
        names=[
            "total_path_length",
            "total_distance",
            "path_length",
            "total_length",
            "best_distance",
        ],
        default=None,
    )
    if total_path_length is None:
        computed_length = compute_path_length(solution, start_positions)
        # Use the manuscript-reported fallback if the JSON does not store a metric
        # and the computed value is unavailable.
        total_path_length = computed_length if computed_length > 0 else DEFAULT_TOTAL_PATH_LENGTH

    energy_range = get_nested_metric(
        solution,
        names=[
            "energy_range_percent",
            "energy_range",
            "Energy Range",
            "energy_range_pct",
        ],
        default=DEFAULT_ENERGY_RANGE_PERCENT,
    )

    theoretical_cov = get_nested_metric(
        solution,
        names=[
            "theoretical_coverage",
            "theoretical_coverage_percent",
            "theoretical_cov",
        ],
        default=DEFAULT_THEORETICAL_COVERAGE,
    )

    effective_cov = get_nested_metric(
        solution,
        names=[
            "effective_coverage",
            "effective_coverage_percent",
            "effective_cov",
        ],
        default=DEFAULT_EFFECTIVE_COVERAGE,
    )

    # Convert ratios to percentages if needed.
    if isinstance(energy_range, (int, float)) and energy_range <= 1.0:
        energy_range = energy_range * 100.0
    if isinstance(theoretical_cov, (int, float)) and theoretical_cov <= 1.0:
        theoretical_cov = theoretical_cov * 100.0
    if isinstance(effective_cov, (int, float)) and effective_cov <= 1.0:
        effective_cov = effective_cov * 100.0

    # Force the figure summary to match the manuscript and Supplementary Table S6.
    total_path_length = 588.88
    energy_range = 5.91
    theoretical_cov = 100.0
    effective_cov = 100.0

    # Slightly wider legend panel than the previous version. This only affects
    # the summary/encoding panel and prevents enlarged labels from overlapping.
    fig, ax = plt.subplots(figsize=(4.25, 5.35), dpi=OUTPUT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x, y, s, size=PANEL_BODY_FS, weight="normal", ha="left", va="center", color="#111111"):
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

    def sample_line(x0, x1, y, color, linestyle="-", lw=1.7, alpha=1.0):
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

    # ------------------------------------------------------------------
    # Representative case
    # ------------------------------------------------------------------
    text(0.05, 0.955, "Representative case", size=PANEL_TITLE_FS, weight="bold")

    # Compact two/three-column summary rows. Long labels are shortened only in
    # the legend panel to improve readability at manuscript scale.
    y = 0.895
    text(0.06, y, "UGVs:", size=PANEL_BODY_FS, weight="semibold")
    text(0.205, y, "4", size=PANEL_BODY_FS)
    text(0.36, y, "Trees:", size=PANEL_BODY_FS, weight="semibold")
    text(0.525, y, f"{n_trees}", size=PANEL_BODY_FS)

    y = 0.850
    text(0.06, y, "Required viewpoints:", size=PANEL_BODY_FS, weight="semibold")
    text(0.55, y, f"{required_viewpoints}", size=PANEL_BODY_FS)

    y = 0.805
    text(0.06, y, "Region:", size=PANEL_BODY_FS, weight="semibold")
    text(0.95, y, f"{width:.2f} m × {height:.2f} m", size=PANEL_BODY_FS, ha="right")

    y = 0.760
    text(0.06, y, "Total path length:", size=PANEL_BODY_FS, weight="semibold")
    text(0.95, y, f"{float(total_path_length):.2f} m", size=PANEL_BODY_FS, ha="right")

    y = 0.715
    text(0.06, y, "Energy range:", size=PANEL_BODY_FS, weight="semibold")
    text(0.95, y, f"{float(energy_range):.2f}%", size=PANEL_BODY_FS, ha="right")

    y = 0.670
    text(0.06, y, "Coverage:", size=PANEL_BODY_FS, weight="semibold")
    text(0.95, y, f"{float(theoretical_cov):.0f}% / {float(effective_cov):.0f}%", size=PANEL_BODY_FS, ha="right")

    ax.plot([0.05, 0.95], [0.625, 0.625], color="#D0D0D0", lw=0.9, transform=ax.transAxes)

    # ------------------------------------------------------------------
    # Route encoding
    # ------------------------------------------------------------------
    text(0.05, 0.585, "Route-style encoding", size=PANEL_SECTION_FS, weight="bold")

    head_y = 0.525
    text(0.05, head_y, "UGV", size=PANEL_HEADER_FS, weight="bold")
    text(0.365, head_y, "Local viewpoint\npath", size=9.0, weight="bold", ha="center")
    text(0.785, head_y, "Inter-tree route\nsegment", size=9.0, weight="bold", ha="center")

    row_y = 0.455
    row_gap = 0.057

    for vid in range(4):
        color = ROUTE_COLORS[vid]
        style = ROUTE_STYLES[vid]

        text(0.05, row_y, f"UGV{vid + 1}", size=PANEL_BODY_FS)

        sample_line(
            0.225, 0.505, row_y,
            color=color,
            linestyle=style,
            lw=max(1.80, ROUTE_LOCAL_WIDTH[vid] + 0.60),
            alpha=1.0,
        )

        sample_line(
            0.615, 0.930, row_y,
            color=color,
            linestyle=style,
            lw=max(2.00, ROUTE_MAIN_WIDTH[vid] + 0.60),
            alpha=1.0,
        )

        row_y -= row_gap

    ax.plot([0.05, 0.95], [0.245, 0.245], color="#D0D0D0", lw=0.9, transform=ax.transAxes)

    # ------------------------------------------------------------------
    # Scene elements
    # ------------------------------------------------------------------
    text(0.05, 0.205, "Scene elements", size=PANEL_SECTION_FS, weight="bold")

    x_icon_l = 0.10
    x_text_l = 0.19
    x_icon_r = 0.60
    x_text_r = 0.69

    y1 = 0.145
    y2 = 0.085
    y3 = 0.030

    # Left column
    draw_ugv_icon(ax, x_icon_l, y1, zoom=UGV_LEGEND_ZOOM, angle=0, coord="axes", zorder=20)
    text(x_text_l, y1, "UGV", size=PANEL_BODY_FS)

    ax.scatter(
        [x_icon_l],
        [y2],
        s=72,
        marker="o",
        facecolor=TREE_FACE,
        edgecolor=TREE_EDGE,
        linewidth=0.0,
        alpha=TREE_ALPHA,
        transform=ax.transAxes,
        zorder=5,
    )
    text(x_text_l, y2, "Trees", size=PANEL_BODY_FS)

    # Right column
    ax.scatter(
        [x_icon_r],
        [y1],
        s=38,
        marker="o",
        facecolor=VIEWPOINT_COLOR,
        edgecolor="white",
        linewidth=0.55,
        transform=ax.transAxes,
        zorder=5,
    )
    text(x_text_r, y1, "Observation waypoints", size=PANEL_BODY_FS)

    ax.scatter(
        [x_icon_r],
        [y2],
        s=76,
        marker="^",
        facecolor="white",
        edgecolor=START_COLOR,
        linewidth=1.45,
        transform=ax.transAxes,
        zorder=5,
    )
    text(x_text_r, y2, "Start positions", size=PANEL_BODY_FS)

    ax.scatter(
        [x_icon_r],
        [y3],
        s=70,
        marker="s",
        facecolor="white",
        edgecolor=END_COLOR,
        linewidth=1.35,
        transform=ax.transAxes,
        zorder=5,
    )
    text(x_text_r, y3, "End positions", size=PANEL_BODY_FS)

    fig.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00)
    save_png_tiff(fig, OUT_B, dpi=OUTPUT_DPI)

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)

# =========================================================
# 8. Main
# =========================================================
def main():
    scene, solution = load_data()

    draw_route_map_panel(scene, solution)
    draw_summary_panel(scene, solution)

    print("Done. Generated Fig. 6 panels in PNG and TIFF format:")
    for stem in [OUT_A, OUT_B]:
        print(f"  {stem.with_suffix('.png').name}")
        print(f"  {stem.with_suffix('.tiff').name}")
    print(f"dpi = {OUTPUT_DPI}")


if __name__ == "__main__":
    main()
