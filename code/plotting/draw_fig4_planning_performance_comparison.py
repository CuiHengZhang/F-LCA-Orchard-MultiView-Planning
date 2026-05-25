# -*- coding: utf-8 -*-
"""
Draw Fig. 4: planning performance comparison with six combined panels.

This version matches the subfigure-label style used for the revised Fig. 1,
Fig. 2, Fig. 5, and Fig. 6.

Panels:
(a) Mean Primary Cost
(b) Primary Cost distribution
(c) Mean Obstacle Penalty
(d) Obstacle Penalty distribution
(e) Mean Total Distance
(f) Total Distance distribution

Input CSV files, placed in the same folder as this script:
  - F-LCA_seed_results.csv
  - SISR-VRP_seed_results.csv
  - ACO_seed_results.csv
  - GA_seed_results.csv
  - PSO_seed_results.csv

Each CSV should include at least:
  seed, primary_cost, obstacle_penalty, total_distance

Output:
  - Fig4_planning_performance_comparison.png
  - Fig4_planning_performance_comparison.tif

Requirements:
  python -m pip install pandas numpy matplotlib pillow
"""

from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 1. Input files
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_FILES = {
    "F-LCA": BASE_DIR / "F-LCA_seed_results.csv",
    "SISR-VRP": BASE_DIR / "SISR-VRP_seed_results.csv",
    "ACO": BASE_DIR / "ACO_seed_results.csv",
    "GA": BASE_DIR / "GA_seed_results.csv",
    "PSO": BASE_DIR / "PSO_seed_results.csv",
}


# =========================================================
# 2. Output files
# =========================================================
PNG_OUT = BASE_DIR / "Fig4_planning_performance_comparison.png"
TIFF_OUT = BASE_DIR / "Fig4_planning_performance_comparison.tif"

OUTPUT_DPI = 600


# =========================================================
# 3. Metric column names
# =========================================================
COLUMN_CANDIDATES = {
    "Primary Cost": [
        "Primary Cost",
        "primary_cost",
        "primary cost",
        "main_cost",
        "main cost",
        "best_cost",
        "best cost",
        "PrimaryCost",
    ],
    "Obstacle Penalty": [
        "Obstacle Penalty",
        "obstacle_penalty",
        "obstacle penalty",
        "obs_penalty",
        "obs penalty",
        "ObstaclePenalty",
    ],
    "Total Distance": [
        "Total Distance",
        "total_distance",
        "total distance",
        "distance",
        "total_dist",
        "total dist",
        "TotalDistance",
    ],
}


# =========================================================
# 4. Algorithm order and colors
# =========================================================
ALGORITHMS = ["F-LCA", "SISR-VRP", "ACO", "GA", "PSO"]

BAR_COLORS = {
    "F-LCA": "#DCE6F1",
    "SISR-VRP": "#C6D6E6",
    "ACO": "#9FBBD8",
    "GA": "#6F97C4",
    "PSO": "#3F73A8",
}

EDGE_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"


# =========================================================
# 5. Figure panel layout
# =========================================================
PANEL_W = 2200
PANEL_H = 1450

MARGIN_X = 55
MARGIN_Y = 40
GAP_X = 140
GAP_Y = 18

LABEL_H = 125
LABEL_TOP_PAD = 14

FIG_W = MARGIN_X * 2 + PANEL_W * 2 + GAP_X
FIG_H = MARGIN_Y * 2 + (PANEL_H + LABEL_H) * 3 + GAP_Y * 2


# =========================================================
# 6. Matplotlib subplot layout inside each panel
# =========================================================
# 关键参数：
# 标签现在按这个绘图区中心居中，而不是按整个 panel 居中。
AX_LEFT = 0.18
AX_RIGHT = 0.97
AX_TOP = 0.95
AX_BOTTOM = 0.105

# 绘图区中心相对于单个 panel 左边界的横向位置
AXIS_CENTER_X = int(((AX_LEFT + AX_RIGHT) / 2.0) * PANEL_W)


# =========================================================
# 7. Font settings
# =========================================================
def load_pil_font(size=76):
    candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/Times New Roman.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/timesi.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


LABEL_FONT_SIZE = 88

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 8. Helper functions
# =========================================================
def normalize_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def find_column(df, candidates):
    existing = {normalize_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_name(cand)
        if key in existing:
            return existing[key]

    raise KeyError(
        f"Cannot find any of these columns: {candidates}\n"
        f"Existing columns are: {list(df.columns)}"
    )


def load_metric_data():
    data = {metric: {} for metric in COLUMN_CANDIDATES.keys()}

    for alg in ALGORITHMS:
        csv_path = DATA_FILES[alg]

        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find file for {alg}: {csv_path}")

        df = pd.read_csv(csv_path)

        for metric, candidates in COLUMN_CANDIDATES.items():
            col = find_column(df, candidates)
            values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

            if len(values) == 0:
                raise ValueError(f"No valid numeric data for {alg} - {metric}")

            data[metric][alg] = values

    return data


def comma_formatter(x, pos):
    return f"{int(x):,}"


def style_axis(ax):
    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.9)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.35)
        spine.set_color("#555555")

    ax.tick_params(
        axis="both",
        labelsize=18,
        width=1.1,
        length=4.0,
    )

    ax.yaxis.set_major_formatter(FuncFormatter(comma_formatter))


def render_matplotlib_panel(draw_func, metric_name, data, ylabel):
    dpi = 300
    fig_w = PANEL_W / dpi
    fig_h = PANEL_H / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    draw_func(
        ax=ax,
        metric_name=metric_name,
        data=data,
        ylabel=ylabel,
    )

    fig.subplots_adjust(
        left=AX_LEFT,
        right=AX_RIGHT,
        top=AX_TOP,
        bottom=AX_BOTTOM,
    )

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    if img.size != (PANEL_W, PANEL_H):
        img = img.resize((PANEL_W, PANEL_H), Image.Resampling.LANCZOS)

    return img


def draw_bar_panel(ax, metric_name, data, ylabel):
    """
    Draw one bar panel.
    No error bars are drawn.
    """
    means = [np.mean(data[metric_name][alg]) for alg in ALGORITHMS]
    x = np.arange(len(ALGORITHMS))

    bars = ax.bar(
        x,
        means,
        width=0.65,
        color=[BAR_COLORS[alg] for alg in ALGORITHMS],
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ALGORITHMS, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=22)

    style_axis(ax)

    ymax = max(means)
    ax.set_ylim(0, ymax * 1.18)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.035,
            f"{mean:,.1f}",
            ha="center",
            va="bottom",
            fontsize=17,
        )


def draw_box_panel(ax, metric_name, data, ylabel):
    box_data = [data[metric_name][alg] for alg in ALGORITHMS]
    x = np.arange(1, len(ALGORITHMS) + 1)

    bp = ax.boxplot(
        box_data,
        positions=x,
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops={
            "color": "black",
            "linewidth": 1.2,
        },
        boxprops={
            "edgecolor": EDGE_COLOR,
            "linewidth": 1.15,
        },
        whiskerprops={
            "color": EDGE_COLOR,
            "linewidth": 1.0,
        },
        capprops={
            "color": EDGE_COLOR,
            "linewidth": 1.0,
        },
        flierprops={
            "marker": "o",
            "markersize": 3.4,
            "markerfacecolor": "white",
            "markeredgecolor": EDGE_COLOR,
            "markeredgewidth": 0.7,
        },
    )

    for patch, alg in zip(bp["boxes"], ALGORITHMS):
        patch.set_facecolor(BAR_COLORS[alg])

    ax.set_xticks(x)
    ax.set_xticklabels(ALGORITHMS, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=22)

    style_axis(ax)

    all_values = np.concatenate(box_data)
    ymin = max(0, np.min(all_values) * 0.88)
    ymax = np.max(all_values) * 1.10
    ax.set_ylim(ymin, ymax)


def fit_label_font(label, max_width, preferred_size=88, min_size=76):
    size = preferred_size

    while size >= min_size:
        font = load_pil_font(size=size)
        dummy = Image.new("RGB", (10, 10), "white")
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]

        if text_w <= max_width - 24:
            return font

        size -= 2

    return load_pil_font(size=min_size)


def draw_axis_centered_label(draw, panel_x, label_y, label):
    """
    Draw subfigure label centered under the actual plotting axis area.

    This fixes the previous issue where right-column labels looked left-shifted
    because the full panel contains a wide y-axis label and tick-label area.
    """
    font = fit_label_font(
        label=label,
        max_width=PANEL_W,
        preferred_size=LABEL_FONT_SIZE,
        min_size=76,
    )

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]

    # 重点：按坐标轴绘图区中心居中
    text_center_x = panel_x + AXIS_CENTER_X
    text_x = int(text_center_x - text_w / 2)
    text_y = label_y + LABEL_TOP_PAD

    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)


def clean_outer_canvas_edges(img: Image.Image, edge_px: int = 3) -> Image.Image:
    """
    Remove possible gray artifacts at the outermost edges of the exported canvas.
    This only touches the outer 1-3 px border of the whole combined figure.
    """
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    draw.rectangle((0, 0, w, edge_px), fill="white")
    draw.rectangle((0, h - edge_px, w, h), fill="white")
    draw.rectangle((0, 0, edge_px, h), fill="white")
    draw.rectangle((w - edge_px, 0, w, h), fill="white")

    return img


# =========================================================
# 9. Main drawing
# =========================================================
def main():
    data = load_metric_data()

    panel_specs = [
        (
            draw_bar_panel,
            "Primary Cost",
            "Primary Cost (a.u.)",
            "(a) Mean Primary Cost",
        ),
        (
            draw_box_panel,
            "Primary Cost",
            "Primary Cost (a.u.)",
            "(b) Primary Cost distribution",
        ),
        (
            draw_bar_panel,
            "Obstacle Penalty",
            "Obstacle Penalty (a.u.)",
            "(c) Mean Obstacle Penalty",
        ),
        (
            draw_box_panel,
            "Obstacle Penalty",
            "Obstacle Penalty (a.u.)",
            "(d) Obstacle Penalty distribution",
        ),
        (
            draw_bar_panel,
            "Total Distance",
            "Total Distance (m)",
            "(e) Mean Total Distance",
        ),
        (
            draw_box_panel,
            "Total Distance",
            "Total Distance (m)",
            "(f) Total Distance distribution",
        ),
    ]

    panels = []

    for draw_func, metric_name, ylabel, label in panel_specs:
        img = render_matplotlib_panel(
            draw_func=draw_func,
            metric_name=metric_name,
            data=data,
            ylabel=ylabel,
        )
        panels.append((img, label))

    canvas = Image.new("RGB", (FIG_W, FIG_H), "white")
    draw = ImageDraw.Draw(canvas)

    positions = [
        # Row 1
        (MARGIN_X, MARGIN_Y),
        (MARGIN_X + PANEL_W + GAP_X, MARGIN_Y),

        # Row 2
        (MARGIN_X, MARGIN_Y + PANEL_H + LABEL_H + GAP_Y),
        (MARGIN_X + PANEL_W + GAP_X, MARGIN_Y + PANEL_H + LABEL_H + GAP_Y),

        # Row 3
        (MARGIN_X, MARGIN_Y + (PANEL_H + LABEL_H + GAP_Y) * 2),
        (MARGIN_X + PANEL_W + GAP_X, MARGIN_Y + (PANEL_H + LABEL_H + GAP_Y) * 2),
    ]

    for (img, label), (x, y) in zip(panels, positions):
        canvas.paste(img, (x, y))

        draw_axis_centered_label(
            draw=draw,
            panel_x=x,
            label_y=y + PANEL_H,
            label=label,
        )

    canvas = clean_outer_canvas_edges(canvas, edge_px=3)

    canvas.save(PNG_OUT, format="PNG")

    canvas.save(
        TIFF_OUT,
        format="TIFF",
        dpi=(OUTPUT_DPI, OUTPUT_DPI),
        compression="tiff_lzw",
    )

    print("Done.")
    print(f"PNG saved to:  {PNG_OUT}")
    print(f"TIFF saved to: {TIFF_OUT}")
    print(f"Final pixel size: {canvas.size}")
    print(f"DPI: {OUTPUT_DPI}")
    print(f"Axis-centered label offset from panel center: {AXIS_CENTER_X - PANEL_W // 2} px")


if __name__ == "__main__":
    main()