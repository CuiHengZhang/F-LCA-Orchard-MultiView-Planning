# -*- coding: utf-8 -*-
"""
Plot convergence curves from repository-relative result files.

Input priority:
1. results/convergence_results/main_convergence_summary_1500iters.csv
   Expected columns: algorithm, iter, best_primary_cost_mean, best_primary_cost_std

Output:
    figures_source_data/supplementary_figures/convergence_all_algorithms_final.*
    figures_source_data/supplementary_figures/convergence_flca_vs_aco_final.*
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures_source_data"

INPUT_CSV = RESULTS_DIR / "convergence_results" / "main_convergence_summary_1500iters.csv"
OUT_DIR = FIGURES_DIR / "supplementary_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = ["algorithm", "iter", "best_primary_cost_mean", "best_primary_cost_std"]

STYLE_MAP = {
    "F-LCA": {"color": "#1f77b4", "linestyle": "-",  "linewidth": 2.2},
    "ACO":   {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.0},
    "GA":    {"color": "#2ca02c", "linestyle": "-.", "linewidth": 2.0},
    "PSO":   {"color": "#d62728", "linestyle": ":",  "linewidth": 2.2},
    "SISR-VRP": {"color": "#9467bd", "linestyle": (0, (3, 1, 1, 1)), "linewidth": 2.0},
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_all_curves(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(require_file(file_path), encoding="utf-8-sig")
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}. Current columns: {df.columns.tolist()}")

    df = df.sort_values(["algorithm", "iter"]).reset_index(drop=True)
    df = df[df["iter"] <= 1500].copy()
    return df


def select_algorithm(df: pd.DataFrame, label: str) -> pd.DataFrame:
    sub = df[df["algorithm"].astype(str) == label].copy()
    if sub.empty:
        raise ValueError(f"No rows found for algorithm '{label}'. Available: {sorted(df['algorithm'].unique())}")
    return sub


def configure_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"


def plot_curve_with_band(ax, df: pd.DataFrame, label: str, alpha_fill: float = 0.10):
    x = df["iter"].to_numpy()
    y = df["best_primary_cost_mean"].to_numpy()
    s = df["best_primary_cost_std"].fillna(0).to_numpy()

    st = STYLE_MAP.get(label, {"color": None, "linestyle": "-", "linewidth": 2.0})
    ax.plot(
        x, y,
        label=label,
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=st["linewidth"],
    )
    ax.fill_between(
        x, y - s, y + s,
        color=st["color"],
        alpha=alpha_fill,
        linewidth=0,
    )


def beautify_axes(ax, xlabel: str, ylabel: str, legend_loc: str = "upper right"):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, color="gray")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    leg = ax.legend(frameon=True, fontsize=10, loc=legend_loc)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)


def save_figure(fig, stem: str):
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_DIR / f'{stem}.png'}")
    print(f"Saved: {OUT_DIR / f'{stem}.svg'}")


def main():
    configure_style()
    all_curves = load_all_curves(INPUT_CSV)

    # Figure 1: all available algorithms in the convergence summary.
    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(111)

    preferred_order = ["F-LCA", "ACO", "GA", "PSO", "SISR-VRP"]
    available = [name for name in preferred_order if name in set(all_curves["algorithm"].astype(str))]
    if not available:
        available = sorted(all_curves["algorithm"].astype(str).unique())

    for label in available:
        plot_curve_with_band(ax, select_algorithm(all_curves, label), label, alpha_fill=0.10)

    ax.set_xlim(1, 1500)
    beautify_axes(ax, "Iteration", "Historical Best Primary Cost (a.u.)")
    plt.tight_layout()
    save_figure(fig, "convergence_all_algorithms_final")
    plt.close(fig)

    # Figure 2: F-LCA vs ACO.
    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(111)

    for label in ["F-LCA", "ACO"]:
        plot_curve_with_band(ax, select_algorithm(all_curves, label), label, alpha_fill=0.12)

    ax.set_xlim(1, 1500)
    ax.set_ylim(1350, 1850)
    beautify_axes(ax, "Iteration", "Historical Best Primary Cost (a.u.)")
    plt.tight_layout()
    save_figure(fig, "convergence_flca_vs_aco_final")
    plt.close(fig)


if __name__ == "__main__":
    main()
