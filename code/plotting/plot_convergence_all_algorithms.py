import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. 文件路径：改成你自己的
# =========================
f_flca = Path(r"D:\pycharm\PythonProject2\flca_20seed_fair_budget\convergence_curve_fixed.csv")
f_aco  = Path(r"D:\pycharm\PythonProject2\aco_20seed_fair_budget\convergence_summary.csv")
f_ga   = Path(r"D:\pycharm\PythonProject2\ga_20seed_fair_budget\convergence_summary.csv")
f_pso  = Path(r"D:\pycharm\PythonProject2\pso_20seed_fair_budget\convergence_summary.csv")

out_dir = Path(r"D:\pycharm\PythonProject2")
out_dir.mkdir(parents=True, exist_ok=True)

def load_curve(file_path, algo_name):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    required_cols = ["iter", "best_primary_cost_mean", "best_primary_cost_std"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{algo_name} 缺少列 {col}，当前列名：{df.columns.tolist()}")
    df = df.sort_values("iter").reset_index(drop=True)
    df = df[df["iter"] <= 1500].copy()
    return df

df_flca = load_curve(f_flca, "F-LCA")
df_aco  = load_curve(f_aco,  "ACO")
df_ga   = load_curve(f_ga,   "GA")
df_pso  = load_curve(f_pso,  "PSO")

# =========================
# 2. 全局风格
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.bbox"] = "tight"

# 颜色 + 线型
style_map = {
    "F-LCA": {"color": "#1f77b4", "linestyle": "-",  "linewidth": 2.2},
    "ACO":   {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.0},
    "GA":    {"color": "#2ca02c", "linestyle": "-.", "linewidth": 2.0},
    "PSO":   {"color": "#d62728", "linestyle": ":",  "linewidth": 2.2},
}

def plot_curve_with_band(ax, df, label, alpha_fill=0.10):
    x = df["iter"].to_numpy()
    y = df["best_primary_cost_mean"].to_numpy()
    s = df["best_primary_cost_std"].to_numpy()

    st = style_map[label]
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
        linewidth=0
    )

def beautify_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, color="gray")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    leg = ax.legend(frameon=True, fontsize=10, loc="upper right")
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)

# =========================
# 图1：四算法整体收敛图
# =========================
fig = plt.figure(figsize=(8, 5), dpi=300)
ax = fig.add_subplot(111)

plot_curve_with_band(ax, df_flca, "F-LCA", alpha_fill=0.10)
plot_curve_with_band(ax, df_aco,  "ACO",   alpha_fill=0.10)
plot_curve_with_band(ax, df_ga,   "GA",    alpha_fill=0.10)
plot_curve_with_band(ax, df_pso,  "PSO",   alpha_fill=0.10)

ax.set_xlim(1, 1500)
beautify_axes(ax, "Iteration", "Historical Best Primary Cost (a.u.)")

out1_png = out_dir / "convergence_all_algorithms_final.png"
out1_svg = out_dir / "convergence_all_algorithms_final.svg"
plt.tight_layout()
plt.savefig(out1_png, dpi=300)
plt.savefig(out1_svg)
plt.show()

print(f"图1已保存：{out1_png}")
print(f"图1已保存：{out1_svg}")

# =========================
# 图2：F-LCA vs ACO 局部图
# =========================
fig = plt.figure(figsize=(8, 5), dpi=300)
ax = fig.add_subplot(111)

plot_curve_with_band(ax, df_flca, "F-LCA", alpha_fill=0.12)
plot_curve_with_band(ax, df_aco,  "ACO",   alpha_fill=0.12)

ax.set_xlim(1, 1500)
ax.set_ylim(1350, 1850)
beautify_axes(ax, "Iteration", "Historical Best Primary Cost (a.u.)")

out2_png = out_dir / "convergence_flca_vs_aco_final.png"
out2_svg = out_dir / "convergence_flca_vs_aco_final.svg"
plt.tight_layout()
plt.savefig(out2_png, dpi=300)
plt.savefig(out2_svg)
plt.show()

print(f"图2已保存：{out2_png}")
print(f"图2已保存：{out2_svg}")