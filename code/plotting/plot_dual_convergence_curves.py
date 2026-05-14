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

# =========================
# 图1：四算法整体收敛图
# =========================
fig = plt.figure(figsize=(8, 5), dpi=300)
ax = fig.add_subplot(111)

for df, label in [
    (df_flca, "F-LCA"),
    (df_aco,  "ACO"),
    (df_ga,   "GA"),
    (df_pso,  "PSO"),
]:
    x = df["iter"].to_numpy()
    y = df["best_primary_cost_mean"].to_numpy()
    s = df["best_primary_cost_std"].to_numpy()
    ax.plot(x, y, linewidth=2.0, label=label)
    ax.fill_between(x, y - s, y + s, alpha=0.12)

ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Best Primary Cost", fontsize=12)
ax.set_xlim(1, 1500)

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, color="gray")

leg = ax.legend(frameon=True, fontsize=10)
leg.get_frame().set_edgecolor("0.7")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_alpha(0.9)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
out1 = Path(r"D:\pycharm\PythonProject2\convergence_all_algorithms_final.png")
plt.savefig(out1, bbox_inches="tight", facecolor="white")
plt.show()

print(f"图1已保存：{out1}")

# =========================
# 图2：F-LCA vs ACO 局部图
# =========================
fig = plt.figure(figsize=(8, 5), dpi=300)
ax = fig.add_subplot(111)

for df, label in [
    (df_flca, "F-LCA"),
    (df_aco,  "ACO"),
]:
    x = df["iter"].to_numpy()
    y = df["best_primary_cost_mean"].to_numpy()
    s = df["best_primary_cost_std"].to_numpy()
    ax.plot(x, y, linewidth=2.0, label=label)
    ax.fill_between(x, y - s, y + s, alpha=0.12)

ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Best Primary Cost", fontsize=12)
ax.set_xlim(1, 1500)
ax.set_ylim(1350, 1850)

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, color="gray")

leg = ax.legend(frameon=True, fontsize=10)
leg.get_frame().set_edgecolor("0.7")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_alpha(0.9)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
out2 = Path(r"D:\pycharm\PythonProject2\convergence_flca_vs_aco_final.png")
plt.savefig(out2, bbox_inches="tight", facecolor="white")
plt.show()

print(f"图2已保存：{out2}")