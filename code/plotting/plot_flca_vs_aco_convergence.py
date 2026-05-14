import pandas as pd
from pathlib import Path

# ===== 改成你的真实文件路径 =====
file_path = Path(r"D:\pycharm\PythonProject2\flca_20seed_fair_budget\convergence_seed.csv")

if not file_path.exists():
    raise FileNotFoundError(f"找不到文件：{file_path}")

# 尝试读取
try:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path)

# 检查列名
print("列名：", df.columns.tolist())

required_cols = {"seed", "algorithm", "iter", "best_primary_cost"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"缺少必要列：{missing}")

# 排序
df = df.sort_values(["algorithm", "seed", "iter"]).reset_index(drop=True)

# 对每个 seed 做累计最小值，修正为“历史最优”
df["best_primary_cost_fixed"] = (
    df.groupby(["algorithm", "seed"])["best_primary_cost"].cummin()
)

# 保存修正后的每 seed 数据
seed_out = file_path.parent / "seed_curve_fixed.csv"
df.to_csv(seed_out, index=False, encoding="utf-8-sig")

# 聚合成均值/标准差收敛曲线
agg = (
    df.groupby(["algorithm", "iter"])["best_primary_cost_fixed"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .rename(columns={
          "mean": "best_primary_cost_mean",
          "std": "best_primary_cost_std",
          "count": "n"
      })
)

curve_out = file_path.parent / "convergence_curve_fixed.csv"
agg.to_csv(curve_out, index=False, encoding="utf-8-sig")

print("处理完成！")
print("修正后的每seed文件：", seed_out)
print("聚合后的收敛曲线文件：", curve_out)