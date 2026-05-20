from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[2]

SEED_DIR = ROOT / "results" / "seed_results"
OUT_DIR = ROOT / "results" / "statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "F-LCA": "F-LCA_seed_results.csv",
    "SISR-VRP": "SISR-VRP_seed_results.csv",
    "ACO": "ACO_seed_results.csv",
    "GA": "GA_seed_results.csv",
    "PSO": "PSO_seed_results.csv",
}


METRIC_ALIASES = {
    "Primary Cost": ["primary_cost", "primary cost", "primarycost"],
    "Total Cost": ["total_cost", "total cost", "totalcost"],
    "Total Distance": ["total_distance", "total distance", "totaldistance"],
    "Obstacle Penalty": ["obstacle_penalty", "obstacle penalty", "obstaclepenalty"],
    "Energy Range": ["energy_range_percent", "energy_range", "energy range", "energyrange"],
    "Runtime": ["runtime_sec", "runtime", "runtime_s"],
    "Theoretical Coverage": ["theoretical_coverage", "theoretical coverage", "theoreticalcoverage"],
    "Effective Coverage": ["effective_coverage", "effective coverage", "effectivecoverage"],
}


SEED_ALIASES = ["seed", "random_seed", "random seed"]


def normalize_col(name: str) -> str:
    return (
        str(name).strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace("%", "percent")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
    )


def find_column(df: pd.DataFrame, aliases):
    normalized = {normalize_col(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_col(alias)
        if key in normalized:
            return normalized[key]
    return None


def format_p(p):
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return "p < 0.01"
    if p < 0.05:
        return "p < 0.05"
    return f"p = {p:.4f}"


def holm_adjust(p_values):
    """
    Holm step-down adjusted p-values.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        raw_adj = (m - rank) * p_values[idx]
        running_max = max(running_max, raw_adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


# Load seed-level data
data = {}

for algorithm, filename in FILES.items():
    path = SEED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    seed_col = find_column(df, SEED_ALIASES)
    if seed_col is not None:
        df = df.sort_values(seed_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    data[algorithm] = df


friedman_rows = []
wilcoxon_rows = []

algorithms = list(FILES.keys())
baseline = "F-LCA"
comparators = [a for a in algorithms if a != baseline]

for metric_name, aliases in METRIC_ALIASES.items():
    values_by_algorithm = {}
    missing = False

    for algorithm in algorithms:
        df = data[algorithm]
        col = find_column(df, aliases)

        if col is None:
            print(f"[Warning] {algorithm}: missing column for {metric_name}")
            missing = True
            break

        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        values_by_algorithm[algorithm] = values

    if missing:
        continue

    lengths = [len(v) for v in values_by_algorithm.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Unequal paired sample lengths for {metric_name}: {lengths}")

    n = lengths[0]

    # If all algorithms have identical values, skip statistical test
    stacked = np.vstack([values_by_algorithm[a] for a in algorithms])
    if np.allclose(stacked, stacked[0, :]):
        friedman_rows.append({
            "Metric": metric_name,
            "Algorithms included": ", ".join(algorithms),
            "Number of paired instances": n,
            "Statistic": "",
            "p-value": "",
            "Formatted result": "Not tested",
            "Interpretation": "All algorithms achieved identical values"
        })
        continue

    # Friedman test
    stat, p = friedmanchisquare(*[values_by_algorithm[a] for a in algorithms])
    friedman_rows.append({
        "Metric": metric_name,
        "Algorithms included": ", ".join(algorithms),
        "Number of paired instances": n,
        "Statistic": round(stat, 6),
        "p-value": p,
        "Formatted result": format_p(p),
        "Interpretation": "Significant overall difference among algorithms" if p < 0.05 else "No significant overall difference among algorithms"
    })

    # Wilcoxon paired tests: F-LCA vs each comparator
    raw_rows = []
    raw_p_values = []

    base_values = values_by_algorithm[baseline]

    for comp in comparators:
        comp_values = values_by_algorithm[comp]

        # Wilcoxon signed-rank test
        try:
            w_stat, raw_p = wilcoxon(base_values, comp_values, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, raw_p = np.nan, 1.0

        base_mean = np.mean(base_values)
        comp_mean = np.mean(comp_values)

        if base_mean < comp_mean:
            direction = "F-LCA lower"
        elif base_mean > comp_mean:
            direction = f"{comp} lower"
        else:
            direction = "Equal mean"

        raw_rows.append({
            "Metric": metric_name,
            "Pairwise comparison": f"F-LCA vs {comp}",
            "Direction of difference": direction,
            "Wilcoxon statistic": w_stat,
            "Raw p-value": raw_p,
            "F-LCA mean": base_mean,
            f"{comp} mean": comp_mean,
        })
        raw_p_values.append(raw_p)

    adjusted_p_values = holm_adjust(raw_p_values)

    for row, adj_p in zip(raw_rows, adjusted_p_values):
        row["Holm-adjusted p-value"] = adj_p
        row["Formatted result after Holm correction"] = format_p(adj_p)

        metric = row["Metric"]
        direction = row["Direction of difference"]

        if adj_p < 0.05:
            row["Interpretation"] = f"Significant difference; {direction}"
        else:
            row["Interpretation"] = f"No significant difference after Holm correction; {direction}"

        wilcoxon_rows.append(row)


friedman_df = pd.DataFrame(friedman_rows)
wilcoxon_df = pd.DataFrame(wilcoxon_rows)

friedman_out = OUT_DIR / "friedman_results.csv"
wilcoxon_out = OUT_DIR / "wilcoxon_holm_results.csv"

friedman_df.to_csv(friedman_out, index=False, encoding="utf-8-sig")
wilcoxon_df.to_csv(wilcoxon_out, index=False, encoding="utf-8-sig")

print("Saved:")
print(friedman_out)
print(wilcoxon_out)