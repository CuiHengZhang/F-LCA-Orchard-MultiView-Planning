from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SEED_DIR = ROOT / "results" / "seed_results"
OUT_DIR = ROOT / "results" / "summary_tables"
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
    "Total Distance / m": ["total_distance", "total distance", "totaldistance"],
    "Theoretical Coverage / %": ["theoretical_coverage", "theoretical coverage", "theoreticalcoverage"],
    "Effective Coverage / %": ["effective_coverage", "effective coverage", "effectivecoverage"],
    "Obstacle Penalty": ["obstacle_penalty", "obstacle penalty", "obstaclepenalty"],
    "Energy Range / %": ["energy_range_percent", "energy_range", "energy range", "energyrange"],
    "Runtime / s": ["runtime_sec", "runtime", "runtime_s", "runtime / s"],
}


def normalize_col(name: str) -> str:
    return (
        name.strip()
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


def mean_std(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series.mean(), series.std(ddof=1)


rows_numeric = []
rows_text = []

for algorithm, filename in FILES.items():
    path = SEED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    numeric_row = {"Algorithm": algorithm}
    text_row = {"Algorithm": algorithm}

    for metric_name, aliases in METRIC_ALIASES.items():
        col = find_column(df, aliases)
        if col is None:
            print(f"[Warning] {algorithm}: column for {metric_name} not found.")
            numeric_row[f"{metric_name} Mean"] = None
            numeric_row[f"{metric_name} Std"] = None
            text_row[metric_name] = ""
            continue

        mean, std = mean_std(df[col])
        numeric_row[f"{metric_name} Mean"] = round(mean, 6)
        numeric_row[f"{metric_name} Std"] = round(std, 6)
        text_row[metric_name] = f"{mean:.2f} ± {std:.2f}"

    rows_numeric.append(numeric_row)
    rows_text.append(text_row)


numeric_df = pd.DataFrame(rows_numeric)
text_df = pd.DataFrame(rows_text)

numeric_out = OUT_DIR / "main_comparison_summary_numeric.csv"
text_out = OUT_DIR / "table_S7_full_algorithm_comparison.csv"

numeric_df.to_csv(numeric_out, index=False, encoding="utf-8-sig")
text_df.to_csv(text_out, index=False, encoding="utf-8-sig")


# Calculate headline improvements: F-LCA vs SISR-VRP
def get_mean(algorithm, metric):
    row = numeric_df[numeric_df["Algorithm"] == algorithm].iloc[0]
    return row[f"{metric} Mean"]


flca_primary = get_mean("F-LCA", "Primary Cost")
sisr_primary = get_mean("SISR-VRP", "Primary Cost")

flca_distance = get_mean("F-LCA", "Total Distance / m")
sisr_distance = get_mean("SISR-VRP", "Total Distance / m")

improvement_rows = [
    {
        "Comparison": "F-LCA vs SISR-VRP",
        "Metric": "Primary Cost",
        "F-LCA": flca_primary,
        "SISR-VRP": sisr_primary,
        "Improvement / %": round((sisr_primary - flca_primary) / sisr_primary * 100, 2),
    },
    {
        "Comparison": "F-LCA vs SISR-VRP",
        "Metric": "Total Distance / m",
        "F-LCA": flca_distance,
        "SISR-VRP": sisr_distance,
        "Improvement / %": round((sisr_distance - flca_distance) / sisr_distance * 100, 2),
    },
]

improvement_df = pd.DataFrame(improvement_rows)
improvement_out = OUT_DIR / "headline_improvements.csv"
improvement_df.to_csv(improvement_out, index=False, encoding="utf-8-sig")

print("Saved:")
print(numeric_out)
print(text_out)
print(improvement_out)