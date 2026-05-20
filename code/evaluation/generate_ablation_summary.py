from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ABLATION_DIR = ROOT / "results" / "ablation_results"
OUT_DIR = ABLATION_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "F-LCA": "flca_full_seed_results.csv",
    "F-LCA w/o hetero": "flca_no_hetero_seed_results.csv",
    "F-LCA w/o orthogonal observation": "flca_no_orthogonal_seed_results.csv",
    "F-LCA w/o late-stage cooperative restructuring": "flca_wo_late_stage_restructuring_seed_results.csv",
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
    path = ABLATION_DIR / filename
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

numeric_out = OUT_DIR / "ablation_summary_numeric.csv"
text_out = OUT_DIR / "table1_ablation_summary.csv"

numeric_df.to_csv(numeric_out, index=False, encoding="utf-8-sig")
text_df.to_csv(text_out, index=False, encoding="utf-8-sig")

print("Saved:")
print(numeric_out)
print(text_out)