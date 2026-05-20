# -*- coding: utf-8 -*-
"""
Convert raw seed-level convergence data into a fixed historical-best convergence summary.

Default input:
    results/convergence_results/convergence_seed.csv

Outputs:
    results/convergence_results/seed_curve_fixed.csv
    results/convergence_results/convergence_curve_fixed.csv

This script is optional if main_convergence_summary_1500iters.csv has already been prepared.
"""

from pathlib import Path

import pandas as pd


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_CSV = RESULTS_DIR / "convergence_results" / "convergence_seed.csv"
OUT_DIR = RESULTS_DIR / "convergence_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    for encoding in ("utf-8-sig", "gbk", None):
        try:
            if encoding is None:
                return pd.read_csv(path)
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def main():
    df = read_csv_auto(INPUT_CSV)

    print("Columns:", df.columns.tolist())

    required_cols = {"seed", "algorithm", "iter", "best_primary_cost"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["algorithm", "seed", "iter"]).reset_index(drop=True)

    # Convert each seed curve into historical best values.
    df["best_primary_cost_fixed"] = (
        df.groupby(["algorithm", "seed"])["best_primary_cost"].cummin()
    )

    seed_out = OUT_DIR / "seed_curve_fixed.csv"
    df.to_csv(seed_out, index=False, encoding="utf-8-sig")

    agg = (
        df.groupby(["algorithm", "iter"])["best_primary_cost_fixed"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .rename(columns={
              "mean": "best_primary_cost_mean",
              "std": "best_primary_cost_std",
              "count": "n",
          })
    )

    curve_out = OUT_DIR / "convergence_curve_fixed.csv"
    agg.to_csv(curve_out, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Fixed seed-level curve:", seed_out)
    print("Aggregated convergence curve:", curve_out)


if __name__ == "__main__":
    main()
