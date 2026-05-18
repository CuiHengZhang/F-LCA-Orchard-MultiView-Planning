import csv
import os
import statistics
import time
from copy import deepcopy
from typing import Dict, Any, List, Tuple

import F_LCA_tiered_refinement_pathboost as flca


# ============================================================
# Release script for F-LCA parameter sensitivity analysis
#
# Parameters:
#   lambda_o: obstacle-penalty weight
#   partition_battery_power: battery-aware partition exponent
#   lambda_b: path-balance weight
#   lambda_c: observation-completeness penalty multiplier
#
# Each parameter is perturbed one at a time using multipliers:
#   0.8x, 0.9x, 1.0x, 1.1x, 1.2x
#
# Each setting is evaluated on 20 matched random instances:
#   seeds 42–61
#
# The script outputs clean seed-level and summary files.
# Auxiliary logging fields not used in the manuscript are not exported.
# ============================================================

SEEDS = list(range(42, 62))
MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.2]
PARAMETER_NAMES = ["lambda_o", "partition_battery_power", "lambda_b", "lambda_c"]

OUT_DIR = "parameter_sensitivity_release"
SEED_CSV = os.path.join(OUT_DIR, "flca_parameter_sensitivity_seed_results_clean.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "flca_parameter_sensitivity_summary_clean.csv")


def build_sensitivity_settings() -> List[Tuple[str, float]]:
    return [(param_name, multiplier) for param_name in PARAMETER_NAMES for multiplier in MULTIPLIERS]


def apply_parameter_change(cfg: flca.AlgorithmConfig, param_name: str, multiplier: float) -> Dict[str, Any]:
    base_values = {
        "lambda_balance": cfg.lambda_balance,
        "lambda_obstacle": cfg.lambda_obstacle,
        "lambda_missing_point": cfg.lambda_missing_point,
        "lambda_tree_incomplete": cfg.lambda_tree_incomplete,
        "partition_battery_power": cfg.partition_battery_power,
    }

    if param_name == "lambda_b":
        cfg.lambda_balance = base_values["lambda_balance"] * multiplier
    elif param_name == "lambda_o":
        cfg.lambda_obstacle = base_values["lambda_obstacle"] * multiplier
    elif param_name == "lambda_c":
        cfg.lambda_missing_point = base_values["lambda_missing_point"] * multiplier
        cfg.lambda_tree_incomplete = base_values["lambda_tree_incomplete"] * multiplier
    elif param_name == "partition_battery_power":
        cfg.partition_battery_power = base_values["partition_battery_power"] * multiplier
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")

    return {
        "lambda_balance": cfg.lambda_balance,
        "lambda_obstacle": cfg.lambda_obstacle,
        "lambda_missing_point": cfg.lambda_missing_point,
        "lambda_tree_incomplete": cfg.lambda_tree_incomplete,
        "partition_battery_power": cfg.partition_battery_power,
    }


def run_one_case(seed: int, param_name: str, multiplier: float) -> Dict[str, Any]:
    env_cfg = flca.EnvironmentConfig(random_seed=seed)
    base_cfg = flca.AlgorithmConfig(run_seed=seed, report_every=10**9)

    cfg = deepcopy(base_cfg)
    changed_values = apply_parameter_change(cfg, param_name, multiplier)

    tag = f"F-LCA-sensitivity-{param_name}-{multiplier:.1f}x"
    cfg.name = tag

    start = time.time()
    result = flca.run_single_case(
        env_cfg=env_cfg,
        alg_cfg=cfg,
        use_layered_refine=True,
        tag=tag,
    )
    runtime = time.time() - start

    return {
        "parameter": param_name,
        "multiplier": float(multiplier),
        "seed": int(seed),

        "lambda_balance": float(changed_values["lambda_balance"]),
        "lambda_obstacle": float(changed_values["lambda_obstacle"]),
        "lambda_missing_point": float(changed_values["lambda_missing_point"]),
        "lambda_tree_incomplete": float(changed_values["lambda_tree_incomplete"]),
        "partition_battery_power": float(changed_values["partition_battery_power"]),

        "primary_cost": float(result.best_primary_cost),
        "total_cost": float(result.best_reference_total_cost),
        "total_distance": float(result.best_total_distance),
        "theoretical_coverage": float(result.theoretical_observation_coverage),
        "effective_coverage": float(result.effective_observation_coverage),
        "obstacle_penalty": float(result.obstacle_penalty),
        "energy_range_percent": float(result.energy_range_percent),
        "incomplete_tree_ratio": float(result.incomplete_tree_ratio),
        "runtime_sec": float(runtime),
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric_cols = {
        "multiplier", "seed", "lambda_balance", "lambda_obstacle",
        "lambda_missing_point", "lambda_tree_incomplete",
        "partition_battery_power", "primary_cost", "total_cost",
        "total_distance", "theoretical_coverage", "effective_coverage",
        "obstacle_penalty", "energy_range_percent", "incomplete_tree_ratio",
        "runtime_sec",
    }
    out = []
    for row in rows:
        new_row = dict(row)
        for col in numeric_cols:
            if col in new_row and new_row[col] != "":
                if col == "seed":
                    new_row[col] = int(float(new_row[col]))
                else:
                    new_row[col] = float(new_row[col])
        out.append(new_row)
    return out


def completed_keys(rows: List[Dict[str, Any]]) -> set:
    return {(r["parameter"], float(r["multiplier"]), int(r["seed"])) for r in rows}


def mean_std(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def actual_value(row: Dict[str, Any]) -> str:
    p = row["parameter"]
    if p == "lambda_o":
        return f"{float(row['lambda_obstacle']):.3f}"
    if p == "partition_battery_power":
        return f"{float(row['partition_battery_power']):.3f}"
    if p == "lambda_b":
        return f"{float(row['lambda_balance']):.3f}"
    if p == "lambda_c":
        return f"{float(row['lambda_missing_point']):.1f} / {float(row['lambda_tree_incomplete']):.1f}"
    return ""


def aggregate_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metric_keys = [
        "primary_cost", "total_cost", "total_distance",
        "theoretical_coverage", "effective_coverage",
        "obstacle_penalty", "energy_range_percent",
        "incomplete_tree_ratio", "runtime_sec",
    ]

    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["parameter"]), float(row["multiplier"])), []).append(row)

    order = {p: i for i, p in enumerate(PARAMETER_NAMES)}
    summary = []
    for (param_name, multiplier), group_rows in sorted(groups.items(), key=lambda x: (order[x[0][0]], x[0][1])):
        first = group_rows[0]
        item = {
            "parameter": param_name,
            "multiplier": float(multiplier),
            "n": len(group_rows),
            "actual_value": actual_value(first),
            "lambda_balance": float(first["lambda_balance"]),
            "lambda_obstacle": float(first["lambda_obstacle"]),
            "lambda_missing_point": float(first["lambda_missing_point"]),
            "lambda_tree_incomplete": float(first["lambda_tree_incomplete"]),
            "partition_battery_power": float(first["partition_battery_power"]),
        }
        for metric in metric_keys:
            vals = [float(r[metric]) for r in group_rows]
            m, s = mean_std(vals)
            item[f"{metric}_mean"] = m
            item[f"{metric}_std"] = s
        summary.append(item)
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = normalize_rows(read_existing_rows(SEED_CSV))
    done = completed_keys(rows)

    settings = build_sensitivity_settings()
    total_runs = len(settings) * len(SEEDS)

    remaining = [
        (param_name, multiplier, seed)
        for param_name, multiplier in settings
        for seed in SEEDS
        if (param_name, float(multiplier), int(seed)) not in done
    ]

    print("===== F-LCA parameter sensitivity release run =====")
    print(f"Expected total runs: {total_runs}")
    print(f"Completed runs already found: {len(done)}")
    print(f"Remaining runs: {len(remaining)}")

    run_idx = len(done)
    for param_name, multiplier, seed in remaining:
        run_idx += 1
        print(f"[{run_idx}/{total_runs}] parameter={param_name}, multiplier={multiplier:.1f}x, seed={seed}")
        row = run_one_case(seed, param_name, multiplier)
        rows.append(row)
        write_csv(SEED_CSV, rows)
        write_csv(SUMMARY_CSV, aggregate_summary(rows))

        print(
            f"    primary={row['primary_cost']:.2f}, "
            f"distance={row['total_distance']:.2f}, "
            f"eff_cov={row['effective_coverage']:.2f}%, "
            f"obs={row['obstacle_penalty']:.2f}, "
            f"energy={row['energy_range_percent']:.2f}%"
        )

    write_csv(SUMMARY_CSV, aggregate_summary(rows))
    print(f"Saved: {SEED_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
