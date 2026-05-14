"""
Run execution-oriented proxy evaluation for SISR-VRP under the same
anchor-preserving A* post-processing pipeline used for F-LCA and ACO.

Outputs:
  sisr_execution_postprocess_20seed_astar_anchor/
    - sisr_execution_seed_results_20seed_astar_anchor.csv
    - sisr_execution_vehicle_results_20seed_astar_anchor.csv
    - sisr_execution_summary_20seed_astar_anchor.csv

Place this file in the same directory as:
  - exp_suite_common_fair_budget_with_sisr.py
  - postprocess_execution_metrics_astar_anchor.py
  - F_LCA_tiered_refinement_pathboost.py
"""

import os
import statistics
import time
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import pandas as pd

from exp_suite_common_fair_budget_with_sisr import (
    ExperimentProblem,
    SEEDS,
    result_row,
    _sisr_initial_solution,
    _sisr_string_removal,
    _sisr_greedy_repair,
    _sisr_limited_two_opt,
)
from postprocess_execution_metrics_astar_anchor import process_algorithm


OUT_DIR = "sisr_execution_postprocess_20seed_astar_anchor"
MAX_RUNS = len(SEEDS)  # 可先改成 3 或 5 试跑


def to_xy_list(points):
    return [[float(x), float(y)] for x, y in points]


def build_scene_dict_from_problem(problem: ExperimentProblem) -> dict:
    """Build a scene dict compatible with process_algorithm()."""
    return {
        "seed": int(problem.seed),
        "width": float(problem.env_cfg.width),
        "height": float(problem.env_cfg.height),
        "vehicle_num": int(problem.env_cfg.vehicle_num),
        "vehicle_battery": [float(v) for v in problem.env_cfg.vehicle_battery],
        "start_positions": to_xy_list(problem.env_cfg.start_positions),
        "trees": [
            {"tree_id": int(i), "x": float(x), "y": float(y)}
            for i, (x, y) in enumerate(problem.trees)
        ],
        "obstacles": [
            {"obstacle_id": int(i), "cx": float(ox), "cy": float(oy), "r": float(rr)}
            for i, (ox, oy, rr) in enumerate(problem.obstacles)
        ],
        "tree_observation_points": {
            str(tid): to_xy_list(pts)
            for tid, pts in problem.evaluator.tree_to_points.items()
        },
        "tree_required_points": {
            str(tid): to_xy_list(pts)
            for tid, pts in problem.evaluator.tree_to_required_points.items()
        },
    }


def run_sisr_vrp_with_details(
    seed: int,
    n_iter: int = 1500,
    min_string_len: int = 2,
    max_string_len: int = 5,
    max_strings: int = 2,
    local_search_every: int = 100,
):
    """
    Same SISR-VRP logic as run_sisr_vrp(), but returns best groups/orders
    so that execution-level anchor paths can be reconstructed.
    """
    problem = ExperimentProblem(seed)
    # Keep the same RNG source as the original SISR-VRP implementation.
    import numpy as np

    rng = np.random.default_rng(seed)

    groups, orders = _sisr_initial_solution(problem)
    current_metrics = problem.evaluate(groups, orders)
    current_cost = problem.scalar_cost(current_metrics)
    best_metrics = deepcopy(current_metrics)
    best_cost = float(current_cost)

    start = time.time()
    for it in range(n_iter):
        cand_groups = deepcopy(groups)
        cand_orders = deepcopy(orders)

        removed = _sisr_string_removal(
            problem,
            rng,
            cand_groups,
            cand_orders,
            min_string_len=min_string_len,
            max_string_len=max_string_len,
            max_strings=max_strings,
        )
        if not removed:
            continue

        cand_groups, cand_orders = _sisr_greedy_repair(
            problem, rng, cand_groups, cand_orders, removed
        )

        if (it + 1) % local_search_every == 0:
            cand_groups, cand_orders, cand_metrics = _sisr_limited_two_opt(
                problem, rng, cand_groups, cand_orders, max_trials=8
            )
        else:
            cand_metrics = problem.evaluate(cand_groups, cand_orders)

        cand_cost = problem.scalar_cost(cand_metrics)

        # Conservative acceptance: accept only improving moves.
        if cand_cost + 1e-9 < current_cost:
            groups, orders = cand_groups, cand_orders
            current_metrics = cand_metrics
            current_cost = float(cand_cost)
            if cand_cost + 1e-9 < best_cost:
                best_metrics = deepcopy(cand_metrics)
                best_cost = float(cand_cost)

    runtime = time.time() - start
    row = result_row(seed, "SISR-VRP", best_metrics, runtime)
    details = {
        "groups": deepcopy(best_metrics["groups"]),
        "orders": deepcopy(best_metrics["orders"]),
    }
    return row, runtime, details, problem


def export_sisr_in_memory(seed: int) -> Tuple[dict, dict, dict]:
    """Return scene, path_json, and original main-metric row for one seed."""
    row, runtime, details, problem = run_sisr_vrp_with_details(seed)
    scene = build_scene_dict_from_problem(problem)

    groups = details["groups"]
    orders = details["orders"]

    best_path = {
        "seed": int(seed),
        "algorithm": "SISR-VRP",
        "summary": {
            "primary_cost": float(row["primary_cost"]),
            "total_cost": float(row["total_cost"]),
            "total_distance": float(row["total_distance"]),
            "effective_coverage": float(row["effective_coverage"]),
            "theoretical_coverage": float(row["theoretical_coverage"]),
            "obstacle_penalty": float(row["obstacle_penalty"]),
            "energy_range_percent": float(row["energy_range_percent"]),
            "runtime_sec": float(runtime),
        },
        "vehicles": {},
    }

    for vid in range(problem.vehicle_num):
        start_pt = problem.env.start_positions[vid]
        order = orders.get(vid, [])
        local_path = problem.evaluator._render_order_to_path(start_pt, order)
        full_path = [start_pt] + local_path

        best_path["vehicles"][str(vid)] = {
            "vehicle_id": int(vid),
            "start": [float(start_pt[0]), float(start_pt[1])],
            "path_points": to_xy_list(full_path),
            "local_path_points": to_xy_list(local_path),
            "tree_order": [int(t) for t in order],
            "assigned_tree_ids": [int(t) for t in groups.get(vid, [])],
        }

    return scene, best_path, row


def aggregate_exec_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate seed-level execution summaries into mean ± std rows."""
    if not rows:
        return []

    metrics = [
        "mean_estimated_execution_time",
        "total_estimated_execution_time",
        "minimum_obstacle_clearance",
        "minimum_safe_clearance",
        "total_turn_violation_count",
        "total_turn_violation_ratio_percent",
        "total_execution_length",
    ]

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["algorithm"]), []).append(row)

    out = []
    for algo, rs in groups.items():
        item = {"algorithm": algo, "n": len(rs)}
        for m in metrics:
            vals = [float(r[m]) for r in rs]
            item[f"{m}_mean"] = float(statistics.fmean(vals))
            item[f"{m}_std"] = float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0
        out.append(item)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_rows: List[Dict[str, Any]] = []
    vehicle_rows: List[pd.DataFrame] = []
    main_metric_rows: List[Dict[str, Any]] = []

    use_seeds = SEEDS[:MAX_RUNS]
    print("\n===== SISR-VRP execution-oriented post-processing =====")
    print("Pipeline: anchor-preserving A* post-processing")
    print(f"Seeds: {use_seeds[0]}-{use_seeds[-1]} | n={len(use_seeds)}")

    for idx, seed in enumerate(use_seeds, start=1):
        print(f"\n--- seed {seed} ({idx}/{len(use_seeds)}) ---")

        scene, path_json, main_row = export_sisr_in_memory(seed)
        bundle, vehicle_df = process_algorithm(scene, path_json, "SISR-VRP")
        s = bundle["summary"]

        seed_rows.append(s)
        vehicle_rows.append(vehicle_df.assign(seed=int(seed)))
        main_metric_rows.append(main_row)

        print(
            f"SISR-VRP | exec_len={s['total_execution_length']:.2f} | "
            f"exec_time={s['total_estimated_execution_time']:.2f} | "
            f"obs_clear={s['minimum_obstacle_clearance']:.2f} | "
            f"safe_clear={s['minimum_safe_clearance']:.2f} | "
            f"turn_ratio={s['total_turn_violation_ratio_percent']:.2f}%"
        )

    seed_df = pd.DataFrame(seed_rows)
    seed_csv = os.path.join(OUT_DIR, "sisr_execution_seed_results_20seed_astar_anchor.csv")
    seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")

    vehicle_df_all = pd.concat(vehicle_rows, ignore_index=True) if vehicle_rows else pd.DataFrame()
    vehicle_csv = os.path.join(OUT_DIR, "sisr_execution_vehicle_results_20seed_astar_anchor.csv")
    vehicle_df_all.to_csv(vehicle_csv, index=False, encoding="utf-8-sig")

    summary_rows = aggregate_exec_rows(seed_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "sisr_execution_summary_20seed_astar_anchor.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    main_metric_df = pd.DataFrame(main_metric_rows)
    main_metric_csv = os.path.join(OUT_DIR, "sisr_main_metric_check_20seed.csv")
    main_metric_df.to_csv(main_metric_csv, index=False, encoding="utf-8-sig")

    print("\n===== SISR-VRP execution summary =====")
    print(summary_df.to_string(index=False))
    print(
        "\nSaved to:\n"
        f"  {seed_csv}\n"
        f"  {vehicle_csv}\n"
        f"  {summary_csv}\n"
        f"  {main_metric_csv}"
    )
    print("\n说明：本脚本只重跑 SISR-VRP 的执行层后处理，不改变主实验、消融实验或统计检验结果。")


if __name__ == "__main__":
    main()
