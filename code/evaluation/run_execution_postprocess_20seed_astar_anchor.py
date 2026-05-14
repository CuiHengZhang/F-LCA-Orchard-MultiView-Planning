import os
import math
import json
import statistics
from copy import deepcopy
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

import F_LCA_tiered_refinement_pathboost as flca
from exp_suite_common_fair_budget import ExperimentProblem, SEEDS, result_row
from postprocess_execution_metrics_astar_anchor import process_algorithm


OUT_DIR = "execution_postprocess_20seed_astar_anchor"


def to_xy_list(points):
    return [[float(x), float(y)] for x, y in points]


def build_scene_dict_from_env(seed: int, env_cfg, env, tree_to_points, tree_to_required_points):
    scene = {
        "seed": int(seed),
        "width": float(env_cfg.width),
        "height": float(env_cfg.height),
        "vehicle_num": int(env_cfg.vehicle_num),
        "vehicle_battery": [float(v) for v in env_cfg.vehicle_battery],
        "start_positions": to_xy_list(env_cfg.start_positions),
        "trees": [
            {"tree_id": int(i), "x": float(x), "y": float(y)}
            for i, (x, y) in enumerate(env.tree_coords)
        ],
        "obstacles": [
            {"obstacle_id": int(i), "cx": float(ox), "cy": float(oy), "r": float(rr)}
            for i, (ox, oy, rr) in enumerate(env.obstacles)
        ],
        "tree_observation_points": {
            str(tid): to_xy_list(pts)
            for tid, pts in tree_to_points.items()
        },
        "tree_required_points": {
            str(tid): to_xy_list(pts)
            for tid, pts in tree_to_required_points.items()
        },
    }
    return scene


def export_flca_in_memory(seed: int) -> Tuple[dict, dict]:
    env_cfg = flca.EnvironmentConfig(random_seed=seed)
    cfg = flca.AlgorithmConfig(run_seed=seed, report_every=10**9)
    cfg.name = f"F-LCA-seed{seed}-batch-export"

    obstacles = flca.build_complex_orchard_obstacles(env_cfg)
    trees = flca.build_complex_orchard_trees(env_cfg, obstacles)
    env = flca.OrchardEnv(
        tree_coords=trees,
        vehicle_num=env_cfg.vehicle_num,
        vehicle_battery=env_cfg.vehicle_battery,
        start_positions=env_cfg.start_positions,
        obstacles=obstacles,
        obs_radius=env_cfg.obs_radius,
        safe_margin=env_cfg.safe_margin,
    )

    solver = flca.FusionLeafcutterAntAlgorithm(env=env, cfg=deepcopy(cfg))
    result = solver.run()

    scene = build_scene_dict_from_env(
        seed=seed,
        env_cfg=env_cfg,
        env=env,
        tree_to_points=solver.tree_to_points,
        tree_to_required_points=solver.tree_to_required_points,
    )

    best_path = {
        "seed": int(seed),
        "algorithm": "F-LCA",
        "summary": {
            "primary_cost": float(result.best_primary_cost),
            "total_cost": float(result.best_reference_total_cost),
            "total_distance": float(result.best_total_distance),
            "effective_coverage": float(result.effective_observation_coverage),
            "theoretical_coverage": float(result.theoretical_observation_coverage),
            "obstacle_penalty": float(result.obstacle_penalty),
            "energy_range_percent": float(result.energy_range_percent),
            "runtime_sec": float(result.runtime_sec),
        },
        "vehicles": {}
    }

    for vid in range(env_cfg.vehicle_num):
        start_pt = env.start_positions[vid]
        local_path = solver.best_paths.get(vid, [])
        full_path = [start_pt] + local_path
        best_path["vehicles"][str(vid)] = {
            "vehicle_id": int(vid),
            "start": [float(start_pt[0]), float(start_pt[1])],
            "path_points": to_xy_list(full_path),
            "local_path_points": to_xy_list(local_path),
            "tree_order": [int(t) for t in solver.best_orders.get(vid, [])],
            "assigned_tree_ids": [int(t) for t in solver.best_tree_groups.get(vid, [])],
        }

    return scene, best_path


def run_aco_with_details(seed: int, n_ants: int = 20, n_iter: int = 1500, candidate_k: int = 8):
    """
    基于你当前 fair-budget ACO 的同口径实现，额外返回 best_metrics，
    便于重建每辆车的访问序列与路径。
    """
    problem = ExperimentProblem(seed)
    rng = np.random.default_rng(seed)
    N, M = problem.n_trees, problem.vehicle_num
    pher_assign = np.ones((N, M), dtype=float)
    best_metrics = None
    best_cost = np.inf
    start = __import__("time").time()

    for _ in range(n_iter):
        iter_best = None
        iter_best_cost = np.inf
        for _ant in range(n_ants):
            remaining = set(range(N))
            groups = {vid: [] for vid in range(M)}
            orders = {vid: [] for vid in range(M)}
            current_pos = {vid: problem.env.start_positions[vid] for vid in range(M)}

            vehicle_cycle = list(rng.permutation(M))
            step = 0

            while remaining:
                vid = int(vehicle_cycle[step % M])
                rem_list = list(remaining)
                dists = np.array(
                    [
                        math.hypot(
                            problem.trees[t][0] - current_pos[vid][0],
                            problem.trees[t][1] - current_pos[vid][1]
                        )
                        for t in rem_list
                    ],
                    dtype=float,
                )
                use_k = min(candidate_k, len(rem_list))
                if len(rem_list) > use_k:
                    idxs = np.argpartition(dists, use_k - 1)[:use_k]
                    cand = [rem_list[i] for i in idxs]
                else:
                    cand = rem_list

                vals = []
                for tid in cand:
                    heur = 1.0 / (
                        1e-6
                        + math.hypot(
                            problem.trees[tid][0] - current_pos[vid][0],
                            problem.trees[tid][1] - current_pos[vid][1]
                        )
                    )
                    vals.append((pher_assign[tid, vid] ** 1.0) * (heur ** 2.0))

                vals = np.array(vals, dtype=float)
                probs = vals / vals.sum() if vals.sum() > 1e-12 else np.full_like(vals, 1.0 / len(vals))
                chosen = cand[int(rng.choice(len(cand), p=probs))]

                groups[vid].append(chosen)
                orders[vid].append(chosen)
                path = problem.evaluator._render_order_to_path(problem.env.start_positions[vid], orders[vid])
                current_pos[vid] = path[-1] if path else current_pos[vid]
                remaining.remove(chosen)
                step += 1

            metrics = problem.evaluate(groups, orders)
            cost = problem.scalar_cost(metrics)
            if cost < iter_best_cost:
                iter_best_cost = cost
                iter_best = metrics
            if cost < best_cost:
                best_cost = cost
                best_metrics = metrics

        pher_assign *= 0.90
        if iter_best is not None:
            deposit = 1.0 / max(1.0, iter_best_cost)
            for vid in range(M):
                for tid in iter_best["groups"][vid]:
                    pher_assign[tid, vid] += deposit

    runtime = __import__("time").time() - start
    assert best_metrics is not None
    row = result_row(seed, "ACO", best_metrics, runtime)
    return row, runtime, best_metrics, problem


def export_aco_in_memory(seed: int) -> Tuple[dict, dict]:
    row, _, best_metrics, problem = run_aco_with_details(seed)

    scene = build_scene_dict_from_env(
        seed=seed,
        env_cfg=problem.env_cfg,
        env=problem.env,
        tree_to_points=problem.evaluator.tree_to_points,
        tree_to_required_points=problem.evaluator.tree_to_required_points,
    )

    groups = best_metrics["groups"]
    orders = best_metrics["orders"]

    best_path = {
        "seed": int(seed),
        "algorithm": "ACO",
        "summary": {
            "primary_cost": float(row["primary_cost"]),
            "total_cost": float(row["total_cost"]),
            "total_distance": float(row["total_distance"]),
            "effective_coverage": float(row["effective_coverage"]),
            "theoretical_coverage": float(row["theoretical_coverage"]),
            "obstacle_penalty": float(row["obstacle_penalty"]),
            "energy_range_percent": float(row["energy_range_percent"]),
            "runtime_sec": float(row["runtime_sec"]),
        },
        "vehicles": {}
    }

    for vid in range(problem.vehicle_num):
        start_pt = problem.env.start_positions[vid]
        local_path = problem.evaluator._render_order_to_path(start_pt, orders[vid])
        full_path = [start_pt] + local_path

        best_path["vehicles"][str(vid)] = {
            "vehicle_id": int(vid),
            "start": [float(start_pt[0]), float(start_pt[1])],
            "path_points": to_xy_list(full_path),
            "local_path_points": to_xy_list(local_path),
            "tree_order": [int(t) for t in orders[vid]],
            "assigned_tree_ids": [int(t) for t in groups[vid]],
        }

    return scene, best_path


def aggregate_exec_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    print("\n===== 20-seed execution-oriented post-processing (anchor-preserving A*) =====")
    print("Algorithms: F-LCA vs ACO")
    print(f"Seeds: {SEEDS[0]}-{SEEDS[-1]}")

    for idx, seed in enumerate(SEEDS, start=1):
        print(f"\n--- seed {seed} ({idx}/{len(SEEDS)}) ---")

        # F-LCA
        scene_flca, path_flca = export_flca_in_memory(seed)
        flca_bundle, _ = process_algorithm(scene_flca, path_flca, "F-LCA")
        s1 = flca_bundle["summary"]
        seed_rows.append(s1)
        print(
            f"F-LCA | exec_len={s1['total_execution_length']:.2f} | "
            f"exec_time={s1['total_estimated_execution_time']:.2f} | "
            f"obs_clear={s1['minimum_obstacle_clearance']:.2f} | "
            f"safe_clear={s1['minimum_safe_clearance']:.2f} | "
            f"turn_ratio={s1['total_turn_violation_ratio_percent']:.2f}%"
        )

        # ACO
        scene_aco, path_aco = export_aco_in_memory(seed)
        aco_bundle, _ = process_algorithm(scene_aco, path_aco, "ACO")
        s2 = aco_bundle["summary"]
        seed_rows.append(s2)
        print(
            f"ACO   | exec_len={s2['total_execution_length']:.2f} | "
            f"exec_time={s2['total_estimated_execution_time']:.2f} | "
            f"obs_clear={s2['minimum_obstacle_clearance']:.2f} | "
            f"safe_clear={s2['minimum_safe_clearance']:.2f} | "
            f"turn_ratio={s2['total_turn_violation_ratio_percent']:.2f}%"
        )

    seed_df = pd.DataFrame(seed_rows)
    seed_csv = os.path.join(OUT_DIR, "execution_seed_results_20seed_astar_anchor.csv")
    seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")

    summary_rows = aggregate_exec_rows(seed_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "execution_summary_20seed_astar_anchor.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== Summary =====")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to:\n  {seed_csv}\n  {summary_csv}")
    print("\n说明：这里只重跑执行层后处理，不改主实验、统计检验与消融实验原始结果。")


if __name__ == "__main__":
    main()
