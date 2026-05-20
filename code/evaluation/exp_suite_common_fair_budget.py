import csv
import math
import os
from pathlib import Path
import random
import statistics
import time
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

import F_LCA_tiered_refinement_pathboost as flca


SEEDS = [42 + i for i in range(20)]

# Repository-aware output handling.
# The script files live in: <repo>/code/evaluation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FAIR_BUDGET_OUTPUT_DIR = OUTPUTS_DIR / "fair_budget_runs"


def resolve_output_path(path):
    """Resolve relative output paths under <repo>/outputs/fair_budget_runs/."""
    p = Path(path)
    if p.is_absolute():
        return p
    return FAIR_BUDGET_OUTPUT_DIR / p



class ExperimentProblem:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self.env_cfg = flca.EnvironmentConfig(random_seed=self.seed)
        self.alg_cfg = flca.AlgorithmConfig(run_seed=self.seed, report_every=10**9)
        self.obstacles = flca.build_complex_orchard_obstacles(self.env_cfg)
        self.trees = flca.build_complex_orchard_trees(self.env_cfg, self.obstacles)
        self.env = flca.OrchardEnv(
            tree_coords=self.trees,
            vehicle_num=self.env_cfg.vehicle_num,
            vehicle_battery=self.env_cfg.vehicle_battery,
            start_positions=self.env_cfg.start_positions,
            obstacles=self.obstacles,
            obs_radius=self.env_cfg.obs_radius,
            safe_margin=self.env_cfg.safe_margin,
        )
        # evaluator object; do not call run()
        self.evaluator = flca.FusionLeafcutterAntAlgorithm(env=self.env, cfg=deepcopy(self.alg_cfg))
        self.n_trees = self.evaluator.n_trees
        self.vehicle_num = self.env_cfg.vehicle_num
        self.tree_complexity = self.evaluator.tree_complexity.copy()

    def normalized_loads(self, groups: Dict[int, List[int]]) -> np.ndarray:
        vals = []
        for vid in range(self.vehicle_num):
            load = sum(float(self.tree_complexity[tid]) for tid in groups.get(vid, []))
            vals.append(load / max(1e-9, self.env.vehicle_battery[vid]))
        return np.array(vals, dtype=float)

    def task_ratios(self, groups: Dict[int, List[int]]) -> List[float]:
        total = max(1, sum(len(v) for v in groups.values()))
        return [100.0 * len(groups.get(vid, [])) / total for vid in range(self.vehicle_num)]

    def repair_groups(self, groups: Dict[int, List[int]]) -> Dict[int, List[int]]:
        # unique ownership
        owner = {}
        for vid in range(self.vehicle_num):
            for tid in groups.get(vid, []):
                owner[tid] = vid
        repaired = {vid: [] for vid in range(self.vehicle_num)}
        for tid in range(self.n_trees):
            vid = owner.get(tid)
            if vid is None:
                # nearest start as fallback
                dists = [
                    math.hypot(self.trees[tid][0] - self.env.start_positions[v][0], self.trees[tid][1] - self.env.start_positions[v][1])
                    for v in range(self.vehicle_num)
                ]
                vid = int(np.argmin(dists))
            repaired[vid].append(tid)
        # ensure non-empty
        for vid in range(self.vehicle_num):
            if repaired[vid]:
                continue
            donor = max(range(self.vehicle_num), key=lambda v: len(repaired[v]))
            if len(repaired[donor]) <= 1:
                continue
            donor_center = np.mean(np.array([self.trees[t] for t in repaired[donor]], dtype=float), axis=0)
            tid = max(
                repaired[donor],
                key=lambda t: math.hypot(self.trees[t][0] - donor_center[0], self.trees[t][1] - donor_center[1]),
            )
            repaired[donor].remove(tid)
            repaired[vid].append(tid)
        return repaired

    def repair_orders(self, groups: Dict[int, List[int]], orders: Dict[int, List[int]]) -> Dict[int, List[int]]:
        out = {}
        for vid in range(self.vehicle_num):
            group_set = set(groups.get(vid, []))
            seq = [t for t in orders.get(vid, []) if t in group_set]
            for tid in groups.get(vid, []):
                if tid not in seq:
                    seq.append(tid)
            out[vid] = seq
        return out

    def homogeneous_partition(self) -> Dict[int, List[int]]:
        centers = np.array(self.env.start_positions, dtype=float)
        quotas = np.full(self.vehicle_num, np.sum(self.tree_complexity) / self.vehicle_num, dtype=float)
        tree_order = list(range(self.n_trees))
        tree_order.sort(
            key=lambda tid: np.min(np.linalg.norm(centers - self.evaluator.tree_coords_np[tid], axis=1)) + 3.5 * self.tree_complexity[tid],
            reverse=True,
        )
        assigned = {i: [] for i in range(self.vehicle_num)}
        for _ in range(6):
            assigned = {i: [] for i in range(self.vehicle_num)}
            loads = np.zeros(self.vehicle_num, dtype=float)
            for tid in tree_order:
                tree = self.evaluator.tree_coords_np[tid]
                scores = []
                for vid in range(self.vehicle_num):
                    dist = np.linalg.norm(tree - centers[vid]) + 1e-6
                    load_after = loads[vid] + self.tree_complexity[tid]
                    load_pressure = load_after / (quotas[vid] + 1e-6)
                    start_bias = math.hypot(tree[0] - self.env.start_positions[vid][0], tree[1] - self.env.start_positions[vid][1])
                    obstacle = self.env.obstacle_pressure((tree[0], tree[1]))
                    overload = max(0.0, load_pressure - 1.03)
                    score = dist * (1.0 + 0.82 * load_pressure) + 0.10 * start_bias + 3.2 * obstacle + 28.0 * overload
                    scores.append(score)
                best_vid = int(np.argmin(scores))
                assigned[best_vid].append(tid)
                loads[best_vid] += self.tree_complexity[tid]
            for vid in range(self.vehicle_num):
                if assigned[vid]:
                    coords = self.evaluator.tree_coords_np[assigned[vid]]
                    weights = self.tree_complexity[assigned[vid]]
                    centers[vid] = np.average(coords, axis=0, weights=weights)
        return self.repair_groups(assigned)

    def decode_from_priority(self, groups: Dict[int, List[int]], priority: np.ndarray) -> Dict[int, List[int]]:
        orders = {}
        for vid in range(self.vehicle_num):
            orders[vid] = sorted(groups.get(vid, []), key=lambda t: float(priority[t]))
        return orders

    def evaluate(self, groups: Dict[int, List[int]], orders: Dict[int, List[int]]) -> Dict[str, Any]:
        groups = self.repair_groups(groups)
        orders = self.repair_orders(groups, orders)
        paths = {}
        for vid in range(self.vehicle_num):
            paths[vid] = self.evaluator._render_order_to_path(self.env.start_positions[vid], orders[vid])
        metrics = self.evaluator._evaluate_paths_metrics(paths, custom_tree_groups=groups)
        metrics["task_ratios"] = self.task_ratios(groups)
        metrics["groups"] = deepcopy(groups)
        metrics["orders"] = deepcopy(orders)
        return metrics

    def scalar_cost(self, metrics: Dict[str, Any]) -> float:
        return float(metrics["total_cost"])


def result_row(seed: int, algo: str, metrics: Dict[str, Any], runtime_sec: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {
        "seed": int(seed),
        "algorithm": algo,
        "primary_cost": float(metrics["primary_cost"]),
        "total_cost": float(metrics["total_cost"]),
        "total_distance": float(metrics["total_distance"]),
        "effective_coverage": float(metrics["effective_cov"] * 100.0),
        "theoretical_coverage": float(metrics["theoretical_cov"] * 100.0),
        "obstacle_penalty": float(metrics["obstacle_penalty"]),
        "energy_range_percent": float(metrics["energy_range_percent"]),
        "incomplete_tree_ratio": float(metrics["incomplete_tree_ratio"] * 100.0),
        "runtime_sec": float(runtime_sec),
        "task_ratio_v1": float(metrics["task_ratios"][0]),
        "task_ratio_v2": float(metrics["task_ratios"][1]),
        "task_ratio_v3": float(metrics["task_ratios"][2]),
        "task_ratio_v4": float(metrics["task_ratios"][3]),
    }
    if extra:
        row.update(extra)
    return row


def aggregate_rows(rows: List[Dict[str, Any]], group_key: str = "algorithm") -> List[Dict[str, Any]]:
    if not rows:
        return []
    numeric_keys = [
        "primary_cost", "total_cost", "total_distance", "effective_coverage", "theoretical_coverage",
        "obstacle_penalty", "energy_range_percent", "incomplete_tree_ratio", "runtime_sec",
        "task_ratio_v1", "task_ratio_v2", "task_ratio_v3", "task_ratio_v4",
    ]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    out = []
    for g, rs in groups.items():
        item = {group_key: g, "n": len(rs)}
        for k in numeric_keys:
            vals = [float(r[k]) for r in rs if k in r]
            if not vals:
                continue
            item[f"{k}_mean"] = float(statistics.fmean(vals))
            item[f"{k}_std"] = float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0
        out.append(item)
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]):
    path = resolve_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def history_rows(seed: int, algo: str, history: List[float]) -> List[Dict[str, Any]]:
    return [
        {
            "seed": int(seed),
            "algorithm": str(algo),
            "iter": int(i + 1),
            "best_primary_cost": float(v),
        }
        for i, v in enumerate(history)
    ]


def aggregate_history_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    groups: Dict[Tuple[str, int], List[float]] = {}
    for row in rows:
        key = (str(row["algorithm"]), int(row["iter"]))
        groups.setdefault(key, []).append(float(row["best_primary_cost"]))

    out = []
    for (algo, it), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        out.append(
            {
                "algorithm": algo,
                "iter": int(it),
                "best_primary_cost_mean": float(statistics.fmean(vals)),
                "best_primary_cost_std": float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0,
                "n": len(vals),
            }
        )
    return out


def save_results_bundle(out_dir: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_dir = resolve_output_path(out_dir)
    seed_csv = out_dir / "seed_results.csv"
    summary_csv = out_dir / "summary.csv"
    write_csv(seed_csv, rows)
    summary = aggregate_rows(rows)
    write_csv(summary_csv, summary)
    return summary


def save_convergence_bundle(out_dir: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_dir = resolve_output_path(out_dir)
    seed_csv = out_dir / "convergence_seed.csv"
    summary_csv = out_dir / "convergence_summary.csv"
    write_csv(seed_csv, rows)
    summary = aggregate_history_rows(rows)
    write_csv(summary_csv, summary)
    return summary


def run_flca(seed: int, ablation: Optional[str] = None, return_history: bool = False):
    env_cfg = flca.EnvironmentConfig(random_seed=seed)
    alg_cfg = flca.AlgorithmConfig(run_seed=seed, report_every=10**9)
    cfg = deepcopy(alg_cfg)
    tag = "F-LCA"
    if ablation == "no_hetero":
        tag = "F-LCA-no-hetero"
        cfg.partition_battery_power = 0.0
        cfg.partition_quota_power = 0.0
        cfg.repartition_energy_gap = 1e9
        cfg.repartition_load_gap = 1e9
        cfg.forced_repartition_after_stagnation = cfg.max_iter + 1
        cfg.layered_refine_after_stagnation = cfg.max_iter + 1
    elif ablation == "no_orthogonal":
        tag = "F-LCA-no-orthogonal"
        cfg.min_points_per_tree = 1
        cfg.preferred_points_per_tree = 1
    elif ablation == "no_layered":
        tag = "F-LCA-no-layered"
        cfg.layered_refine_after_stagnation = cfg.max_iter + 1
        cfg.forced_repartition_after_stagnation = cfg.max_iter + 1
        cfg.exchange_after_stagnation = cfg.max_iter + 1
        cfg.rebuild_cooldown = cfg.max_iter + 1
    cfg.name = tag
    start = time.time()
    result = flca.run_single_case(env_cfg, cfg, use_layered_refine=(ablation != "no_layered"), tag=tag)
    runtime = time.time() - start
    metrics = {
        "primary_cost": result.best_primary_cost,
        "total_cost": result.best_reference_total_cost,
        "total_distance": result.best_total_distance,
        "effective_cov": result.effective_observation_coverage / 100.0 if result.effective_observation_coverage > 1.0 else result.effective_observation_coverage,
        "theoretical_cov": result.theoretical_observation_coverage / 100.0 if result.theoretical_observation_coverage > 1.0 else result.theoretical_observation_coverage,
        "obstacle_penalty": result.obstacle_penalty,
        "energy_range_percent": result.energy_range_percent,
        "incomplete_tree_ratio": result.incomplete_tree_ratio / 100.0 if result.incomplete_tree_ratio > 1.0 else result.incomplete_tree_ratio,
        "task_ratios": [0.0, 0.0, 0.0, 0.0],
    }
    # recover last task ratios from report rows if available
    if result.report_rows:
        last = result.report_rows[-1]
        metrics["task_ratios"] = [
            float(last.get("best_task_ratio_v1", 0.0)),
            float(last.get("best_task_ratio_v2", 0.0)),
            float(last.get("best_task_ratio_v3", 0.0)),
            float(last.get("best_task_ratio_v4", 0.0)),
        ]
    extra = {
        "layered_refine_trigger_count": int(getattr(result, "layered_refine_trigger_count", 0)),
        "layered_refine_success_count": int(getattr(result, "layered_refine_success_count", 0)),
    }
    history = list(getattr(result, "history_best_primary", []))
    if not history:
        history = [float(result.best_primary_cost)]
    row = result_row(seed, tag, metrics, runtime, extra)
    if return_history:
        return row, runtime, history
    return row, runtime


# ---------------------------- Baselines ----------------------------

def _candidate_from_assignment(problem: ExperimentProblem, assignment: List[int], priority: np.ndarray) -> Dict[str, Any]:
    groups = {vid: [] for vid in range(problem.vehicle_num)}
    for tid, vid in enumerate(assignment):
        groups[int(vid)].append(tid)
    groups = problem.repair_groups(groups)
    orders = problem.decode_from_priority(groups, priority)
    return problem.evaluate(groups, orders)


def run_pso(seed: int, n_particles: int = 20, n_iter: int = 1500, return_history: bool = False):
    problem = ExperimentProblem(seed)
    rng = np.random.default_rng(seed)
    N, M = problem.n_trees, problem.vehicle_num

    pos_assign = rng.normal(size=(n_particles, N, M))
    vel_assign = np.zeros_like(pos_assign)
    pos_order = rng.normal(size=(n_particles, N))
    vel_order = np.zeros_like(pos_order)

    pbest_cost = np.full(n_particles, np.inf)
    pbest_assign = pos_assign.copy()
    pbest_order = pos_order.copy()
    gbest_cost = np.inf
    gbest_assign = pos_assign[0].copy()
    gbest_order = pos_order[0].copy()
    gbest_metrics = None
    history: List[float] = []

    start = time.time()
    for _ in range(n_iter):
        for i in range(n_particles):
            assignment = np.argmax(pos_assign[i], axis=1)
            metrics = _candidate_from_assignment(problem, assignment.tolist(), pos_order[i])
            cost = problem.scalar_cost(metrics)
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_assign[i] = pos_assign[i].copy()
                pbest_order[i] = pos_order[i].copy()
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_assign = pos_assign[i].copy()
                gbest_order = pos_order[i].copy()
                gbest_metrics = metrics
        w, c1, c2 = 0.72, 1.45, 1.45
        r1 = rng.random(size=vel_assign.shape)
        r2 = rng.random(size=vel_assign.shape)
        vel_assign = w * vel_assign + c1 * r1 * (pbest_assign - pos_assign) + c2 * r2 * (gbest_assign - pos_assign)
        pos_assign = pos_assign + vel_assign
        r1o = rng.random(size=vel_order.shape)
        r2o = rng.random(size=vel_order.shape)
        vel_order = w * vel_order + c1 * r1o * (pbest_order - pos_order) + c2 * r2o * (gbest_order - pos_order)
        pos_order = pos_order + vel_order
        if gbest_metrics is not None:
            history.append(float(gbest_metrics["primary_cost"]))
    runtime = time.time() - start
    assert gbest_metrics is not None
    row = result_row(seed, "PSO", gbest_metrics, runtime)
    if return_history:
        return row, runtime, history
    return row, runtime


def _init_ga_population(problem: ExperimentProblem, rng: np.random.Generator, pop_size: int) -> List[Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, Any]]]:
    pop = []
    for _ in range(pop_size):
        groups = problem.homogeneous_partition()
        # random perturb assignments
        for _ in range(6):
            donor = int(rng.integers(problem.vehicle_num))
            receiver = int(rng.integers(problem.vehicle_num))
            if donor == receiver or len(groups[donor]) <= 1:
                continue
            tid = int(rng.choice(groups[donor]))
            groups[donor].remove(tid)
            groups[receiver].append(tid)
        groups = problem.repair_groups(groups)
        priority = rng.normal(size=problem.n_trees)
        orders = problem.decode_from_priority(groups, priority)
        metrics = problem.evaluate(groups, orders)
        pop.append((groups, orders, metrics))
    return pop


def _ga_crossover(problem: ExperimentProblem, rng: np.random.Generator, p1, p2):
    g1, o1, _ = p1
    g2, o2, _ = p2
    owner1 = {}
    owner2 = {}
    for vid in range(problem.vehicle_num):
        for tid in g1[vid]: owner1[tid] = vid
        for tid in g2[vid]: owner2[tid] = vid
    child_assign = []
    for tid in range(problem.n_trees):
        child_assign.append(owner1[tid] if rng.random() < 0.5 else owner2[tid])
    groups = {vid: [] for vid in range(problem.vehicle_num)}
    for tid, vid in enumerate(child_assign):
        groups[int(vid)].append(tid)
    groups = problem.repair_groups(groups)
    orders = {}
    for vid in range(problem.vehicle_num):
        seq = [t for t in o1[vid] if t in groups[vid]] + [t for t in o2[vid] if t in groups[vid] and t not in o1[vid]]
        for t in groups[vid]:
            if t not in seq:
                seq.append(t)
        orders[vid] = seq
    return groups, orders


def _ga_mutate(problem: ExperimentProblem, rng: np.random.Generator, groups, orders):
    groups = deepcopy(groups)
    orders = deepcopy(orders)
    for _ in range(3):
        op = int(rng.integers(3))
        if op == 0:
            donor = int(rng.integers(problem.vehicle_num))
            receiver = int(rng.integers(problem.vehicle_num))
            if donor != receiver and len(groups[donor]) > 1:
                tid = int(rng.choice(groups[donor]))
                groups[donor].remove(tid)
                groups[receiver].append(tid)
        else:
            vid = int(rng.integers(problem.vehicle_num))
            if len(orders[vid]) >= 2:
                i, j = sorted(rng.choice(len(orders[vid]), size=2, replace=False).tolist())
                if op == 1:
                    orders[vid][i], orders[vid][j] = orders[vid][j], orders[vid][i]
                else:
                    orders[vid][i:j+1] = list(reversed(orders[vid][i:j+1]))
    groups = problem.repair_groups(groups)
    orders = problem.repair_orders(groups, orders)
    return groups, orders


def run_ga(seed: int, pop_size: int = 20, n_gen: int = 1500, return_history: bool = False):
    problem = ExperimentProblem(seed)
    rng = np.random.default_rng(seed)
    pop = _init_ga_population(problem, rng, pop_size)

    best = None
    best_cost = np.inf
    history: List[float] = []

    start = time.time()
    for _ in range(n_gen):
        pop.sort(key=lambda x: problem.scalar_cost(x[2]))
        elites = pop[: max(2, pop_size // 5)]
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1 = elites[int(rng.integers(len(elites)))]
            p2 = elites[int(rng.integers(len(elites)))]
            groups, orders = _ga_crossover(problem, rng, p1, p2)
            groups, orders = _ga_mutate(problem, rng, groups, orders)
            metrics = problem.evaluate(groups, orders)
            new_pop.append((groups, orders, metrics))
        pop = new_pop
        pop.sort(key=lambda x: problem.scalar_cost(x[2]))
        current_best = pop[0][2]
        current_cost = problem.scalar_cost(current_best)
        if current_cost < best_cost:
            best_cost = current_cost
            best = current_best
        history.append(float(best["primary_cost"]))

    runtime = time.time() - start
    assert best is not None
    row = result_row(seed, "GA", best, runtime)
    if return_history:
        return row, runtime, history
    return row, runtime

def run_aco(
    seed: int,
    n_ants: int = 20,
    n_iter: int = 1500,
    candidate_k: int = 8,
    return_history: bool = False,
    return_details: bool = False,
):
    """
    Fair ACO baseline:
    - same environment / observation model / evaluator as F-LCA
    - NO battery-aware vehicle selection
    - NO explicit load-balance reward
    - vehicles construct routes in a neutral cyclic schedule
    """
    problem = ExperimentProblem(seed)
    rng = np.random.default_rng(seed)
    N, M = problem.n_trees, problem.vehicle_num
    pher_assign = np.ones((N, M), dtype=float)
    best_metrics = None
    best_cost = np.inf
    history: List[float] = []
    start = time.time()

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
                    [math.hypot(problem.trees[t][0] - current_pos[vid][0], problem.trees[t][1] - current_pos[vid][1]) for t in rem_list],
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
                    heur = 1.0 / (1e-6 + math.hypot(problem.trees[tid][0] - current_pos[vid][0], problem.trees[tid][1] - current_pos[vid][1]))
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
        if best_metrics is not None:
            history.append(float(best_metrics["primary_cost"]))

    runtime = time.time() - start
    assert best_metrics is not None
    row = result_row(seed, "ACO", best_metrics, runtime)
    details = {
        "groups": deepcopy(best_metrics["groups"]),
        "orders": deepcopy(best_metrics["orders"]),
    }
    if return_history and return_details:
        return row, runtime, history, details
    if return_history:
        return row, runtime, history
    if return_details:
        return row, runtime, details
    return row, runtime


def print_brief(rows: List[Dict[str, Any]]):
    for row in aggregate_rows(rows):
        print(
            f"{row['algorithm']}: "
            f"primary={row.get('primary_cost_mean', float('nan')):.2f}±{row.get('primary_cost_std', 0.0):.2f}, "
            f"distance={row.get('total_distance_mean', float('nan')):.2f}±{row.get('total_distance_std', 0.0):.2f}, "
            f"energy={row.get('energy_range_percent_mean', float('nan')):.2f}±{row.get('energy_range_percent_std', 0.0):.2f}, "
            f"time={row.get('runtime_sec_mean', float('nan')):.2f}s"
        )


def print_saved_paths(out_dir: str):
    out_dir = resolve_output_path(out_dir)
    print(f"\nSaved to:\n  {out_dir / 'seed_results.csv'}\n  {out_dir / 'summary.csv'}")
