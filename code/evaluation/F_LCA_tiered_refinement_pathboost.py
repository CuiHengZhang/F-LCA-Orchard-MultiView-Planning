
import csv
import math
import os
from pathlib import Path
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EnvironmentConfig:
    width: float = 120.0
    height: float = 120.0
    tree_num: int = 80
    vehicle_num: int = 4
    vehicle_battery: List[float] = field(default_factory=lambda: [100.0, 80.0, 60.0, 40.0])
    start_positions: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (120.0, 0.0), (0.0, 120.0), (120.0, 120.0)]
    )
    # 这里只保留占位，真正复杂障碍由 build_complex_orchard_obstacles() 生成
    obstacles: List[Tuple[float, float, float]] = field(default_factory=list)
    obs_radius: float = 2.0
    safe_margin: float = 1.5
    random_seed: int = 42


@dataclass
class AlgorithmConfig:
    max_iter: int = 1500
    report_every: int = 100
    grid_step: float = 1.0

    base_candidate_k: int = 10
    late_candidate_k: int = 9
    explore_candidate_boost: int = 4
    route_candidates: int = 4

    alpha_start: float = 1.00
    alpha_end: float = 1.20
    beta_start: float = 2.80
    beta_end: float = 3.80
    evap_start: float = 0.94
    evap_end: float = 0.90

    lambda_balance: float = 0.68
    lambda_obstacle: float = 0.19
    lambda_missing_point: float = 18.0
    lambda_tree_incomplete: float = 22.0
    lambda_energy_guard: float = 18.0

    min_points_per_tree: int = 4
    preferred_points_per_tree: int = 4
    max_fallback_ring: int = 4

    local_search_every: int = 20
    local_search_every_late: int = 12
    local_search_trials: int = 10
    local_search_trials_late: int = 20
    best_polish_trials: int = 24

    stagnation_limit: int = 100
    stagnation_delta: float = 0.12
    diversity_threshold: float = 0.25

    layered_refine_after_stagnation: int = 90
    layered_refine_cooldown: int = 100
    layered_refine_move_count: int = 4
    layered_refine_energy_gain: float = 0.8
    layered_refine_primary_relax: float = 1.020

    destroy_after_stagnation: int = 85
    destroy_ratio: float = 0.25
    anti_pheromone_strength: float = 0.20
    anti_pheromone_decay: float = 0.92

    repartition_check_every: int = 25
    repartition_energy_gap: float = 0.08
    repartition_load_gap: float = 0.10
    repair_rounds: int = 3
    repair_top_ratio: float = 0.35
    repair_gain_eps: float = 0.02
    boundary_pool_ratio: float = 0.50
    repartition_move_count: int = 6

    greedy_start: float = 0.20
    greedy_mid: float = 0.68
    greedy_end: float = 0.88
    late_stage_ratio: float = 0.50

    exchange_after_stagnation: int = 30
    exchange_cooldown: int = 25
    exchange_top_ratio: float = 0.55
    exchange_try_count: int = 8

    partition_quota_power: float = 1.00
    partition_battery_power: float = 0.85
    rebuild_cooldown: int = 100
    rebuild_primary_relax: float = 1.020
    exchange_primary_relax: float = 1.020
    exchange_energy_drop: float = 1.5
    early_stop_min_iter: int = 700
    early_stop_patience: int = 240
    enable_early_stop: bool = False

    target_energy_percent: float = 18.0
    forced_repartition_after_stagnation: int = 90
    forced_repartition_cooldown: int = 70

    name: str = "F-LCA-Tiered-Refinement-PathBoost"
    run_seed: int = 42


@dataclass
class RunResult:
    best_primary_cost: float
    best_reference_total_cost: float
    best_total_distance: float
    runtime_sec: float
    effective_observation_coverage: float
    theoretical_observation_coverage: float
    obstacle_penalty: float
    energy_range_percent: float
    layered_refine_trigger_count: int
    layered_refine_success_count: int
    incomplete_tree_ratio: float
    report_rows: List[Dict[str, float]]
    history_best_primary: List[float]


class OrchardEnv:
    def __init__(
        self,
        tree_coords: List[Tuple[float, float]],
        vehicle_num: int,
        vehicle_battery: List[float],
        start_positions: List[Tuple[float, float]],
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        obs_radius: float = 2.0,
        safe_margin: float = 1.2,
    ):
        self.tree_coords = [(float(x), float(y)) for x, y in tree_coords]
        self.vehicle_num = int(vehicle_num)
        self.vehicle_battery = [float(v) for v in vehicle_battery]
        self.start_positions = [(float(x), float(y)) for x, y in start_positions]
        self.obstacles = obstacles if obstacles else []
        self.obs_radius = float(obs_radius)
        self.safe_margin = float(safe_margin)

        total_battery = max(1e-9, sum(self.vehicle_battery))
        self.battery_weights = [v / total_battery for v in self.vehicle_battery]
        self.tree_coords_np = np.array(self.tree_coords, dtype=float)

    def line_intersects_circle(self, p1, p2, center, radius) -> bool:
        px, py = p1
        qx, qy = p2
        cx, cy = center
        dx = qx - px
        dy = qy - py
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return math.hypot(px - cx, py - cy) <= radius
        t = clip(((cx - px) * dx + (cy - py) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        nx = px + t * dx
        ny = py + t * dy
        return math.hypot(nx - cx, ny - cy) <= radius

    def point_is_safe(self, p) -> bool:
        for ox, oy, rr in self.obstacles:
            if math.hypot(p[0] - ox, p[1] - oy) <= rr + self.safe_margin:
                return False
        return True

    def point_clearance(self, p) -> float:
        if not self.obstacles:
            return 1e9
        return min(math.hypot(p[0] - ox, p[1] - oy) - (rr + self.safe_margin) for ox, oy, rr in self.obstacles)

    def obstacle_pressure(self, p) -> float:
        if not self.obstacles:
            return 0.0
        vals = []
        for ox, oy, rr in self.obstacles:
            d = math.hypot(p[0] - ox, p[1] - oy)
            vals.append(max(0.0, (rr + self.safe_margin + 3.0) - d))
        return float(sum(vals))

    def segment_obstacle_penalty(self, p1, p2, hard_penalty=110.0, soft_gain=4.5) -> float:
        penalty = 0.0
        for ox, oy, rr in self.obstacles:
            safe_r = rr + self.safe_margin
            if self.line_intersects_circle(p1, p2, (ox, oy), safe_r):
                penalty += hard_penalty
            else:
                px, py = p1
                qx, qy = p2
                dx = qx - px
                dy = qy - py
                if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                    dist = math.hypot(px - ox, py - oy)
                else:
                    t = clip(((ox - px) * dx + (oy - py) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
                    dist = math.hypot(px + t * dx - ox, py + t * dy - oy)
                threshold = safe_r + 1.4
                if dist < threshold:
                    penalty += soft_gain * (threshold - dist)
        return penalty


class FusionLeafcutterAntAlgorithm:
    def __init__(self, env: OrchardEnv, cfg: AlgorithmConfig):
        self.env = env
        self.cfg = cfg
        random.seed(cfg.run_seed)
        np.random.seed(cfg.run_seed)

        self.n_trees = len(self.env.tree_coords)
        self.tree_coords_np = np.array(self.env.tree_coords, dtype=float)

        self.vehicle_tree_groups: Dict[int, List[int]] = {i: [] for i in range(self.env.vehicle_num)}
        self.vehicle_paths: Dict[int, List[Tuple[float, float]]] = {}
        self.vehicle_orders: Dict[int, List[int]] = {}
        self.vehicle_raw_distance: Dict[int, float] = {}
        self.vehicle_energy_ratio: Dict[int, float] = {}

        self.best_global_cost = float("inf")  # 参考综合值，不再作为主判优显示
        self.best_primary_cost = float("inf")  # 主判优指标：总距离 + 障碍加权
        self.best_global_distance = float("inf")
        self.best_paths: Dict[int, List[Tuple[float, float]]] = {}
        self.best_orders: Dict[int, List[int]] = {}
        self.best_vehicle_distances: List[float] = []
        self.best_effective_coverage_ratio = 0.0
        self.best_theoretical_coverage_ratio = 0.0
        self.best_obstacle_penalty = float("inf")
        self.best_energy_range_percent = 0.0
        self.best_incomplete_tree_ratio = 1.0
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_tree_groups: Dict[int, List[int]] = {}
        self.best_task_ratios: List[float] = [0.0 for _ in range(env.vehicle_num)]

        self.last_improve_iter = 0
        self.report_rows: List[Dict[str, float]] = []
        self.history_best_primary: List[float] = []
        self.relax_until = -1
        self.recent_visit_grid = None
        self.last_exchange_iter = -10**9
        self.last_rebuild_iter = -10**9
        self.last_layered_refine_iter = -10**9

        self.layered_refine_trigger_count = 0
        self.layered_refine_success_count = 0

        self._generate_observation_points()
        self._init_pheromone()
        self._precompute_grid_indices()
        self._build_node_graph()
        self._precompute_required_point_helpers()
        self.tree_complexity = self._estimate_tree_complexity()
        self.recent_visit_grid = np.zeros_like(self.dynamic_grid, dtype=float)

        # 惰性衰减：避免每一代都对整张网格做全量乘法
        self.dynamic_grid_scale = 1.0
        self.recent_visit_scale = 1.0

        # 单车顺序评估缓存：减少局部搜索/重构阶段的重复重建与重复打分
        self.order_eval_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[float, List[Tuple[float, float]]]] = {}

    def _generate_observation_points(self):
        self.tree_to_points: Dict[int, List[Tuple[float, float]]] = {}
        self.tree_to_required_points: Dict[int, List[Tuple[float, float]]] = {}
        self.tree_to_available_count: Dict[int, int] = {}
        self.tree_to_required_count: Dict[int, int] = {}

        base_angles = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
        rotate_offsets = [0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 8.0, -math.pi / 8.0]

        for tid, (x, y) in enumerate(self.env.tree_coords):
            selected: List[Tuple[float, float]] = []
            used = set()

            for base in base_angles:
                found = None
                for ring in range(self.cfg.max_fallback_ring + 1):
                    radius = max(0.65, self.env.obs_radius - 0.32 * ring)
                    for delta in rotate_offsets:
                        ang = base + delta
                        p = (x + radius * math.cos(ang), y + radius * math.sin(ang))
                        key = (round(p[0], 6), round(p[1], 6))
                        if key in used:
                            continue
                        if self.env.point_is_safe(p):
                            found = p
                            used.add(key)
                            break
                    if found is not None:
                        break
                if found is not None:
                    selected.append(found)

            if len(selected) < self.cfg.preferred_points_per_tree:
                extra_angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
                for ring in range(self.cfg.max_fallback_ring + 2):
                    radius = max(0.55, self.env.obs_radius - 0.22 * ring)
                    for ang in extra_angles:
                        p = (x + radius * math.cos(float(ang)), y + radius * math.sin(float(ang)))
                        key = (round(p[0], 6), round(p[1], 6))
                        if key in used:
                            continue
                        if self.env.point_is_safe(p):
                            selected.append(p)
                            used.add(key)
                            if len(selected) >= self.cfg.preferred_points_per_tree:
                                break
                    if len(selected) >= self.cfg.preferred_points_per_tree:
                        break

            if not selected:
                selected = [(x, y)]

            required_count = min(self.cfg.min_points_per_tree, len(selected))
            required_count = max(required_count, 1)

            selected.sort(key=lambda p: math.atan2(p[1] - y, p[0] - x))
            self.tree_to_points[tid] = selected
            self.tree_to_required_points[tid] = selected[:required_count]
            self.tree_to_available_count[tid] = len(selected)
            self.tree_to_required_count[tid] = required_count

    def _init_pheromone(self):
        xs = [x for x, _ in self.env.tree_coords] + [x for x, _ in self.env.start_positions]
        ys = [y for _, y in self.env.tree_coords] + [y for _, y in self.env.start_positions]
        self.xmin = min(xs) - 10.0
        self.ymin = min(ys) - 10.0
        width = int((max(xs) - self.xmin + 10.0) / self.cfg.grid_step) + 1
        height = int((max(ys) - self.ymin + 10.0) / self.cfg.grid_step) + 1
        self.dynamic_grid = np.full((width, height), 0.12, dtype=float)

    def _precompute_grid_indices(self):
        idxs = []
        for (x, y) in self.env.tree_coords:
            i = int(clip(int((x - self.xmin) / self.cfg.grid_step), 0, self.dynamic_grid.shape[0] - 1))
            j = int(clip(int((y - self.ymin) / self.cfg.grid_step), 0, self.dynamic_grid.shape[1] - 1))
            idxs.append((i, j))
        self.tree_grid_idx = np.array(idxs, dtype=int)

    def _point_key(self, p):
        return round(float(p[0]), 6), round(float(p[1]), 6)

    def _precompute_required_point_helpers(self):
        self.tree_required_point_indices: Dict[int, List[int]] = {}
        self.tree_required_points_sorted: Dict[int, List[Tuple[float, float]]] = {}
        for tid, pts in self.tree_to_required_points.items():
            self.tree_required_point_indices[tid] = [self._node_idx(p) for p in pts]
            center = self.tree_coords_np[tid]
            self.tree_required_points_sorted[tid] = sorted(
                pts,
                key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]),
            )

    def _materialize_dynamic_grid(self):
        if abs(self.dynamic_grid_scale - 1.0) > 1e-12:
            self.dynamic_grid *= self.dynamic_grid_scale
            self.dynamic_grid_scale = 1.0

    def _materialize_recent_grid(self):
        if abs(self.recent_visit_scale - 1.0) > 1e-12:
            self.recent_visit_grid *= self.recent_visit_scale
            self.recent_visit_scale = 1.0

    def _maybe_normalize_grid_scales(self):
        if self.dynamic_grid_scale < 1e-4 or self.dynamic_grid_scale > 1e4:
            self._materialize_dynamic_grid()
        if self.recent_visit_scale < 1e-4 or self.recent_visit_scale > 1e4:
            self._materialize_recent_grid()

    def _build_node_graph(self):
        self.node_coords: List[Tuple[float, float]] = []
        self.node_index: Dict[Tuple[float, float], int] = {}

        def add_point(p):
            key = self._point_key(p)
            if key not in self.node_index:
                self.node_index[key] = len(self.node_coords)
                self.node_coords.append((float(p[0]), float(p[1])))
            return self.node_index[key]

        self.start_node_idx = [add_point(p) for p in self.env.start_positions]
        self.tree_point_idx = {tid: [add_point(p) for p in pts] for tid, pts in self.tree_to_points.items()}

        n = len(self.node_coords)
        self.node_dist = np.zeros((n, n), dtype=float)
        self.node_obs = np.zeros((n, n), dtype=float)
        self.node_clear_bonus = np.zeros(n, dtype=float)

        for i, p in enumerate(self.node_coords):
            self.node_clear_bonus[i] = min(2.0, max(0.0, self.env.point_clearance(p))) * 0.08

        for i in range(n):
            for j in range(i + 1, n):
                pi = self.node_coords[i]
                pj = self.node_coords[j]
                d = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
                obs = self.env.segment_obstacle_penalty(pi, pj)
                self.node_dist[i, j] = self.node_dist[j, i] = d
                self.node_obs[i, j] = self.node_obs[j, i] = obs

    def _node_idx(self, p):
        return self.node_index[self._point_key(p)]

    def _distance(self, a, b) -> float:
        return float(self.node_dist[self._node_idx(a), self._node_idx(b)])

    def _edge_obstacle(self, a, b) -> float:
        return float(self.node_obs[self._node_idx(a), self._node_idx(b)])

    def _dist(self, path: List[Tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(self._distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    def _path_obstacle_penalty(self, path: List[Tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(self._edge_obstacle(path[i], path[i + 1]) for i in range(len(path) - 1))

    def _estimate_tree_complexity(self) -> np.ndarray:
        vals = []
        for tid, (x, y) in enumerate(self.env.tree_coords):
            available = self.tree_to_available_count[tid]
            miss = max(0, self.cfg.preferred_points_per_tree - available)
            pressure = self.env.obstacle_pressure((x, y))
            vals.append(1.0 + 0.18 * miss + 0.06 * pressure)
        return np.array(vals, dtype=float)

    def _tree_weight(self, tid: int) -> float:
        return float(self.tree_complexity[tid])

    def _required_tree_load(self, tree_ids: List[int]) -> float:
        return float(sum(self._tree_weight(tid) for tid in tree_ids))

    def _partition_score(self, tid: int, vid: int, centers: np.ndarray, loads: np.ndarray, quotas: np.ndarray) -> float:
        tree = self.tree_coords_np[tid]
        dist = np.linalg.norm(tree - centers[vid]) + 1e-6
        load_after = loads[vid] + self._tree_weight(tid)
        load_pressure = load_after / (quotas[vid] + 1e-6)
        start_bias = math.hypot(tree[0] - self.env.start_positions[vid][0], tree[1] - self.env.start_positions[vid][1])
        obstacle = self.env.obstacle_pressure((tree[0], tree[1]))
        overload = max(0.0, load_pressure - 1.03)
        battery_term = max(1e-9, self.env.battery_weights[vid] ** self.cfg.partition_battery_power)
        return (
            dist * (1.0 + 0.82 * load_pressure)
            + 0.10 * start_bias
            + 3.2 * obstacle
            + 28.0 * overload
        ) / battery_term

    def battery_partition(self):
        centers = np.array(self.env.start_positions, dtype=float)
        quota_w = np.power(np.array(self.env.vehicle_battery, dtype=float), self.cfg.partition_quota_power)
        quota_w = quota_w / max(1e-9, float(np.sum(quota_w)))
        quotas = quota_w * max(1.0, float(np.sum(self.tree_complexity)))
        loads = np.zeros(self.env.vehicle_num, dtype=float)

        tree_order = list(range(self.n_trees))
        tree_order.sort(
            key=lambda tid: np.min(np.linalg.norm(centers - self.tree_coords_np[tid], axis=1))
            + 3.5 * self.tree_complexity[tid],
            reverse=True,
        )

        for _ in range(7):
            assigned = {i: [] for i in range(self.env.vehicle_num)}
            loads[:] = 0.0
            for tid in tree_order:
                scores = [self._partition_score(tid, vid, centers, loads, quotas) for vid in range(self.env.vehicle_num)]
                best_vid = int(np.argmin(scores))
                assigned[best_vid].append(tid)
                loads[best_vid] += self._tree_weight(tid)

            for vid in range(self.env.vehicle_num):
                if assigned[vid]:
                    coords = self.tree_coords_np[assigned[vid]]
                    weights = self.tree_complexity[assigned[vid]]
                    centers[vid] = np.average(coords, axis=0, weights=weights)

        self.vehicle_tree_groups = self._repair_partition(assigned, quotas, compute_centers=False)

    def _repair_partition(self, assigned: Dict[int, List[int]], quotas: np.ndarray, compute_centers: bool = False):
        centers = []
        loads = np.zeros(self.env.vehicle_num, dtype=float)

        for vid in range(self.env.vehicle_num):
            group = assigned[vid]
            loads[vid] = self._required_tree_load(group)
            if compute_centers and self.best_paths.get(vid):
                centers.append(np.mean(np.array(self.best_paths[vid], dtype=float), axis=0))
            elif group:
                coords = self.tree_coords_np[group]
                weights = self.tree_complexity[group]
                centers.append(np.average(coords, axis=0, weights=weights))
            else:
                centers.append(np.array(self.env.start_positions[vid], dtype=float))
        centers = np.array(centers, dtype=float)

        for _ in range(self.cfg.repair_rounds):
            moved = False
            for vid in range(self.env.vehicle_num):
                group = assigned[vid]
                if len(group) <= 1:
                    continue
                center = centers[vid]
                ranked = sorted(
                    group,
                    key=lambda tid: np.linalg.norm(self.tree_coords_np[tid] - center) + 2.0 * self.tree_complexity[tid],
                    reverse=True,
                )
                topn = max(1, int(len(group) * self.cfg.repair_top_ratio))
                for tid in ranked[:topn]:
                    best_vid = vid
                    best_score = self._partition_score(tid, vid, centers, loads, quotas)
                    for oid in range(self.env.vehicle_num):
                        if oid == vid:
                            continue
                        if loads[oid] + self._tree_weight(tid) > max(quotas[oid] * 1.20, quotas[oid] + 1.5):
                            continue
                        cand_score = self._partition_score(tid, oid, centers, loads, quotas)
                        if cand_score < best_score * (1.0 - self.cfg.repair_gain_eps):
                            best_score = cand_score
                            best_vid = oid
                    if best_vid != vid:
                        assigned[vid].remove(tid)
                        assigned[best_vid].append(tid)
                        loads[vid] -= self._tree_weight(tid)
                        loads[best_vid] += self._tree_weight(tid)
                        moved = True
            if not moved:
                break
        return assigned

    def _needs_repartition(self, it: int) -> bool:
        if (it + 1) % self.cfg.repartition_check_every != 0:
            return False
        if len(self.vehicle_raw_distance) < self.env.vehicle_num:
            return False
        ratios = [self.vehicle_energy_ratio.get(vid, 0.0) for vid in range(self.env.vehicle_num)]
        if not ratios:
            return False
        energy_gap = max(ratios) - min(ratios)
        loads = [self._required_tree_load(self.vehicle_tree_groups.get(vid, [])) for vid in range(self.env.vehicle_num)]
        avg_load = max(1e-9, float(np.mean(loads)))
        load_gap = (max(loads) - min(loads)) / avg_load
        return energy_gap >= self.cfg.repartition_energy_gap or load_gap >= self.cfg.repartition_load_gap

    def _adaptive_repartition(self):
        if not self.vehicle_raw_distance:
            return
        energy_ratio = np.array([self.vehicle_energy_ratio.get(vid, 0.0) for vid in range(self.env.vehicle_num)], dtype=float)
        if len(energy_ratio) == 0:
            return

        quota_w = np.power(np.array(self.env.vehicle_battery, dtype=float), self.cfg.partition_quota_power)
        quota_w = quota_w / max(1e-9, float(np.sum(quota_w)))
        quotas = quota_w * max(1.0, float(np.sum(self.tree_complexity)))
        moved_any = False

        for _ in range(self.cfg.repartition_move_count):
            high = int(np.argmax(energy_ratio))
            low = int(np.argmin(energy_ratio))
            if high == low or len(self.vehicle_tree_groups[high]) <= 1:
                break

            high_center = np.mean(self.tree_coords_np[self.vehicle_tree_groups[high]], axis=0)
            candidates = sorted(
                self.vehicle_tree_groups[high],
                key=lambda tid: np.linalg.norm(self.tree_coords_np[tid] - high_center) + 0.8 * self.tree_complexity[tid],
                reverse=True,
            )
            pool = candidates[: max(1, int(len(candidates) * self.cfg.boundary_pool_ratio))]

            moved = False
            for tid in pool:
                curr_p = self.tree_coords_np[tid]
                curr_score = (
                    math.hypot(curr_p[0] - self.env.start_positions[high][0], curr_p[1] - self.env.start_positions[high][1])
                    / max(1e-9, self.env.vehicle_battery[high])
                )
                new_score = (
                    math.hypot(curr_p[0] - self.env.start_positions[low][0], curr_p[1] - self.env.start_positions[low][1])
                    / max(1e-9, self.env.vehicle_battery[low])
                )
                if new_score < curr_score * 1.85:
                    self.vehicle_tree_groups[high].remove(tid)
                    self.vehicle_tree_groups[low].append(tid)
                    energy_ratio[high] -= self._tree_weight(tid) / max(1e-9, self.env.vehicle_battery[high])
                    energy_ratio[low] += self._tree_weight(tid) / max(1e-9, self.env.vehicle_battery[low])
                    moved = True
                    moved_any = True
                    break
            if not moved:
                break

        if moved_any:
            self.vehicle_tree_groups = self._repair_partition(self.vehicle_tree_groups, quotas, compute_centers=True)

    def _load_ratios_from_groups(self, groups: Optional[Dict[int, List[int]]] = None) -> np.ndarray:
        use_groups = self.vehicle_tree_groups if groups is None else groups
        vals = []
        for vid in range(self.env.vehicle_num):
            load = self._required_tree_load(use_groups.get(vid, []))
            vals.append(load / max(1e-9, self.env.vehicle_battery[vid]))
        return np.array(vals, dtype=float)

    def _medium_ant_boundary_refinement(self) -> bool:
        groups = deepcopy(self.vehicle_tree_groups)
        moved_any = False
        for _ in range(self.cfg.layered_refine_move_count):
            ratios = self._load_ratios_from_groups(groups)
            donor = int(np.argmax(ratios))
            receiver = int(np.argmin(ratios))
            current_gap = float(ratios.max() - ratios.min())
            if donor == receiver or len(groups.get(donor, [])) <= 2:
                break

            donor_center = np.mean(self.tree_coords_np[groups[donor]], axis=0) if groups[donor] else np.array(self.env.start_positions[donor], dtype=float)
            receiver_center = np.mean(self.tree_coords_np[groups[receiver]], axis=0) if groups[receiver] else np.array(self.env.start_positions[receiver], dtype=float)
            ranked = sorted(
                groups[donor],
                key=lambda tid: (
                    np.linalg.norm(self.tree_coords_np[tid] - donor_center)
                    - 0.85 * np.linalg.norm(self.tree_coords_np[tid] - receiver_center)
                    + 0.35 * self.tree_complexity[tid]
                ),
                reverse=True,
            )
            pool = ranked[: max(1, int(len(ranked) * 0.5))]
            moved = False
            best_choice = None
            best_gap = current_gap
            for tid in pool:
                cand_groups = deepcopy(groups)
                cand_groups[donor].remove(tid)
                cand_groups[receiver].append(tid)
                new_ratios = self._load_ratios_from_groups(cand_groups)
                new_gap = float(new_ratios.max() - new_ratios.min())
                donor_dist = math.hypot(
                    self.tree_coords_np[tid][0] - self.env.start_positions[donor][0],
                    self.tree_coords_np[tid][1] - self.env.start_positions[donor][1],
                )
                receiver_dist = math.hypot(
                    self.tree_coords_np[tid][0] - self.env.start_positions[receiver][0],
                    self.tree_coords_np[tid][1] - self.env.start_positions[receiver][1],
                )
                if receiver_dist <= donor_dist * 1.55 and new_gap + 1e-9 < best_gap:
                    best_gap = new_gap
                    best_choice = tid
            if best_choice is not None:
                groups[donor].remove(best_choice)
                groups[receiver].append(best_choice)
                moved = True
                moved_any = True
            if not moved:
                break

        if moved_any:
            quota_w = np.power(np.array(self.env.vehicle_battery, dtype=float), self.cfg.partition_quota_power)
            quota_w = quota_w / max(1e-9, float(np.sum(quota_w)))
            quotas = quota_w * max(1.0, float(np.sum(self.tree_complexity)))
            self.vehicle_tree_groups = self._repair_partition(groups, quotas, compute_centers=True)
            return True
        return False

    def _progress(self, it: int) -> float:
        return 1.0 if self.cfg.max_iter <= 1 else it / float(self.cfg.max_iter - 1)

    def _stagnation_len(self, it: int) -> int:
        return max(0, (it + 1) - self.last_improve_iter)

    def _in_escape_window(self, it: int) -> bool:
        return (it + 1) <= self.relax_until

    def _alpha(self, it: int) -> float:
        t = self._progress(it)
        return self.cfg.alpha_start + (self.cfg.alpha_end - self.cfg.alpha_start) * t

    def _beta(self, it: int) -> float:
        t = self._progress(it)
        base = self.cfg.beta_start + (self.cfg.beta_end - self.cfg.beta_start) * t
        if self._in_escape_window(it):
            base -= 0.25
        return base

    def _evap(self, it: int) -> float:
        t = self._progress(it)
        base = self.cfg.evap_start + (self.cfg.evap_end - self.cfg.evap_start) * t
        if self._in_escape_window(it):
            base += 0.03
        return float(clip(base, 0.82, 0.98))

    def _greedy_ratio(self, it: int, candidate_index: int = 0) -> float:
        t = self._progress(it)
        if t <= 0.55:
            ratio = self.cfg.greedy_start + (self.cfg.greedy_mid - self.cfg.greedy_start) * (t / 0.55)
        else:
            ratio = self.cfg.greedy_mid + (self.cfg.greedy_end - self.cfg.greedy_mid) * ((t - 0.55) / 0.45)
        if self._in_escape_window(it):
            ratio -= 0.18
        ratio -= 0.12 * min(candidate_index, 2)
        return float(clip(ratio, 0.18, 0.92))

    def _current_candidate_k(self, it: int) -> int:
        if self._in_escape_window(it):
            return self.cfg.base_candidate_k + self.cfg.explore_candidate_boost
        if it >= self.cfg.max_iter * self.cfg.late_stage_ratio:
            return self.cfg.late_candidate_k
        return self.cfg.base_candidate_k

    def _route_candidate_count(self, it: int) -> int:
        if it >= self.cfg.max_iter * 0.72 or self._stagnation_len(it) >= self.cfg.destroy_after_stagnation:
            return self.cfg.route_candidates + 1
        return self.cfg.route_candidates

    def _current_local_search_every(self, it: int) -> int:
        if self._in_escape_window(it):
            return 10**9
        if it >= self.cfg.max_iter * self.cfg.late_stage_ratio:
            return self.cfg.local_search_every_late
        return self.cfg.local_search_every

    def _tree_leg_info(self, curr, tree_id):
        pts = self.tree_to_required_points[tree_id]
        point_indices = self.tree_required_point_indices[tree_id]
        idx_curr = self._node_idx(curr)

        best_idx = 0
        best_val = float("inf")
        for local_idx, idx_p in enumerate(point_indices):
            leg = self.node_dist[idx_curr, idx_p]
            obs = self.node_obs[idx_curr, idx_p]
            clearance_bonus = self.node_clear_bonus[idx_p]
            val = float(leg + self.cfg.lambda_obstacle * obs - clearance_bonus)
            if val < best_val:
                best_val = val
                best_idx = local_idx
        return best_val, pts[best_idx]

    def _ordered_points_with_lookahead(self, tree_id: int, entry_point, reference_pt=None):
        pts = self.tree_to_required_points[tree_id]
        if len(pts) <= 1:
            return pts[:]

        pts_sorted = self.tree_required_points_sorted[tree_id]
        start_idx = int(np.argmin([math.hypot(entry_point[0] - p[0], entry_point[1] - p[1]) for p in pts_sorted]))
        order_ccw = [pts_sorted[(start_idx + i) % len(pts_sorted)] for i in range(len(pts_sorted))]
        order_cw = [pts_sorted[(start_idx - i) % len(pts_sorted)] for i in range(len(pts_sorted))]

        def order_cost(order):
            base = math.hypot(entry_point[0] - order[0][0], entry_point[1] - order[0][1])
            intra = sum(math.hypot(order[i][0] - order[i + 1][0], order[i][1] - order[i + 1][1]) for i in range(len(order) - 1))
            exit_cost = 0.0
            if reference_pt is not None:
                exit_cost = 0.55 * math.hypot(order[-1][0] - reference_pt[0], order[-1][1] - reference_pt[1])
            return base + intra + exit_cost

        return order_ccw if order_cost(order_ccw) <= order_cost(order_cw) else order_cw

    def _select_candidate_infos(self, curr, rem_list: List[int], it: int):
        k = min(self._current_candidate_k(it), len(rem_list))
        if k <= 0:
            return []

        centers = self.tree_coords_np[rem_list]
        center_dists = np.hypot(centers[:, 0] - curr[0], centers[:, 1] - curr[1]) + 1e-12
        coarse_k = min(len(rem_list), max(k + 3, 2 * k))
        if len(rem_list) > coarse_k:
            coarse_ids = np.argpartition(center_dists, coarse_k - 1)[:coarse_k]
            coarse_tids = [rem_list[i] for i in coarse_ids]
        else:
            coarse_tids = rem_list[:]

        if self._stagnation_len(it) >= self.cfg.destroy_after_stagnation and len(rem_list) > coarse_k:
            far_count = min(max(2, k // 2), len(rem_list) - coarse_k)
            far_pool = list(np.argsort(center_dists)[-far_count:])
            extra_tids = [rem_list[i] for i in far_pool]
            coarse_tids = list(dict.fromkeys(coarse_tids + extra_tids))

        alpha = self._alpha(it)
        beta = self._beta(it)
        recent_max = max(1e-9, float(self.recent_visit_grid.max()) * self.recent_visit_scale)
        infos = []
        for tid in coarse_tids:
            best_leg, entry_point = self._tree_leg_info(curr, tid)
            gi, gj = self.tree_grid_idx[tid]
            pheromone = self.dynamic_grid[gi, gj] * self.dynamic_grid_scale + 1e-6
            recent_val = self.recent_visit_grid[gi, gj] * self.recent_visit_scale
            anti = 1.0 + self.cfg.anti_pheromone_strength * (recent_val / recent_max)
            if self._stagnation_len(it) < self.cfg.destroy_after_stagnation:
                anti = 1.0 + 0.5 * (anti - 1.0)
            complexity = 1.0 + 0.10 * self.tree_complexity[tid]
            score = ((pheromone / anti) ** alpha) * ((1.0 / (best_leg + 1e-9)) ** beta) / complexity
            infos.append((tid, best_leg, entry_point, score))

        infos.sort(key=lambda x: x[1])
        return infos[:k]

    def _sample_tree(self, cand_infos, greedy_ratio: float):
        if not cand_infos:
            raise RuntimeError("No candidate infos available.")
        if random.random() < greedy_ratio:
            return cand_infos[int(np.argmax([x[3] for x in cand_infos]))]
        scores = np.array([x[3] for x in cand_infos], dtype=float)
        if float(scores.sum()) <= 1e-12:
            probs = np.full_like(scores, 1.0 / len(scores))
        else:
            probs = scores / float(scores.sum())
        return cand_infos[int(np.random.choice(len(cand_infos), p=probs))]

    def _single_vehicle_order_cost_from_path(self, vid: int, path: List[Tuple[float, float]], order: List[int]) -> float:
        start = self.env.start_positions[vid]
        full = [start] + path

        d = self._dist(full)
        p_obs = self._path_obstacle_penalty(full)
        req_points = sum(self.tree_to_required_count.get(tid, 0) for tid in order)
        got_points = len(path)
        missing_points = max(0, req_points - got_points)
        incomplete_tree_count = sum(
            1 for tid in order
            if self.tree_to_required_count.get(tid, 0) > len(self.tree_to_required_points[tid])
        )
        incomplete_ratio = incomplete_tree_count / max(1, len(order))
        primary_cost = d + self.cfg.lambda_obstacle * p_obs
        total_cost = (
            primary_cost
            + self.cfg.lambda_missing_point * missing_points
            + self.cfg.lambda_tree_incomplete * incomplete_ratio * max(1, len(order))
        )
        return float(total_cost)

    def _route_objective(self, vid: int, path: List[Tuple[float, float]], order: List[int]) -> float:
        return self._single_vehicle_order_cost_from_path(vid, path, order)

    def build_route_once(self, vid: int, it: int, candidate_index: int = 0):
        curr = self.env.start_positions[vid]
        path: List[Tuple[float, float]] = []
        visit_order: List[int] = []
        remaining = set(self.vehicle_tree_groups[vid])
        greedy_ratio = self._greedy_ratio(it, candidate_index)

        while remaining:
            cand_infos = self._select_candidate_infos(curr, list(remaining), it)
            chosen_tid, _, entry_point, _ = self._sample_tree(cand_infos, greedy_ratio)
            ref_pt = None
            if len(remaining) > 1:
                rem_wo = list(remaining)
                rem_wo.remove(chosen_tid)
                # 候选0更偏近邻，候选1/2更偏全局质心，保留多样性
                if candidate_index == 0:
                    ref_pt = np.mean(self.tree_coords_np[rem_wo], axis=0)
                else:
                    idxs = np.random.choice(len(rem_wo), size=min(3, len(rem_wo)), replace=False)
                    ref_pt = np.mean(self.tree_coords_np[[rem_wo[i] for i in idxs]], axis=0)
            ordered_pts = self._ordered_points_with_lookahead(chosen_tid, entry_point, ref_pt)
            path.extend(ordered_pts)
            visit_order.append(chosen_tid)
            curr = ordered_pts[-1]
            remaining.remove(chosen_tid)

        return path, visit_order

    def build_route_candidates(self, vid: int, it: int):
        candidates = []
        for cid in range(self._route_candidate_count(it)):
            path, order = self.build_route_once(vid, it, candidate_index=cid)
            score = self._route_objective(vid, path, order)
            candidates.append((score, path, order))
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    def _render_order_to_path(self, start, order: List[int]):
        curr = start
        path = []
        for idx, tid in enumerate(order):
            _, entry_point = self._tree_leg_info(curr, tid)
            ref_pt = self.tree_coords_np[order[idx + 1]] if idx + 1 < len(order) else None
            ordered_pts = self._ordered_points_with_lookahead(tid, entry_point, ref_pt)
            path.extend(ordered_pts)
            curr = ordered_pts[-1]
        return path

    def _evaluate_order_cost(self, vid: int, order: List[int]):
        key = (vid, tuple(order))
        cached = self.order_eval_cache.get(key)
        if cached is not None:
            return cached[0], cached[1][:]

        path = self._render_order_to_path(self.env.start_positions[vid], order)
        total_cost = self._single_vehicle_order_cost_from_path(vid, path, order)
        self.order_eval_cache[key] = (total_cost, path[:])
        return total_cost, path

    def _local_improve_order(self, vid: int, order: List[int], max_trials=12, allow_insert=False):
        if len(order) < 4:
            return order[:], self._render_order_to_path(self.env.start_positions[vid], order)
        best_order = order[:]
        best_cost, best_path = self._evaluate_order_cost(vid, best_order)
        n = len(order)
        trials = 0

        for i in range(n - 2):
            for j in range(i + 1, min(n, i + 6)):
                new_order = best_order[:]
                new_order[i:j + 1] = reversed(new_order[i:j + 1])
                new_cost, new_path = self._evaluate_order_cost(vid, new_order)
                trials += 1
                if new_cost + 1e-9 < best_cost:
                    best_cost, best_order, best_path = new_cost, new_order, new_path
                if trials >= max_trials:
                    return best_order, best_path

        if allow_insert:
            for i in range(n):
                for j in range(max(0, i - 3), min(n, i + 4)):
                    if i == j:
                        continue
                    new_order = best_order[:]
                    node = new_order.pop(i)
                    new_order.insert(j, node)
                    new_cost, new_path = self._evaluate_order_cost(vid, new_order)
                    trials += 1
                    if new_cost + 1e-9 < best_cost:
                        best_cost, best_order, best_path = new_cost, new_order, new_path
                    if trials >= 3 * max_trials:
                        return best_order, best_path

        return best_order, best_path

    def _evaluate_paths_metrics(
        self,
        vehicle_paths: Dict[int, List[Tuple[float, float]]],
        custom_start: Optional[Dict[int, Tuple[float, float]]] = None,
        custom_tree_groups: Optional[Dict[int, List[int]]] = None,
    ):
        dists = []
        eff_covs = []
        theo_covs = []
        penalty = 0.0
        total = 0.0
        energy_ratios = []
        missing_points_total = 0
        required_points_total = 0
        incomplete_trees = 0
        tree_total = 0

        groups = self.vehicle_tree_groups if custom_tree_groups is None else custom_tree_groups
        if custom_start is None:
            starts = {vid: self.env.start_positions[vid] for vid in groups.keys()}
        else:
            starts = custom_start

        for vid in groups.keys():
            path = vehicle_paths.get(vid, [])
            start = starts[vid]
            full = [start] + path
            d = self._dist(full)
            p_obs = self._path_obstacle_penalty(full)
            assigned = groups.get(vid, [])
            req_points = sum(self.tree_to_required_count.get(tid, 0) for tid in assigned)
            got_points = len(path)
            missing_points = max(0, req_points - got_points)
            incomplete_tree_count = sum(
                1 for tid in assigned
                if self.tree_to_required_count.get(tid, 0) > len(self.tree_to_required_points[tid])
            )

            eff = got_points / max(1, sum(self.tree_to_available_count.get(tid, 1) for tid in assigned))
            theo = got_points / max(1, req_points)

            dists.append(d)
            total += d
            penalty += p_obs
            eff_covs.append(float(clip(eff, 0.0, 1.0)))
            theo_covs.append(float(clip(theo, 0.0, 1.0)))
            missing_points_total += missing_points
            required_points_total += req_points
            incomplete_trees += incomplete_tree_count
            tree_total += len(assigned)

            battery = self.env.vehicle_battery[vid] if vid < len(self.env.vehicle_battery) else max(self.env.vehicle_battery)
            energy_ratios.append(d / max(1e-9, battery))

        mean_eff = float(np.mean(eff_covs)) if eff_covs else 1.0
        mean_theo = float(np.mean(theo_covs)) if theo_covs else 1.0
        e_vals = np.array(energy_ratios, dtype=float) if energy_ratios else np.array([0.0])
        e_range = float(e_vals.max() - e_vals.min()) if len(e_vals) > 0 else 0.0
        route_range = max(dists) - min(dists) if dists else 0.0
        incomplete_ratio = incomplete_trees / max(1, tree_total)
        primary_cost = total + self.cfg.lambda_obstacle * penalty
        balance_cost = self.cfg.lambda_balance * (0.60 * route_range + 0.40 * e_range * 100.0)
        guard_cost = self.cfg.lambda_energy_guard * max(0.0, e_range - 0.10) * max(1, self.n_trees)
        total_cost = (
            primary_cost
            + balance_cost
            + self.cfg.lambda_missing_point * missing_points_total
            + self.cfg.lambda_tree_incomplete * incomplete_ratio * max(1, tree_total)
            + guard_cost
        )

        return {
            "total_cost": float(total_cost),
            "primary_cost": float(primary_cost),
            "balance_cost": float(balance_cost + guard_cost),
            "total_distance": float(total),
            "effective_cov": mean_eff,
            "theoretical_cov": mean_theo if required_points_total > 0 else 1.0,
            "obstacle_penalty": float(penalty),
            "energy_range_percent": float((e_range / max(1e-9, e_vals.max())) * 100.0) if len(e_vals) > 0 else 0.0,
            "energy_range_raw": float(e_range),
            "vehicle_dists": dists,
            "missing_points_total": int(missing_points_total),
            "required_points_total": int(required_points_total),
            "incomplete_tree_ratio": float(incomplete_ratio),
        }

    def _layered_score_key(self, metrics: Dict[str, float]):
        energy_pct = float(metrics.get("energy_range_percent", 1e9))
        feasible_energy = 0 if energy_pct <= self.cfg.target_energy_percent else 1
        major_cost = float(metrics.get("primary_cost", metrics.get("total_cost", 0.0))) if feasible_energy == 0 else float(metrics.get("total_cost", 0.0))
        minor_cost = float(metrics.get("total_cost", 0.0)) if feasible_energy == 0 else float(metrics.get("primary_cost", metrics.get("total_cost", 0.0)))
        return (
            int(metrics.get("missing_points_total", 0)),
            round(float(metrics.get("incomplete_tree_ratio", 0.0)), 8),
            feasible_energy,
            round(major_cost, 8),
            round(energy_pct, 8),
            round(minor_cost, 8),
        )

    def _is_metrics_better(self, cand: Dict[str, float], ref: Optional[Dict[str, float]]) -> bool:
        if ref is None:
            return True
        return self._layered_score_key(cand) < self._layered_score_key(ref)

    def _best_metrics_proxy(self) -> Optional[Dict[str, float]]:
        if self.best_metrics is None:
            return None
        return deepcopy(self.best_metrics)

    def _evaluate_current_solution(self):
        metrics = self._evaluate_paths_metrics(self.vehicle_paths)
        for vid in range(self.env.vehicle_num):
            d = metrics["vehicle_dists"][vid] if vid < len(metrics["vehicle_dists"]) else 0.0
            self.vehicle_raw_distance[vid] = d
            self.vehicle_energy_ratio[vid] = d / max(1e-9, self.env.vehicle_battery[vid])
        return metrics

    def _polish_solution(self, vehicle_orders: Dict[int, List[int]], best_only=False):
        polished_orders = deepcopy(vehicle_orders)
        polished_paths = {}
        for vid in range(self.env.vehicle_num):
            trials = self.cfg.best_polish_trials if best_only else (
                self.cfg.local_search_trials_late if len(vehicle_orders[vid]) >= 8 else self.cfg.local_search_trials
            )
            order, path = self._local_improve_order(
                vid,
                polished_orders[vid],
                max_trials=trials,
                allow_insert=True if best_only else False,
            )
            polished_orders[vid] = order
            polished_paths[vid] = path
        metrics = self._evaluate_paths_metrics(polished_paths)
        return metrics, polished_orders, polished_paths

    def _destroy_and_repair_from_best(self):
        if not self.best_orders:
            return None

        new_orders = deepcopy(self.best_orders)
        for vid in range(self.env.vehicle_num):
            order = new_orders.get(vid, [])
            if len(order) < 6:
                continue
            remove_n = max(2, int(len(order) * self.cfg.destroy_ratio))
            remove_n = min(remove_n, len(order) - 2)
            start_idx = random.randint(0, len(order) - remove_n)
            removed = order[start_idx:start_idx + remove_n]
            remain = order[:start_idx] + order[start_idx + remove_n:]
            random.shuffle(removed)

            for tid in removed:
                best_pos = 0
                best_cost = float("inf")
                sample_positions = list(range(len(remain) + 1))
                if len(sample_positions) > 10:
                    sample_positions = sorted(set([0, len(remain)] + random.sample(sample_positions[1:-1], 8)))
                for pos in sample_positions:
                    candidate = remain[:pos] + [tid] + remain[pos:]
                    cost, _ = self._evaluate_order_cost(vid, candidate)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                remain.insert(best_pos, tid)
            new_orders[vid] = remain

        metrics, new_orders, new_paths = self._polish_solution(new_orders, best_only=False)
        return metrics, new_orders, new_paths

    def _nearest_other_vehicle(self, vid: int) -> int:
        sx, sy = self.env.start_positions[vid]
        best_oid = vid
        best_d = float("inf")
        for oid, (ox, oy) in enumerate(self.env.start_positions):
            if oid == vid:
                continue
            d = math.hypot(sx - ox, sy - oy)
            if d < best_d:
                best_d = d
                best_oid = oid
        return best_oid

    def _select_boundary_tree(self, from_vid: int, to_vid: int) -> Optional[int]:
        group = self.vehicle_tree_groups.get(from_vid, [])
        if len(group) <= 2:
            return None
        ranked = sorted(
            group,
            key=lambda tid: (
                math.hypot(self.tree_coords_np[tid][0] - self.env.start_positions[from_vid][0], self.tree_coords_np[tid][1] - self.env.start_positions[from_vid][1])
                - 0.70 * math.hypot(self.tree_coords_np[tid][0] - self.env.start_positions[to_vid][0], self.tree_coords_np[tid][1] - self.env.start_positions[to_vid][1])
                + 1.2 * self.tree_complexity[tid]
            ),
            reverse=True,
        )
        topn = max(1, int(len(ranked) * self.cfg.exchange_top_ratio))
        return ranked[0] if topn <= 1 else random.choice(ranked[:topn])

    def _accept_boundary_exchange(self, donor: int, receiver: int, tid: int, base_metrics: Dict[str, float]):
        if tid is None or tid not in self.vehicle_tree_groups.get(donor, []):
            return None

        cand_groups = deepcopy(self.vehicle_tree_groups)
        cand_groups[donor].remove(tid)
        cand_groups[receiver].append(tid)

        donor_order = [x for x in self.vehicle_orders.get(donor, []) if x != tid]
        if not donor_order:
            donor_order = cand_groups[donor][:]
        receiver_base = [x for x in self.vehicle_orders.get(receiver, []) if x in cand_groups[receiver] and x != tid]

        best_receiver_order = None
        best_receiver_cost = float("inf")
        sample_positions = list(range(len(receiver_base) + 1))
        if len(sample_positions) > 8:
            sample_positions = sorted(set([0, len(receiver_base)] + random.sample(sample_positions[1:-1], 6)))
        for pos in sample_positions:
            cand_order = receiver_base[:pos] + [tid] + receiver_base[pos:]
            cand_cost, _ = self._evaluate_order_cost(receiver, cand_order)
            if cand_cost < best_receiver_cost:
                best_receiver_cost = cand_cost
                best_receiver_order = cand_order
        if best_receiver_order is None:
            best_receiver_order = receiver_base + [tid]

        donor_order, donor_path = self._local_improve_order(donor, donor_order, max_trials=10, allow_insert=True)
        receiver_order, receiver_path = self._local_improve_order(receiver, best_receiver_order, max_trials=10, allow_insert=True)

        cand_orders = deepcopy(self.vehicle_orders)
        cand_paths = deepcopy(self.vehicle_paths)
        cand_orders[donor] = donor_order
        cand_orders[receiver] = receiver_order
        cand_paths[donor] = donor_path
        cand_paths[receiver] = receiver_path

        cand_metrics = self._evaluate_paths_metrics(cand_paths, custom_tree_groups=cand_groups)
        primary_ok = cand_metrics["primary_cost"] <= base_metrics["primary_cost"] * self.cfg.exchange_primary_relax
        total_ok = cand_metrics["total_cost"] <= base_metrics["total_cost"] * 1.01
        energy_better = cand_metrics["energy_range_percent"] + self.cfg.exchange_energy_drop < base_metrics["energy_range_percent"]
        improved = self._is_metrics_better(cand_metrics, base_metrics) or ((primary_ok or total_ok) and energy_better)
        if not improved:
            return None
        return cand_metrics, cand_groups, cand_orders, cand_paths

    def _try_boundary_exchange(self, it: int, base_metrics: Dict[str, float]):
        if self._stagnation_len(it) < self.cfg.exchange_after_stagnation:
            return None
        if (it + 1 - self.last_exchange_iter) < self.cfg.exchange_cooldown:
            return None
        if len(self.vehicle_energy_ratio) < self.env.vehicle_num:
            return None

        ratios = [self.vehicle_energy_ratio.get(vid, 0.0) for vid in range(self.env.vehicle_num)]
        donor_candidates = sorted(range(self.env.vehicle_num), key=lambda v: ratios[v], reverse=True)[:2]
        receiver_candidates = sorted(range(self.env.vehicle_num), key=lambda v: ratios[v])[:2]
        tries = 0
        best_found = None
        for donor in donor_candidates:
            for receiver in receiver_candidates:
                if donor == receiver:
                    continue
                tries += 1
                tid = self._select_boundary_tree(donor, receiver)
                result = self._accept_boundary_exchange(donor, receiver, tid, base_metrics)
                if result is not None:
                    best_found = result
                    break
                if tries >= self.cfg.exchange_try_count:
                    break
            if best_found is not None or tries >= self.cfg.exchange_try_count:
                break
        if best_found is not None:
            self.last_exchange_iter = it + 1
        return best_found

    def _reinforce_path_on_grid(
        self,
        full: List[Tuple[float, float]],
        amount: float,
        target_grid: Optional[np.ndarray] = None,
    ):
        if target_grid is None:
            grid = self.dynamic_grid
            scale = self.dynamic_grid_scale
        else:
            grid = target_grid
            scale = self.recent_visit_scale if target_grid is self.recent_visit_grid else 1.0

        arr = np.array(full, dtype=float)
        is_arr = np.clip(((arr[:, 0] - self.xmin) / self.cfg.grid_step).astype(int), 0, grid.shape[0] - 1)
        js_arr = np.clip(((arr[:, 1] - self.ymin) / self.cfg.grid_step).astype(int), 0, grid.shape[1] - 1)
        ui, counts = np.unique(is_arr * grid.shape[1] + js_arr, return_counts=True)
        grid[ui // grid.shape[1], ui % grid.shape[1]] += (amount / max(scale, 1e-12)) * counts

    def _update_recent_grid(self):
        self.recent_visit_scale *= self.cfg.anti_pheromone_decay
        self._maybe_normalize_grid_scales()
        for vid, path in self.vehicle_paths.items():
            if not path:
                continue
            self._reinforce_path_on_grid([self.env.start_positions[vid]] + path, 1.0, target_grid=self.recent_visit_grid)

    def _update_pheromone(self, it: int):
        evap = self._evap(it)
        self.dynamic_grid_scale *= evap
        self._maybe_normalize_grid_scales()

        ranked = sorted(
            [(vid, self.vehicle_raw_distance.get(vid, 0.0) + 15.0 * self.vehicle_energy_ratio.get(vid, 0.0))
             for vid in range(self.env.vehicle_num)],
            key=lambda x: x[1],
        )
        elite_count = max(1, int(math.ceil(self.env.vehicle_num * 0.5)))
        for vid, _ in ranked[:elite_count]:
            path = self.vehicle_paths.get(vid, [])
            if path:
                full = [self.env.start_positions[vid]] + path
                self._reinforce_path_on_grid(full, 8.0 / (max(1e-6, self._dist(full)) / max(1, len(full))))

        for vid, path in self.best_paths.items():
            if path:
                full = [self.env.start_positions[vid]] + path
                self._reinforce_path_on_grid(full, 3.8 / (max(1e-6, self.best_global_distance) / max(1, len(full))))

        if self._stagnation_len(it) >= self.cfg.destroy_after_stagnation:
            recent_nonzero = np.nonzero(self.recent_visit_grid)
            if len(recent_nonzero[0]) > 0:
                recent_max = max(1e-9, float(self.recent_visit_grid.max()) * self.recent_visit_scale)
                penalty_vals = np.clip(
                    (self.recent_visit_grid[recent_nonzero] * self.recent_visit_scale) / recent_max,
                    0.0,
                    1.0,
                )
                self.dynamic_grid[recent_nonzero] *= (1.0 - 0.12 * penalty_vals)

        if (it + 1) % 12 == 0 or self.dynamic_grid_scale < 0.35:
            self._materialize_dynamic_grid()
            self.dynamic_grid = np.clip(self.dynamic_grid, 0.03, 28.0)

    def _should_trigger_layered_refinement(self, it: int) -> bool:
        current_iter = it + 1
        if (current_iter - self.last_layered_refine_iter) < self.cfg.layered_refine_cooldown:
            return False
        if self._stagnation_len(it) < self.cfg.layered_refine_after_stagnation:
            return False
        if len(self.history_best_primary) < self.cfg.stagnation_limit:
            return False
        recent = self.history_best_primary[-self.cfg.stagnation_limit:]
        recent_improve = recent[0] - recent[-1]
        recent_diversity = float(np.std(np.diff(recent))) if len(recent) >= 2 else 0.0
        return recent_improve <= self.cfg.stagnation_delta and recent_diversity <= self.cfg.diversity_threshold

    def _apply_layered_progressive_refinement(self, it: int, base_metrics: Dict[str, float]):
        self.layered_refine_trigger_count += 1
        self.last_layered_refine_iter = it + 1

        original_groups = deepcopy(self.vehicle_tree_groups)
        moved = self._medium_ant_boundary_refinement()
        active_groups = deepcopy(self.vehicle_tree_groups)

        candidate_orders: Dict[int, List[int]] = {}
        candidate_paths: Dict[int, List[Tuple[float, float]]] = {}
        for vid in range(self.env.vehicle_num):
            path, order = self.build_route_candidates(vid, it)
            order, path = self._local_improve_order(
                vid,
                order,
                max_trials=self.cfg.local_search_trials_late if it >= self.cfg.max_iter * self.cfg.late_stage_ratio else self.cfg.local_search_trials,
                allow_insert=False,
            )
            candidate_orders[vid] = order
            candidate_paths[vid] = path

        candidate_metrics = self._evaluate_paths_metrics(candidate_paths, custom_tree_groups=active_groups)
        primary_ok = candidate_metrics["primary_cost"] <= base_metrics["primary_cost"] * self.cfg.layered_refine_primary_relax
        energy_better = candidate_metrics["energy_range_percent"] + self.cfg.layered_refine_energy_gain < base_metrics["energy_range_percent"]
        improved = self._is_metrics_better(candidate_metrics, base_metrics) or (primary_ok and energy_better)

        if improved:
            self.layered_refine_success_count += 1
            return candidate_metrics, active_groups, candidate_orders, candidate_paths

        if moved:
            self.vehicle_tree_groups = original_groups
        return None

    def _late_best_path_boost(self, it: int):
        if not self.best_orders or not self.best_tree_groups:
            return None
        if it + 1 < int(self.cfg.max_iter * 0.70):
            return None
        if self._stagnation_len(it) < 70:
            return None
        if (it + 1 - self.last_rebuild_iter) < self.cfg.rebuild_cooldown:
            return None

        cand_orders = deepcopy(self.best_orders)
        cand_paths = {}
        for vid in range(self.env.vehicle_num):
            order, path = self._local_improve_order(
                vid,
                cand_orders[vid],
                max_trials=self.cfg.best_polish_trials,
                allow_insert=True,
            )
            cand_orders[vid] = order
            cand_paths[vid] = path

        cand_metrics = self._evaluate_paths_metrics(cand_paths, custom_tree_groups=self.best_tree_groups)
        better_global = self._is_metrics_better(cand_metrics, self._best_metrics_proxy())
        better_primary = cand_metrics["primary_cost"] + 1e-9 < self.best_primary_cost and cand_metrics["energy_range_percent"] <= self.best_energy_range_percent + 1.0
        if better_global or better_primary:
            self.last_rebuild_iter = it + 1
            return cand_metrics, deepcopy(self.best_tree_groups), cand_orders, cand_paths
        return None

    def _task_ratio_list(self, groups: Optional[Dict[int, List[int]]] = None):
        total = max(1, self.n_trees)
        use_groups = self.vehicle_tree_groups if groups is None else groups
        return [100.0 * len(use_groups.get(vid, [])) / total for vid in range(self.env.vehicle_num)]

    def _task_ratio_text(self, groups: Optional[Dict[int, List[int]]] = None, ratios: Optional[List[float]] = None):
        use_ratios = self._task_ratio_list(groups) if ratios is None else ratios
        return " | ".join([f"V{vid + 1}:{use_ratios[vid]:.1f}%" for vid in range(self.env.vehicle_num)])

    def _append_report_row(self, it: int, elapsed: float):
        ratios = self.best_task_ratios if self.best_tree_groups else self._task_ratio_list()
        row = {
            "iter": int(it + 1),
            "best_primary_cost": float(self.best_primary_cost),
            "best_reference_total_cost": float(self.best_global_cost),
            "best_total_distance": float(self.best_global_distance),
            "theoretical_coverage": float(self.best_theoretical_coverage_ratio * 100.0),
            "effective_coverage": float(self.best_effective_coverage_ratio * 100.0),
            "obstacle_penalty": float(self.best_obstacle_penalty),
            "energy_range_percent": float(self.best_energy_range_percent),
            "incomplete_tree_ratio": float(self.best_incomplete_tree_ratio * 100.0),
            "elapsed_sec": float(elapsed),
            "best_task_ratio_detail": self._task_ratio_text(ratios=ratios),
            "layered_refine_trigger_count": int(self.layered_refine_trigger_count),
            "layered_refine_success_count": int(self.layered_refine_success_count),
            "layered_refine_success_rate": float(100.0 * self.layered_refine_success_count / max(1, self.layered_refine_trigger_count)),
            "stagnation_len": int(self._stagnation_len(it)),
        }
        for vid, ratio in enumerate(ratios, start=1):
            row[f"best_task_ratio_v{vid}"] = float(ratio)
        self.report_rows.append(row)

    def run(self) -> RunResult:
        self.battery_partition()
        start_time = time.time()

        for it in range(self.cfg.max_iter):
            self.order_eval_cache.clear()

            if self._needs_repartition(it):
                self._adaptive_repartition()
            elif (
                self._stagnation_len(it) >= self.cfg.forced_repartition_after_stagnation
                and (it + 1 - self.last_exchange_iter) >= self.cfg.forced_repartition_cooldown
            ):
                if self._medium_ant_boundary_refinement():
                    self.last_exchange_iter = it + 1

            self.vehicle_paths = {}
            self.vehicle_orders = {}

            for vid in range(self.env.vehicle_num):
                path, order = self.build_route_candidates(vid, it)
                if (it + 1) % self._current_local_search_every(it) == 0:
                    order, path = self._local_improve_order(
                        vid,
                        order,
                        max_trials=self.cfg.local_search_trials_late if it >= self.cfg.max_iter * self.cfg.late_stage_ratio else self.cfg.local_search_trials,
                        allow_insert=(it >= self.cfg.max_iter * self.cfg.late_stage_ratio),
                    )
                self.vehicle_paths[vid] = path
                self.vehicle_orders[vid] = order

            metrics = self._evaluate_current_solution()

            # 停滞后低频触发一次破坏-重构，避免后半程空转且控制耗时
            if (
                self._stagnation_len(it) >= self.cfg.destroy_after_stagnation
                and self.best_orders
                and (it + 1 - self.last_rebuild_iter) >= self.cfg.rebuild_cooldown
            ):
                rebuilt = self._destroy_and_repair_from_best()
                self.last_rebuild_iter = it + 1
                if rebuilt is not None:
                    rebuilt_metrics = rebuilt[0]
                    primary_ok = rebuilt_metrics["primary_cost"] <= metrics["primary_cost"] * self.cfg.rebuild_primary_relax
                    energy_better = rebuilt_metrics["energy_range_percent"] + 2.0 < metrics["energy_range_percent"]
                    if self._is_metrics_better(rebuilt_metrics, metrics) or (primary_ok and energy_better):
                        metrics, self.vehicle_orders, self.vehicle_paths = rebuilt

            exchanged = self._try_boundary_exchange(it, metrics)
            if exchanged is not None and self._is_metrics_better(exchanged[0], metrics):
                metrics, self.vehicle_tree_groups, self.vehicle_orders, self.vehicle_paths = exchanged

            if self._is_metrics_better(metrics, self._best_metrics_proxy()):
                candidate_metrics = metrics
                candidate_orders = deepcopy(self.vehicle_orders)
                candidate_paths = deepcopy(self.vehicle_paths)

                polish_metrics, polish_orders, polish_paths = self._polish_solution(candidate_orders, best_only=True)
                if self._is_metrics_better(polish_metrics, candidate_metrics):
                    candidate_metrics = polish_metrics
                    candidate_orders = polish_orders
                    candidate_paths = polish_paths

                self.best_metrics = deepcopy(candidate_metrics)
                self.best_global_cost = candidate_metrics["total_cost"]
                self.best_primary_cost = candidate_metrics["primary_cost"]
                self.best_global_distance = candidate_metrics["total_distance"]
                self.best_paths = candidate_paths
                self.best_orders = candidate_orders
                self.best_tree_groups = deepcopy(self.vehicle_tree_groups)
                self.best_task_ratios = self._task_ratio_list(self.best_tree_groups)
                self.best_effective_coverage_ratio = candidate_metrics["effective_cov"]
                self.best_theoretical_coverage_ratio = candidate_metrics["theoretical_cov"]
                self.best_obstacle_penalty = candidate_metrics["obstacle_penalty"]
                self.best_energy_range_percent = candidate_metrics["energy_range_percent"]
                self.best_vehicle_distances = candidate_metrics["vehicle_dists"][:]
                self.best_incomplete_tree_ratio = candidate_metrics["incomplete_tree_ratio"]
                self.last_improve_iter = it + 1

            layered = None
            if self._should_trigger_layered_refinement(it):
                layered = self._apply_layered_progressive_refinement(it, metrics)
            if layered is not None:
                metrics, self.vehicle_tree_groups, self.vehicle_orders, self.vehicle_paths = layered
                if self._is_metrics_better(metrics, self._best_metrics_proxy()):
                    candidate_metrics = metrics
                    candidate_orders = deepcopy(self.vehicle_orders)
                    candidate_paths = deepcopy(self.vehicle_paths)
                    polish_metrics, polish_orders, polish_paths = self._polish_solution(candidate_orders, best_only=True)
                    if self._is_metrics_better(polish_metrics, candidate_metrics):
                        candidate_metrics = polish_metrics
                        candidate_orders = polish_orders
                        candidate_paths = polish_paths
                    self.best_metrics = deepcopy(candidate_metrics)
                    self.best_global_cost = candidate_metrics["total_cost"]
                    self.best_primary_cost = candidate_metrics["primary_cost"]
                    self.best_global_distance = candidate_metrics["total_distance"]
                    self.best_paths = candidate_paths
                    self.best_orders = candidate_orders
                    self.best_tree_groups = deepcopy(self.vehicle_tree_groups)
                    self.best_task_ratios = self._task_ratio_list(self.best_tree_groups)
                    self.best_effective_coverage_ratio = candidate_metrics["effective_cov"]
                    self.best_theoretical_coverage_ratio = candidate_metrics["theoretical_cov"]
                    self.best_obstacle_penalty = candidate_metrics["obstacle_penalty"]
                    self.best_energy_range_percent = candidate_metrics["energy_range_percent"]
                    self.best_vehicle_distances = candidate_metrics["vehicle_dists"][:]
                    self.best_incomplete_tree_ratio = candidate_metrics["incomplete_tree_ratio"]
                    self.last_improve_iter = it + 1

            late_boost = self._late_best_path_boost(it)
            if late_boost is not None:
                boost_metrics, boost_groups, boost_orders, boost_paths = late_boost
                self.vehicle_tree_groups = deepcopy(boost_groups)
                self.vehicle_orders = deepcopy(boost_orders)
                self.vehicle_paths = deepcopy(boost_paths)
                self.best_metrics = deepcopy(boost_metrics)
                self.best_global_cost = boost_metrics["total_cost"]
                self.best_primary_cost = boost_metrics["primary_cost"]
                self.best_global_distance = boost_metrics["total_distance"]
                self.best_paths = deepcopy(boost_paths)
                self.best_orders = deepcopy(boost_orders)
                self.best_tree_groups = deepcopy(boost_groups)
                self.best_task_ratios = self._task_ratio_list(self.best_tree_groups)
                self.best_effective_coverage_ratio = boost_metrics["effective_cov"]
                self.best_theoretical_coverage_ratio = boost_metrics["theoretical_cov"]
                self.best_obstacle_penalty = boost_metrics["obstacle_penalty"]
                self.best_energy_range_percent = boost_metrics["energy_range_percent"]
                self.best_vehicle_distances = boost_metrics["vehicle_dists"][:]
                self.best_incomplete_tree_ratio = boost_metrics["incomplete_tree_ratio"]
                self.last_improve_iter = it + 1

            self.history_best_primary.append(self.best_primary_cost)

            self._update_recent_grid()
            self._update_pheromone(it)

            if self.cfg.enable_early_stop and (it + 1) >= self.cfg.early_stop_min_iter and self._stagnation_len(it) >= self.cfg.early_stop_patience:
                elapsed = time.time() - start_time
                self._append_report_row(it, elapsed)
                print(
                    f"[{self.cfg.name}] 提前停止 | "
                    f"迭代={it + 1} | "
                    f"最优主目标={self.best_primary_cost:.2f} | "
                    f"最优总距离={self.best_global_distance:.2f} | "
                    f"能耗极差={self.best_energy_range_percent:.2f}% | "
                    f"停滞长度={self._stagnation_len(it)} | "
                    f"耗时={elapsed:.1f}s"
                )
                break

            if (it + 1) % self.cfg.report_every == 0:
                elapsed = time.time() - start_time
                self._append_report_row(it, elapsed)
                print(
                    f"[{self.cfg.name}] 迭代{it + 1:4d} | "
                    f"最优主目标={self.best_primary_cost:.2f} | "
                    f"参考综合值={self.best_global_cost:.2f} | "
                    f"最优总距离={self.best_global_distance:.2f} | "
                    f"理论覆盖率={self.best_theoretical_coverage_ratio * 100.0:.1f}% | "
                    f"有效覆盖率={self.best_effective_coverage_ratio * 100.0:.1f}% | "
                    f"不完整树比例={self.best_incomplete_tree_ratio * 100.0:.1f}% | "
                    f"障碍惩罚={self.best_obstacle_penalty:.1f} | "
                    f"最佳任务占比={self._task_ratio_text(ratios=self.best_task_ratios if self.best_tree_groups else self._task_ratio_list())} | "
                    f"分层精修触发/成功={self.layered_refine_trigger_count}/{self.layered_refine_success_count} | "
                    f"停滞长度={self._stagnation_len(it)} | "
                    f"耗时={elapsed:.1f}s"
                )

        self._materialize_dynamic_grid()
        self._materialize_recent_grid()

        return RunResult(
            best_primary_cost=float(self.best_primary_cost),
            best_reference_total_cost=float(self.best_global_cost),
            best_total_distance=float(self.best_global_distance),
            runtime_sec=float(time.time() - start_time),
            effective_observation_coverage=float(self.best_effective_coverage_ratio * 100.0),
            theoretical_observation_coverage=float(self.best_theoretical_coverage_ratio * 100.0),
            obstacle_penalty=float(self.best_obstacle_penalty),
            energy_range_percent=float(self.best_energy_range_percent),
            layered_refine_trigger_count=int(self.layered_refine_trigger_count),
            layered_refine_success_count=int(self.layered_refine_success_count),
            incomplete_tree_ratio=float(self.best_incomplete_tree_ratio * 100.0),
            report_rows=deepcopy(self.report_rows),
            history_best_primary=[float(x) for x in self.history_best_primary],
        )



def build_complex_orchard_obstacles(env_cfg: EnvironmentConfig) -> List[Tuple[float, float, float]]:
    # 两条近似“障碍带”+ 若干散布障碍，制造绕行、瓶颈和局部最优陷阱
    obs = [
        (28.0, 26.0, 8.5), (44.0, 38.0, 8.0), (60.0, 50.0, 8.0), (76.0, 62.0, 8.0),
        (92.0, 74.0, 8.5), (38.0, 86.0, 7.5), (58.0, 82.0, 8.0), (78.0, 78.0, 8.0),
        (98.0, 72.0, 7.5), (22.0, 96.0, 6.5), (102.0, 26.0, 6.5)
    ]
    return obs



def sample_safe_point(rng: np.random.Generator, env_cfg: EnvironmentConfig, obstacles, center=None, spread=10.0):
    def is_safe(x, y):
        for ox, oy, rr in obstacles:
            if math.hypot(x - ox, y - oy) <= rr + env_cfg.safe_margin + 1.8:
                return False
        return True

    # 第一层：围绕目标中心做逐步放宽采样
    if center is not None:
        for scale in [1.0, 1.5, 2.0, 3.0, 4.5]:
            cur_spread = spread * scale
            for _ in range(400):
                x = float(np.clip(rng.normal(center[0], cur_spread), 4.0, env_cfg.width - 4.0))
                y = float(np.clip(rng.normal(center[1], cur_spread), 4.0, env_cfg.height - 4.0))
                if is_safe(x, y):
                    return (x, y)

        # 第二层：以目标点为圆心做确定性半径搜索，避免完全靠随机碰运气
        for radius in [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0]:
            for deg in range(0, 360, 15):
                rad = math.radians(deg)
                x = float(np.clip(center[0] + radius * math.cos(rad), 4.0, env_cfg.width - 4.0))
                y = float(np.clip(center[1] + radius * math.sin(rad), 4.0, env_cfg.height - 4.0))
                if is_safe(x, y):
                    return (x, y)

    # 第三层：退回全局安全采样
    for _ in range(4000):
        x = float(rng.uniform(4.0, env_cfg.width - 4.0))
        y = float(rng.uniform(4.0, env_cfg.height - 4.0))
        if is_safe(x, y):
            return (x, y)

    raise RuntimeError("复杂环境采样失败：安全点不足，请适当减少障碍密度或放宽安全边界。")


def build_complex_orchard_trees(env_cfg: EnvironmentConfig, obstacles) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(env_cfg.random_seed)

    # 4 个簇 + 2 条走廊带点，尽量诱发前期较好、后期难跳的结构
    cluster_centers = [
        (18.0, 18.0), (100.0, 20.0), (22.0, 100.0), (100.0, 98.0),
        (55.0, 28.0), (68.0, 96.0)
    ]
    cluster_counts = [12, 12, 12, 12, 10, 10]  # 共 68
    trees = []

    for c, n in zip(cluster_centers, cluster_counts):
        for _ in range(n):
            trees.append(sample_safe_point(rng, env_cfg, obstacles, center=c, spread=9.5))

    # 12 个“瓶颈走廊”点：逼迫算法在障碍缝里做取舍
    corridor_points = [
        (48.0, 20.0), (55.0, 34.0), (63.0, 46.0), (71.0, 58.0),
        (81.0, 66.0), (90.0, 79.0), (30.0, 74.0), (42.0, 80.0),
        (54.0, 86.0), (66.0, 83.0), (80.0, 76.0), (92.0, 69.0)
    ]
    for px, py in corridor_points:
        # 沿走廊附近轻微扰动
        trees.append(sample_safe_point(rng, env_cfg, obstacles, center=(px, py), spread=2.5))

    if len(trees) != env_cfg.tree_num:
        raise RuntimeError(f"复杂环境树数量不匹配：{len(trees)} != {env_cfg.tree_num}")
    return trees


def run_single_case(env_cfg: EnvironmentConfig, alg_cfg: AlgorithmConfig, use_layered_refine: bool, tag: str):
    obstacles = build_complex_orchard_obstacles(env_cfg)
    trees = build_complex_orchard_trees(env_cfg, obstacles)

    env = OrchardEnv(
        tree_coords=trees,
        vehicle_num=env_cfg.vehicle_num,
        vehicle_battery=env_cfg.vehicle_battery,
        start_positions=env_cfg.start_positions,
        obstacles=obstacles,
        obs_radius=env_cfg.obs_radius,
        safe_margin=env_cfg.safe_margin,
    )

    cfg = deepcopy(alg_cfg)
    cfg.name = f"{alg_cfg.name}-{tag}"
    if not use_layered_refine:
        cfg.layered_refine_after_stagnation = cfg.max_iter + 1
        cfg.forced_repartition_after_stagnation = cfg.max_iter + 1

    solver = FusionLeafcutterAntAlgorithm(env=env, cfg=cfg)
    result = solver.run()
    return result


def print_case_summary(title: str, result: RunResult):
    print(f"\n[{title}]")
    print(f"最优主目标(距离+障碍加权): {result.best_primary_cost:.2f}")
    print(f"参考综合值: {result.best_reference_total_cost:.2f}")
    print(f"最优总距离: {result.best_total_distance:.2f}")
    print(f"理论覆盖率: {result.theoretical_observation_coverage:.2f}%")
    print(f"有效覆盖率: {result.effective_observation_coverage:.2f}%")
    print(f"能耗极差: {result.energy_range_percent:.2f}%")
    print(f"不完整树比例: {result.incomplete_tree_ratio:.2f}%")
    print(f"障碍惩罚: {result.obstacle_penalty:.2f}")
    print(f"分层精修触发次数: {result.layered_refine_trigger_count}")
    print(f"分层精修成功次数: {result.layered_refine_success_count}")
    rate = 100.0 * result.layered_refine_success_count / max(1, result.layered_refine_trigger_count)
    print(f"分层精修成功率: {rate:.2f}%")
    print(f"总耗时: {result.runtime_sec:.2f}s")






def main():
    env_cfg = EnvironmentConfig()
    alg_cfg = AlgorithmConfig()

    np.random.seed(env_cfg.random_seed)
    random.seed(env_cfg.random_seed)

    print("开始运行 F-LCA 三级工蚁分层协同版（复杂环境，异质分工 + 四点观测 + 渐进精修）...")
    print("复杂环境特征：80树 + 11障碍 + 瓶颈走廊 + 电池=100/80/60/40")

    obstacles = build_complex_orchard_obstacles(env_cfg)
    trees = build_complex_orchard_trees(env_cfg, obstacles)

    env = OrchardEnv(
        tree_coords=trees,
        vehicle_num=env_cfg.vehicle_num,
        vehicle_battery=env_cfg.vehicle_battery,
        start_positions=env_cfg.start_positions,
        obstacles=obstacles,
        obs_radius=env_cfg.obs_radius,
        safe_margin=env_cfg.safe_margin,
    )

    solver = FusionLeafcutterAntAlgorithm(env=env, cfg=alg_cfg)
    result = solver.run()

    out_dir = OUTPUTS_DIR / "flca_tiered_refinement_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.report_rows:
        with (out_dir / "report.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=result.report_rows[0].keys())
            writer.writeheader()
            writer.writerows(result.report_rows)

    print("\n运行结束！")
    print(f"最优主目标(距离+障碍加权): {result.best_primary_cost:.2f}")
    print(f"参考综合值(仅作辅助查看): {result.best_reference_total_cost:.2f}")
    print(f"最优总距离: {result.best_total_distance:.2f}")
    print(f"理论覆盖率: {result.theoretical_observation_coverage:.2f}%")
    print(f"有效覆盖率: {result.effective_observation_coverage:.2f}%")
    print(f"能耗极差: {result.energy_range_percent:.2f}%")
    print(f"不完整树比例: {result.incomplete_tree_ratio:.2f}%")
    print(f"障碍惩罚: {result.obstacle_penalty:.2f}")
    print(f"分层精修触发次数: {result.layered_refine_trigger_count}")
    print(f"分层精修成功次数: {result.layered_refine_success_count}")
    rate = 100.0 * result.layered_refine_success_count / max(1, result.layered_refine_trigger_count)
    print(f"分层精修成功率: {rate:.2f}%")
    print(f"总耗时: {result.runtime_sec:.2f}s")
    print(f"数据已导出至: {out_dir}")


if __name__ == "__main__":
    main()
