import json
import math
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 用户可调参数
# =========================
SCENE_JSON = "aco_seed44_export/scene_seed44.json"
FLCA_JSON = "flca_seed44_export/best_path_seed44.json"
ACO_JSON = "aco_seed44_export/best_path_seed44_aco.json"
OUT_DIR = "execution_postprocess_seed44_astar_anchor"

GRID_RES = 0.5               # 栅格分辨率（m）
VEHICLE_RADIUS = 0.6         # 车体等效半径（m）
SAFE_MARGIN = 1.5            # 安全边界（m）
MIN_TURN_RADIUS = 2.0        # 最小转弯半径（m）

V_LIN = 0.8                  # 平均线速度（m/s）
T_STOP = 3.0                 # 单观测点停留时间（s）

SMOOTH_WINDOW = 3            # 段内轻量平滑窗口（建议奇数；1表示不平滑）
SHOW_RISK_MARKERS = True
SHOW_TURN_MARKERS = True


# =========================
# 基础工具
# =========================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])


def polyline_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))


def point_to_segment_distance(p: Tuple[float, float],
                              a: Tuple[float, float],
                              b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def point_clearance_to_obstacles(point: Tuple[float, float],
                                 obstacles: List[Tuple[float, float, float]],
                                 inflate: float = 0.0) -> float:
    x, y = point
    ds = [math.hypot(x - ox, y - oy) - (r + inflate) for ox, oy, r in obstacles]
    return min(ds) if ds else float("inf")


def segment_clearance_to_obstacles(a: Tuple[float, float],
                                   b: Tuple[float, float],
                                   obstacles: List[Tuple[float, float, float]],
                                   inflate: float = 0.0) -> float:
    ds = []
    for ox, oy, r in obstacles:
        d = point_to_segment_distance((ox, oy), a, b) - (r + inflate)
        ds.append(d)
    return min(ds) if ds else float("inf")


def min_clearance_polyline(points: List[Tuple[float, float]],
                           obstacles: List[Tuple[float, float, float]],
                           inflate: float = 0.0) -> float:
    if len(points) == 0:
        return float("inf")
    vals = [point_clearance_to_obstacles(p, obstacles, inflate) for p in points]
    if len(points) >= 2:
        vals += [segment_clearance_to_obstacles(points[i], points[i + 1], obstacles, inflate)
                 for i in range(len(points) - 1)]
    return min(vals) if vals else float("inf")


def circumradius(p0: Tuple[float, float],
                 p1: Tuple[float, float],
                 p2: Tuple[float, float]) -> float:
    a = euclidean(p0, p1)
    b = euclidean(p1, p2)
    c = euclidean(p0, p2)
    if a < 1e-9 or b < 1e-9 or c < 1e-9:
        return float("inf")
    area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    if area2 < 1e-9:
        return float("inf")
    area = 0.5 * area2
    return (a * b * c) / (4.0 * area)


def moving_average_segment(points: List[Tuple[float, float]], window: int) -> List[Tuple[float, float]]:
    """仅在单段内部轻量平滑，保留端点。"""
    if window <= 1 or len(points) <= 2:
        return points[:]
    if window % 2 == 0:
        window += 1
    k = window // 2
    arr = np.array(points, dtype=float)
    out = arr.copy()
    for i in range(1, len(points) - 1):
        l = max(0, i - k)
        r = min(len(points), i + k + 1)
        out[i] = arr[l:r].mean(axis=0)
    out[0] = arr[0]
    out[-1] = arr[-1]
    return [tuple(map(float, x)) for x in out]


# =========================
# 栅格与 A*
# =========================
class OccupancyGrid:
    def __init__(self, width: float, height: float, obstacles: List[Tuple[float, float, float]], res: float):
        self.width = float(width)
        self.height = float(height)
        self.res = float(res)
        self.nx = int(math.ceil(self.width / self.res)) + 1
        self.ny = int(math.ceil(self.height / self.res)) + 1
        self.obstacles = obstacles

    def world_to_grid(self, p: Tuple[float, float]) -> Tuple[int, int]:
        x, y = p
        gx = int(round(x / self.res))
        gy = int(round(y / self.res))
        gx = max(0, min(self.nx - 1, gx))
        gy = max(0, min(self.ny - 1, gy))
        return gx, gy

    def grid_to_world(self, g: Tuple[int, int]) -> Tuple[float, float]:
        gx, gy = g
        return gx * self.res, gy * self.res

    def is_free_world(self, p: Tuple[float, float], inflate: float) -> bool:
        x, y = p
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        return point_clearance_to_obstacles(p, self.obstacles, inflate) >= 0.0

    def is_free_grid(self, g: Tuple[int, int], inflate: float) -> bool:
        return self.is_free_world(self.grid_to_world(g), inflate)


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def astar_grid(grid: OccupancyGrid,
               start_xy: Tuple[float, float],
               goal_xy: Tuple[float, float],
               inflate: float) -> Optional[List[Tuple[float, float]]]:
    start = grid.world_to_grid(start_xy)
    goal = grid.world_to_grid(goal_xy)

    # 若锚点位于膨胀障碍边界内，允许锚点本身，但不允许中间路径进入
    def neighbors(node):
        x, y = node
        for dx, dy, c in [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                          (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.nx and 0 <= ny < grid.ny:
                ng = (nx, ny)
                if ng == goal or ng == start or grid.is_free_grid(ng, inflate):
                    yield ng, c

    pq = []
    heapq.heappush(pq, (0.0, start))
    gscore = {start: 0.0}
    parent = {start: None}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = []
            node = cur
            while node is not None:
                path.append(grid.grid_to_world(node))
                node = parent[node]
            path.reverse()
            path[0] = start_xy
            path[-1] = goal_xy
            return path

        for nb, cost in neighbors(cur):
            ng = gscore[cur] + cost
            if nb not in gscore or ng < gscore[nb]:
                gscore[nb] = ng
                parent[nb] = cur
                f = ng + heuristic(nb, goal)
                heapq.heappush(pq, (f, nb))
    return None


def line_is_safe(a: Tuple[float, float], b: Tuple[float, float],
                 obstacles: List[Tuple[float, float, float]], inflate: float) -> bool:
    return segment_clearance_to_obstacles(a, b, obstacles, inflate) >= 0.0


def visibility_simplify(points: List[Tuple[float, float]],
                        obstacles: List[Tuple[float, float, float]],
                        inflate: float) -> List[Tuple[float, float]]:
    """只在单段内部做可见性简化，保留端点。"""
    if len(points) <= 2:
        return points[:]
    out = [points[0]]
    i = 0
    n = len(points)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if line_is_safe(points[i], points[j], obstacles, inflate):
                break
            j -= 1
        out.append(points[j])
        i = j
    return out


def connect_anchor_pair(a: Tuple[float, float],
                        b: Tuple[float, float],
                        grid: OccupancyGrid,
                        obstacles: List[Tuple[float, float, float]],
                        inflate: float,
                        smooth_window: int) -> List[Tuple[float, float]]:
    """对相邻锚点单独连接，禁止跨锚点 shortcut。"""
    if line_is_safe(a, b, obstacles, inflate):
        seg = [a, b]
    else:
        raw = astar_grid(grid, a, b, inflate)
        if raw is None:
            seg = [a, b]
        else:
            seg = visibility_simplify(raw, obstacles, inflate)
            seg = moving_average_segment(seg, smooth_window)
            # 平滑后若出现侵入，则回退到简化后路径
            if min_clearance_polyline(seg, obstacles, inflate) < 0:
                seg = visibility_simplify(raw, obstacles, inflate)

    seg[0] = a
    seg[-1] = b
    return seg


# =========================
# 执行轨迹构造（锚点保留版）
# =========================
def build_execution_path(anchor_points: List[Tuple[float, float]],
                         grid: OccupancyGrid,
                         obstacles: List[Tuple[float, float, float]],
                         inflate: float,
                         smooth_window: int) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    返回：
    - 执行轨迹点
    - 锚点在执行轨迹中的索引位置（用于后续分析/可视化）
    """
    if len(anchor_points) <= 1:
        return anchor_points[:], [0]

    exec_pts: List[Tuple[float, float]] = []
    anchor_indices: List[int] = []

    for i in range(len(anchor_points) - 1):
        a = anchor_points[i]
        b = anchor_points[i + 1]
        seg = connect_anchor_pair(a, b, grid, obstacles, inflate, smooth_window)

        if i == 0:
            exec_pts.extend(seg)
            anchor_indices.append(0)
            anchor_indices.append(len(exec_pts) - 1)
        else:
            # 拼接时保留当前段终点锚点，去掉重复起点
            exec_pts.extend(seg[1:])
            anchor_indices.append(len(exec_pts) - 1)

    return exec_pts, anchor_indices


# =========================
# 指标计算
# =========================
def turn_violation_stats(points: List[Tuple[float, float]], min_turn_radius: float) -> Tuple[int, List[int]]:
    if len(points) < 3:
        return 0, []
    bad_ids = []
    for i in range(1, len(points) - 1):
        R = circumradius(points[i - 1], points[i], points[i + 1])
        if math.isfinite(R) and R < min_turn_radius:
            bad_ids.append(i)
    return len(bad_ids), bad_ids


def process_algorithm(scene: dict, path_json: dict, algorithm_name: str) -> Tuple[dict, pd.DataFrame]:
    width = float(scene["width"])
    height = float(scene["height"])
    obstacles = [(float(o["cx"]), float(o["cy"]), float(o["r"])) for o in scene["obstacles"]]
    grid = OccupancyGrid(width, height, obstacles, GRID_RES)

    inflate_safety = SAFE_MARGIN + VEHICLE_RADIUS

    vehicles = path_json["vehicles"]
    rows = []

    total_exec_len = 0.0
    total_exec_time = 0.0
    overall_min_obstacle_clear = float("inf")
    overall_min_safe_clear = float("inf")
    total_turn_viol = 0
    total_turn_windows = 0
    exec_count = 0
    collision_free_count = 0

    # 为画图保留
    plot_bundle = []

    for vid_str in sorted(vehicles.keys(), key=lambda x: int(x)):
        v = vehicles[vid_str]
        raw_anchor = [tuple(map(float, p)) for p in v["path_points"]]   # 起点 + 必经观测点
        n_obs = max(0, len(raw_anchor) - 1)

        exec_path, anchor_ids = build_execution_path(
            anchor_points=raw_anchor,
            grid=grid,
            obstacles=obstacles,
            inflate=inflate_safety,
            smooth_window=SMOOTH_WINDOW
        )

        exec_len = polyline_length(exec_path)
        exec_time = exec_len / V_LIN + n_obs * T_STOP

        min_entity_clear = min_clearance_polyline(exec_path, obstacles, inflate=VEHICLE_RADIUS)
        min_safe_clear = min_clearance_polyline(exec_path, obstacles, inflate=inflate_safety)

        n_viol, bad_ids = turn_violation_stats(exec_path, MIN_TURN_RADIUS)
        turn_windows = max(0, len(exec_path) - 2)
        viol_ratio = (n_viol / turn_windows * 100.0) if turn_windows > 0 else 0.0

        executable = (n_viol == 0)
        collision_free = (min_safe_clear >= 0.0)

        total_exec_len += exec_len
        total_exec_time += exec_time
        overall_min_obstacle_clear = min(overall_min_obstacle_clear, min_entity_clear)
        overall_min_safe_clear = min(overall_min_safe_clear, min_safe_clear)
        total_turn_viol += n_viol
        total_turn_windows += turn_windows
        exec_count += 1 if executable else 0
        collision_free_count += 1 if collision_free else 0

        rows.append({
            "algorithm": algorithm_name,
            "vehicle_id": int(v["vehicle_id"]),
            "n_obs": int(n_obs),
            "execution_length": exec_len,
            "estimated_execution_time": exec_time,
            "min_obstacle_clearance": min_entity_clear,
            "min_safe_clearance": min_safe_clear,
            "turn_violation_count": n_viol,
            "turn_violation_ratio_percent": viol_ratio,
            "executable": int(executable),
            "collision_free": int(collision_free),
        })

        plot_bundle.append({
            "vehicle_id": int(v["vehicle_id"]),
            "raw_anchor": raw_anchor,
            "exec_path": exec_path,
            "bad_ids": bad_ids,
            "entity_risk_pts": [p for p in exec_path if point_clearance_to_obstacles(p, obstacles, VEHICLE_RADIUS) < 1.0],
            "safe_risk_pts": [p for p in exec_path if point_clearance_to_obstacles(p, obstacles, inflate_safety) < 0.5],
        })

    df = pd.DataFrame(rows)
    summary = {
        "seed": int(path_json["seed"]),
        "algorithm": algorithm_name,
        "mean_estimated_execution_time": float(df["estimated_execution_time"].mean()),
        "total_estimated_execution_time": float(total_exec_time),
        "minimum_obstacle_clearance": float(overall_min_obstacle_clear),
        "minimum_safe_clearance": float(overall_min_safe_clear),
        "total_turn_violation_count": int(total_turn_viol),
        "total_turn_violation_ratio_percent": float((total_turn_viol / total_turn_windows * 100.0) if total_turn_windows > 0 else 0.0),
        "executable_trajectory_rate_percent": float(exec_count / len(df) * 100.0),
        "collision_free_success_rate_percent": float(collision_free_count / len(df) * 100.0),
        "total_execution_length": float(total_exec_len),
    }
    return {"summary": summary, "plot_bundle": plot_bundle}, df


# =========================
# 绘图
# =========================
def plot_algorithm(scene: dict, bundle: dict, out_png: str, title: str):
    obstacles = [(float(o["cx"]), float(o["cy"]), float(o["r"])) for o in scene["obstacles"]]
    trees = [(float(t["x"]), float(t["y"])) for t in scene["trees"]]
    starts = [tuple(map(float, s)) for s in scene["start_positions"]]

    fig, ax = plt.subplots(figsize=(14, 12), dpi=180)

    # 障碍实体与安全圈
    for ox, oy, r in obstacles:
        circ_entity = plt.Circle((ox, oy), r, fill=False, color="black", lw=1.2)
        circ_safe = plt.Circle((ox, oy), r + SAFE_MARGIN, fill=False, color="gray", lw=1.2, ls="--")
        ax.add_patch(circ_entity)
        ax.add_patch(circ_safe)

    if trees:
        tx, ty = zip(*trees)
        ax.scatter(tx, ty, s=20, label="Trees", alpha=0.7)

    if starts:
        sx, sy = zip(*starts)
        ax.scatter(sx, sy, s=120, marker="s", label="Starts")

    colors = ["tab:orange", "tab:red", "tab:brown", "tab:gray", "tab:green", "tab:blue"]
    risk_colors = ["tab:green", "tab:purple", "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]
    turn_markers = ["^", "^", "^", "^", "^", "^"]

    for idx, item in enumerate(bundle["plot_bundle"]):
        vid = item["vehicle_id"]
        c = colors[idx % len(colors)]
        rc = risk_colors[idx % len(risk_colors)]

        raw_anchor = item["raw_anchor"]
        exec_path = item["exec_path"]

        if len(raw_anchor) >= 2:
            x, y = zip(*raw_anchor)
            ax.plot(x, y, "--", color=c, alpha=0.55, lw=1.2, label=f"V{vid+1} raw anchors")

        if len(exec_path) >= 2:
            x, y = zip(*exec_path)
            ax.plot(x, y, "-", color=c, lw=2.0, label=f"V{vid+1} exec path")

        if SHOW_RISK_MARKERS and item["safe_risk_pts"]:
            rx, ry = zip(*item["safe_risk_pts"])
            ax.scatter(rx, ry, marker="x", s=35, color=rc, label=f"V{vid+1} safety risk")

        if SHOW_TURN_MARKERS and item["bad_ids"]:
            pts = [exec_path[i] for i in item["bad_ids"]]
            if pts:
                tx, ty = zip(*pts)
                ax.scatter(tx, ty, marker=turn_markers[idx % len(turn_markers)], s=40,
                           color=c, label=f"V{vid+1} turn violation")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-2, float(scene["width"]) + 2)
    ax.set_ylim(-2, float(scene["height"]) + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=9, ncol=1, framealpha=0.9)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =========================
# 主流程
# =========================
def save_json(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_json(SCENE_JSON)
    flca_json = load_json(FLCA_JSON)
    aco_json = load_json(ACO_JSON)

    # F-LCA
    flca_bundle, flca_df = process_algorithm(scene, flca_json, "F-LCA")
    save_json(out_dir / "f_lca_execution_metrics_astar_anchor.json", flca_bundle["summary"])
    flca_df.to_csv(out_dir / "f_lca_vehicle_metrics_astar_anchor.csv", index=False, encoding="utf-8-sig")
    plot_algorithm(
        scene, flca_bundle,
        str(out_dir / "f_lca_execution_plot_astar_anchor.png"),
        f"F-LCA Execution-oriented Post-processing with Anchor-preserving A* (seed={flca_json['seed']})"
    )

    # ACO
    aco_bundle, aco_df = process_algorithm(scene, aco_json, "ACO")
    save_json(out_dir / "aco_execution_metrics_astar_anchor.json", aco_bundle["summary"])
    aco_df.to_csv(out_dir / "aco_vehicle_metrics_astar_anchor.csv", index=False, encoding="utf-8-sig")
    plot_algorithm(
        scene, aco_bundle,
        str(out_dir / "aco_execution_plot_astar_anchor.png"),
        f"ACO Execution-oriented Post-processing with Anchor-preserving A* (seed={aco_json['seed']})"
    )

    # 汇总表
    summary_rows = [flca_bundle["summary"], aco_bundle["summary"]]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "execution_comparison_summary_astar_anchor.csv",
                      index=False, encoding="utf-8-sig")

    # 终端输出
    for s in summary_rows:
        print(f"\n===== {s['algorithm']} =====")
        print(f"seed = {s['seed']}")
        print(f"估算执行时间均值 = {s['mean_estimated_execution_time']:.3f}")
        print(f"估算执行时间总和 = {s['total_estimated_execution_time']:.3f}")
        print(f"障碍实体最小净空 = {s['minimum_obstacle_clearance']:.3f}")
        print(f"安全边界最小净空 = {s['minimum_safe_clearance']:.3f}")
        print(f"曲率违例总数 = {s['total_turn_violation_count']}")
        print(f"曲率违例比例 = {s['total_turn_violation_ratio_percent']:.2f}%")
        print(f"可执行轨迹率 = {s['executable_trajectory_rate_percent']:.2f}%")
        print(f"无碰撞成功率 = {s['collision_free_success_rate_percent']:.2f}%")
        print(f"执行轨迹总长度 = {s['total_execution_length']:.3f}")

    print(f"\n结果已保存到: {out_dir}")
    print("说明：本脚本只影响执行层后处理，不影响主实验、统计检验与消融实验的原始结果。")


if __name__ == "__main__":
    main()
