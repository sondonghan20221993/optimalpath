#!/usr/bin/env python3
"""
mast3r L자 물체의 초기 경로 형태 시각화
- Greedy (실제 pbnbv_path) vs Orbit vs Random 비교
"""

import numpy as np
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUT_DIR = Path(__file__).parent / "results" / "mast3r_pathcomparison"
PBNBV_JSON = Path(__file__).parent / "results" / "realtest_pbnbv_path" / "realtest_pbnbv_path.json"

# ─────────────────────────────────────────────────────────────────────────────
# 경로 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_orbit(target, n_points, radius=5.0, alt=1.0):
    """Orbit-N: 균등한 원형 경로"""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    positions = []
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = target[0] + radius * np.cos(az_rad)
        y = target[1] + radius * np.sin(az_rad)
        z = target[2] - alt  # NED: 음수 = 위
        positions.append([x, y, z])
    return np.array(positions)

def make_random(target, n_points, radius=5.0, alt=1.0, seed=42):
    """Random-N: 무작위 선택"""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(n_points):
        th = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(3.0, 8.0)
        a = rng.uniform(0.5, 3.0)
        x = target[0] + r * np.cos(th)
        y = target[1] + r * np.sin(th)
        z = target[2] - a
        positions.append([x, y, z])
    return np.array(positions)

def load_greedy_from_json(json_path):
    """pbnbv_path.json에서 Greedy 경로 로드"""
    with open(json_path) as f:
        data = json.load(f)

    positions = []
    for wp in data['waypoints']:
        positions.append(wp['position'])
    return np.array(positions)

# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def visualize_paths():
    """3개 경로를 비교하는 3D 시각화"""

    # 데이터 로드
    greedy_path = load_greedy_from_json(PBNBV_JSON)
    n_points = len(greedy_path)

    with open(PBNBV_JSON) as f:
        data = json.load(f)
    target = np.array(data['target_position'])

    # Orbit & Random 생성
    orbit_path = make_orbit(target, n_points, radius=5.0, alt=1.0)
    random_path = make_random(target, n_points, radius=5.0, alt=1.0)

    # 3개 경로 시각화
    fig = plt.figure(figsize=(18, 5))

    paths = {
        'Orbit (균등 원형)': orbit_path,
        'Greedy (형상 적응)': greedy_path,
        'Random (무작위)': random_path
    }

    for idx, (name, path) in enumerate(paths.items(), 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')

        # 타겟
        ax.scatter(*target, color='red', s=200, marker='*',
                  label='Target (L자 물체)', zorder=100)

        # 경로
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'o-',
               linewidth=2, markersize=6, label='Path', alpha=0.7)

        # 시작점 & 끝점
        ax.scatter(*path[0], color='green', s=100, marker='s', label='Start')
        ax.scatter(*path[-1], color='blue', s=100, marker='^', label='End')

        # 레이아웃
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (NED)', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 같은 축 범위
        all_pts = np.vstack([path, target])
        pad = 2
        ax.set_xlim([all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad])
        ax.set_ylim([all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad])
        ax.set_zlim([all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad])

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mast3r_paths_3d_comparison.png", dpi=150, bbox_inches='tight')
    print(f"✅ 저장: {OUT_DIR / 'mast3r_paths_3d_comparison.png'}")
    plt.close()

def visualize_2d_toplevel():
    """위에서 본 2D 경로 (XY 평면)"""

    # 데이터 로드
    greedy_path = load_greedy_from_json(PBNBV_JSON)
    n_points = len(greedy_path)

    with open(PBNBV_JSON) as f:
        data = json.load(f)
    target = np.array(data['target_position'])

    # Orbit & Random 생성
    orbit_path = make_orbit(target, n_points, radius=5.0, alt=1.0)
    random_path = make_random(target, n_points, radius=5.0, alt=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    paths = [
        ('Orbit (균등 원형)', orbit_path),
        ('Greedy (형상 적응)', greedy_path),
        ('Random (무작위)', random_path)
    ]

    for ax, (name, path) in zip(axes, paths):
        # 타겟
        ax.scatter(target[0], target[1], color='red', s=200, marker='*',
                  label='L자 물체 중심', zorder=100)

        # 경로
        ax.plot(path[:, 0], path[:, 1], 'o-',
               linewidth=2, markersize=8, label='Path', alpha=0.7)

        # 시작점 & 끝점
        ax.scatter(path[0, 0], path[0, 1], color='green', s=100, marker='s', label='Start')
        ax.scatter(path[-1, 0], path[-1, 1], color='blue', s=100, marker='^', label='End')

        # 레이아웃
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # 축 범위
        all_pts = np.vstack([path, target])
        pad = 2
        ax.set_xlim([all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad])
        ax.set_ylim([all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad])

    plt.tight_layout()
    plt.savefig(OUT_DIR / "mast3r_paths_2d_topview.png", dpi=150, bbox_inches='tight')
    print(f"✅ 저장: {OUT_DIR / 'mast3r_paths_2d_topview.png'}")
    plt.close()

def print_path_stats():
    """경로 통계"""

    greedy_path = load_greedy_from_json(PBNBV_JSON)
    n_points = len(greedy_path)

    with open(PBNBV_JSON) as f:
        data = json.load(f)
    target = np.array(data['target_position'])

    orbit_path = make_orbit(target, n_points, radius=5.0, alt=1.0)
    random_path = make_random(target, n_points, radius=5.0, alt=1.0)

    def compute_stats(path, name):
        # 경로 길이
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_dist = np.sum(dists)

        # 타겟까지의 거리
        target_dists = np.linalg.norm(path - target, axis=1)
        mean_dist_to_target = np.mean(target_dists)

        # 경로의 높이 분산
        z_std = np.std(path[:, 2])

        print(f"\n{name}:")
        print(f"  경로 길이: {total_dist:.2f}m")
        print(f"  타겟까지 평균 거리: {mean_dist_to_target:.2f}m")
        print(f"  높이 변화 (Std): {z_std:.2f}m")
        print(f"  포인트 수: {n_points}")

    print("=" * 60)
    print("mast3r L자 물체의 초기 경로 형태 분석")
    print("=" * 60)

    compute_stats(orbit_path, "Orbit (균등 원형)")
    compute_stats(greedy_path, "Greedy (형상 적응)")
    compute_stats(random_path, "Random (무작위)")

    print("\n" + "=" * 60)
    print("📌 핵심 관찰")
    print("=" * 60)
    print("✅ Orbit: 균등한 원형 → 모든 방향에서 균등하게 관측")
    print("✅ Greedy: 형상 보고 적응 → L자 물체의 특정 부분에 집중")
    print("✅ Random: 무작위 → 예측 불가능한 분포")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("mast3r L자 물체의 초기 경로 형태 시각화")
    print("=" * 60)

    print("\n[1/3] 3D 경로 비교...")
    visualize_paths()

    print("[2/3] 2D 위에서 본 경로...")
    visualize_2d_toplevel()

    print("[3/3] 통계 분석...")
    print_path_stats()

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)
