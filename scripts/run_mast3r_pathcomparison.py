#!/usr/bin/env python3
"""
mast3r 복원물체 (L자 모양)에 대한 경로 비교 실험
step2_v3과 동일한 방식: Orbit vs Greedy vs Random

목표: 초기 경로가 어떤 형태로 나오는지 확인
"""

import numpy as np
import json
import math
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import open3d as o3d
except ImportError:
    o3d = None

# ─────────────────────────────────────────────────────────────────────────────
# 경로 & 파라미터
# ─────────────────────────────────────────────────────────────────────────────

MAST3R_DIR = Path(r"/mnt/c/Users/sdh97/Desktop/3d_results/blue_1")
PLY_FILE = MAST3R_DIR / "mast3r_tsdf_clean.ply"
POSES_FILE = MAST3R_DIR / "poses.npy"
FOCALS_FILE = MAST3R_DIR / "focals.npy"

OUT_DIR = Path(__file__).parent / "results" / "mast3r_pathcomparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 알고리즘 파라미터 (pbnbv_path.py 참조)
FOV_DEG = 89.9
MIN_DIST = 4.0
MAX_DIST = 13.0
VOXEL_SIZES = [0.15, 0.075]
N_BUDGET = [4, 6, 8, 12, 16]

# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티: PLY 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_ply(ply_path, sample_n=50000):
    """PLY 파일 로드 및 점 서브샘플링 (open3d 없이)"""
    import struct

    points = []
    try:
        with open(ply_path, 'rb') as f:
            # Header 읽기
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break

            # vertex 개수 파싱
            n_vertices = 0
            for line in header_lines:
                if line.startswith('element vertex'):
                    n_vertices = int(line.split()[-1])
                    break

            # vertex 데이터 읽기 (x, y, z만)
            for _ in range(n_vertices):
                x, y, z = struct.unpack('<fff', f.read(12))
                points.append([x, y, z])

    except Exception as e:
        print(f"Warning: PLY load failed ({e}), using open3d")
        if o3d is not None:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points).tolist()
        else:
            raise

    points = np.array(points)

    # NaN/Inf 제거
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]

    if len(points) > sample_n:
        indices = np.random.choice(len(points), sample_n, replace=False)
        points = points[indices]

    print(f"Loaded {len(points)} points from {ply_path.name}")
    return points

def estimate_normals(points, k=10):
    """법선 추정: k-nearest neighbor로 PCA"""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    normals = []

    # k-NN 찾기 (자동으로 거리 결정)
    for i, p in enumerate(points):
        distances, indices = tree.query(p, k=k)
        neighbors = points[indices]

        # PCA로 법선 추정
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        normal = pca.components_[-1]  # 가장 작은 고유값의 고유벡터

        # 평균 카메라 위치 (0, 0, 5) 방향으로 향하도록
        camera_dir = np.array([0, 0, 1])
        if np.dot(normal, camera_dir) < 0:
            normal = -normal

        normals.append(normal)

    return np.array(normals)

# ─────────────────────────────────────────────────────────────────────────────
# Voxelization & PCA
# ─────────────────────────────────────────────────────────────────────────────

def voxelize_and_pca(points, point_normals, voxel_size):
    """Voxelize 및 voxel 레벨 법선 생성"""
    origin = points.min(0) - voxel_size
    voxel_indices = np.floor((points - origin) / voxel_size).astype(int)

    unique_voxels, inv = np.unique(voxel_indices, axis=0, return_inverse=True)
    voxel_centers = unique_voxels * voxel_size + origin + voxel_size/2

    # Voxel-level normals
    voxel_normals = np.zeros((len(unique_voxels), 3))
    for i in range(len(unique_voxels)):
        mask = inv == i
        avg = point_normals[mask].mean(0)
        norm = np.linalg.norm(avg)
        voxel_normals[i] = avg / (norm + 1e-9)

    # PCA flatness
    pca = PCA()
    pca.fit(voxel_centers)
    var_ratio = pca.explained_variance_ratio_
    flatness = var_ratio[-1] / var_ratio[0]

    return {
        'n_surface_voxels': len(unique_voxels),
        'voxel_centers': voxel_centers,
        'voxel_normals': voxel_normals,
        'flatness': float(flatness),
        'pca_components': pca.components_,
        'pca_mean': pca.mean_
    }

# ─────────────────────────────────────────────────────────────────────────────
# 경로 생성 (step2_v3과 동일)
# ─────────────────────────────────────────────────────────────────────────────

def make_candidates(voxel_centers):
    """후보 시점 생성: 물체 주변의 구 좌표"""
    target = voxel_centers.mean(0)
    cands = []

    for alt in np.arange(1.0, 9.5, 1.0):
        z = target[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2 * math.pi * rad / 1.5))
            for i in range(n_az):
                th = 2 * math.pi * i / n_az
                cands.append([target[0] + rad * math.cos(th),
                             target[1] + rad * math.sin(th), z])
    return np.array(cands), target

def get_visible(cam_pos, target, voxel_centers, voxel_normals):
    """cam_pos에서 보이는 voxel 불리언 마스크"""
    cam_dir = target - cam_pos
    cam_dir /= np.linalg.norm(cam_dir) + 1e-9

    to = voxel_centers - cam_pos
    dist = np.linalg.norm(to, axis=1)

    in_range = (dist >= MIN_DIST) & (dist <= MAX_DIST)
    in_fov = (to * cam_dir).sum(1) / (dist + 1e-9) >= math.cos(math.radians(FOV_DEG / 2))
    front = (voxel_normals * (cam_pos - voxel_centers)).sum(1) > 0

    return in_range & in_fov & front

def make_orbit(target, n_points, radius=7.0, alt=4.0):
    """Orbit-N: 균등 원형"""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    positions = []
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = target[0] + radius * np.cos(az_rad)
        y = target[1] + radius * np.sin(az_rad)
        z = target[2] - alt
        positions.append([x, y, z])
    return np.array(positions)

def make_greedy(candidates, target, voxel_centers, voxel_normals, n_budget):
    """Greedy-N: oracle set-cover greedy"""
    vis_masks = np.array([
        get_visible(c, target, voxel_centers, voxel_normals)
        for c in candidates
    ])

    covered = np.zeros(len(voxel_centers), dtype=bool)
    selected = []

    for _ in range(n_budget):
        gains = np.sum(vis_masks & ~covered, axis=1)
        best = int(np.argmax(gains))
        selected.append(candidates[best])
        covered |= vis_masks[best]

    return np.array(selected)

def make_random(candidates, n_budget, rng):
    """Random-N: 무작위 선택"""
    indices = rng.choice(len(candidates), n_budget, replace=False)
    return candidates[indices]

# ─────────────────────────────────────────────────────────────────────────────
# 커버리지 평가
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_coverage(path, target, voxel_centers, voxel_normals):
    """경로의 누적 커버리지"""
    covered = np.zeros(len(voxel_centers), dtype=bool)

    for cam_pos in path:
        visible = get_visible(cam_pos, target, voxel_centers, voxel_normals)
        covered |= visible

    return np.mean(covered)

def path_distance(path):
    """경로의 총 이동 거리"""
    dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.sum(dists)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("mast3r 복원물체 경로 비교 실험")
    print("=" * 80)

    # 점군 로드
    print("\n[1/4] mast3r 점군 로드...")
    try:
        points = load_ply(PLY_FILE, sample_n=50000)
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"자동 대체: 임시 테스트 물체 생성")
        points = np.random.randn(10000, 3) * 2

    # 법선 추정
    print("[2/4] 법선 추정...")
    normals = estimate_normals(points, k=15)

    results = []

    for voxel_size in VOXEL_SIZES:
        print(f"\n[3/4] Voxelize (size={voxel_size}m)...")
        vox_info = voxelize_and_pca(points, normals, voxel_size)

        voxel_centers = vox_info['voxel_centers']
        voxel_normals = vox_info['voxel_normals']
        target = vox_info['pca_mean']

        print(f"  → {vox_info['n_surface_voxels']} surface voxels")
        print(f"  → flatness={vox_info['flatness']:.3f}")

        # 후보 생성
        candidates, _ = make_candidates(voxel_centers)
        print(f"  → {len(candidates)} candidate viewpoints")

        rng = np.random.RandomState(42)

        for n_budget in N_BUDGET:
            print(f"\n  N={n_budget}:")

            # 3가지 경로 생성
            orbit_path = make_orbit(target, n_budget)
            greedy_path = make_greedy(candidates, target, voxel_centers, voxel_normals, n_budget)
            random_path = make_random(candidates, n_budget, rng)

            # 커버리지 평가
            orbit_cov = evaluate_coverage(orbit_path, target, voxel_centers, voxel_normals)
            greedy_cov = evaluate_coverage(greedy_path, target, voxel_centers, voxel_normals)
            random_cov = evaluate_coverage(random_path, target, voxel_centers, voxel_normals)

            orbit_dist = path_distance(orbit_path)
            greedy_dist = path_distance(greedy_path)
            random_dist = path_distance(random_path)

            print(f"    Orbit:  {orbit_cov:.1%} coverage, {orbit_dist:.1f}m distance")
            print(f"    Greedy: {greedy_cov:.1%} coverage, {greedy_dist:.1f}m distance")
            print(f"    Random: {random_cov:.1%} coverage, {random_dist:.1f}m distance")

            results.append({
                'voxel_size': voxel_size,
                'flatness': vox_info['flatness'],
                'n_budget': n_budget,
                'orbit_coverage': float(orbit_cov),
                'greedy_coverage': float(greedy_cov),
                'random_coverage': float(random_cov),
                'orbit_distance': float(orbit_dist),
                'greedy_distance': float(greedy_dist),
                'random_distance': float(random_dist),
            })

    # 결과 저장
    print("\n[4/4] 결과 저장...")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 시각화
    voxel_sizes_unique = sorted(set(r['voxel_size'] for r in results))
    fig, axes = plt.subplots(1, len(voxel_sizes_unique), figsize=(14, 5))

    if len(voxel_sizes_unique) == 1:
        axes = [axes]

    for ax, vs in zip(axes, voxel_sizes_unique):
        subset = [r for r in results if r['voxel_size'] == vs]
        ns = [r['n_budget'] for r in subset]

        orbit_covs = [r['orbit_coverage'] for r in subset]
        greedy_covs = [r['greedy_coverage'] for r in subset]
        random_covs = [r['random_coverage'] for r in subset]

        ax.plot(ns, orbit_covs, 'o-', label='Orbit', linewidth=2, markersize=8)
        ax.plot(ns, greedy_covs, 's-', label='Greedy', linewidth=2, markersize=8)
        ax.plot(ns, random_covs, '^-', label='Random', linewidth=2, markersize=8)

        ax.set_xlabel('Path Length (N)', fontsize=11)
        ax.set_ylabel('Coverage', fontsize=11)
        ax.set_title(f'mast3r (voxel={vs}m)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(OUT_DIR / "coverage_comparison.png", dpi=150)
    print(f"  → {OUT_DIR / 'coverage_comparison.png'}")

    print("\n" + "=" * 80)
    print(f"✅ 완료! 결과: {OUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
