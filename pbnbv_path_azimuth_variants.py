#!/usr/bin/env python3
"""
pbnbv_path.py 변형: 방위각 항 추가하여 여러 버전 실행
- λ = 0.0 (기존, 커버리지만)
- λ = 0.25, 0.5, 1.0, 2.0 (Multi-objective)
- Hierarchical (coverage >= 95% 이후 azimuth 우선)

각 버전의 방위각 간격을 측정하고 비교
"""

import json
import math
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# 경로 설정
BASE_DIR = Path(r"/mnt/c/Users/sdh97/Desktop/blue_1_fhd_sfm(pp팍스 mast3r결과)")
PLY_PATH = BASE_DIR / "pointcloud.ply"
POSES_PATH = BASE_DIR / "poses.npy"
FOCALS_PATH = BASE_DIR / "focals.npy"
OUT_DIR = Path("results/pbnbv_azimuth_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 알고리즘 파라미터
N_PCD_SAMPLE = 40_000
MAX_DIST = 8.0
UNDEROBS_THRESH = 3
N_CANDIDATES = 150
N_SELECT = 10
LOOKAHEAD = 2
ORBIT_ALTITUDES = [0.5, 1.0, 1.5]
ORBIT_RADII = [2.0, 3.5, 5.0]
IMG_W, IMG_H = 1920, 1080

print("=" * 80)
print("pbnbv_path - 방위각 항 추가 실험")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드 (간단히)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(PLY_PATH))
    pts = np.asarray(pcd.points)
    if len(pts) > N_PCD_SAMPLE:
        idx = np.random.choice(len(pts), N_PCD_SAMPLE, replace=False)
        pts = pts[idx]
    print(f"\n✓ 점군 로드: {len(pts)} points")
except:
    print("⚠ open3d 로드 실패. 테스트 데이터 사용.")
    pts = np.random.randn(1000, 3) * 5

poses = np.load(POSES_PATH)
focals = np.load(FOCALS_PATH)

cam_positions = poses[:, :3, 3]
cam_rotations = poses[:, :3, :3]
cam_forwards = np.array([R @ np.array([0., 0., 1.]) for R in cam_rotations])

look_pts = cam_positions + cam_forwards * 3.0
target = look_pts.mean(axis=0)

print(f"✓ 카메라 수: {len(poses)}")
print(f"✓ 타겟: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 후보 생성 (기존과 동일)
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates(target, altitudes, radii, n_total):
    """기존과 동일한 후보 생성"""
    cam_z_mean = cam_positions[:, 2].mean()
    candidates = []
    n_per = max(1, n_total // (len(altitudes) * len(radii)))
    for alt in altitudes:
        z = cam_z_mean - alt
        for rad in radii:
            for i in range(n_per):
                theta = 2 * math.pi * i / n_per
                x = target[0] + rad * math.cos(theta)
                y = target[1] + rad * math.sin(theta)
                candidates.append([x, y, z])
    return np.array(candidates)

candidates = generate_candidates(target, ORBIT_ALTITUDES, ORBIT_RADII, N_CANDIDATES)
print(f"✓ 후보 생성: {len(candidates)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 점수 함수 (여러 버전)
# ─────────────────────────────────────────────────────────────────────────────

def point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist):
    """기존 함수"""
    v = pts - cam_pos
    dist = np.linalg.norm(v, axis=1)
    valid_dist = (dist > 0.1) & (dist < max_dist)
    v_norm = v / (dist[:, None] + 1e-8)
    cos_half = math.cos(math.radians(fov_deg / 2.0))
    dot = v_norm @ cam_dir
    in_cone = dot > cos_half
    return valid_dist & in_cone

fov_h = 2 * np.degrees(np.arctan(IMG_W / 2 / focals.mean()))

def coverage_gain(cam_pos, already_visible):
    """커버리지 이득"""
    to_target = target - cam_pos
    cam_dir = to_target / (np.linalg.norm(to_target) + 1e-8)
    visible = point_in_frustum(cam_pos, cam_dir, fov_h, pts, MAX_DIST)
    new_visible = visible & ~already_visible
    return np.sum(new_visible)

def azimuth_spread_gain(cam_pos, selected_positions):
    """방위각 분산 이득

    현재까지 선택된 위치들과 새 후보의 방위각을 고려
    새 후보를 추가했을 때 최소 인접 간격이 얼마나 늘어나는가
    """
    if len(selected_positions) == 0:
        return 1.0  # 첫 번째는 최대 가치

    # 선택된 위치들의 방위각
    rel_selected = selected_positions - target
    az_selected = np.degrees(np.arctan2(rel_selected[:, 1], rel_selected[:, 0]))

    # 새 후보의 방위각
    rel_new = cam_pos - target
    az_new = np.degrees(np.arctan2(rel_new[1], rel_new[0]))

    # 새 후보 추가 후 정렬
    all_az = np.sort(np.concatenate([az_selected, [az_new]]))

    # 인접 간격 계산
    gaps = []
    for i in range(len(all_az)):
        next_i = (i + 1) % len(all_az)
        if next_i == 0:
            gap = (all_az[0] + 360) - all_az[i]
        else:
            gap = all_az[next_i] - all_az[i]
        gaps.append(gap)

    min_gap = np.min(gaps)  # 새 후보로 인해 가장 좁혀진 간격
    # 이득: 간격이 클수록 높은 점수 (100° 목표)
    spread_score = min(min_gap, 100.0)

    return spread_score

# ─────────────────────────────────────────────────────────────────────────────
# 4. Greedy 실행 (여러 λ 버전)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_path(lambda_weight, method_name):
    """Greedy 경로 생성

    lambda_weight: 방위각 항의 가중치 (0 = 커버리지만, ∞ = 방위각만)
    """
    selected = []
    already_visible = np.zeros(len(pts), dtype=bool)

    for step in range(N_SELECT):
        best_idx = -1
        best_score = -1

        for i, cand in enumerate(candidates):
            cov_gain = coverage_gain(cand, already_visible)

            if lambda_weight == float('inf'):
                # Hierarchical: 이미 95% 이상 커버면 방위각만
                current_coverage = np.mean(already_visible)
                if current_coverage >= 0.95 or cov_gain == 0:
                    score = azimuth_spread_gain(cand, np.array(selected))
                else:
                    score = cov_gain
            else:
                # Multi-objective: λ * coverage + (1-normalized) * azimuth
                az_gain = azimuth_spread_gain(cand, np.array(selected) if selected else np.array([]))
                score = lambda_weight * cov_gain + (1 - lambda_weight) * az_gain

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append(candidates[best_idx])
            to_target = target - candidates[best_idx]
            cam_dir = to_target / (np.linalg.norm(to_target) + 1e-8)
            visible = point_in_frustum(candidates[best_idx], cam_dir, fov_h, pts, MAX_DIST)
            already_visible |= visible

    selected = np.array(selected)
    coverage = np.mean(already_visible)

    return selected, coverage

# ─────────────────────────────────────────────────────────────────────────────
# 5. 방위각 간격 측정
# ─────────────────────────────────────────────────────────────────────────────

def measure_azimuth_gaps(positions):
    """방위각 간격 측정"""
    rel = positions - target
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    sorted_az = np.sort(az)

    gaps = []
    for i in range(len(sorted_az)):
        next_i = (i + 1) % len(sorted_az)
        if next_i == 0:
            gap = (sorted_az[0] + 360) - sorted_az[i]
        else:
            gap = sorted_az[next_i] - sorted_az[i]
        gaps.append(gap)

    return sorted_az, gaps

# ─────────────────────────────────────────────────────────────────────────────
# 6. 실행 및 결과 비교
# ─────────────────────────────────────────────────────────────────────────────

results = {}

# Multi-objective variants
lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"\n【 Multi-objective variants (λ * coverage + (1-λ) * azimuth) 】")
for lam in lambdas:
    print(f"\n  λ = {lam:.2f}...")
    path, cov = greedy_path(lam, f"multi_lambda_{lam}")
    sorted_az, gaps = measure_azimuth_gaps(path)

    results[f"Multi-λ{lam}"] = {
        "lambda": lam,
        "method": "multi-objective",
        "coverage": cov,
        "positions": path,
        "azimuths": sorted_az.tolist(),
        "gaps": gaps,
        "max_gap": np.max(gaps),
        "min_gap": np.min(gaps),
        "mean_gap": np.mean(gaps),
        "std_gap": np.std(gaps),
    }
    print(f"    Coverage: {cov:.1%}")
    print(f"    Max gap: {np.max(gaps):.1f}° | Min gap: {np.min(gaps):.1f}° | Mean gap: {np.mean(gaps):.1f}°")

# Hierarchical variant
print(f"\n【 Hierarchical variant (coverage >= 95% then azimuth) 】")
path, cov = greedy_path(float('inf'), "hierarchical")
sorted_az, gaps = measure_azimuth_gaps(path)

results["Hierarchical"] = {
    "lambda": float('inf'),
    "method": "hierarchical",
    "coverage": cov,
    "positions": path,
    "azimuths": sorted_az.tolist(),
    "gaps": gaps,
    "max_gap": np.max(gaps),
    "min_gap": np.min(gaps),
    "mean_gap": np.mean(gaps),
    "std_gap": np.std(gaps),
}
print(f"  Coverage: {cov:.1%}")
print(f"  Max gap: {np.max(gaps):.1f}° | Min gap: {np.min(gaps):.1f}° | Mean gap: {np.mean(gaps):.1f}°")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 결과 비교 & 저장
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("비교 요약")
print("=" * 80)
print(f"\n{'Method':<20} {'Coverage':<12} {'Max Gap':<12} {'Mean Gap':<12} {'Std Gap':<12}")
print("-" * 68)

for name, data in sorted(results.items()):
    print(f"{name:<20} {data['coverage']:>10.1%}  {data['max_gap']:>10.1f}°  {data['mean_gap']:>10.1f}°  {data['std_gap']:>10.1f}°")

# JSON 저장 (위치 제외 - 너무 크니까)
results_json = {}
for name, data in results.items():
    results_json[name] = {
        k: v for k, v in data.items() if k != "positions"
    }

with open(OUT_DIR / "azimuth_variants.json", "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\n✓ 결과 저장: {OUT_DIR / 'azimuth_variants.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. 판정
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("판정: 원형인가 vs 집중인가?")
print("=" * 80)

for name, data in sorted(results.items()):
    max_gap = data["max_gap"]
    if max_gap < 70:
        verdict = "✅ 균등 분포 (8개 고루 퍼짐)"
    elif max_gap > 100:
        verdict = "❌ 비균등/집중 (한쪽 비어있음)"
    else:
        verdict = "⚠️  경계 사례"
    print(f"{name:<20} Max gap {max_gap:>6.1f}°  →  {verdict}")

print("\n" + "=" * 80)

EOF
