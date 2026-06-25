#!/usr/bin/env python3
"""
간단한 테스트: real_test 물체로 여러 λ 값의 Greedy 실행
방위각 간격 측정 및 비교
"""
import numpy as np
import math
from pathlib import Path

# real_test 데이터
data = np.load('real_test/real_test_pts_normals.npz')
voxel_centers = data['points']
voxel_normals = data['normals']

TARGET = np.mean(voxel_centers, axis=0)
FOV_DEG = 89.9
MIN_DIST = 4.0
MAX_DIST = 13.0

print("=" * 80)
print("real_test 물체 - 여러 λ 값 Greedy 비교")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 함수
# ─────────────────────────────────────────────────────────────────────────────

def make_candidates():
    cands = []
    for alt in np.arange(1.0, 9.5, 1.0):
        z = TARGET[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2 * math.pi * rad / 1.5))
            for i in range(n_az):
                th = 2 * math.pi * i / n_az
                cands.append([TARGET[0] + rad * math.cos(th),
                             TARGET[1] + rad * math.sin(th), z])
    return np.array(cands)

def get_visible(cam_pos):
    cam_dir = TARGET - cam_pos
    cam_dir /= np.linalg.norm(cam_dir) + 1e-9
    to = voxel_centers - cam_pos
    dist = np.linalg.norm(to, axis=1)
    in_range = (dist >= MIN_DIST) & (dist <= MAX_DIST)
    in_fov = (to * cam_dir).sum(1) / (dist + 1e-9) >= math.cos(math.radians(FOV_DEG / 2))
    front = (voxel_normals * (cam_pos - voxel_centers)).sum(1) > 0
    return in_range & in_fov & front

def azimuth_spread_gain(cam_pos, selected):
    """방위각 분산 이득"""
    if len(selected) == 0:
        return 45.0  # 첫 번째는 이상적 값

    rel_sel = selected - TARGET
    az_sel = np.degrees(np.arctan2(rel_sel[:, 1], rel_sel[:, 0]))

    rel_new = cam_pos - TARGET
    az_new = np.degrees(np.arctan2(rel_new[1], rel_new[0]))

    all_az = np.sort(np.concatenate([az_sel, [az_new]]))

    gaps = []
    for i in range(len(all_az)):
        next_i = (i + 1) % len(all_az)
        if next_i == 0:
            gap = (all_az[0] + 360) - all_az[i]
        else:
            gap = all_az[next_i] - all_az[i]
        gaps.append(gap)

    # 최소 간격이 커질수록 이득 (최대 45° 추구)
    return np.max(gaps)

def greedy_lambda(candidates, lam_weight, n_budget=8):
    """λ-weighted Greedy"""
    covered = np.zeros(len(voxel_centers), dtype=bool)
    selected = []

    for step in range(n_budget):
        best_idx = -1
        best_score = -1

        for i, cand in enumerate(candidates):
            cov_gain = np.sum(get_visible(cand) & ~covered)
            az_gain = azimuth_spread_gain(cand, np.array(selected) if selected else np.array([]))

            score = lam_weight * cov_gain + (1 - lam_weight) * az_gain

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append(candidates[best_idx])
            covered |= get_visible(candidates[best_idx])

    return np.array(selected)

def measure_gaps(positions):
    """방위각 간격 측정"""
    rel = positions - TARGET
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
# 실행
# ─────────────────────────────────────────────────────────────────────────────

candidates = make_candidates()

lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
results = {}

print(f"\n【 Multi-objective: λ * coverage + (1-λ) * azimuth 】\n")
print(f"{'λ':<6} {'Max Gap':<12} {'Min Gap':<12} {'Mean Gap':<12} {'Verdict':<30}")
print("-" * 72)

for lam in lambdas:
    path = greedy_lambda(candidates, lam)
    sorted_az, gaps = measure_gaps(path)

    max_gap = np.max(gaps)
    min_gap = np.min(gaps)
    mean_gap = np.mean(gaps)

    if max_gap < 70:
        verdict = "✅ 균등"
    elif max_gap > 100:
        verdict = "❌ 집중"
    else:
        verdict = "⚠️  경계"

    results[f"λ={lam}"] = {
        "max_gap": max_gap,
        "min_gap": min_gap,
        "mean_gap": mean_gap,
        "verdict": verdict
    }

    print(f"{lam:<6.2f} {max_gap:>10.1f}°  {min_gap:>10.1f}°  {mean_gap:>10.1f}°  {verdict:<30}")

print("\n" + "=" * 80)

EOF
