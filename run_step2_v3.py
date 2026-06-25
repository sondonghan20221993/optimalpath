#!/usr/bin/env python3
"""
STEP 2 v3: 최종 수정 버전

수정 사항:
  1. voxel 레벨 법선 사용 (generate_test_objects.py에서 저장)
  2. front check 공식: (SURF_NRM*(cam_pos-SURF_CEN)).sum(1) > 0  (pbnbv_paper.py와 동일)
  3. cumulative 커버리지: observed |= visible → 최종 np.mean(observed)
  4. Greedy: 진짜 set-cover greedy (순차 선택)
"""
import numpy as np
import json
import math
from pathlib import Path

TARGET = np.array([-33.67, -50.83, 0.18])
RADIUS = 7.0
FOV_DEG = 89.9
MIN_DIST = 4.0
MAX_DIST = 13.0
START = np.array([-27.06, -50.51, -6.13])

OBJECTS = ['flat-simple', 'flat-complex', 'box', 'sphere-small', 'sphere-large', 'occluded-chair']
VOXEL_SIZES = [0.15, 0.075]
N_BUDGET = [4, 6, 8, 12, 16]

# ─────────────────────────────────────────────────────────────────────────────
# 관측 모델 (pbnbv_paper.py와 동일 공식)
# ─────────────────────────────────────────────────────────────────────────────

def get_visible(cam_pos, voxel_centers, voxel_normals):
    """cam_pos에서 보이는 voxel 불리언 마스크"""
    cam_dir = TARGET - cam_pos
    cam_dir /= np.linalg.norm(cam_dir) + 1e-9

    to = voxel_centers - cam_pos
    dist = np.linalg.norm(to, axis=1)

    in_range = (dist >= MIN_DIST) & (dist <= MAX_DIST)
    in_fov = (to * cam_dir).sum(1) / (dist + 1e-9) >= math.cos(math.radians(FOV_DEG / 2))
    front = (voxel_normals * (cam_pos - voxel_centers)).sum(1) > 0  # pbnbv_paper.py L95

    return in_range & in_fov & front

# ─────────────────────────────────────────────────────────────────────────────
# 후보 생성 (pbnbv_paper.py와 동일)
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

CANDIDATES = make_candidates()

# ─────────────────────────────────────────────────────────────────────────────
# 경로 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_orbit(n_points, alt=4.0):
    """Orbit-N: 균등 원형"""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    positions = []
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = TARGET[0] + RADIUS * np.cos(az_rad)
        y = TARGET[1] + RADIUS * np.sin(az_rad)
        z = TARGET[2] - alt
        positions.append([x, y, z])
    return np.array(positions)

def make_greedy(voxel_centers, voxel_normals, n_budget):
    """Greedy-N: oracle set-cover greedy"""
    # 모든 후보에 대해 visibility mask 선계산
    vis_masks = np.array([
        get_visible(c, voxel_centers, voxel_normals)
        for c in CANDIDATES
    ])  # shape: (n_candidates, n_voxels)

    covered = np.zeros(len(voxel_centers), dtype=bool)
    selected = []

    for _ in range(n_budget):
        # 각 후보의 marginal gain
        gains = np.sum(vis_masks & ~covered, axis=1)
        best = int(np.argmax(gains))
        selected.append(CANDIDATES[best])
        covered |= vis_masks[best]

    return np.array(selected)

def make_random(n_budget, rng):
    """Random-N: 무작위 선택"""
    idx = rng.choice(len(CANDIDATES), min(n_budget, len(CANDIDATES)), replace=False)
    return CANDIDATES[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 평가: 최종 누적 커버리지
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_path(positions, voxel_centers, voxel_normals):
    """N 시점 방문 후 최종 커버리지"""
    observed = np.zeros(len(voxel_centers), dtype=bool)
    for pos in positions:
        observed |= get_visible(pos, voxel_centers, voxel_normals)
    return float(np.mean(observed))

def pathlen(positions):
    d = 0.0
    cur = START
    for p in positions:
        d += np.linalg.norm(np.array(p) - cur)
        cur = np.array(p)
    return d

# ─────────────────────────────────────────────────────────────────────────────
# 단일 실험
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(obj_name, voxel_size, n_budget, path_type, rng):
    obj_file = Path(f'data/test_objects/{obj_name}_{voxel_size:.3f}.npz')
    if not obj_file.exists():
        raise FileNotFoundError(f"{obj_file} 없음")

    data = np.load(obj_file)
    voxel_centers = data['voxel_centers']
    voxel_normals = data['voxel_normals']

    if path_type == 'orbit':
        positions = make_orbit(n_budget)
    elif path_type == 'greedy':
        positions = make_greedy(voxel_centers, voxel_normals, n_budget)
    elif path_type == 'random':
        positions = make_random(n_budget, rng)
    else:
        raise ValueError(f"unknown path_type: {path_type}")

    coverage = evaluate_path(positions, voxel_centers, voxel_normals)
    distance = pathlen(positions)

    return {
        'object': obj_name,
        'voxel_size': voxel_size,
        'n_budget': n_budget,
        'path_type': path_type,
        'coverage': coverage,
        'distance': distance,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = Path('results/step2_v3')
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    print("=" * 80)
    print("STEP 2 v3: 비교 실험")
    print(f"후보 시점: {len(CANDIDATES)}개")
    print("=" * 80)

    all_results = []
    total = len(OBJECTS) * len(VOXEL_SIZES) * len(N_BUDGET) * 3
    done = 0

    for obj_name in OBJECTS:
        for voxel_size in VOXEL_SIZES:
            for n_budget in N_BUDGET:
                for path_type in ['orbit', 'greedy', 'random']:
                    try:
                        r = run_experiment(obj_name, voxel_size, n_budget, path_type, rng)
                        all_results.append(r)
                        done += 1
                        if done % 10 == 0 or done == total:
                            print(f"  [{done}/{total}] {obj_name} {voxel_size} N={n_budget} {path_type}: {r['coverage']*100:.1f}%")
                    except Exception as e:
                        print(f"  ERROR {obj_name} {voxel_size} N={n_budget} {path_type}: {e}")

    print(f"\n완료: {len(all_results)} / {total}")

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ {out_dir / 'results.json'}")

    # 요약 테이블 (0.15m, N=8)
    print("\n" + "=" * 80)
    print("요약 (0.15m, N=8)")
    print("=" * 80)
    print(f"{'물체':<22} {'Orbit':>8} {'Greedy':>8} {'Random':>8}")
    print("-" * 50)

    for obj in OBJECTS:
        def get(pt):
            return next((r['coverage'] for r in all_results
                        if r['object'] == obj and r['voxel_size'] == 0.15
                        and r['n_budget'] == 8 and r['path_type'] == pt), None)
        o, g, rd = get('orbit'), get('greedy'), get('random')
        o_s = f"{o*100:.1f}%" if o is not None else "-"
        g_s = f"{g*100:.1f}%" if g is not None else "-"
        r_s = f"{rd*100:.1f}%" if rd is not None else "-"
        print(f"{obj:<22} {o_s:>8} {g_s:>8} {r_s:>8}")

    # std 비교 (N=8, 0.15m)
    print("\n" + "=" * 80)
    print("강건성 분석 (std, 0.15m)")
    print("=" * 80)
    print(f"{'N':>4} {'std(Orbit)':>12} {'std(Greedy)':>12} {'std(Random)':>12}")
    print("-" * 44)

    for n in N_BUDGET:
        def get_covs(pt):
            return [r['coverage'] for r in all_results
                    if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == pt]
        o_covs = get_covs('orbit')
        g_covs = get_covs('greedy')
        r_covs = get_covs('random')
        if o_covs and g_covs and r_covs:
            print(f"{n:>4} {np.std(o_covs)*100:>11.2f}% {np.std(g_covs)*100:>11.2f}% {np.std(r_covs)*100:>11.2f}%")
