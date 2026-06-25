#!/usr/bin/env python3
"""
STEP 2 v2: 수정된 비교 실험

누적 관측 기반 정확한 voxel 커버리지 계산
"""
import numpy as np
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

TARGET = np.array([-33.67, -50.83, 0.18])
RADIUS = 7.0
START = np.array([-27.06, -50.51, -6.13])

OBJECTS = ['flat-simple', 'flat-complex', 'box', 'sphere-small', 'sphere-large', 'occluded-chair']
VOXEL_SIZES = [0.15, 0.075]
N_BUDGET = [4, 6, 8, 12, 16]

# ─────────────────────────────────────────────────────────────────────────────
# 경로 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_orbit(n_points, alt=4.0):
    """Orbit-N"""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    positions = []
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = TARGET[0] + RADIUS * np.cos(az_rad)
        y = TARGET[1] + RADIUS * np.sin(az_rad)
        z = TARGET[2] - alt
        positions.append([x, y, z])
    return np.array(positions)

def make_greedy(voxel_centers, normals, n_budget):
    """Greedy-N: oracle로 top N 선택"""
    # 모든 후보 생성
    candidates = []
    for alt in np.arange(1.0, 9.5, 1.0):
        z = TARGET[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2*math.pi*rad / 1.5))
            for i in range(n_az):
                th = 2*math.pi*i/n_az
                candidates.append([TARGET[0]+rad*math.cos(th), TARGET[1]+rad*math.sin(th), z])

    candidates = np.array(candidates)

    # 각 후보의 가시 voxel 수 계산
    gains = []
    for cand in candidates:
        dist = np.linalg.norm(cand - voxel_centers, axis=1)
        in_range = (dist >= 4.0) & (dist <= 13.0)

        cam_dir = TARGET - cand
        cam_dir /= np.linalg.norm(cam_dir) + 1e-9

        to = voxel_centers - cand
        to_norm = np.linalg.norm(to, axis=1, keepdims=True) + 1e-9
        cos_angle = np.sum(cam_dir * (to / to_norm), axis=1)
        in_fov = cos_angle >= np.cos(np.radians(89.9/2))

        front = np.dot(normals * (cand - voxel_centers), np.ones(normals.shape[1])) > 0

        visible = in_range & in_fov & front
        gains.append(np.sum(visible))

    gains = np.array(gains)
    top_indices = np.argsort(gains)[-n_budget:]
    return candidates[top_indices]

def make_random(n_budget):
    """Random-N"""
    candidates = []
    for alt in np.arange(1.0, 9.5, 1.0):
        z = TARGET[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2*math.pi*rad / 1.5))
            for i in range(n_az):
                th = 2*math.pi*i/n_az
                candidates.append([TARGET[0]+rad*math.cos(th), TARGET[1]+rad*math.sin(th), z])

    candidates = np.array(candidates)
    selected_idx = np.random.choice(len(candidates), min(n_budget, len(candidates)), replace=False)
    return candidates[selected_idx]

# ─────────────────────────────────────────────────────────────────────────────
# 평가 (누적 voxel 관측)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_path(positions, voxel_centers, normals):
    """경로의 누적 커버리지 계산"""
    observed = np.zeros(len(voxel_centers), dtype=bool)
    coverages = []

    for pos in positions:
        # 이 위치에서 보이는 voxel
        dist = np.linalg.norm(pos - voxel_centers, axis=1)
        in_range = (dist >= 4.0) & (dist <= 13.0)

        cam_dir = TARGET - pos
        cam_dir /= np.linalg.norm(cam_dir) + 1e-9

        to = voxel_centers - pos
        to_norm = np.linalg.norm(to, axis=1, keepdims=True) + 1e-9
        cos_angle = np.sum(cam_dir * (to / to_norm), axis=1)
        in_fov = cos_angle >= np.cos(np.radians(89.9/2))

        front = np.dot(normals * (pos - voxel_centers), np.ones(normals.shape[1])) > 0

        visible = in_range & in_fov & front
        observed |= visible

        cov = np.mean(observed)
        coverages.append(cov)

    return np.mean(coverages) if coverages else 0.0

def pathlen(positions):
    """경로 길이"""
    d = 0.0
    cur = START
    for p in positions:
        d += np.linalg.norm(p - cur)
        cur = p
    return d

# ─────────────────────────────────────────────────────────────────────────────
# 실험
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(obj_name, voxel_size, n_budget, path_type):
    """단일 실험"""
    try:
        obj_file = Path(f'data/test_objects/{obj_name}_{voxel_size:.3f}.npz')
        if not obj_file.exists():
            return None

        data = np.load(obj_file)
        voxel_centers = data['voxel_centers']
        normals = data['normals']
        points = data['points']

        # 경로 생성
        if path_type == 'orbit':
            positions = make_orbit(n_budget)
        elif path_type == 'greedy':
            positions = make_greedy(voxel_centers, normals, n_budget)
        elif path_type == 'random':
            positions = make_random(n_budget)
        else:
            return None

        # 평가
        coverage = evaluate_path(positions, voxel_centers, normals)
        distance = pathlen(positions)

        return {
            'object': obj_name,
            'voxel_size': voxel_size,
            'n_budget': n_budget,
            'path_type': path_type,
            'coverage': float(coverage),
            'distance': float(distance),
            'n_waypoints': len(positions)
        }

    except Exception as e:
        print(f"Error {obj_name} {voxel_size} {n_budget} {path_type}: {str(e)[:50]}")
        return None

if __name__ == '__main__':
    out_dir = Path('results/step2_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 2 v2: 누적 voxel 관측 기반 실험")
    print("=" * 80)

    all_results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {}

        for obj_name in OBJECTS:
            for voxel_size in VOXEL_SIZES:
                for n_budget in N_BUDGET:
                    for path_type in ['orbit', 'greedy', 'random']:
                        future = executor.submit(
                            run_experiment,
                            obj_name, voxel_size, n_budget, path_type
                        )
                        futures[future] = (obj_name, voxel_size, n_budget, path_type)

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)
                completed += 1
                if completed % 20 == 0:
                    print(f"진행: {completed} / {len(futures)}")

    print(f"\n완료: {len(all_results)} 결과")

    # 저장
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ {out_dir / 'results.json'}")

    # 요약
    print("\n" + "=" * 80)
    print("요약 (0.15m, N=8)")
    print("=" * 80)
    print(f"{'물체':<20} {'Orbit':>10} {'Greedy':>10} {'Random':>10}")
    print("-" * 80)

    for obj in OBJECTS:
        orbit = next((r['coverage'] for r in all_results if r['object']==obj and r['voxel_size']==0.15 and r['n_budget']==8 and r['path_type']=='orbit'), None)
        greedy = next((r['coverage'] for r in all_results if r['object']==obj and r['voxel_size']==0.15 and r['n_budget']==8 and r['path_type']=='greedy'), None)
        random_cov = next((r['coverage'] for r in all_results if r['object']==obj and r['voxel_size']==0.15 and r['n_budget']==8 and r['path_type']=='random'), None)

        o_str = f"{orbit*100:.1f}%" if orbit else "-"
        g_str = f"{greedy*100:.1f}%" if greedy else "-"
        r_str = f"{random_cov*100:.1f}%" if random_cov else "-"

        print(f"{obj:<20} {o_str:>10} {g_str:>10} {r_str:>10}")

