#!/usr/bin/env python3
"""
STEP 2: 비교 실험 (유일한 실행 단계)

6물체 × 2해상도 × 3경로(Orbit/Greedy/Random) × 5N(4,6,8,12,16)
= 180 실험

각 실험: 커버리지, 경로 길이, 온라인 NBV 필요 스텝 측정
"""
import numpy as np
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 기존 pbnbv_paper.py의 함수 재사용
import pbnbv_paper as P

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

OBJECTS = ['flat-simple', 'flat-complex', 'box', 'sphere-small', 'sphere-large', 'occluded-chair']
VOXEL_SIZES = [0.15, 0.075]
N_BUDGET = [4, 6, 8, 12, 16]  # 시점 예산
TARGET = np.array([-33.67, -50.83, 0.18])
RADIUS = 7.0
START = np.array([-27.06, -50.51, -6.13])

# ─────────────────────────────────────────────────────────────────────────────
# 경로 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_orbit(n_points, alt=4.0):
    """Orbit-N: 균등 원형 (N등분)"""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    positions = []

    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = TARGET[0] + RADIUS * np.cos(az_rad)
        y = TARGET[1] + RADIUS * np.sin(az_rad)
        z = TARGET[2] - alt
        positions.append([x, y, z])

    return np.array(positions)

def make_greedy(object_data, n_budget):
    """Greedy-N: oracle로 상위 N장 선택"""
    points, normals = object_data
    origin = points.min(0) - 0.15  # 기본 voxel size
    voxel_size = 0.15

    # 모든 후보 생성
    candidates = []
    for alt in np.arange(1.0, 9.5, 1.0):
        z = TARGET[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2*math.pi*rad / 1.5))
            for i in range(n_az):
                th = 2*math.pi*i/n_az
                candidates.append([
                    TARGET[0] + rad*math.cos(th),
                    TARGET[1] + rad*math.sin(th),
                    z
                ])

    candidates = np.array(candidates)

    # Oracle: 각 후보의 gain 계산
    gains = []
    for cand in candidates:
        # 간단히: frustum 내 점 수 (정확한 구현은 pbnbv_paper.observed_by 사용)
        dist = np.linalg.norm(cand - points, axis=1)
        in_range = (dist >= 4.0) & (dist <= 13.0)
        cam_dir = TARGET - cand
        cam_dir /= np.linalg.norm(cam_dir) + 1e-9
        to = points - cand
        angle = np.arccos(np.clip(np.dot(to, cam_dir) / (np.linalg.norm(to, axis=1) + 1e-9), -1, 1))
        in_fov = angle <= np.radians(89.9/2)
        front = np.dot(normals * (cand - points), np.ones((len(points), 3))) > 0

        gain = np.sum(in_range & in_fov & front)
        gains.append(gain)

    gains = np.array(gains)
    top_indices = np.argsort(gains)[-n_budget:]
    selected = candidates[top_indices]

    return selected

def make_random(n_budget):
    """Random-N: 무작위 선택"""
    candidates = []
    for alt in np.arange(1.0, 9.5, 1.0):
        z = TARGET[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2*math.pi*rad / 1.5))
            for i in range(n_az):
                th = 2*math.pi*i/n_az
                candidates.append([
                    TARGET[0] + rad*math.cos(th),
                    TARGET[1] + rad*math.sin(th),
                    z
                ])

    candidates = np.array(candidates)
    selected_indices = np.random.choice(len(candidates), n_budget, replace=False)
    return candidates[selected_indices]

# ─────────────────────────────────────────────────────────────────────────────
# 평가 (커버리지)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_path(positions, points, normals, voxel_size):
    """경로의 커버리지 측정"""
    origin = points.min(0) - voxel_size
    observed_count = 0

    for pos in positions:
        dist = np.linalg.norm(pos - points, axis=1)
        in_range = (dist >= 4.0) & (dist <= 13.0)

        cam_dir = (points - pos) / (np.linalg.norm(points - pos, axis=1, keepdims=True) + 1e-9)
        to = points - pos
        cos_angle = np.sum(cam_dir * to / np.linalg.norm(to, axis=1, keepdims=True), axis=1)
        in_fov = cos_angle >= np.cos(np.radians(89.9/2))

        front = np.dot(normals * (pos - points), np.ones(normals.shape[1])) > 0

        visible = in_range & in_fov & front
        observed_count = max(observed_count, np.sum(visible))

    # Voxel 기반 coverage (간단 근사)
    voxel_indices = np.floor((points - origin) / voxel_size).astype(int)
    n_surface_voxels = len(np.unique(voxel_indices, axis=0))

    coverage = min(observed_count / max(1, n_surface_voxels), 1.0)

    return coverage

def pathlen(positions):
    """경로 길이"""
    d = 0
    cur = START
    for p in positions:
        d += np.linalg.norm(p - cur)
        cur = p
    return d

# ─────────────────────────────────────────────────────────────────────────────
# 병렬 실험
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(obj_name, voxel_size, n_budget, path_type):
    """단일 실험 (Orbit/Greedy/Random 중 하나)"""
    try:
        # 물체 데이터 로드
        obj_file = Path(f'data/test_objects/{obj_name}_{voxel_size:.3f}.npz')
        if not obj_file.exists():
            return None

        data = np.load(obj_file)
        points = data['points']
        normals = data['normals']

        # 경로 생성
        if path_type == 'orbit':
            positions = make_orbit(n_budget)
        elif path_type == 'greedy':
            positions = make_greedy((points, normals), n_budget)
        elif path_type == 'random':
            positions = make_random(n_budget)
        else:
            return None

        # 평가
        coverage = evaluate_path(positions, points, normals, voxel_size)
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
        print(f"Error {obj_name} {voxel_size} {n_budget} {path_type}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = Path('results/step2_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 2: 비교 실험 시작")
    print("=" * 80)
    print(f"물체 {len(OBJECTS)} × 해상도 {len(VOXEL_SIZES)} × 예산 {len(N_BUDGET)} × 경로 3")
    print(f"= {len(OBJECTS) * len(VOXEL_SIZES) * len(N_BUDGET) * 3} 실험")
    print()

    all_results = []

    # 병렬 실행
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {}

        for obj_name in OBJECTS:
            for voxel_size in VOXEL_SIZES:
                for n_budget in N_BUDGET:
                    for path_type in ['orbit', 'greedy', 'random']:
                        future = executor.submit(
                            run_single_experiment,
                            obj_name, voxel_size, n_budget, path_type
                        )
                        futures[future] = (obj_name, voxel_size, n_budget, path_type)

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_results.append(result)
                completed += 1
                if completed % 20 == 0:
                    print(f"진행: {completed} / {len(futures)}")

    print(f"\n완료: {len(all_results)} / {len(OBJECTS) * len(VOXEL_SIZES) * len(N_BUDGET) * 3}")

    # 결과 저장
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ 결과 저장: {out_dir / 'results.json'}")

    # 요약 테이블 (N=8, 0.15m)
    print("\n" + "=" * 80)
    print("요약 (0.15m, N=8)")
    print("=" * 80)
    print(f"{'물체':<20} {'Orbit':<15} {'Greedy':<15} {'Random':<15}")
    print("-" * 80)

    for obj_name in OBJECTS:
        orbit_cov = next((r['coverage'] for r in all_results
                         if r['object'] == obj_name and r['voxel_size'] == 0.15
                         and r['n_budget'] == 8 and r['path_type'] == 'orbit'), None)
        greedy_cov = next((r['coverage'] for r in all_results
                          if r['object'] == obj_name and r['voxel_size'] == 0.15
                          and r['n_budget'] == 8 and r['path_type'] == 'greedy'), None)
        random_cov = next((r['coverage'] for r in all_results
                          if r['object'] == obj_name and r['voxel_size'] == 0.15
                          and r['n_budget'] == 8 and r['path_type'] == 'random'), None)

        if orbit_cov and greedy_cov and random_cov:
            print(f"{obj_name:<20} {orbit_cov*100:>13.1f}% {greedy_cov*100:>13.1f}% {random_cov*100:>13.1f}%")
