#!/usr/bin/env python3
"""
미지의 물체 복원 시뮬레이션

가정: real_test를 "미지의 물체"로 취급 (형상/크기 모름)
→ 초기 경로(blind choice) 설정
→ 그 경로로 부분 관측
→ frontier 노출 후 온라인 NBV로 보강
→ 최종 커버리지, 스텝 수, 경로 길이 비교

객관적 기준: 최종 커버리지 도달 efficiency (스텝 수, 거리)
"""
import json, math
import numpy as np
from pathlib import Path
import pbnbv_paper as P

TARGET = P.TARGET

def make_orbit(alt, n_points, radius=7.0):
    """고도 alt, 시점 n개 원형."""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    pos = []
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = TARGET[0] + radius * np.cos(az_rad)
        y = TARGET[1] + radius * np.sin(az_rad)
        z = TARGET[2] - alt
        pos.append([x, y, z])
    return np.array(pos), azimuths

def make_spiral(alt_start, alt_end, n_points, radius=7.0):
    """나선형: 고도가 점진적으로 상승."""
    alts = np.linspace(alt_start, alt_end, n_points)
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    pos = []
    for az_deg, alt in zip(azimuths, alts):
        az_rad = np.radians(az_deg)
        x = TARGET[0] + radius * np.cos(az_rad)
        y = TARGET[1] + radius * np.sin(az_rad)
        z = TARGET[2] - alt
        pos.append([x, y, z])
    return np.array(pos), azimuths

def pathlen(pts, start=None):
    """경로 길이."""
    if start is None:
        start = np.array([-27.06, -50.51, -6.13])
    d = 0
    cur = start
    for p in pts:
        d += np.linalg.norm(p - cur)
        cur = p
    return d

def run_scenario(name, initial_path, max_online_steps=10):
    """
    초기 경로 실행 → frontier 노출 → 온라인 NBV 보강
    """
    start = np.array([-27.06, -50.51, -6.13])
    observed = np.zeros(P.N_SURF, dtype=bool)
    path = []
    total_steps = 0

    # Phase 1: 초기 경로 (blind, 반드시 따름)
    print(f"\n▶ {name}")
    print(f"  초기 경로: {len(initial_path)}시점")

    for step, cam_pos in enumerate(initial_path):
        newobs = P.observed_by(cam_pos)
        gained = (~observed[newobs]).sum()
        observed[newobs] = True
        cov = observed.sum() / P.N_SURF

        alt = TARGET[2] - cam_pos[2]
        az = math.degrees(math.atan2(cam_pos[1]-TARGET[1], cam_pos[0]-TARGET[0]))

        path.append({
            'step': step+1,
            'phase': 'initial',
            'pos': cam_pos.tolist(),
            'alt': round(alt, 1),
            'azimuth': round(az, 0),
            'gained': int(gained),
            'coverage': round(cov, 4)
        })
        total_steps += 1

        if gained == 0:
            print(f"    step{step+1}: alt={alt:.1f}m az={az:.0f}° +{gained:3d}voxel cov={cov*100:5.1f}% (정보 포화)")
        else:
            print(f"    step{step+1}: alt={alt:.1f}m az={az:.0f}° +{gained:3d}voxel cov={cov*100:5.1f}%")

    init_cov = observed.sum() / P.N_SURF
    init_gain = sum(p['gained'] for p in path)
    print(f"  ▼ 초기 후: cov={init_cov*100:.1f}%, 누적 +{init_gain} voxel, 거리={pathlen(initial_path, start):.1f}m")

    # Phase 2: 온라인 NBV (frontier 노출된 부분 추가)
    print(f"\n  온라인 NBV:")
    cands = P.make_candidates()
    used = set(range(len(initial_path)))  # 초기 경로 위치 재사용 금지 (대략적)

    for online_step in range(max_online_steps):
        fro_idx = P.compute_frontier(observed)
        if len(fro_idx) == 0:
            print(f"    frontier 없음 (완전 커버)")
            break

        occ_pts = P.SURF_CEN[observed]
        occ_nrm = P.SURF_NRM[observed]
        fro_pts = P.SURF_CEN[fro_idx]
        fro_nrm = P.SURF_NRM[fro_idx]

        occ_ell = P.fit_ellipsoids(occ_pts, occ_nrm)
        fro_ell = P.fit_ellipsoids(fro_pts, fro_nrm)

        # greedy: gain 최대 후보
        best_i = -1
        best_gain = -1
        for i, c in enumerate(cands):
            newobs_i = P.observed_by(c)
            gain_i = (~observed[newobs_i]).sum()
            if gain_i > best_gain:
                best_gain = gain_i
                best_i = i

        if best_gain <= 0:
            print(f"    online step{online_step+1}: 추가 정보 없음")
            break

        chosen = cands[best_i]
        newobs = P.observed_by(chosen)
        observed[newobs] = True
        cov = observed.sum() / P.N_SURF

        alt = TARGET[2] - chosen[2]
        az = math.degrees(math.atan2(chosen[1]-TARGET[1], chosen[0]-TARGET[0]))

        path.append({
            'step': len(path)+1,
            'phase': 'online_nbv',
            'pos': chosen.tolist(),
            'alt': round(alt, 1),
            'azimuth': round(az, 0),
            'gained': int(best_gain),
            'coverage': round(cov, 4)
        })
        total_steps += 1

        print(f"    online step{online_step+1}: alt={alt:.1f}m az={az:.0f}° +{best_gain:3d}voxel cov={cov*100:5.1f}%")

    final_cov = observed.sum() / P.N_SURF
    final_gain = sum(p['gained'] for p in path)
    init_dist = pathlen(initial_path, start)
    total_pos = np.array([p['pos'] for p in path])
    total_dist = pathlen(total_pos, start)

    return {
        'name': name,
        'initial_n_waypoints': len(initial_path),
        'initial_coverage': round(init_cov, 4),
        'initial_distance_m': round(init_dist, 1),
        'initial_gain': int(init_gain),
        'total_steps': int(total_steps),
        'final_coverage': round(final_cov, 4),
        'total_distance_m': round(total_dist, 1),
        'total_gain': int(final_gain),
        'path': path
    }

# ─────────────────────────────────────────────────────────────────────────────
# 여러 초기 경로 시나리오

scenarios = {}

# 1. 원형 1바퀴 (저고도)
orb_1m_8, _ = make_orbit(1.0, 8)
scenarios['1m 원형 8시점'] = orb_1m_8

# 2. 원형 1바퀴 (중고도)
orb_4m_8, _ = make_orbit(4.0, 8)
scenarios['4m 원형 8시점'] = orb_4m_8

# 3. 원형 1바퀴 (고고도)
orb_7m_8, _ = make_orbit(7.0, 8)
scenarios['7m 원형 8시점'] = orb_7m_8

# 4. 원형 1바퀴 (저고도, 시점 적음)
orb_2m_4, _ = make_orbit(2.0, 4)
scenarios['2m 원형 4시점'] = orb_2m_4

# 5. 나선형 (저→고 상승)
spi_1to5, _ = make_spiral(1.0, 5.0, 8)
scenarios['나선형 1~5m 8시점'] = spi_1to5

# 6. 나선형 (고→저 하강)
spi_5to1, _ = make_spiral(5.0, 1.0, 8)
scenarios['나선형 5~1m 8시점'] = spi_5to1

# 7. 원형 2바퀴 (저고도 + 고고도)
orb_2m_4b, _ = make_orbit(2.0, 4)
orb_6m_4b, _ = make_orbit(6.0, 4)
orb_2orbit = np.vstack([orb_2m_4b, orb_6m_4b])
scenarios['2바퀴 (2m 4시점 + 6m 4시점)'] = orb_2orbit

# 8. 최적 고고도 (이전 greedy 결과)
orb_4m_3, _ = make_orbit(4.0, 3)
scenarios['4m 원형 3시점 (최적?)'] = orb_4m_3

# ─────────────────────────────────────────────────────────────────────────────
# 실행

results = {}
for name, initial_path in scenarios.items():
    result = run_scenario(name, initial_path, max_online_steps=15)
    results[name] = result

# ─────────────────────────────────────────────────────────────────────────────
# 비교표

print("\n" + "="*100)
print(f"{'경로':30} {'초기 WP':>8} {'초기 cov':>10} {'초기 dist':>10} {'최종 step':>10} {'최종 cov':>10} {'총 dist':>10}")
print("-"*100)

for name in sorted(results.keys(), key=lambda n: -results[n]['final_coverage']):
    r = results[name]
    print(f"{name:30} {r['initial_n_waypoints']:8d} "
          f"{r['initial_coverage']*100:9.1f}% {r['initial_distance_m']:9.1f}m "
          f"{r['total_steps']:10d} {r['final_coverage']*100:9.1f}% {r['total_distance_m']:9.1f}m")

print("="*100)

# JSON 저장
out = Path('results/unknown_object_simulation.json')
json.dump(results, open(out, 'w'), indent=2)
print(f"\n✓ {out}")
print("\n📊 해석:")
print("  - '초기 cov': blind 경로만으로 몇 % 커버됐나")
print("  - '최종 step': frontier 노출 후 온라인 NBV로 100% 도달까지 총 몇 스텝")
print("  - '총 dist': 초기 경로 + 온라인 경로의 전체 거리")
print("  - 목표: 초기 거리 짧으면서 최종 coverage 빠르게 도달")
