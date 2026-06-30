#!/usr/bin/env python3
"""
순수 coverage greedy: F(투영면적+가중치)를 제거.
매 스텝마다 '새로 관측하는 voxel 수'만으로 다음 후보 선택.
각 고도별 실행 → 시점 수, 최종 커버리지, 방위 분포를 보여줌.
"""
import json, math
import numpy as np
from pathlib import Path
import pbnbv_paper as P

TARGET = P.TARGET

# 고도 고정하고 후보 생성
def make_candidates_fixed_alt(alt_m):
    """고도를 alt_m으로 고정, 반경·방위는 변함."""
    z = TARGET[2] - alt_m
    cands = []
    for rad in [4.5, 6.0, 7.5, 9.0]:
        n_az = max(6, int(2*math.pi*rad / 1.5))
        for i in range(n_az):
            th = 2*math.pi*i/n_az
            cands.append([TARGET[0]+rad*math.cos(th),
                         TARGET[1]+rad*math.sin(th), z])
    return np.array(cands)

def run_coverage_greedy(alt_m, max_steps=12):
    """고도 alt_m에서 순수 coverage greedy 실행."""
    cands = make_candidates_fixed_alt(alt_m)
    observed = np.zeros(P.N_SURF, dtype=bool)
    path = []
    used = np.zeros(len(cands), dtype=bool)

    for step in range(max_steps):
        # 각 후보가 새로 관측할 voxel 수
        gains = np.array([
            (~observed[P.observed_by(c)]).sum() if not used[i] else -999
            for i, c in enumerate(cands)
        ])
        best = int(np.argmax(gains))
        if gains[best] < 0:
            break

        used[best] = True
        chosen = cands[best]
        newobs = P.observed_by(chosen)
        gained = (~observed[newobs]).sum()
        observed[newobs] = True
        cov = observed.sum() / P.N_SURF

        alt = TARGET[2] - chosen[2]
        az = math.degrees(math.atan2(chosen[1]-TARGET[1], chosen[0]-TARGET[0]))

        path.append({
            'step': step+1,
            'pos': chosen.tolist(),
            'alt': round(alt, 2),
            'azimuth': round(az, 1),
            'gained': int(gained),
            'coverage': round(cov, 4)
        })

        if gained == 0:
            break

    return path

# 각 고도 1~7m 실행
results = {}
for alt in range(1, 8):
    print(f"\n=== 고도 {alt}m ===")
    path = run_coverage_greedy(alt)
    results[alt] = path

    if path:
        print(f"시점 수: {len(path)}, 최종커버: {path[-1]['coverage']*100:.1f}%")
        print(f"방위: {[p['azimuth'] for p in path]}")
        print(f"gained: {[p['gained'] for p in path]}")

    # JSON 저장
    out = Path(f'results/pbnbv_paper/alt_{alt}m_greedy.json')
    json.dump({'altitude': alt, 'path': path}, open(out, 'w'), indent=2)

# 비교표
print("\n" + "="*70)
print(f"{'고도':>4} {'시점수':>5} {'최종커버':>8} {'방위들':>35}")
print("-"*70)
for alt in range(1, 8):
    path = results[alt]
    if path:
        n = len(path)
        cov = path[-1]['coverage']*100
        azs = [f"{p['azimuth']:.0f}" for p in path]
        print(f"{alt}m  {n:5d}  {cov:7.1f}%   {', '.join(azs)}")
    else:
        print(f"{alt}m     0      0.0%     (실패)")

print("\n✓ 결과: results/pbnbv_paper/alt_*m_greedy.json")
