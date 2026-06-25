#!/usr/bin/env python3
"""
STEP 1: Orbit 포화점 분석 및 형상 복잡도 관계 분석
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

with open('results/step2_v3/results.json') as f:
    all_results = json.load(f)

# PCA flatness 로드
with open('data/test_objects/metadata.json') as f:
    metadata = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Orbit 커버리지 곡선 추출
# ─────────────────────────────────────────────────────────────────────────────

OBJECTS = ['flat-simple', 'flat-complex', 'box', 'sphere-small', 'sphere-large', 'occluded-chair']
VOXEL_SIZES = [0.15, 0.075]
N_BUDGET = [4, 6, 8, 12, 16]

orbit_curves = {}

for obj in OBJECTS:
    for vs in VOXEL_SIZES:
        key = f"{obj}_{vs:.3f}"
        coverages = []
        for n in N_BUDGET:
            r = next((r for r in all_results
                     if r['object'] == obj and r['voxel_size'] == vs
                     and r['n_budget'] == n and r['path_type'] == 'orbit'), None)
            if r:
                coverages.append(r['coverage'])
            else:
                coverages.append(None)
        orbit_curves[key] = coverages

# ─────────────────────────────────────────────────────────────────────────────
# 포화점 탐지: 증가분 < 1%p (0.01)
# ─────────────────────────────────────────────────────────────────────────────

saturation_points = {}
SATURATION_THRESHOLD = 0.01

print("=" * 80)
print("STEP 1: Orbit 포화점 분석")
print("=" * 80)
print(f"포화 기준: 증가분 < {SATURATION_THRESHOLD*100:.1f}%p\n")

for obj in OBJECTS:
    print(f"{obj}:")
    for vs in VOXEL_SIZES:
        key = f"{obj}_{vs:.3f}"
        covs = orbit_curves[key]

        # 유효한 커버리지만
        valid = [c for c in covs if c is not None]

        print(f"  {vs:.3f}m: {[f'{c*100:.1f}%' for c in valid]}")

        # 포화점 탐지
        sat_n = None
        for i in range(1, len(valid)):
            delta = valid[i] - valid[i-1]
            if delta < SATURATION_THRESHOLD:
                sat_n = N_BUDGET[i]
                print(f"    → 포화점: N={sat_n} (증가분 {delta*100:.2f}%p)")
                break

        if sat_n is None:
            print(f"    → 포화 미도달 (검증 필요)")

        saturation_points[key] = sat_n

# ─────────────────────────────────────────────────────────────────────────────
# 의자 N=16 검증
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("의자(occluded-chair) N=16 검증")
print("=" * 80)

chair_015 = orbit_curves['occluded-chair_0.150']
chair_0075 = orbit_curves['occluded-chair_0.075']

print(f"0.15m: {[f'{c*100:.1f}%' for c in chair_015]}")
if len(chair_015) >= 5:
    delta = chair_015[4] - chair_015[3]  # N=16 증가분
    print(f"  N=12→16 증가분: {delta*100:.2f}%p", end="")
    if delta < 0.01:
        print(" (포화 도달 ✓)")
    else:
        print(f" → N=20, 24 확장 필요")

print(f"0.075m: {[f'{c*100:.1f}%' for c in chair_0075]}")
if len(chair_0075) >= 5:
    delta = chair_0075[4] - chair_0075[3]
    print(f"  N=12→16 증가분: {delta*100:.2f}%p", end="")
    if delta < 0.01:
        print(" (포화 도달 ✓)")
    else:
        print(f" → N=20, 24 확장 필요")

# ─────────────────────────────────────────────────────────────────────────────
# 형상 복잡도 vs 포화점 산점도
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("형상 복잡도 vs 포화점")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, vs in enumerate(VOXEL_SIZES):
    ax = axes[ax_idx]

    flatness_vals = []
    saturation_vals = []
    labels = []

    for obj in OBJECTS:
        key = f"{obj}_{vs:.3f}"
        # 메타데이터 키 형식: "0.15m" 또는 "0.075m"
        vs_key = "0.15m" if vs == 0.15 else "0.075m"
        f_val = metadata['objects'][obj][vs_key]['flatness']
        s_val = saturation_points.get(key)

        if s_val is not None:
            flatness_vals.append(f_val)
            saturation_vals.append(s_val)
            labels.append(obj)

    ax.scatter(flatness_vals, saturation_vals, s=200, alpha=0.6)
    for i, label in enumerate(labels):
        ax.annotate(label, (flatness_vals[i], saturation_vals[i]),
                   fontsize=9, ha='right')

    ax.set_xlabel('PCA Flatness (작을수록 평탄)', fontsize=10)
    ax.set_ylabel('Saturation N', fontsize=10)
    ax.set_title(f'Voxel Size {vs}m', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 20])

plt.tight_layout()
plt.savefig('results/step2_v3/saturation_analysis.png', dpi=150)
print(f"✓ results/step2_v3/saturation_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# Orbit 커버리지 곡선 시각화
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, obj in enumerate(OBJECTS):
    ax = axes[idx]

    for vs, style in [(0.15, 'o-'), (0.075, 's--')]:
        key = f"{obj}_{vs:.3f}"
        covs = orbit_curves[key]
        ns_valid = [N_BUDGET[i] for i in range(len(covs)) if covs[i] is not None]
        covs_valid = [c*100 for c in covs if c is not None]

        ax.plot(ns_valid, covs_valid, style, label=f'{vs}m', markersize=8)

    ax.set_xlabel('N (시점 수)', fontsize=10)
    ax.set_ylabel('Coverage (%)', fontsize=10)
    ax.set_title(obj, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('results/step2_v3/orbit_curves.png', dpi=150)
print(f"✓ results/step2_v3/orbit_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# std 비교 (강건성)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("주력 지표: std(커버리지) — 강건성 분석")
print("=" * 80)
print(f"{'N':>4} {'std(Orbit)':>12} {'std(Greedy)':>12} {'Orbit<Greedy?'}")
print("-" * 50)

for n in N_BUDGET:
    orbit_covs = [r['coverage'] for r in all_results
                  if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == 'orbit']
    greedy_covs = [r['coverage'] for r in all_results
                   if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == 'greedy']

    if orbit_covs and greedy_covs:
        std_o = np.std(orbit_covs)
        std_g = np.std(greedy_covs)
        sign = "✓" if std_o < std_g else "✗"
        print(f"{n:>4} {std_o*100:>11.2f}% {std_g*100:>11.2f}%  {sign}")

print("\n결론 판정: 모든 N에서 std(Orbit) < std(Greedy)인가?")
all_better = all(
    np.std([r['coverage'] for r in all_results
            if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == 'orbit'])
    < np.std([r['coverage'] for r in all_results
              if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == 'greedy'])
    for n in N_BUDGET
)

if all_better:
    print("✅ YES → Orbit이 더 강건함")
else:
    print("❌ NO → Greedy가 더 강건함")
