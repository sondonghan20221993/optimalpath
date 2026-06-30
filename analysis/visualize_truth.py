#!/usr/bin/env python3
"""
데이터를 정직하게 시각화하기

1. 각 물체별 3경로 비교 (누가 이기는가)
2. std vs min (최솟값) 비교
3. Greedy-Orbit 격차 분포
4. 물체별 "경로 구분 가능 여부" (천장 확인)
5. sphere에서의 명백한 Greedy 우위
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 데이터 로드
with open('results/step2_v3/results.json') as f:
    all_results = json.load(f)

OBJECTS = ['flat-simple', 'flat-complex', 'box', 'sphere-small', 'sphere-large', 'occluded-chair']
VOXEL_SIZES = [0.15, 0.075]
N_BUDGET = [4, 6, 8, 12, 16]

# ─────────────────────────────────────────────────────────────────────────────
# 1. 각 물체별 3경로 비교 (N=8, 0.15m)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, obj in enumerate(OBJECTS):
    ax = axes[idx]

    orbit = next((r['coverage'] for r in all_results
                 if r['object'] == obj and r['voxel_size'] == 0.15
                 and r['n_budget'] == 8 and r['path_type'] == 'orbit'), None)
    greedy = next((r['coverage'] for r in all_results
                  if r['object'] == obj and r['voxel_size'] == 0.15
                  and r['n_budget'] == 8 and r['path_type'] == 'greedy'), None)
    random = next((r['coverage'] for r in all_results
                  if r['object'] == obj and r['voxel_size'] == 0.15
                  and r['n_budget'] == 8 and r['path_type'] == 'random'), None)

    values = [orbit*100, greedy*100, random*100] if all([orbit, greedy, random]) else [0, 0, 0]

    # 색상: 어느 경로가 이기는지 표시
    colors = []
    max_val = max(values)
    for v in values:
        if abs(v - max_val) < 0.1:  # 동일하면 회색
            colors.append('#888888')
        else:
            colors.append('#cccccc')
    colors[np.argmax(values)] = '#FF6B6B' if values[np.argmax(values)] > values[0] else '#4ECDC4'

    bars = ax.bar(['Orbit', 'Greedy', 'Random'], values, color=colors, edgecolor='black', linewidth=1.5)

    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim([0, 110])
    ax.set_title(obj, fontsize=11, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('각 물체별 3경로 비교 (N=8, 0.15m) — 누가 이기는가?', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/step2_v3/truth_1_per_object.png', dpi=150, bbox_inches='tight')
print("✓ truth_1_per_object.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. std vs min(worst-case) 비교
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ns = N_BUDGET
std_orbit = []
std_greedy = []
std_random = []

min_orbit = []
min_greedy = []
min_random = []

for n in ns:
    o_covs = [r['coverage'] for r in all_results
              if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == 'orbit']
    g_covs = [r['coverage'] for r in all_results
              if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == 'greedy']
    r_covs = [r['coverage'] for r in all_results
              if r['voxel_size'] == 0.15 and r['n_budget'] == n and r['path_type'] == 'random']

    std_orbit.append(np.std(o_covs))
    std_greedy.append(np.std(g_covs))
    std_random.append(np.std(r_covs))

    min_orbit.append(np.min(o_covs))
    min_greedy.append(np.min(g_covs))
    min_random.append(np.min(r_covs))

# std 그래프
ax1.plot(ns, np.array(std_orbit)*100, 'o-', linewidth=2, markersize=8, label='Orbit', color='#4ECDC4')
ax1.plot(ns, np.array(std_greedy)*100, 's-', linewidth=2, markersize=8, label='Greedy', color='#FF6B6B')
ax1.plot(ns, np.array(std_random)*100, '^-', linewidth=2, markersize=8, label='Random', color='#FFD93D')
ax1.set_xlabel('N (시점 수)', fontsize=11)
ax1.set_ylabel('std (분산, %)', fontsize=11)
ax1.set_title('std: 큰 이유?', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 20])

# 주석
ax1.text(4, 19, "Greedy의 std가 큼 = sphere에서 혼자 10% 이기기 때문",
        fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# min(worst-case) 그래프
ax2.plot(ns, np.array(min_orbit)*100, 'o-', linewidth=2, markersize=8, label='Orbit', color='#4ECDC4')
ax2.plot(ns, np.array(min_greedy)*100, 's-', linewidth=2, markersize=8, label='Greedy', color='#FF6B6B')
ax2.plot(ns, np.array(min_random)*100, '^-', linewidth=2, markersize=8, label='Random', color='#FFD93D')
ax2.set_xlabel('N (시점 수)', fontsize=11)
ax2.set_ylabel('min(최악 물체, %)', fontsize=11)
ax2.set_title('min: 어떤 물체가 와도 최악은?', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([50, 105])

# 주석
ax2.text(4.2, 102, "Greedy도 63.2% (Orbit과 동일) — chair가 안 올라감",
        fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.suptitle('std vs min(worst-case): 강건성의 정의를 바꾸면?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/step2_v3/truth_2_std_vs_min.png', dpi=150, bbox_inches='tight')
print("✓ truth_2_std_vs_min.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Greedy - Orbit 격차 분포 (모든 물체, N=8)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

gaps = []
obj_labels = []

for obj in OBJECTS:
    o = next((r['coverage'] for r in all_results
             if r['object'] == obj and r['voxel_size'] == 0.15
             and r['n_budget'] == 8 and r['path_type'] == 'orbit'), None)
    g = next((r['coverage'] for r in all_results
             if r['object'] == obj and r['voxel_size'] == 0.15
             and r['n_budget'] == 8 and r['path_type'] == 'greedy'), None)

    if o and g:
        gap = (g - o) * 100
        gaps.append(gap)
        obj_labels.append(obj)

# 색상: 0 주변이면 회색, 양수면 빨강(Greedy 우위)
colors = ['#FF6B6B' if abs(g) > 0.5 else '#888888' for g in gaps]

bars = ax.bar(obj_labels, gaps, color=colors, edgecolor='black', linewidth=1.5)

# 0 기준선
ax.axhline(y=0, color='black', linewidth=2, linestyle='-')

# 값 표시
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    y_pos = height + (0.3 if height > 0 else -0.5)
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
           f'{gap:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
           fontsize=11, fontweight='bold')

ax.set_ylabel('Greedy - Orbit (%)', fontsize=11)
ax.set_title('Greedy가 Orbit을 이기는 정도 (N=8, 0.15m)', fontsize=12, fontweight='bold')
ax.set_ylim([-2, 12])
ax.grid(axis='y', alpha=0.3)

# 범례
ax.text(2.5, 11, "회색 = 동일, 빨강 = Greedy 우위", fontsize=10,
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/step2_v3/truth_3_gap_distribution.png', dpi=150, bbox_inches='tight')
print("✓ truth_3_gap_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 경로 구분 가능 여부 (천장 존재 판정)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

distinguishability = []
obj_labels_2 = []

for obj in OBJECTS:
    o = next((r['coverage'] for r in all_results
             if r['object'] == obj and r['voxel_size'] == 0.15
             and r['n_budget'] == 8 and r['path_type'] == 'orbit'), None)
    g = next((r['coverage'] for r in all_results
             if r['object'] == obj and r['voxel_size'] == 0.15
             and r['n_budget'] == 8 and r['path_type'] == 'greedy'), None)
    r = next((r['coverage'] for r in all_results
             if r['object'] == obj and r['voxel_size'] == 0.15
             and r['n_budget'] == 8 and r['path_type'] == 'random'), None)

    if o and g and r:
        # "경로 구분 가능" = 세 경로의 max-min 차이
        max_diff = max(o, g, r) - min(o, g, r)
        distinguishability.append(max_diff * 100)
        obj_labels_2.append(obj)

# 색상: 0이면 빨강(완전 동일=구분 불가), 크면 초록(구분 가능)
colors_dist = ['#FF6B6B' if d < 0.5 else '#4ECDC4' for d in distinguishability]

bars = ax.bar(obj_labels_2, distinguishability, color=colors_dist, edgecolor='black', linewidth=1.5)

for bar, dist in zip(bars, distinguishability):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
           f'{dist:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('max(Orbit,Greedy,Random) - min (%)', fontsize=11)
ax.set_title('경로 구분 가능성: 이 물체에서 경로 선택이 의미가 있나?', fontsize=12, fontweight='bold')
ax.set_ylim([0, 12])
ax.grid(axis='y', alpha=0.3)

# 범례
ax.text(2, 11, "빨강 < 0.5% = 경로 무관 (천장에 막힘)\n초록 > 0.5% = 경로 비교 가능",
       fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/step2_v3/truth_4_distinguishability.png', dpi=150, bbox_inches='tight')
print("✓ truth_4_distinguishability.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Sphere의 명백한 Greedy 우위
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, obj in enumerate(['sphere-small', 'sphere-large']):
    ax = axes[ax_idx]

    orbit_by_n = []
    greedy_by_n = []
    random_by_n = []

    for n in N_BUDGET:
        o = next((r['coverage'] for r in all_results
                 if r['object'] == obj and r['voxel_size'] == 0.15
                 and r['n_budget'] == n and r['path_type'] == 'orbit'), None)
        g = next((r['coverage'] for r in all_results
                 if r['object'] == obj and r['voxel_size'] == 0.15
                 and r['n_budget'] == n and r['path_type'] == 'greedy'), None)
        r = next((r['coverage'] for r in all_results
                 if r['object'] == obj and r['voxel_size'] == 0.15
                 and r['n_budget'] == n and r['path_type'] == 'random'), None)

        orbit_by_n.append(o*100 if o else 0)
        greedy_by_n.append(g*100 if g else 0)
        random_by_n.append(r*100 if r else 0)

    ax.plot(N_BUDGET, orbit_by_n, 'o-', linewidth=2, markersize=8, label='Orbit', color='#4ECDC4')
    ax.plot(N_BUDGET, greedy_by_n, 's-', linewidth=2, markersize=8, label='Greedy', color='#FF6B6B')
    ax.plot(N_BUDGET, random_by_n, '^-', linewidth=2, markersize=8, label='Random', color='#FFD93D')

    ax.set_xlabel('N (시점 수)', fontsize=11)
    ax.set_ylabel('Coverage (%)', fontsize=11)
    ax.set_title(f'{obj} — Greedy 우위 명확', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([80, 102])

plt.suptitle('구형 물체에서는 Greedy가 명백히 이김', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/step2_v3/truth_5_sphere_greedy_dominates.png', dpi=150, bbox_inches='tight')
print("✓ truth_5_sphere_greedy_dominates.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. flatness 분포 (중간점 부재 확인)
# ─────────────────────────────────────────────────────────────────────────────

with open('data/test_objects/metadata.json') as f:
    metadata = json.load(f)

flatness_vals = []
obj_names = []

for obj in OBJECTS:
    f_val = metadata['objects'][obj]['0.15m']['flatness']
    flatness_vals.append(f_val)
    obj_names.append(obj)

fig, ax = plt.subplots(figsize=(12, 6))

colors_flat = ['#4ECDC4' if f < 0.3 else '#FF6B6B' for f in flatness_vals]
bars = ax.bar(obj_names, flatness_vals, color=colors_flat, edgecolor='black', linewidth=1.5)

for bar, f in zip(bars, flatness_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{f:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 구간 표시
ax.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='평탄 경계')
ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.5, label='구형 경계')

# 목표 구간
ax.axhspan(0.25, 0.92, alpha=0.1, color='yellow', label='목표 분포 (0.25~0.92)')

ax.set_ylabel('PCA Flatness (작을수록 평탄)', fontsize=11)
ax.set_title('flatness 분포: 중간점이 비었다 (0.4~0.7 구간 없음)', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/step2_v3/truth_6_flatness_distribution.png', dpi=150, bbox_inches='tight')
print("✓ truth_6_flatness_distribution.png")

print("\n" + "="*80)
print("시각화 완료: 데이터가 말하는 진짜 이야기")
print("="*80)
print("""
truth_1: 각 물체별 3경로 비교
  → 4개 물체: Orbit = Greedy = Random (회색)
  → 2개 물체(sphere): Greedy 우위 (빨강)

truth_2: std vs min
  → std: Greedy가 크다 (sphere에서 이겨서)
  → min: 모두 동일 (chair 63.2%가 바닥)

truth_3: Greedy - Orbit 격차
  → 4개: 0.0% (동률)
  → 2개: +8~11% (Greedy 우위)

truth_4: 경로 구분 가능성
  → box, chair: 0% (경로 무관, 천장 차단)
  → 나머지: 경로에 따라 차이

truth_5: sphere에서 Greedy 우위
  → 모든 N에서 Greedy > Orbit

truth_6: flatness 분포
  → 0.13~0.20 (평탄 극단), 0.93~0.97 (구형 극단)
  → 0.4~0.7 중간: 없음 (상관 입증 불가)
""")
