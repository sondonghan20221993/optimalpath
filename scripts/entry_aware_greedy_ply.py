#!/usr/bin/env python3
"""
Entry-aware Greedy path on drone_real PLY data
New project 5 알고리즘 적용: coverage + novelty + distance_penalty + entry_awareness
45도 카메라 제약 포함
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── 파라미터 ──────────────────────────────────────────────────────────
PLY_PATH   = "/mnt/c/Users/sdh97/Desktop/학교/캔위성/2차 발표자료/drone_real_sfm/3m_1/pointcloud.ply"
N_STEPS    = 8
N_AZ       = 24
RADII      = [2.0, 3.0, 4.0]       # 반경 (m)
MAX_TILT   = 45.0                   # 카메라 최대 틸트각 (°)
FOV_DEG    = 86.0
VOXEL      = 0.1                    # voxel 크기 (m)

# 점수 가중치 (New project 5 기반)
W_COVERAGE  = 0.70
W_NOVELTY   = 0.25
W_DIST_PEN  = 0.08
W_ENTRY_D   = 0.35
W_ENTRY_H   = 0.20
W_ENTRY_ANG = 0.25

# ── PLY 로드 (샘플링) ─────────────────────────────────────────────────
print("Loading PLY...")
pts = []
with open(PLY_PATH, 'r') as f:
    for line in f:
        if line.strip() == 'end_header':
            break
    for i, line in enumerate(f):
        if i % 5 == 0:   # 1/5 샘플링
            v = line.strip().split()
            if len(v) >= 3:
                pts.append([float(v[0]), float(v[1]), float(v[2])])

pts = np.array(pts)
print(f"Loaded {len(pts)} pts (sampled)")

# ── 물체 중심 추정 (밀도 높은 영역) ───────────────────────────────────
# Z 중앙값 기준 위쪽 절반에서 XY 중심 추출
z_med = np.median(pts[:,2])
obj_pts = pts[pts[:,2] > z_med]
cx = np.median(obj_pts[:,0])
cy = np.median(obj_pts[:,1])
cz = np.median(obj_pts[:,2])
TARGET = np.array([cx, cy, cz])
print(f"Target center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")

# ── Voxel 맵 ──────────────────────────────────────────────────────────
origin = pts.min(0) - VOXEL
vox_idx = np.floor((pts - origin) / VOXEL).astype(np.int32)
vox_set = set(map(tuple, vox_idx))
total_voxels = len(vox_set)
print(f"Total voxels: {total_voxels}")

def get_visible_voxels(pos, observed_set):
    """간단한 가시성: 카메라 방향 FOV 콘 안의 voxel"""
    dir_to_target = TARGET - pos
    dir_norm = dir_to_target / (np.linalg.norm(dir_to_target) + 1e-8)
    fov_rad = math.radians(FOV_DEG / 2)
    visible = set()
    for v in vox_set:
        vp = np.array(v) * VOXEL + origin
        dv = vp - pos
        dist = np.linalg.norm(dv)
        if dist < 0.1:
            continue
        dv_norm = dv / dist
        angle = math.acos(np.clip(np.dot(dir_norm, dv_norm), -1, 1))
        if angle < fov_rad:
            visible.add(v)
    return visible

# ── 후보 생성 (45° 제약) ──────────────────────────────────────────────
candidates = []
for rad in RADII:
    max_alt = rad * math.tan(math.radians(MAX_TILT))
    alt = min(max_alt, 2.5)         # 최대 2.5m above target
    z = cz + alt
    for i in range(N_AZ):
        th = 2 * math.pi * i / N_AZ
        x = cx + rad * math.cos(th)
        y = cy + rad * math.sin(th)
        candidates.append(np.array([x, y, z]))

candidates = np.array(candidates)
print(f"Candidates: {len(candidates)}")

# ── Entry-aware Greedy ────────────────────────────────────────────────
print("\nRunning Entry-aware Greedy...")
observed   = set()
path       = []
scores_log = []

# 시작점: target 정면 (x+ 방향, 반경 3m)
start_idx = N_AZ  # 두 번째 반경(3m)의 0° 방향
current   = candidates[start_idx].copy()
path.append(current)

# 시작점의 가시 voxel
vis = get_visible_voxels(current, observed)
observed |= vis

for step in range(1, N_STEPS):
    best_score = -1e9
    best_idx   = 0
    best_vis   = set()

    prev = path[-1]
    prev_dir = prev[:2] - TARGET[:2]
    prev_dir_norm = prev_dir / (np.linalg.norm(prev_dir) + 1e-8)

    for ci, cand in enumerate(candidates):
        # 이미 방문한 위치 건너뜀 (가까운 곳)
        if any(np.linalg.norm(cand - p) < 0.5 for p in path):
            continue

        vis_new = get_visible_voxels(cand, observed)
        new_vox = vis_new - observed

        # Coverage score
        cov = len(new_vox) / max(total_voxels, 1)

        # Novelty: 이전 방문 위치들과의 최소 각도 (멀수록 좋음)
        cand_dir = cand[:2] - TARGET[:2]
        cand_dir_norm = cand_dir / (np.linalg.norm(cand_dir) + 1e-8)
        min_ang_diff = math.pi
        for p in path:
            pd = p[:2] - TARGET[:2]
            pd_n = pd / (np.linalg.norm(pd) + 1e-8)
            ang = math.acos(np.clip(np.dot(cand_dir_norm, pd_n), -1, 1))
            min_ang_diff = min(min_ang_diff, ang)
        novelty = min_ang_diff / math.pi

        # Distance penalty
        dist = np.linalg.norm(cand - prev)
        max_dist = max(RADII) * 2 * math.pi
        dist_pen = dist / max_dist

        # Entry awareness: heading change
        cand_dir_from_prev = (cand - prev)[:2]
        cand_dir_from_prev_n = cand_dir_from_prev / (np.linalg.norm(cand_dir_from_prev) + 1e-8)
        heading_change = 1.0 - abs(np.dot(prev_dir_norm, cand_dir_from_prev_n))

        # Entry height penalty
        height_diff = abs(cand[2] - prev[2])

        score = (W_COVERAGE  * cov
               + W_NOVELTY   * novelty
               - W_DIST_PEN  * dist_pen
               - W_ENTRY_ANG * heading_change
               - W_ENTRY_H   * (height_diff / 3.0))

        if score > best_score:
            best_score = score
            best_idx   = ci
            best_vis   = vis_new

    path.append(candidates[best_idx])
    observed |= best_vis
    scores_log.append(best_score)
    cov_pct = len(observed) / total_voxels * 100
    print(f"  Step {step+1}: pos=({candidates[best_idx][0]:.1f},{candidates[best_idx][1]:.1f},{candidates[best_idx][2]:.1f})"
          f"  score={best_score:.3f}  coverage={cov_pct:.1f}%")

path = np.array(path)

# ── 시각화 ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor('#111')

# 1. Top view (XY)
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_facecolor('#1a1a2e')
# point cloud (희박하게)
stride = max(1, len(pts)//5000)
ax1.scatter(pts[::stride,0], pts[::stride,1], s=0.3, c='#aaaaaa', alpha=0.3)
# candidates
ax1.scatter(candidates[:,0], candidates[:,1], s=15, c='#444466', alpha=0.5)
# path
ax1.plot(path[:,0], path[:,1], '-', color='#ff5a36', linewidth=2, alpha=0.8)
ax1.scatter(path[:,0], path[:,1], s=120, c=[plt.cm.plasma(i/N_STEPS) for i in range(len(path))],
            edgecolors='white', linewidth=1.2, zorder=5)
for i, p in enumerate(path):
    ax1.annotate(str(i+1), (p[0], p[1]), fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')
ax1.scatter([cx], [cy], s=200, c='#00d26a', marker='*', zorder=6)
ax1.set_xlabel('X (m)', color='white'); ax1.set_ylabel('Y (m)', color='white')
ax1.set_title('Top View (XY)', color='white', fontsize=12)
ax1.tick_params(colors='white'); ax1.set_aspect('equal')
for spine in ax1.spines.values(): spine.set_color('#444')

# 2. Side view (XZ)
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_facecolor('#1a1a2e')
ax2.scatter(pts[::stride,0], pts[::stride,2], s=0.3, c='#aaaaaa', alpha=0.3)
ax2.plot(path[:,0], path[:,2], '-', color='#ff5a36', linewidth=2, alpha=0.8)
ax2.scatter(path[:,0], path[:,2], s=120, c=[plt.cm.plasma(i/N_STEPS) for i in range(len(path))],
            edgecolors='white', linewidth=1.2, zorder=5)
for i, p in enumerate(path):
    ax2.annotate(str(i+1), (p[0], p[2]), fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')
# 45도 제약선
for rad in RADII:
    max_alt = rad * math.tan(math.radians(MAX_TILT))
    ax2.axhline(cz + max_alt, color='#ffaa00', alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_xlabel('X (m)', color='white'); ax2.set_ylabel('Z (m)', color='white')
ax2.set_title('Side View (XZ) | dashed=45° limit', color='white', fontsize=12)
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_color('#444')

# 3. 3D scatter
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.set_facecolor('#111')
stride3 = max(1, len(pts)//3000)
ax3.scatter(pts[::stride3,0], pts[::stride3,1], pts[::stride3,2],
            s=0.3, c='#888888', alpha=0.2)
ax3.scatter(path[:,0], path[:,1], path[:,2],
            s=150, c=[plt.cm.plasma(i/N_STEPS) for i in range(len(path))],
            edgecolors='white', linewidth=1, zorder=5)
for i, p in enumerate(path):
    ax3.text(p[0], p[1], p[2], str(i+1), fontsize=8, color='white',
             ha='center', va='center', fontweight='bold')
ax3.scatter([cx], [cy], [cz], s=300, c='#00d26a', marker='*')
ax3.set_xlabel('X', color='white'); ax3.set_ylabel('Y', color='white'); ax3.set_zlabel('Z', color='white')
ax3.set_title('3D View', color='white', fontsize=12)
ax3.tick_params(colors='white')
ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False

plt.suptitle(f'Entry-aware Greedy Path  |  drone_real_3m  |  N={N_STEPS}  |  45° constraint',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()

out = "/mnt/c/Users/sdh97/Documents/GitHub/optimalpath/results/entry_aware_greedy_ply.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#111')
print(f"\n✓ Saved: {out}")
plt.close()

# ── 결과 출력 ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("WAYPOINTS")
print("="*60)
print(f"{'Step':<6} {'X':>7} {'Y':>7} {'Z':>7}  {'Az':>8}  {'Tilt':>6}")
print("-"*60)
for i, p in enumerate(path):
    az = math.degrees(math.atan2(p[1]-cy, p[0]-cx)) % 360
    horiz = math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)
    tilt = math.degrees(math.atan2(p[2]-cz, horiz))
    print(f"{i+1:<6} {p[0]:>7.2f} {p[1]:>7.2f} {p[2]:>7.2f}  {az:>7.1f}°  {tilt:>5.1f}°")

final_cov = len(observed)/total_voxels*100
print(f"\nFinal coverage: {final_cov:.1f}%")
