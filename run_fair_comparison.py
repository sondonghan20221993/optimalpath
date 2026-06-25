#!/usr/bin/env python3
"""
공정한 비교 실험: Orbit-N vs PB-NBV-N (독립 실행)

핵심 수정:
- Greedy(PB-NBV)는 Orbit 없이 처음부터 독립적으로 실행
- 동일 N (16), 동일 물체, 동일 고도 조건
"""

import json, math
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

# ── 설정 ─────────────────────────────────────────────────────────────────────
TARGET   = np.array([-33.67, -50.83, 0.18])
VOXEL    = 0.15
FOV_DEG  = 89.9
IMG_W, IMG_H = 1920, 1080
MIN_DIST = 4.0
MAX_DIST = 13.0
N_SELECT = 16        # 비교할 waypoint 수
MAX_ELLIPSOIDS = 8
SEED = 0
np.random.seed(SEED)

OUT_DIR = Path("results/fair_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
d = np.load('real_test/real_test_pts_normals.npz')
PTS, NRM = d['points'], d['normals']
ORIGIN = PTS.min(0) - VOXEL

def pt_to_vox(p):
    return tuple(np.floor((p - ORIGIN) / VOXEL).astype(int))

surface_vox = {}
for p, n in zip(PTS, NRM):
    v = pt_to_vox(p)
    surface_vox.setdefault(v, []).append(n)
SURF = {v: np.mean(ns, axis=0) for v, ns in surface_vox.items()}
SURF_KEYS = list(SURF.keys())
SURF_CEN = np.array([(np.array(v) + 0.5) * VOXEL + ORIGIN for v in SURF_KEYS])
SURF_NRM = np.array([SURF[v] for v in SURF_KEYS])
N_SURF = len(SURF_KEYS)
print(f"GT surface voxels: {N_SURF}")

# ── 카메라 유틸 ───────────────────────────────────────────────────────────────
def make_K():
    fx = (IMG_W / 2) / math.tan(math.radians(FOV_DEG / 2))
    return np.array([[fx, 0, IMG_W/2], [0, fx, IMG_H/2], [0, 0, 1]])
K = make_K()
FX, FY, CX, CY = K[0,0], K[1,1], K[0,2], K[1,2]

def look_at_R(cam_pos, target):
    f = target - cam_pos; f /= np.linalg.norm(f) + 1e-9
    up0 = np.array([0, 0, -1.0])
    r = np.cross(up0, f)
    if np.linalg.norm(r) < 1e-6: up0 = np.array([0, 1.0, 0]); r = np.cross(up0, f)
    r /= np.linalg.norm(r) + 1e-9
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=0)

def observed_by(cam_pos):
    cam_dir = TARGET - cam_pos; cam_dir /= np.linalg.norm(cam_dir) + 1e-9
    to = SURF_CEN - cam_pos
    dist = np.linalg.norm(to, axis=1)
    cos_half = math.cos(math.radians(FOV_DEG / 2))
    in_range = (dist >= MIN_DIST) & (dist <= MAX_DIST)
    in_fov = (to * cam_dir).sum(1) / (dist + 1e-9) >= cos_half
    front = (SURF_NRM * (cam_pos - SURF_CEN)).sum(1) > 0
    return np.where(in_range & in_fov & front)[0]

NEIGH = [(dx,dy,dz) for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1)
         if not(dx==0 and dy==0 and dz==0)]
SURF_SET = set(SURF_KEYS)

def compute_frontier(observed_mask):
    obs_keys = set(SURF_KEYS[i] for i in np.where(observed_mask)[0])
    frontier_idx = []
    for i, v in enumerate(SURF_KEYS):
        if observed_mask[i]: continue
        for dxyz in NEIGH:
            nb = (v[0]+dxyz[0], v[1]+dxyz[1], v[2]+dxyz[2])
            if nb in obs_keys:
                frontier_idx.append(i); break
    return np.array(frontier_idx, dtype=int)

def fit_ellipsoids(points, normals=None):
    if len(points) < 3:
        if len(points) == 0: return []
        nn = normals.mean(0) if normals is not None and len(normals) > 0 else None
        return [(points.mean(0), np.eye(3) * (VOXEL**2), nn)]
    best = None; best_bic = np.inf
    for k in range(1, min(MAX_ELLIPSOIDS, len(points)) + 1):
        try:
            gm = GaussianMixture(k, covariance_type='full', random_state=SEED).fit(points)
            b = gm.bic(points)
            if b < best_bic: best_bic = b; best = gm
        except: pass
    if best is None:
        nn = normals.mean(0) if normals is not None and len(normals) > 0 else None
        return [(points.mean(0), np.cov(points.T) + np.eye(3)*1e-3, nn)]
    labels = best.predict(points)
    out = []
    for k, (c, cov) in enumerate(zip(best.means_, best.covariances_)):
        nn = None
        if normals is not None:
            m = labels == k
            if m.sum() > 0:
                nn = normals[m].mean(0); nn /= np.linalg.norm(nn) + 1e-9
        out.append((c, cov + np.eye(3)*1e-4, nn))
    return out

def project_area_depth(center, cov, cam_pos, R):
    c_cam = R @ (center - cam_pos)
    z = c_cam[2]
    if z <= MIN_DIST * 0.3: return 0.0, np.inf
    u = FX*c_cam[0]/z + CX; v = FY*c_cam[1]/z + CY
    if not (-IMG_W*0.5 <= u <= IMG_W*1.5 and -IMG_H*0.5 <= v <= IMG_H*1.5):
        return 0.0, np.inf
    J = np.array([[FX/z, 0, -FX*c_cam[0]/z**2],
                  [0, FY/z, -FY*c_cam[1]/z**2]])
    Sig_cam = R @ cov @ R.T
    S2 = J @ Sig_cam @ J.T
    det = np.linalg.det(S2)
    if det <= 0: return 0.0, np.inf
    return float(math.pi * math.sqrt(det)), float(z)

def ellipsoid_visible(center, cam_pos, mean_normal):
    d = np.linalg.norm(cam_pos - center)
    if d < MIN_DIST or d > MAX_DIST: return False
    if mean_normal is None: return True
    return float((mean_normal * (cam_pos - center)).sum()) > 0

def evaluate(cam_pos, occ_ell, fro_ell):
    R = look_at_R(cam_pos, TARGET)
    items = []
    for c, cov, nrm in fro_ell:
        if not ellipsoid_visible(c, cam_pos, nrm): continue
        a, z = project_area_depth(c, cov, cam_pos, R)
        if a > 0: items.append((z, a, True))
    for c, cov, nrm in occ_ell:
        if not ellipsoid_visible(c, cam_pos, nrm): continue
        a, z = project_area_depth(c, cov, cam_pos, R)
        if a > 0: items.append((z, a, False))
    if not items: return 0.0
    items.sort(key=lambda t: t[0])
    F = 0.0
    for r, (z, a, is_f) in enumerate(items):
        W = 0.5**r
        F += (a*W) if is_f else -(a*W)
    return F

def make_candidates(fixed_alt=None):
    """fixed_alt=2.0 이면 고도 2m 고정 후보만 생성"""
    cands = []
    alts = [fixed_alt] if fixed_alt is not None else np.arange(1.0, 9.5, 1.0)
    for alt in alts:
        z = TARGET[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2*math.pi*rad / 1.5))
            for i in range(n_az):
                th = 2*math.pi*i/n_az
                cands.append([TARGET[0]+rad*math.cos(th),
                              TARGET[1]+rad*math.sin(th), z])
    return np.array(cands)

def azimuth_stats(positions):
    angles = sorted([math.degrees(math.atan2(p[1]-TARGET[1], p[0]-TARGET[0])) % 360
                     for p in positions])
    gaps = [(angles[(i+1) % len(angles)] - angles[i]) % 360
            for i in range(len(angles))]
    return angles, gaps, max(gaps), np.mean(gaps), np.std(gaps)

# ═══════════════════════════════════════════════════════════════════
# 실험 1: Orbit-16 (순수 기하학)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("실험 1: Orbit-16 (N=16, 고도 고정 2m)")
print("="*60)

ORBIT_ALT = 2.0
ORBIT_RAD = 6.0
orbit_path = []
observed_orbit = np.zeros(N_SURF, dtype=bool)

for i in range(N_SELECT):
    angle = 2 * math.pi * i / N_SELECT
    pos = np.array([
        TARGET[0] + ORBIT_RAD * math.cos(angle),
        TARGET[1] + ORBIT_RAD * math.sin(angle),
        TARGET[2] - ORBIT_ALT
    ])
    obs = observed_by(pos)
    gained = (~observed_orbit[obs]).sum()
    observed_orbit[obs] = True
    cov = observed_orbit.sum() / N_SURF
    az = math.degrees(math.atan2(pos[1]-TARGET[1], pos[0]-TARGET[0])) % 360
    orbit_path.append({'step': i+1, 'pos': pos.tolist(), 'azimuth': round(az, 1),
                       'gained': int(gained), 'coverage': round(cov, 4)})
    print(f"  step{i+1:2d}: az={az:6.1f}° +{gained:3d}voxel cov={cov*100:5.1f}%")

angles_o, gaps_o, max_o, mean_o, std_o = azimuth_stats([p['pos'] for p in orbit_path])
print(f"\n  Max gap: {max_o:.1f}° | Mean: {mean_o:.1f}° | Std: {std_o:.1f}°")
print(f"  Final coverage: {orbit_path[-1]['coverage']*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# 실험 2: PB-NBV-16 독립 실행 (처음부터, Orbit 없이)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("실험 2: PB-NBV-16 독립 실행 (처음부터 시작)")
print("="*60)

cands = make_candidates(fixed_alt=ORBIT_ALT)  # 고도 2m 고정 — Orbit과 동일 조건
print(f"  후보 시점: {len(cands)}개 (고도 {ORBIT_ALT}m 고정)")

observed_nbv = np.zeros(N_SURF, dtype=bool)  # 처음부터 빈 상태
used = np.zeros(len(cands), dtype=bool)
nbv_path = []

for step in range(N_SELECT):
    fro_idx = compute_frontier(observed_nbv)
    occ_pts = SURF_CEN[observed_nbv]; occ_nrm = SURF_NRM[observed_nbv]
    fro_pts = SURF_CEN[fro_idx] if len(fro_idx) > 0 else np.empty((0,3))
    fro_nrm = SURF_NRM[fro_idx] if len(fro_idx) > 0 else np.empty((0,3))

    occ_ell = fit_ellipsoids(occ_pts, occ_nrm)
    if len(fro_pts) > 0:
        fro_ell = fit_ellipsoids(fro_pts, fro_nrm)
    else:
        fro_ell = []

    # 첫 스텝: 전체를 frontier로
    if len(occ_pts) == 0 and len(fro_pts) == 0:
        fro_ell = fit_ellipsoids(SURF_CEN, SURF_NRM)

    scores = np.array([(-1e18 if used[i] else evaluate(c, occ_ell, fro_ell))
                       for i, c in enumerate(cands)])
    best = int(np.argmax(scores))
    used[best] = True
    chosen = cands[best]

    obs = observed_by(chosen)
    gained = (~observed_nbv[obs]).sum()
    observed_nbv[obs] = True
    cov = observed_nbv.sum() / N_SURF
    alt = TARGET[2] - chosen[2]
    az = math.degrees(math.atan2(chosen[1]-TARGET[1], chosen[0]-TARGET[0])) % 360

    nbv_path.append({'step': step+1, 'pos': chosen.tolist(), 'azimuth': round(az, 1),
                     'alt': round(alt, 2), 'score': round(float(scores[best]), 1),
                     'gained': int(gained), 'coverage': round(cov, 4)})
    print(f"  step{step+1:2d}: az={az:6.1f}° alt={alt:.1f}m F={scores[best]:8.1f} "
          f"+{gained:3d}voxel cov={cov*100:5.1f}%")

    if cov > 0.999:
        print("  (완전 커버 달성)")
        break

angles_n, gaps_n, max_n, mean_n, std_n = azimuth_stats([p['pos'] for p in nbv_path])
print(f"\n  Max gap: {max_n:.1f}° | Mean: {mean_n:.1f}° | Std: {std_n:.1f}°")
print(f"  Final coverage: {nbv_path[-1]['coverage']*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# 비교 요약
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("비교 요약")
print("="*60)
print(f"{'':20} {'Orbit-16':>15} {'PB-NBV-16':>15}")
print(f"{'Max azimuth gap':20} {max_o:>14.1f}° {max_n:>14.1f}°")
print(f"{'Mean gap':20} {mean_o:>14.1f}° {mean_n:>14.1f}°")
print(f"{'Std gap':20} {std_o:>14.1f}° {std_n:>14.1f}°")
print(f"{'Final coverage':20} {orbit_path[-1]['coverage']*100:>13.1f}% {nbv_path[-1]['coverage']*100:>13.1f}%")
print(f"{'고도 고정':20} {'Yes':>15} {'No':>15}")
print(f"{'방위각 균등':20} {'Yes':>15} {'?':>15}")

# 저장
result = {
    'experiment': '공정 비교: Orbit-16 vs PB-NBV-16 독립 실행',
    'orbit': {
        'max_gap': round(max_o, 2), 'mean_gap': round(mean_o, 2),
        'std_gap': round(std_o, 2),
        'coverage': orbit_path[-1]['coverage'],
        'azimuths': [round(a, 1) for a in angles_o],
        'gaps': [round(g, 1) for g in gaps_o],
        'path': orbit_path
    },
    'pbnbv': {
        'max_gap': round(max_n, 2), 'mean_gap': round(mean_n, 2),
        'std_gap': round(std_n, 2),
        'coverage': nbv_path[-1]['coverage'],
        'azimuths': [round(a, 1) for a in angles_n],
        'gaps': [round(g, 1) for g in gaps_n],
        'path': nbv_path
    }
}

out_file = OUT_DIR / "fair_comparison_result.json"
with open(out_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n✓ 저장: {out_file}")
