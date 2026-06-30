#!/usr/bin/env python3
"""
Greedy vs PB-NBV on ground_first — M=8
우리 계획(Greedy) vs 논문 알고리즘(PB-NBV) 공정 비교
"""
import json, math
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

TARGET   = np.array([0.0, 0.0, 0.0])
VOXEL    = 0.4
FOV_DEG  = 86.0       # fx=fy=1029, W=1920 → H-FOV≈86°
IMG_W, IMG_H = 1920, 1080
MIN_DIST = 8.0
MAX_DIST = 20.0
N_STEPS  = 8
MAX_ELLIPSOIDS = 8
MAX_TILT_DEG = 45.0   # 카메라 최대 틸트각 (수평 기준 아래로)
RAD = 15.0            # 반경 15m
# 거리 가중치 (공정 비교를 위해 Greedy/PB-NBV 모두 적용 가능)
GREEDY_ALPHA = 1.0
PBNBV_ALPHA = 1.0
PBNBV_DIST_MODE = "divide"   # "none" | "divide" | "subtract"
PBNBV_LAMBDA = 50.0          # subtract 모드에서만 사용
DIST_EPS = 1e-9
# 45° 제약: alt <= rad * tan(45°) = rad
ALT = min(5.0, RAD * math.tan(math.radians(MAX_TILT_DEG)))
SEED = 0
np.random.seed(SEED)

OUT_DIR = Path("results/greedy_vs_pbnbv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────────
d = np.load('real_test/ground_first_pts_normals.npz')
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
SURF_CEN  = np.array([(np.array(v)+0.5)*VOXEL+ORIGIN for v in SURF_KEYS])
SURF_NRM  = np.array([SURF[v] for v in SURF_KEYS])
N_SURF    = len(SURF_KEYS)
print(f"Surface voxels: {N_SURF:,}  (voxel={VOXEL}m)")

# ── 카메라 유틸 ────────────────────────────────────────────────────
def make_K():
    fx = (IMG_W/2) / math.tan(math.radians(FOV_DEG/2))
    return np.array([[fx,0,IMG_W/2],[0,fx,IMG_H/2],[0,0,1]])
K  = make_K()
FX, FY, CX, CY = K[0,0], K[1,1], K[0,2], K[1,2]

def look_at_R(cam_pos, target):
    f = target - cam_pos; f /= np.linalg.norm(f)+1e-9
    up0 = np.array([0.0, 0.0, 1.0])
    r = np.cross(up0, f)
    if np.linalg.norm(r) < 1e-6: up0 = np.array([0.0, 1.0, 0.0]); r = np.cross(up0, f)
    r /= np.linalg.norm(r)+1e-9
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=0)

def observed_by(cam_pos):
    to   = SURF_CEN - cam_pos
    dist = np.linalg.norm(to, axis=1)
    cam_dir = TARGET - cam_pos; cam_dir /= np.linalg.norm(cam_dir)+1e-9
    cos_half = math.cos(math.radians(FOV_DEG/2))
    in_range = (dist >= MIN_DIST) & (dist <= MAX_DIST)
    in_fov   = (to * cam_dir).sum(1) / (dist+1e-9) >= cos_half
    front    = (SURF_NRM * (cam_pos - SURF_CEN)).sum(1) > 0
    return np.where(in_range & in_fov & front)[0]

NEIGH = [(dx,dy,dz) for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1)
         if not(dx==0 and dy==0 and dz==0)]

def compute_frontier(observed_mask):
    obs_keys = set(SURF_KEYS[i] for i in np.where(observed_mask)[0])
    out = []
    for i, v in enumerate(SURF_KEYS):
        if observed_mask[i]: continue
        for dxyz in NEIGH:
            nb = (v[0]+dxyz[0], v[1]+dxyz[1], v[2]+dxyz[2])
            if nb in obs_keys:
                out.append(i); break
    return np.array(out, dtype=int)

def fit_ellipsoids(points, normals=None):
    if len(points) < 3:
        if len(points) == 0: return []
        nn = normals.mean(0) if normals is not None and len(normals)>0 else None
        return [(points.mean(0), np.eye(3)*(VOXEL**2), nn)]
    best=None; best_bic=np.inf
    for k in range(1, min(MAX_ELLIPSOIDS, len(points))+1):
        try:
            gm = GaussianMixture(k, covariance_type='full', random_state=SEED).fit(points)
            b  = gm.bic(points)
            if b < best_bic: best_bic=b; best=gm
        except: pass
    if best is None:
        nn = normals.mean(0) if normals is not None and len(normals)>0 else None
        return [(points.mean(0), np.cov(points.T)+np.eye(3)*1e-3, nn)]
    labels = best.predict(points)
    out = []
    for k,(c,cov) in enumerate(zip(best.means_, best.covariances_)):
        nn = None
        if normals is not None:
            m = labels==k
            if m.sum()>0: nn = normals[m].mean(0); nn /= np.linalg.norm(nn)+1e-9
        out.append((c, cov+np.eye(3)*1e-4, nn))
    return out

def ellipsoid_visible(center, cam_pos, mean_normal):
    d = np.linalg.norm(cam_pos - center)
    if d < MIN_DIST or d > MAX_DIST: return False
    if mean_normal is None: return True
    return float((mean_normal * (cam_pos - center)).sum()) > 0

def project_area_depth(center, cov, cam_pos, R):
    c_cam = R @ (center - cam_pos)
    z = c_cam[2]
    if z <= MIN_DIST * 0.3: return 0.0, np.inf
    u = FX*c_cam[0]/z + CX; v = FY*c_cam[1]/z + CY
    if not (-IMG_W*0.5 <= u <= IMG_W*1.5 and -IMG_H*0.5 <= v <= IMG_H*1.5): return 0.0, np.inf
    J = np.array([[FX/z, 0, -FX*c_cam[0]/z**2],[0, FY/z, -FY*c_cam[1]/z**2]])
    S2 = J @ (R @ cov @ R.T) @ J.T
    det = np.linalg.det(S2)
    if det <= 0: return 0.0, np.inf
    return float(math.pi * math.sqrt(det)), float(z)

def evaluate_pbnbv(cam_pos, occ_ell, fro_ell):
    R = look_at_R(cam_pos, TARGET)
    items = []
    for c,cov,nrm in fro_ell:
        if not ellipsoid_visible(c, cam_pos, nrm): continue
        a, z = project_area_depth(c, cov, cam_pos, R)
        if a > 0: items.append((z, a, True))
    for c,cov,nrm in occ_ell:
        if not ellipsoid_visible(c, cam_pos, nrm): continue
        a, z = project_area_depth(c, cov, cam_pos, R)
        if a > 0: items.append((z, a, False))
    if not items: return 0.0
    items.sort(key=lambda t: t[0])
    F = sum((a*0.5**r if is_f else -a*0.5**r) for r,(z,a,is_f) in enumerate(items))
    return F

def apply_distance_weight(base_score, step_dist, alpha, mode="none", lam=0.0):
    if step_dist is None:
        return float(base_score)
    if mode == "none":
        return float(base_score)
    if mode == "divide":
        if alpha <= 0:
            return float(base_score)
        return float(base_score) / ((float(step_dist) ** alpha) + DIST_EPS)
    if mode == "subtract":
        return float(base_score) - float(lam) * float(step_dist)
    raise ValueError(f"Unknown distance mode: {mode}")

# ── 후보 시점: 고도 ALT 고정, 반경 RAD, 방위 24등분 ───────────────────
def make_candidates():
    """45° 제약: 각 반경에서 z = rad * tan(MAX_TILT_DEG) 이하"""
    cands = []
    for rad in [RAD, RAD+3, RAD-3]:
        max_alt = rad * math.tan(math.radians(MAX_TILT_DEG))
        z = TARGET[2] + min(ALT, max_alt)
        n_az = 24
        for i in range(n_az):
            th = 2*math.pi*i/n_az
            cands.append([TARGET[0]+rad*math.cos(th),
                          TARGET[1]+rad*math.sin(th), z])
    return np.array(cands)

CANDS = make_candidates()
print(f"Candidates: {len(CANDS)}")

def az_of(pos):
    return math.degrees(math.atan2(pos[1]-TARGET[1], pos[0]-TARGET[0])) % 360

# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print(f"[1] Greedy (우리 계획) — M={N_STEPS}")
print("="*55)

observed_g = np.zeros(N_SURF, dtype=bool)
used_g     = np.zeros(len(CANDS), dtype=bool)
greedy_path = []
cur_pos_g = None

for step in range(N_STEPS):
    scores = np.full(len(CANDS), -1e18, dtype=float)
    gains = np.full(len(CANDS), -1, dtype=int)
    for i, c in enumerate(CANDS):
        if used_g[i]:
            continue
        gain = int((~observed_g[observed_by(c)]).sum())
        gains[i] = gain
        if cur_pos_g is None:
            scores[i] = float(gain)
        else:
            step_dist = float(np.linalg.norm(c - cur_pos_g))
            scores[i] = float(gain) / ((step_dist ** GREEDY_ALPHA) + DIST_EPS) if GREEDY_ALPHA > 0 else float(gain)

    best = int(np.argmax(scores))
    used_g[best] = True
    obs     = observed_by(CANDS[best])
    gained  = int((~observed_g[obs]).sum())
    observed_g[obs] = True
    step_dist = 0.0 if cur_pos_g is None else float(np.linalg.norm(CANDS[best] - cur_pos_g))
    cur_pos_g = CANDS[best]
    cov = observed_g.sum() / N_SURF
    az  = az_of(CANDS[best])
    greedy_path.append({
        'step': step+1, 'pos': CANDS[best].tolist(),
        'azimuth': round(az,1), 'gained': gained,
        'coverage': round(float(cov), 4),
        'step_dist': round(step_dist, 3),
        'score': round(float(scores[best]), 6)
    })
    print(f"  step{step+1}: az={az:6.1f}°  d={step_dist:5.2f}m  s={scores[best]:9.4f}  +{gained:4d}voxel  cov={cov*100:5.1f}%")

print(f"\n  Final coverage: {greedy_path[-1]['coverage']*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print(f"[2] PB-NBV (논문 알고리즘) — M={N_STEPS}")
print("="*55)

observed_n = np.zeros(N_SURF, dtype=bool)
used_n     = np.zeros(len(CANDS), dtype=bool)
nbv_path   = []
cur_pos_n  = None

for step in range(N_STEPS):
    fro_idx = compute_frontier(observed_n)
    occ_pts = SURF_CEN[observed_n];     occ_nrm = SURF_NRM[observed_n]
    fro_pts = SURF_CEN[fro_idx] if len(fro_idx)>0 else np.empty((0,3))
    fro_nrm = SURF_NRM[fro_idx] if len(fro_idx)>0 else np.empty((0,3))
    occ_ell = fit_ellipsoids(occ_pts, occ_nrm)
    fro_ell = fit_ellipsoids(fro_pts, fro_nrm) if len(fro_pts)>0 else []
    if len(occ_pts)==0 and len(fro_pts)==0:
        fro_ell = fit_ellipsoids(SURF_CEN, SURF_NRM)

    raw_scores = np.full(len(CANDS), -1e18, dtype=float)
    scores = np.full(len(CANDS), -1e18, dtype=float)
    for i, c in enumerate(CANDS):
        if used_n[i]:
            continue
        raw = evaluate_pbnbv(c, occ_ell, fro_ell)
        step_dist = None if cur_pos_n is None else float(np.linalg.norm(c - cur_pos_n))
        score = apply_distance_weight(raw, step_dist, PBNBV_ALPHA, PBNBV_DIST_MODE, PBNBV_LAMBDA)
        raw_scores[i] = raw
        scores[i] = score

    best = int(np.argmax(scores))
    used_n[best] = True
    obs    = observed_by(CANDS[best])
    gained = int((~observed_n[obs]).sum())
    observed_n[obs] = True
    step_dist = 0.0 if cur_pos_n is None else float(np.linalg.norm(CANDS[best] - cur_pos_n))
    cur_pos_n = CANDS[best]
    cov = observed_n.sum() / N_SURF
    az  = az_of(CANDS[best])
    nbv_path.append({
        'step': step+1, 'pos': CANDS[best].tolist(),
        'azimuth': round(az,1), 'gained': gained,
        'coverage': round(float(cov), 4),
        'score_raw': round(float(raw_scores[best]), 2),
        'score': round(float(scores[best]), 2),
        'step_dist': round(step_dist, 3)
    })
    print(f"  step{step+1}: az={az:6.1f}°  d={step_dist:5.2f}m  F_raw={raw_scores[best]:9.1f}  F={scores[best]:9.1f}  +{gained:4d}voxel  cov={cov*100:5.1f}%")

print(f"\n  Final coverage: {nbv_path[-1]['coverage']*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("비교 요약")
print("="*55)
g_cov = [p['coverage']*100 for p in greedy_path]
n_cov = [p['coverage']*100 for p in nbv_path]
print(f"{'':10} {'Greedy':>10} {'PB-NBV':>10}")
for s in range(N_STEPS):
    print(f"  step{s+1:2d}:  {g_cov[s]:7.1f}%  {n_cov[s]:7.1f}%")
print(f"  {'Final':8}  {g_cov[-1]:7.1f}%  {n_cov[-1]:7.1f}%")

result = {
    'greedy': {'path': greedy_path, 'final_coverage': greedy_path[-1]['coverage']},
    'pbnbv':  {'path': nbv_path,    'final_coverage': nbv_path[-1]['coverage']},
    'params': {
        'M': N_STEPS, 'alt': ALT, 'rad': RAD, 'voxel': VOXEL, 'fov': FOV_DEG,
        'greedy_alpha': GREEDY_ALPHA,
        'pbnbv_dist_mode': PBNBV_DIST_MODE,
        'pbnbv_alpha': PBNBV_ALPHA,
        'pbnbv_lambda': PBNBV_LAMBDA
    }
}
with open(OUT_DIR/'result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n✓ {OUT_DIR/'result.json'}")
