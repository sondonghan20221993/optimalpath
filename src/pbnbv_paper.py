#!/usr/bin/env python3
"""
pbnbv_paper.py — 논문 PB-NBV (arXiv:2501.10663) 충실 구현

논문 핵심 3요소:
  1. Voxel 분류: 관측된 표면 = Occupied, 그 경계의 미관측 표면 = Frontier
  2. GMM 클러스터링 → 각 클러스터를 Ellipsoid로 피팅
  3. Projection-based 평가: ellipsoid를 이미지 평면에 투영,
     depth rank 가중치 W=0.5^r, 점수 F = Σ(frontier 투영) − Σ(occupied 투영)

NBV 반복 루프:
  매 스텝 최고 F 후보 선택 → 관측 갱신 → frontier 재계산 → 반복

데이터: real_test GT 점군(2438점)을 오라클로 사용.
        "관측" = 후보 시점 frustum 안 + front-facing GT 점의 voxel을 observed로 마킹.
"""
import sys, types
sys.modules.setdefault("open3d", types.ModuleType("open3d"))
import json
import math
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as _A

# ── 설정 ─────────────────────────────────────────────────────────────────────
NPZ      = HERE.parent / "real_test" / "airsim_gt_pts.npz"  # FBX 메시 GT (카메라 독립)
_pts_raw = np.load(NPZ)["points"].astype(float)
TARGET   = _pts_raw.mean(axis=0)   # 공동 통제: 포인트클라우드 centroid
VOXEL    = 0.05      # 포인트 간격(~0.035m)에 맞춤 (기존 0.15m는 과대뭉침)
FOV_DEG  = 89.9
IMG_W, IMG_H = 1920, 1080
MIN_DIST = 0.1
MAX_DIST = 8.0        # 공동 통제: pbnbv_path 와 동일
N_STEPS  = 20
MAX_ELLIPSOIDS = 8
SEED     = 0
np.random.seed(SEED)

# ── 후보 시점: tilt=45° 고정, 공동 통제조건 ───────────────────────────────────
def make_candidates():
    obj_z_top = _pts_raw[:,2].min()
    z_levels  = [round(obj_z_top - dz, 2) for dz in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    return _A.gen_candidates_tilt45(TARGET, z_levels, n_az=36, max_dist=MAX_DIST)

# ── 카메라 intrinsic ─────────────────────────────────────────────────────────
def make_K():
    fx = (IMG_W/2) / math.tan(math.radians(FOV_DEG/2))
    fy = fx
    return np.array([[fx,0,IMG_W/2],[0,fy,IMG_H/2],[0,0,1]])
K = make_K()

def look_at_R(cam_pos, target):
    """카메라 좌표계: +z = 광축(타겟 방향). world->cam 회전 R 반환."""
    f = target - cam_pos; f /= np.linalg.norm(f)+1e-9   # forward
    up0 = np.array([0,0,-1.0])  # NED: 위쪽
    r = np.cross(up0, f);
    if np.linalg.norm(r) < 1e-6: up0 = np.array([0,1.0,0]); r = np.cross(up0,f)
    r /= np.linalg.norm(r)+1e-9
    u = np.cross(f, r)
    R = np.stack([r, u, f], axis=0)   # rows = cam axes in world
    return R

# ── GT 점군 로드 & voxel화 ───────────────────────────────────────────────────
d = np.load(str(NPZ))  # airsim_gt_pts.npz
PTS, NRM = d['points'], d['normals']
ORIGIN = PTS.min(0) - VOXEL

def pt_to_vox(p):
    return tuple(np.floor((p-ORIGIN)/VOXEL).astype(int))

# GT surface voxel 집합 (오라클) + 각 voxel 대표 법선
surface_vox = {}
for p, n in zip(PTS, NRM):
    v = pt_to_vox(p)
    surface_vox.setdefault(v, []).append(n)
SURF = {v: np.mean(ns,axis=0) for v,ns in surface_vox.items()}
SURF_KEYS = list(SURF.keys())
SURF_CEN = np.array([ (np.array(v)+0.5)*VOXEL+ORIGIN for v in SURF_KEYS ])  # voxel 중심
SURF_NRM = np.array([ SURF[v] for v in SURF_KEYS ])
N_SURF = len(SURF_KEYS)
print(f"GT surface voxels: {N_SURF}")

# ── 관측 모델: 후보 시점이 보는 surface voxel 인덱스 ─────────────────────────
def observed_by(cam_pos):
    cam_dir = TARGET - cam_pos; cam_dir/=np.linalg.norm(cam_dir)+1e-9
    to = SURF_CEN - cam_pos
    dist = np.linalg.norm(to,axis=1)
    in_range = (dist>=MIN_DIST)&(dist<=MAX_DIST)
    cos_half = math.cos(math.radians(FOV_DEG/2))
    in_fov = (to*cam_dir).sum(1)/(dist+1e-9) >= cos_half
    front = (SURF_NRM*(cam_pos-SURF_CEN)).sum(1) > 0   # 법선이 카메라 향함
    return np.where(in_range&in_fov&front)[0]

# ── Frontier: 미관측 surface voxel 중 관측된 voxel에 인접한 것 ───────────────
NEIGH = [(dx,dy,dz) for dx in(-1,0,1) for dy in(-1,0,1) for dz in(-1,0,1)
         if not(dx==0 and dy==0 and dz==0)]
SURF_SET = set(SURF_KEYS)
def compute_frontier(observed_mask):
    obs_keys = set(SURF_KEYS[i] for i in np.where(observed_mask)[0])
    frontier_idx = []
    for i,v in enumerate(SURF_KEYS):
        if observed_mask[i]: continue
        # 인접 voxel 중 관측된 surface voxel 있으면 frontier
        for dxyz in NEIGH:
            nb = (v[0]+dxyz[0], v[1]+dxyz[1], v[2]+dxyz[2])
            if nb in obs_keys:
                frontier_idx.append(i); break
    return np.array(frontier_idx, dtype=int)

# ── GMM 클러스터링 → ellipsoid (center, cov, mean_normal) 리스트 ────────────
def fit_ellipsoids(points, normals=None):
    if len(points) < 3:
        if len(points)==0: return []
        c = points.mean(0); cov = np.eye(3)*(VOXEL**2)
        nn = normals.mean(0) if normals is not None and len(normals)>0 else None
        return [(c,cov,nn)]
    best=None; best_bic=np.inf
    for k in range(1, min(MAX_ELLIPSOIDS,len(points))+1):
        try:
            gm=GaussianMixture(k,covariance_type='full',random_state=SEED).fit(points)
            b=gm.bic(points)
            if b<best_bic: best_bic=b; best=gm
        except Exception: pass
    if best is None:
        nn = normals.mean(0) if normals is not None and len(normals)>0 else None
        return [(points.mean(0), np.cov(points.T)+np.eye(3)*1e-3, nn)]
    labels = best.predict(points)
    out=[]
    for k,(c,cov) in enumerate(zip(best.means_, best.covariances_)):
        nn=None
        if normals is not None:
            m = labels==k
            if m.sum()>0:
                nn = normals[m].mean(0); nn/=np.linalg.norm(nn)+1e-9
        out.append((c, cov+np.eye(3)*1e-4, nn))
    return out

# ── Ellipsoid 투영 면적 (closed-form: 공분산 perspective Jacobian) ────────────
# 논문의 dual-quadric 투영을 1차(Gaussian) 근사로 닫힌형 계산 → ray-casting 없음
FX, FY, CX, CY = K[0,0], K[1,1], K[0,2], K[1,2]
def project_area_depth(center, cov, cam_pos, R):
    c_cam = R @ (center - cam_pos)      # world→cam
    z = c_cam[2]
    if z <= MIN_DIST*0.3: return 0.0, np.inf
    u = FX*c_cam[0]/z + CX
    v = FY*c_cam[1]/z + CY
    # 투영 중심이 화면 밖이면(여유 50%) 컷
    if not (-IMG_W*0.5<=u<=IMG_W*1.5 and -IMG_H*0.5<=v<=IMG_H*1.5):
        return 0.0, np.inf
    # perspective projection Jacobian (2x3)
    J = np.array([[FX/z, 0, -FX*c_cam[0]/z**2],
                  [0, FY/z, -FY*c_cam[1]/z**2]])
    Sig_cam = R @ cov @ R.T
    S2 = J @ Sig_cam @ J.T              # 이미지 평면 2D 공분산
    det = np.linalg.det(S2)
    if det <= 0: return 0.0, np.inf
    area = math.pi * math.sqrt(det)     # 1σ 타원 면적
    return float(area), float(z)

# ── ellipsoid 가시성 게이팅: 클러스터 평균 법선이 카메라를 향하는지 ──────────
def ellipsoid_visible(center, cam_pos, mean_normal):
    """클러스터 중심의 평균 법선이 카메라 쪽을 향하고, 거리 범위 안인지."""
    d = np.linalg.norm(cam_pos-center)
    if d<MIN_DIST or d>MAX_DIST: return False
    if mean_normal is None: return True
    return float((mean_normal*(cam_pos-center)).sum()) > 0

# ── 후보 시점 평가 ───────────────────────────────────────────────────────────
# frontier_only=False (논문 baseline): F = Σ(frontier·W) − Σ(occupied·W)
#   → 상대 argmax 용도. 거리로 나누지 말 것(음수 가능).
# frontier_only=True  (드론 거리가중치 용): F = Σ(frontier·W) ≥ 0
#   → occupied 타원체는 depth rank를 차지해 가림은 모델링하되 면적을 빼지 않음.
#     Bircher utility U=F/dist 가 성립하도록 분자를 비음수로 유지.
def evaluate(cam_pos, occ_ell, fro_ell, frontier_only=False):
    R = look_at_R(cam_pos, TARGET)
    items=[]  # (depth, area, is_frontier)
    for c,cov,nrm in fro_ell:
        if not ellipsoid_visible(c,cam_pos,nrm): continue
        a,z = project_area_depth(c,cov,cam_pos,R)
        if a>0: items.append((z,a,True))
    for c,cov,nrm in occ_ell:
        if not ellipsoid_visible(c,cam_pos,nrm): continue
        a,z = project_area_depth(c,cov,cam_pos,R)
        if a>0: items.append((z,a,False))
    if not items: return 0.0
    items.sort(key=lambda t:t[0])   # depth 오름차순(가까운 것 먼저)
    F=0.0
    for r,(z,a,is_f) in enumerate(items):
        W = 0.5**r                  # observability weight (가림 모델: rank=가림)
        if is_f:
            F += a*W                # frontier 면적만 가산 (항상 ≥0)
        elif not frontier_only:
            F -= a*W                # 논문 baseline: occupied 감산
        # frontier_only=True: occupied는 rank 슬롯만 소비(가림) → 면적 미반영
    return F

# ── NBV 반복 루프 ────────────────────────────────────────────────────────────
def run_nbv():
    cands = make_candidates()
    print(f"후보 시점: {len(cands)}개")
    observed = np.zeros(N_SURF, dtype=bool)
    path=[]; cov_curve=[]
    start = np.array([-27.06,-50.51,-6.13])  # 마지막 기존 카메라 근사
    cur = start
    used = np.zeros(len(cands), dtype=bool)   # 재방문 금지 마스크
    for step in range(N_STEPS):
        fro_idx = compute_frontier(observed)
        # ellipsoid 피팅 (법선 포함)
        occ_pts = SURF_CEN[observed];  occ_nrm = SURF_NRM[observed]
        if len(fro_idx)>0:
            fro_pts = SURF_CEN[fro_idx]; fro_nrm = SURF_NRM[fro_idx]
        else:
            fro_pts = np.empty((0,3)); fro_nrm = np.empty((0,3))
        occ_ell = fit_ellipsoids(occ_pts, occ_nrm)
        fro_ell = fit_ellipsoids(fro_pts, fro_nrm) if len(fro_pts)>0 else []
        # 첫 스텝: 관측 전무 → 전체 surface를 frontier로 부트스트랩
        if len(occ_pts)==0 and len(fro_pts)==0:
            fro_ell = fit_ellipsoids(SURF_CEN, SURF_NRM)
        # 후보 평가 (이미 쓴 후보는 -inf)
        scores=np.array([(-1e18 if used[i] else evaluate(c, occ_ell, fro_ell))
                         for i,c in enumerate(cands)])
        best=int(np.argmax(scores))
        chosen=cands[best]
        newobs = observed_by(chosen)
        gained = (~observed[newobs]).sum()
        # 종료조건: 최고 후보가 새 voxel을 하나도 못 보면 중단 (헛스텝 방지)
        if gained == 0:
            print(f"  step{step+1}: 최고 후보 gain=0 → 종료")
            break
        used[best]=True
        observed[newobs]=True
        cov=observed.sum()/N_SURF
        cov_curve.append(cov)
        alt=TARGET[2]-chosen[2]
        az=math.degrees(math.atan2(chosen[1]-TARGET[1], chosen[0]-TARGET[0]))
        path.append({'step':step+1,'pos':chosen.tolist(),'alt':round(alt,2),
                     'azimuth':round(az,1),'score':round(float(scores[best]),1),
                     'gained':int(gained),'coverage':round(cov,4),
                     'n_frontier':int(len(fro_idx))})
        print(f"  step{step+1:2d}: alt={alt:4.1f}m az={az:+6.1f}° "
              f"F={scores[best]:8.1f} +{gained:3d}voxel cov={cov*100:5.1f}% "
              f"frontier={len(fro_idx)}")
        cur=chosen
        if cov>0.999: print("  (완전 커버)"); break
    return path, cov_curve, cands

def eval_on_points(wps):
    """공동 통제: 실제 포인트 2438개 + raycast 기준 커버리지."""
    _A.RAYCAST_OCCLUSION = True
    live = np.ones(len(_pts_raw), dtype=bool)
    curve = []
    for wp in wps:
        _, vis = _A.information_gain(np.array(wp), TARGET, _pts_raw, live,
                                     FOV_DEG, MAX_DIST)
        live &= ~vis
        curve.append(int(len(_pts_raw) - live.sum()))
    return curve, int(live.sum())


if __name__=='__main__':
    path, curve, cands = run_nbv()

    # 공동 통제 평가: 실제 포인트 기준 (voxel 100%여도 포인트는 다를 수 있음)
    wps = [p['pos'] for p in path]
    pt_curve, pt_left = eval_on_points(wps)
    pt_total = len(_pts_raw)
    pt_cov   = (pt_total - pt_left) / pt_total
    print(f"\n[공동통제] 실제 포인트 기준 커버리지:")
    for i, c in enumerate(pt_curve):
        print(f"  WP{i+1}: {c}/{pt_total} ({100*c/pt_total:.1f}%)")

    out=Path('results/pbnbv_paper'); out.mkdir(parents=True,exist_ok=True)
    json.dump({'algorithm':'PB-NBV (paper: voxel+ellipsoid projection)',
               'voxel_size':VOXEL,'n_surface_voxels':N_SURF,
               'final_coverage_voxel':curve[-1],
               'final_coverage_points':round(pt_cov,4),
               'points_covered':pt_total-pt_left,'points_total':pt_total,
               'path':path},
              open(out/'pbnbv_paper.json','w'), indent=2)
    print(f"\n최종 voxel coverage : {curve[-1]*100:.1f}%")
    print(f"최종 point coverage : {pt_cov*100:.1f}%  ({pt_total-pt_left}/{pt_total})")
    print(f"WP 수: {len(path)}개")
    print(f"고도 분포: ", end='')
    alts=[round(p['alt']) for p in path]
    for a in sorted(set(alts)): print(f"{a}m×{alts.count(a)} ",end='')
    print(f"\nJSON: {out/'pbnbv_paper.json'}")
