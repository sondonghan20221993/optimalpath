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
import json
import math
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture

# ── 설정 ─────────────────────────────────────────────────────────────────────
TARGET   = np.array([-33.67, -50.83, 0.18])
VOXEL    = 0.15      # voxel 크기 (m)
FOV_DEG  = 89.9
IMG_W, IMG_H = 1920, 1080
MIN_DIST = 4.0
MAX_DIST = 13.0
N_STEPS  = 12        # NBV 반복 횟수
MAX_ELLIPSOIDS = 8   # 클러스터당 최대 ellipsoid (BIC로 자동 결정)
SEED     = 0
np.random.seed(SEED)

# ── 후보 시점: 편향 줄이려 반구+측면 조밀 격자 (논문은 반구 800개) ───────────
def make_candidates():
    cands = []
    # 고도 1m~9m, 0.5m 간격 / 반경 4.5~9m
    for alt in np.arange(1.0, 9.5, 1.0):
        z = TARGET[2] - alt
        for rad in [4.5, 6.0, 7.5, 9.0]:
            n_az = max(6, int(2*math.pi*rad / 1.5))  # 둘레 비례 방위 분배
            for i in range(n_az):
                th = 2*math.pi*i/n_az
                cands.append([TARGET[0]+rad*math.cos(th),
                              TARGET[1]+rad*math.sin(th), z])
    return np.array(cands)

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
d = np.load('real_test/real_test_pts_normals.npz')
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

# ── 후보 시점 평가: F = Σ(frontier proj·W) − Σ(occupied proj·W) ─────────────
def evaluate(cam_pos, occ_ell, fro_ell):
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
        W = 0.5**r                  # observability weight (가림 모델)
        F += (a*W) if is_f else -(a*W)
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
        used[best]=True
        chosen=cands[best]
        newobs = observed_by(chosen)
        gained = (~observed[newobs]).sum()
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

if __name__=='__main__':
    path, curve, cands = run_nbv()
    out=Path('results/pbnbv_paper'); out.mkdir(parents=True,exist_ok=True)
    json.dump({'algorithm':'PB-NBV (paper: voxel+ellipsoid projection)',
               'voxel_size':VOXEL,'n_surface_voxels':N_SURF,
               'final_coverage':curve[-1],'path':path},
              open(out/'pbnbv_paper.json','w'), indent=2)
    print(f"\n최종 coverage: {curve[-1]*100:.1f}%")
    print(f"고도 분포: ", end='')
    alts=[round(p['alt']) for p in path]
    for a in sorted(set(alts)): print(f"{a}m×{alts.count(a)} ",end='')
    print(f"\nJSON: {out/'pbnbv_paper.json'}")
