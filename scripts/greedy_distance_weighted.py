#!/usr/bin/env python3
"""
거리 가중 Greedy — score(v) = new_voxels(v) / distance(current→v)^alpha
alpha 값별로 경로/커버리지/총 이동거리 비교
"""
import json, math
import numpy as np
from pathlib import Path

TARGET   = np.array([0.0, 0.0, 0.0])
VOXEL    = 0.4
FOV_DEG  = 86.0
IMG_W, IMG_H = 1920, 1080
MIN_DIST = 8.0
MAX_DIST = 20.0
N_STEPS  = 8
MAX_TILT_DEG = 45.0
RAD = 15.0
ALT = min(5.0, RAD * math.tan(math.radians(MAX_TILT_DEG)))
SEED = 0
np.random.seed(SEED)

# 거리 가중 파라미터
ALPHAS = [0.0, 0.5, 1.0, 2.0]   # 비교할 alpha 값들
MAX_STEP_DIST = None             # 하드 컷오프 (None=비활성, 예: 25.0)

OUT_DIR = Path("results/greedy_distance_weighted")
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
print(f"Surface voxels: {N_SURF:,}")

def observed_by(cam_pos):
    to   = SURF_CEN - cam_pos
    dist = np.linalg.norm(to, axis=1)
    cam_dir = TARGET - cam_pos; cam_dir /= np.linalg.norm(cam_dir)+1e-9
    cos_half = math.cos(math.radians(FOV_DEG/2))
    in_range = (dist >= MIN_DIST) & (dist <= MAX_DIST)
    in_fov   = (to * cam_dir).sum(1) / (dist+1e-9) >= cos_half
    front    = (SURF_NRM * (cam_pos - SURF_CEN)).sum(1) > 0
    return np.where(in_range & in_fov & front)[0]

def make_candidates():
    cands = []
    for rad in [RAD, RAD+3, RAD-3]:
        max_alt = rad * math.tan(math.radians(MAX_TILT_DEG))
        z = TARGET[2] + min(ALT, max_alt)
        for i in range(24):
            th = 2*math.pi*i/24
            cands.append([TARGET[0]+rad*math.cos(th),
                          TARGET[1]+rad*math.sin(th), z])
    return np.array(cands)

CANDS = make_candidates()

def az_of(pos):
    return math.degrees(math.atan2(pos[1]-TARGET[1], pos[0]-TARGET[0])) % 360

# ── 미리 각 후보의 관측 voxel 캐싱 ──────────────────────────────────
OBS_CACHE = [observed_by(c) for c in CANDS]

# ── 거리 가중 Greedy ───────────────────────────────────────────────
def run_greedy(alpha, max_step_dist=None):
    observed = np.zeros(N_SURF, dtype=bool)
    used     = np.zeros(len(CANDS), dtype=bool)
    # 시작 위치: 첫 스텝은 거리 무시(현재 위치 없음)하고 순수 gain 최대
    cur_pos  = None
    path = []
    total_travel = 0.0

    for step in range(N_STEPS):
        best_i, best_score, best_gain = -1, -np.inf, 0
        for i, c in enumerate(CANDS):
            if used[i]:
                continue
            gain = int((~observed[OBS_CACHE[i]]).sum())
            if cur_pos is None:
                score = gain                      # 첫 스텝: 거리 항 없음
                dist  = 0.0
            else:
                dist = float(np.linalg.norm(c - cur_pos))
                if max_step_dist is not None and dist > max_step_dist:
                    continue                      # 하드 컷오프
                score = gain / (dist**alpha + 1e-9) if alpha > 0 else gain
            if score > best_score:
                best_score, best_i, best_gain = score, i, gain

        if best_i < 0:   # 컷오프로 후보 없으면 거리 무시하고 최대 gain
            cand_gains = [(-1 if used[i] else int((~observed[OBS_CACHE[i]]).sum()))
                          for i in range(len(CANDS))]
            best_i = int(np.argmax(cand_gains)); best_gain = cand_gains[best_i]

        c = CANDS[best_i]
        step_dist = 0.0 if cur_pos is None else float(np.linalg.norm(c - cur_pos))
        total_travel += step_dist
        used[best_i] = True
        observed[OBS_CACHE[best_i]] = True
        cur_pos = c
        cov = observed.sum() / N_SURF
        path.append({
            'step': step+1, 'pos': c.tolist(), 'azimuth': round(az_of(c),1),
            'gained': best_gain, 'coverage': round(float(cov),4),
            'step_dist': round(step_dist,2), 'cum_dist': round(total_travel,2)
        })
    return path, total_travel

# ── 실행 ───────────────────────────────────────────────────────────
results = {}
print("\n" + "="*70)
print(f"{'alpha':>6} | {'final cov':>9} | {'총 이동거리':>12} | 경로 방위(°)")
print("="*70)
for a in ALPHAS:
    path, travel = run_greedy(a, MAX_STEP_DIST)
    results[f"alpha_{a}"] = {
        'alpha': a, 'path': path,
        'final_coverage': path[-1]['coverage'],
        'total_travel': round(travel,2)
    }
    azs = " → ".join(f"{p['azimuth']:.0f}" for p in path)
    print(f"{a:>6.1f} | {path[-1]['coverage']*100:>8.1f}% | {travel:>10.1f}m | {azs}")

print("="*70)

# ── 스텝별 상세 (alpha 비교) ────────────────────────────────────────
print("\n[스텝별 누적 이동거리 / 커버리지]")
print(f"{'step':>4}", end="")
for a in ALPHAS:
    print(f" | a={a}: dist/cov", end="")
print()
for s in range(N_STEPS):
    print(f"{s+1:>4}", end="")
    for a in ALPHAS:
        p = results[f"alpha_{a}"]['path'][s]
        print(f" | {p['cum_dist']:>5.1f}m/{p['coverage']*100:>4.0f}%", end="")
    print()

with open(OUT_DIR/'result.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ {OUT_DIR/'result.json'}")
