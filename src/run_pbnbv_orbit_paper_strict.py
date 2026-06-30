"""
run_pbnbv_orbit_paper_strict.py
논문 PB-NBV 방식 + '규제 강화': observed 판정에 입사각(incidence angle) 제한 추가.

논문 원본: observed = in_range & in_fov & (normal·view > 0)   ← θ<90° 만
강화안   : observed = in_range & in_fov & (normal·view > cos θ_max)  ← θ<θ_max
  → 빗겨보는(grazing) 면을 '미관측'으로 처리 = raycast 없이 가림을 부분 근사.

θ_max 를 sweep 하며 (90°=원본, 70, 60, 50) 보완 시점 수/커버리지 변화를 관찰.
raycast 없음(논문 정신 유지). 평가만 실제 점 raycast 로 공정 비교.
"""
import sys, types, json, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A
import pbnbv_paper as P

ROOT = HERE.parent
META = sorted((ROOT / "real_test" / "meta").glob("*.json"))
OUT  = ROOT / "results" / "pbnbv_orbit_paper_strict.json"
START   = np.array([-27.06, -50.51, -6.13])
N_STEPS = 20


def load_orbit_positions():
    pos = []
    for m in META:
        p = json.load(open(m))["camera"]["pose"]["position"]
        pos.append([p["x"], p["y"], p["z"]])
    return np.array(pos)


def observed_by_inc(cam_pos, inc_max_deg):
    """논문 observed_by + 입사각 제한 (raycast 없음)."""
    cam_dir = P.TARGET - cam_pos; cam_dir /= np.linalg.norm(cam_dir)+1e-9
    to   = P.SURF_CEN - cam_pos
    dist = np.linalg.norm(to, axis=1)
    in_range = (dist >= P.MIN_DIST) & (dist <= P.MAX_DIST)
    cos_half = math.cos(math.radians(P.FOV_DEG/2))
    in_fov   = (to*cam_dir).sum(1)/(dist+1e-9) >= cos_half
    # 입사각: voxel 법선과 (카메라→voxel 반대방향=voxel→카메라) 사이 각
    view = cam_pos - P.SURF_CEN
    view /= (np.linalg.norm(view, axis=1, keepdims=True)+1e-9)
    cos_inc = (P.SURF_NRM*view).sum(1)
    front_strict = cos_inc >= math.cos(math.radians(inc_max_deg))
    return np.where(in_range & in_fov & front_strict)[0]


def run_one(cam_pos, inc_max_deg, verbose=False):
    observed = np.zeros(P.N_SURF, dtype=bool)
    for c in cam_pos:
        observed[observed_by_inc(c, inc_max_deg)] = True
    n_orbit = int(observed.sum())

    cands   = P.make_candidates()
    used    = np.zeros(len(cands), dtype=bool)
    sel     = []
    cur_pos = cam_pos[-1].copy()   # 궤도 마지막 카메라 → 출발점

    for step in range(N_STEPS):
        fro_idx = P.compute_frontier(observed)
        if len(fro_idx) == 0: break
        occ_ell = P.fit_ellipsoids(P.SURF_CEN[observed], P.SURF_NRM[observed])
        fro_ell = P.fit_ellipsoids(P.SURF_CEN[fro_idx], P.SURF_NRM[fro_idx])
        # U(v) = F(v) / dist(cur_pos → v)  [Bircher 2016: IG / travel_cost]
        # frontier_only=True → F≥0 (occupied는 가림만, 감산 안 함) → 거리 나눗셈 유효
        scores = np.array([
            -1e18 if used[i] else
            P.evaluate(c, occ_ell, fro_ell, frontier_only=True) / (np.linalg.norm(c - cur_pos) + 1e-6)
            for i, c in enumerate(cands)
        ])
        best   = int(np.argmax(scores))
        newobs = observed_by_inc(cands[best], inc_max_deg)
        gained = int((~observed[newobs]).sum())
        if gained == 0: break
        used[best] = True; observed[newobs] = True
        cur_pos = cands[best].copy()   # 위치 업데이트
        sel.append(cands[best])
    n_final = int(observed.sum())

    # Greedy + 실제 점 raycast 평가
    A.RAYCAST_OCCLUSION = True
    pts = P._pts_raw
    if sel:
        order = A.greedy_path(np.array(sel), cam_pos[-1])
        cur, plen = cam_pos[-1].copy(), 0.0
        for wp in order:
            plen += np.linalg.norm(np.asarray(wp)-cur); cur = np.asarray(wp)
    else:
        order, plen = [], 0.0
    live = np.ones(len(pts), dtype=bool)
    for c in cam_pos:
        _, vis = A.information_gain(c, P.TARGET, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    orbit_real = len(pts)-int(live.sum())
    for wp in order:
        _, vis = A.information_gain(np.asarray(wp), P.TARGET, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    final_real = len(pts)-int(live.sum())

    return {
        "inc_max_deg": inc_max_deg,
        "voxel_orbit": n_orbit, "voxel_final": n_final, "n_surf": P.N_SURF,
        "voxel_cov_orbit": round(n_orbit/P.N_SURF,4),
        "voxel_cov_final": round(n_final/P.N_SURF,4),
        "supplementary": len(order),
        "path_m": round(plen,3),
        "real_orbit": orbit_real, "real_final": final_real, "n_pts": len(pts),
        "real_cov_orbit": round(orbit_real/len(pts),4),
        "real_cov_final": round(final_real/len(pts),4),
        "waypoints": [np.asarray(w).tolist() for w in order],
    }


def main():
    cam_pos = load_orbit_positions()
    print(f"원형 궤도 {len(cam_pos)}개 카메라\n")
    print(f"{'θ_max':>6} | {'voxel(궤도)':>10} {'voxel(최종)':>10} | "
          f"{'보완WP':>6} {'경로m':>7} | {'real(궤도)':>9} {'real(최종)':>9} {'Δ':>6}")
    print("-"*82)
    results = []
    for inc in [90, 70, 60, 50, 40]:
        r = run_one(cam_pos, inc)
        results.append(r)
        d = 100*(r['real_cov_final']-r['real_cov_orbit'])
        print(f"{inc:>5}° | {100*r['voxel_cov_orbit']:>9.1f}% {100*r['voxel_cov_final']:>9.1f}% | "
              f"{r['supplementary']:>6} {r['path_m']:>7.1f} | "
              f"{100*r['real_cov_orbit']:>8.1f}% {100*r['real_cov_final']:>8.1f}% {d:>+5.1f}%p")
    json.dump(results, open(OUT,"w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
