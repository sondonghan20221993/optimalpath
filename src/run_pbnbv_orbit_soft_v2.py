"""
run_pbnbv_orbit_soft_v2.py — Soft 규제 + 일관된 선택기준 (원형 arc 분산)

v1(run_pbnbv_orbit_soft.py)의 문제:
  선택은 ellipsoid F-score(투영면적)로, 보상은 soft cos¹ 누적으로 했더니
  F-score가 한 방위(az=270°)를 ~10배 뻥튀기 → 경로가 한 점에 붕괴.

v2 수정:
  선택기준 U(v) = (실제 신규 soft-cover gain) / dist(cur→v)
  → 선택과 보상이 동일 척도. 투영 아티팩트 제거. 경로가 frontier 따라 arc로 분산.

나머지(alpha=1 Lambert, TAU=cos70°, tilt=45° 고정고도)는 v1과 동일.
"""
import sys, types, json, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A
import pbnbv_paper as P
import run_pbnbv_orbit_soft as S   # soft_weight, hard_observed, TAU, ALPHA 재사용

ROOT = HERE.parent
META = sorted((ROOT / "real_test" / "meta").glob("*.json"))
OUT  = ROOT / "results" / "pbnbv_orbit_soft_v2_path.json"

DIST_POW = 1.0   # 거리 가중 지수 (드론 이동비용)
N_AZ     = 36
N_STEPS  = 20


def load_orbit():
    pos = []
    for m in META:
        p = json.load(open(m))["camera"]["pose"]["position"]
        pos.append([p["x"], p["y"], p["z"]])
    return np.array(pos)


def main():
    cam_pos = load_orbit()
    target  = P.TARGET
    pts     = P._pts_raw
    fix_z   = round(pts[:, 2].min() - 3.0, 2)

    print(f"[1] 고정고도 z={fix_z}m, tilt=45°, Soft Lambert cos^{S.ALPHA:.0f}, TAU=cos70°={S.TAU:.3f}")
    print(f"    선택기준 U = 실제_soft_gain / dist^{DIST_POW}  (F-score 아티팩트 제거)")

    # 궤도 누적 soft + hard
    obs_w = np.zeros(P.N_SURF)
    obs_h = np.zeros(P.N_SURF, dtype=bool)
    for c in cam_pos:
        obs_w += S.soft_weight(c)
        obs_h[S.hard_observed(c)] = True
    soft_orbit = int((obs_w >= S.TAU).sum())
    hard_orbit = int(obs_h.sum())
    print(f"[2] 궤도 커버: soft {100*soft_orbit/P.N_SURF:.1f}% | hard {100*hard_orbit/P.N_SURF:.1f}%")

    cands = A.gen_candidates_tilt45(target, [fix_z], n_az=N_AZ, max_dist=P.MAX_DIST)
    used  = np.zeros(len(cands), dtype=bool)
    sel   = []
    cur   = cam_pos[-1].copy()
    print(f"[3] 후보 {len(cands)}개. 출발 ({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f})")
    print(f"[4] Soft NBV (gain/dist):")

    for step in range(N_STEPS):
        prev = (obs_w >= S.TAU)
        if int(prev.sum()) >= P.N_SURF:
            print(f"  step{step+1}: soft 100% → 종료"); break
        # 실제 신규 soft-gain / dist^p 로 선택
        best, bestU, bestGain = -1, -1e18, 0
        for i, c in enumerate(cands):
            if used[i]:
                continue
            dist = np.linalg.norm(c - cur) + 1e-6
            nw   = obs_w + S.soft_weight(c)
            gain = int(((nw >= S.TAU) & ~prev).sum())
            U    = gain / (dist ** DIST_POW)
            if U > bestU:
                bestU, best, bestGain = U, i, gain
        if bestGain == 0:
            print(f"  step{step+1}: gain=0 → 종료"); break
        travel = float(np.linalg.norm(cands[best] - cur))
        obs_w += S.soft_weight(cands[best])
        used[best] = True
        cur = cands[best].copy()
        sel.append(cands[best])
        az  = math.degrees(math.atan2(cands[best][1]-target[1], cands[best][0]-target[0])) % 360
        cov = (obs_w >= S.TAU).sum() / P.N_SURF
        print(f"  WP{step+1:02d}: U={bestU:.3f} +{bestGain}vox travel={travel:.1f}m "
              f"az={az:.0f}° softcov={cov*100:.1f}%")

    # Greedy 재정렬 + raycast 평가
    A.RAYCAST_OCCLUSION = True
    if sel:
        order = A.greedy_path(np.array(sel), cam_pos[-1])
        cur2, plen = cam_pos[-1].copy(), 0.0
        for wp in order:
            plen += np.linalg.norm(np.asarray(wp) - cur2); cur2 = np.asarray(wp)
    else:
        order, plen = [], 0.0

    live = np.ones(len(pts), dtype=bool)
    for c in cam_pos:
        _, vis = A.information_gain(c, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    orbit_real = len(pts) - int(live.sum())
    for wp in order:
        _, vis = A.information_gain(np.asarray(wp), target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    final_real = len(pts) - int(live.sum())

    soft_final = int((obs_w >= S.TAU).sum())
    azs = [round(math.degrees(math.atan2(np.asarray(w)[1]-target[1],
                  np.asarray(w)[0]-target[0])) % 360) for w in order]
    print(f"\n[5] 결과:")
    print(f"    보완 WP: {len(order)}개  경로: {plen:.2f}m  az 분포: {azs}")
    print(f"    az 범위(spread): {max(azs)-min(azs) if azs else 0}°")
    print(f"    soft voxel: {100*soft_orbit/P.N_SURF:.1f}% → {100*soft_final/P.N_SURF:.1f}%")
    print(f"    real(raycast): {100*orbit_real/len(pts):.1f}% → {100*final_real/len(pts):.2f}% "
          f"(남은 {len(pts)-final_real}점)")
    print(f"    [참고] hard(theta<70°): {100*hard_orbit/P.N_SURF:.1f}%")

    waypoints = []
    for i, wp in enumerate(order):
        wp = np.asarray(wp)
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(A.tilt_deg_to_target(wp, target), 1),
                          "azimuth_deg": round(math.degrees(
                              math.atan2(wp[1]-target[1], wp[0]-target[0])) % 360, 1)})

    out = {
        "algorithm": f"Orbit + Soft PB-NBV v2 (gain/dist^{DIST_POW}, Lambert cos1, TAU=cos70)",
        "selection": "real new soft-coverage gain / distance (no F-score artifact)",
        "fix_z": fix_z, "tilt_deg": 45, "alpha": S.ALPHA, "tau": round(S.TAU, 4),
        "soft_cov_orbit": round(soft_orbit/P.N_SURF, 4),
        "soft_cov_final": round(soft_final/P.N_SURF, 4),
        "hard_cov_orbit": round(hard_orbit/P.N_SURF, 4),
        "real_cov_orbit": round(orbit_real/len(pts), 4),
        "real_cov_final": round(final_real/len(pts), 4),
        "az_distribution": azs, "az_spread_deg": (max(azs)-min(azs)) if azs else 0,
        "n_waypoints": len(order), "path_length_m": round(plen, 3),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
