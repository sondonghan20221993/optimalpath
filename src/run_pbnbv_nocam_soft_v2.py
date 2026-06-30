"""
run_pbnbv_nocam_soft_v2.py — Case B와 같은 no-camera 조건 + v2 선택기준

목적: Case B(Soft 0.5^r depth-rank, F-score 선택)와 "같은 시나리오"에서
      v2 방식(Lambert cos¹ 누적 + 실제 gain/dist 선택)을 돌려 직접 비교.

공동 조건 (Case B와 동일):
  - no-camera (전체 점 미관측, 맨바닥부터)
  - 다중 고도 후보 z = top-{1..6}m, tilt=45°, MAX_DIST=8m
  - raycast 기준 최종 평가

차이 (= 비교 포인트):
  Case B : 선택 = 점별 0.5^r depth-rank F-score
  v2     : 선택 = 실제 신규 soft-cover gain / dist  (Lambert cos¹, W>=TAU=cos70°)
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
OUT  = ROOT / "results" / "pbnbv_nocam_soft_v2_path.json"

DIST_POW = 1.0
N_AZ     = 36
N_STEPS  = 20
START    = np.array([-27.06, -50.51, -6.13])   # Case B와 동일 출발점


def main():
    target = P.TARGET
    pts    = P._pts_raw
    obj_z_top = pts[:, 2].min()
    z_levels  = [round(obj_z_top - dz, 2) for dz in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

    print(f"[1] no-camera, 다중고도 z={z_levels}, tilt=45°")
    print(f"    Soft Lambert cos^{S.ALPHA:.0f}, TAU=cos70°={S.TAU:.3f}")
    print(f"    선택기준 U = 실제_soft_gain / dist^{DIST_POW}")

    cands = A.gen_candidates_tilt45(target, z_levels, n_az=N_AZ, max_dist=P.MAX_DIST)
    print(f"[2] 후보 {len(cands)}개")

    obs_w = np.zeros(P.N_SURF)   # 맨바닥: 누적 관측 0
    used  = np.zeros(len(cands), dtype=bool)
    sel   = []
    cur   = START.copy()
    print(f"[3] Soft NBV (gain/dist), 출발 ({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f}):")

    for step in range(N_STEPS):
        prev = (obs_w >= S.TAU)
        if int(prev.sum()) >= P.N_SURF:
            print(f"  step{step+1}: soft 100% → 종료"); break
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
              f"az={az:.0f}° z={cands[best][2]:.2f} softcov={cov*100:.1f}%")

    # Greedy 재정렬 + raycast 평가 (Case B와 동일 기준)
    A.RAYCAST_OCCLUSION = True
    if sel:
        order = A.greedy_path(np.array(sel), START)
        cur2, plen = START.copy(), 0.0
        for wp in order:
            plen += np.linalg.norm(np.asarray(wp) - cur2); cur2 = np.asarray(wp)
    else:
        order, plen = [], 0.0

    live = np.ones(len(pts), dtype=bool)
    for wp in order:
        _, vis = A.information_gain(np.asarray(wp), target, pts, live, P.FOV_DEG, P.MAX_DIST)
        live &= ~vis
    final_real = len(pts) - int(live.sum())
    soft_final = int((obs_w >= S.TAU).sum())

    azs = [round(math.degrees(math.atan2(np.asarray(w)[1]-target[1],
                  np.asarray(w)[0]-target[0])) % 360) for w in order]
    print(f"\n[4] 결과:")
    print(f"    WP: {len(order)}개  경로: {plen:.2f}m  az 분포: {azs}")
    print(f"    az 범위(spread): {max(azs)-min(azs) if azs else 0}°")
    print(f"    soft voxel: {100*soft_final/P.N_SURF:.1f}%")
    print(f"    real(raycast): {100*final_real/len(pts):.2f}% (남은 {len(pts)-final_real}점)")

    waypoints = []
    for i, wp in enumerate(order):
        wp = np.asarray(wp)
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(A.tilt_deg_to_target(wp, target), 1),
                          "azimuth_deg": round(math.degrees(
                              math.atan2(wp[1]-target[1], wp[0]-target[0])) % 360, 1)})

    out = {
        "algorithm": f"no-camera + Soft PB-NBV v2 (gain/dist^{DIST_POW}, Lambert cos1, TAU=cos70)",
        "mode": "no_camera_soft_v2",
        "selection": "real new soft-coverage gain / distance",
        "n_points": len(pts), "tilt_deg": 45, "alpha": S.ALPHA, "tau": round(S.TAU, 4),
        "soft_cov_final": round(soft_final/P.N_SURF, 4),
        "real_cov_final": round(final_real/len(pts), 4),
        "az_distribution": azs, "az_spread_deg": (max(azs)-min(azs)) if azs else 0,
        "n_waypoints": len(order), "path_length_m": round(plen, 3),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
