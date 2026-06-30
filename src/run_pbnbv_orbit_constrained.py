"""
run_pbnbv_orbit_constrained.py — 원형 궤도 후, 궤도 끝점에서 이어가는 constrained PB-NBV

구성 (드론 실제 운용 흐름):
  1) 원형 궤도 34대로 한 바퀴 → 대부분 커버 + 매끄러운 궤적(낮은 ATE)
  2) 궤도 마지막 위치(cam_pos[-1])에서 출발 → 사각지대만 overlap 제약 PB-NBV 보완
     → 궤도 끝과 보완 시작이 연속(overlap 유지) → 전 구간 ATE 안정.

차이점 (run_pbnbv_constrained.py 대비):
  - 출발점 = 궤도 마지막 카메라 (임의 START 아님)
  - obs_w 를 궤도 누적관측으로 seed → 보완은 "궤도가 못 본 곳"만 노림
"""
import sys, types, json, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A
import pbnbv_paper as P
import run_pbnbv_orbit_soft as S

ROOT = HERE.parent
OUT  = ROOT / "results" / "pbnbv_orbit_constrained_path.json"

TAU_OVL  = 0.30
DIST_POW = 1.0
N_AZ     = 36
N_STEPS  = 30


def vis_mask(c):
    return S.soft_weight(c) > 0.0


def az_of(p, t):
    return math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360


def main():
    cam_pos = S.load_orbit_positions()
    target  = P.TARGET
    pts     = P._pts_raw
    fix_z   = round(pts[:, 2].min() - 3.0, 2)

    print(f"[1] 원형 궤도 {len(cam_pos)}대 → 끝점에서 보완 시작")
    print(f"    단일 고도 링 z={fix_z}m, tilt=45°, overlap>={TAU_OVL}, soft cos1 TAU=cos70°={S.TAU:.3f}")

    # 궤도 누적 관측으로 seed
    obs_w = np.zeros(P.N_SURF)
    for c in cam_pos:
        obs_w += S.soft_weight(c)
    soft_orbit = int((obs_w >= S.TAU).sum())
    print(f"[2] 궤도 커버(seed): soft {100*soft_orbit/P.N_SURF:.1f}%")

    cands = A.gen_candidates_tilt45(target, [fix_z], n_az=N_AZ, max_dist=P.MAX_DIST)
    vmask = [vis_mask(c) for c in cands]

    used = np.zeros(len(cands), dtype=bool)
    sel, ovl_log = [], []
    cur     = cam_pos[-1].copy()       # ★ 궤도 마지막 지점에서 출발 (거리 기준점)
    cur_vis = vis_mask(cam_pos[-1])    # 궤도 끝 가시집합 (이 데이터에선 0 = 추적권 밖)
    # 궤도 끝이 물체를 못 보면(가시 0) 첫 시점은 재진입 handoff → overlap 제약 면제
    if int(cur_vis.sum()) == 0:
        cur_vis = None
        print("    [주의] 궤도 끝 카메라 가시 voxel=0(>MAX_DIST) → 첫 보완은 handoff(제약면제)")
    print(f"[3] 출발 = 궤도 끝 ({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f}), az={az_of(cur,target):.0f}°")

    for step in range(N_STEPS):
        prev = (obs_w >= S.TAU)
        best, bestU, bestGain, bestOvl = -1, -1e18, 0, 0.0
        for i, c in enumerate(cands):
            if used[i]:
                continue
            if cur_vis is not None:
                inter = int((cur_vis & vmask[i]).sum())
                base  = int(cur_vis.sum()) + 1e-9
                ovl   = inter / base
                if ovl < TAU_OVL:
                    continue
            else:
                ovl = 1.0
            dist = np.linalg.norm(c - cur) + 1e-6
            nw   = obs_w + S.soft_weight(c)
            gain = int(((nw >= S.TAU) & ~prev).sum())
            U    = gain / (dist ** DIST_POW)
            if U > bestU:
                bestU, best, bestGain, bestOvl = U, i, gain, ovl
        if best < 0:
            print(f"  step{step+1}: 제약 만족 후보 없음 → 종료"); break
        if bestGain == 0:
            print(f"  step{step+1}: gain=0 → 종료"); break
        travel = float(np.linalg.norm(cands[best] - cur))
        obs_w += S.soft_weight(cands[best])
        used[best] = True
        cur = cands[best].copy()
        cur_vis = vmask[best]
        sel.append(cands[best]); ovl_log.append(bestOvl)
        cov = (obs_w >= S.TAU).sum() / P.N_SURF
        print(f"  WP{step+1:02d}: U={bestU:.2f} +{bestGain}vox ovl={bestOvl:.2f} "
              f"travel={travel:.1f}m az={az_of(cands[best],target):.0f}° softcov={cov*100:.1f}%")

    # 경로 = 궤도끝 → 보완 순서 (연속)
    plen, cur2 = 0.0, cam_pos[-1].copy()
    for wp in sel:
        plen += np.linalg.norm(wp - cur2); cur2 = wp

    # raycast 평가 (궤도 + 보완)
    A.RAYCAST_OCCLUSION = True
    live = np.ones(len(pts), dtype=bool)
    for c in cam_pos:
        _, vis = A.information_gain(c, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    orbit_real = len(pts) - int(live.sum())
    for wp in sel:
        _, vis = A.information_gain(wp, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    final_real = len(pts) - int(live.sum())
    soft_final = int((obs_w >= S.TAU).sum())

    azs = [round(az_of(w, target)) for w in sel]
    daz = [min(abs(azs[i+1]-azs[i]), 360-abs(azs[i+1]-azs[i])) for i in range(len(azs)-1)]
    print(f"\n[4] 결과:")
    print(f"    보완 WP: {len(sel)}개  보완경로: {plen:.2f}m  az: {azs}")
    print(f"    궤도끝 az={az_of(cam_pos[-1],target):.0f}° → 첫 보완 az={azs[0] if azs else '-'}°")
    print(f"    overlap: min={min(ovl_log) if ovl_log else 0:.2f} mean={np.mean(ovl_log) if ovl_log else 0:.2f}")
    print(f"    soft voxel: {100*soft_orbit/P.N_SURF:.1f}% → {100*soft_final/P.N_SURF:.1f}%")
    print(f"    real(raycast): {100*orbit_real/len(pts):.1f}% → {100*final_real/len(pts):.2f}% "
          f"(남은 {len(pts)-final_real}점)")

    waypoints = []
    for i, wp in enumerate(sel):
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(A.tilt_deg_to_target(wp, target), 1),
                          "azimuth_deg": round(az_of(wp, target), 1),
                          "overlap_with_prev": round(ovl_log[i], 3)})

    out = {
        "algorithm": f"Orbit-end + Constrained PB-NBV (overlap>={TAU_OVL}, soft cos1, gain/dist)",
        "start": "last orbit camera position",
        "orbit_cams": len(cam_pos),
        "tau_overlap": TAU_OVL, "fix_z": fix_z, "tilt_deg": 45, "tau": round(S.TAU, 4),
        "orbit_end_az": round(az_of(cam_pos[-1], target), 1),
        "soft_cov_orbit": round(soft_orbit/P.N_SURF, 4),
        "soft_cov_final": round(soft_final/P.N_SURF, 4),
        "real_cov_orbit": round(orbit_real/len(pts), 4),
        "real_cov_final": round(final_real/len(pts), 4),
        "az_distribution": azs, "max_adjacent_daz_deg": max(daz) if daz else 0,
        "overlap_min": round(float(min(ovl_log)), 3) if ovl_log else 0,
        "overlap_mean": round(float(np.mean(ovl_log)), 3) if ovl_log else 0,
        "n_waypoints": len(sel), "path_length_m": round(plen, 3),
        "orbit_positions": cam_pos.tolist(),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
