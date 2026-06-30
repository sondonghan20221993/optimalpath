"""
run_pbnbv_orbit45_constrained.py — 일관된 tilt=45° 시스템

앞선 문제: real_test 원본 궤도는 tilt 30~42°, 거리 8m+ (21/34대 MAX_DIST 초과,
           9대 가시 0) → "tilt=45° 궤도"라 부를 수 없고 끝점 handoff 문제 발생.

수정: 궤도도 보완과 동일하게 tilt=45°, 정상 standoff(z=top-3m, 거리 ~4.5m)로 재생성.
  → 궤도 전부 in-range, 끝 카메라도 물체를 봄 → handoff 면제 불필요.
  → 궤도·보완 같은 거리/각도 = 일관된 시스템.

구성:
  1) tilt=45° 원형 궤도 (방위 24개, 균등) — 방위순으로 한 바퀴
  2) 궤도 끝점에서 overlap 제약 constrained PB-NBV 보완 (사각지대만)
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
OUT  = ROOT / "results" / "pbnbv_sparse_constrained_path.json"

ORBIT_N_AZ = 4      # 궤도 방위 수 (균등 원형)
SUPP_N_AZ  = 36      # 보완 후보 방위 수 (더 촘촘하게 infill)
TAU_OVL    = 0.30
DIST_POW   = 1.0
N_STEPS    = 30


def vis_mask(c):
    return S.soft_weight(c) > 0.0


def az_of(p, t):
    return math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360


def main():
    target = P.TARGET
    pts    = P._pts_raw
    fix_z  = round(pts[:, 2].min() - 3.0, 2)

    # ── 1) tilt=45° 정상 궤도 생성 (방위순 정렬) ──
    orbit = A.gen_candidates_tilt45(target, [fix_z], n_az=ORBIT_N_AZ, max_dist=P.MAX_DIST)
    orbit = np.array(sorted(orbit, key=lambda c: az_of(c, target)))   # 방위순 = 원형 비행
    odist = np.linalg.norm(orbit - target, axis=1)
    print(f"[1] tilt=45° 궤도 {len(orbit)}대, z={fix_z}m")
    print(f"    거리: {odist.min():.2f}~{odist.max():.2f}m (모두<=MAX_DIST={P.MAX_DIST}), tilt=45° 일관")

    # 궤도 누적 관측
    obs_w = np.zeros(P.N_SURF)
    for c in orbit:
        obs_w += S.soft_weight(c)
    soft_orbit = int((obs_w >= S.TAU).sum())
    vis_counts = [int((S.soft_weight(c) > 0).sum()) for c in orbit]
    print(f"[2] 궤도 커버(seed): soft {100*soft_orbit/P.N_SURF:.1f}%  "
          f"(가시 voxel 0개 카메라: {sum(1 for v in vis_counts if v==0)}대)")

    # ── 2) 보완 후보 (같은 링, 더 촘촘) ──
    cands = A.gen_candidates_tilt45(target, [fix_z], n_az=SUPP_N_AZ, max_dist=P.MAX_DIST)
    vmask = [vis_mask(c) for c in cands]

    used = np.zeros(len(cands), dtype=bool)
    # 궤도와 겹치는 후보 제외 (같은 위치 재촬영 = soft 이중계산 방지)
    for i, c in enumerate(cands):
        if any(np.allclose(c, o, atol=1e-3) for o in orbit):
            used[i] = True
    print(f"    (궤도 겹침 후보 {int(used.sum())}개 제외)")
    sel, ovl_log = [], []
    cur     = orbit[-1].copy()        # 궤도 마지막(최고 방위)에서 출발
    cur_vis = vis_mask(orbit[-1])     # 이제 in-range → 가시 voxel 있음
    print(f"[3] 출발 = 궤도 끝 az={az_of(orbit[-1],target):.0f}°, "
          f"가시 voxel {int(cur_vis.sum())}개 (handoff 불필요)")

    for step in range(N_STEPS):
        prev = (obs_w >= S.TAU)
        best, bestU, bestGain, bestOvl = -1, -1e18, 0, 0.0
        for i, c in enumerate(cands):
            if used[i]:
                continue
            inter = int((cur_vis & vmask[i]).sum())
            base  = int(cur_vis.sum()) + 1e-9
            ovl   = inter / base
            if ovl < TAU_OVL:
                continue
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
        cur = cands[best].copy(); cur_vis = vmask[best]
        sel.append(cands[best]); ovl_log.append(bestOvl)
        cov = (obs_w >= S.TAU).sum() / P.N_SURF
        print(f"  WP{step+1:02d}: U={bestU:.2f} +{bestGain}vox ovl={bestOvl:.2f} "
              f"travel={travel:.1f}m az={az_of(cands[best],target):.0f}° softcov={cov*100:.1f}%")

    # ── 경로/평가 ──
    plen, cur2 = 0.0, orbit[-1].copy()
    for wp in sel:
        plen += np.linalg.norm(wp - cur2); cur2 = wp
    # 궤도 자체 비행거리(방위순)
    olen, c2 = 0.0, orbit[0].copy()
    for c in orbit[1:]:
        olen += np.linalg.norm(c - c2); c2 = c

    A.RAYCAST_OCCLUSION = True
    live = np.ones(len(pts), dtype=bool)
    for c in orbit:
        _, vis = A.information_gain(c, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    orbit_real = len(pts) - int(live.sum())
    for wp in sel:
        _, vis = A.information_gain(wp, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    final_real = len(pts) - int(live.sum())
    soft_final = int((obs_w >= S.TAU).sum())

    azs = [round(az_of(w, target)) for w in sel]
    daz = [min(abs(azs[i+1]-azs[i]), 360-abs(azs[i+1]-azs[i])) for i in range(len(azs)-1)]
    print(f"\n[4] 결과:")
    print(f"    궤도: {len(orbit)}대, 비행 {olen:.2f}m, soft {100*soft_orbit/P.N_SURF:.1f}% / real {100*orbit_real/len(pts):.1f}%")
    print(f"    보완: {len(sel)} WP, {plen:.2f}m, az {azs}")
    print(f"    overlap: min={min(ovl_log) if ovl_log else 1:.2f} mean={np.mean(ovl_log) if ovl_log else 1:.2f}")
    print(f"    최종: soft {100*soft_final/P.N_SURF:.1f}% / real raycast {100*final_real/len(pts):.2f}% (남은 {len(pts)-final_real}점)")
    print(f"    총 비행: {olen+plen:.2f}m  (궤도 {olen:.1f} + 보완 {plen:.1f})")

    waypoints = [{"idx": i+1, "pos": wp.tolist(),
                  "tilt_deg": round(A.tilt_deg_to_target(wp, target), 1),
                  "azimuth_deg": round(az_of(wp, target), 1),
                  "overlap_with_prev": round(ovl_log[i], 3)} for i, wp in enumerate(sel)]

    out = {
        "algorithm": f"Sparse tilt45 Orbit (demonstrates NBV value) ({ORBIT_N_AZ}) + Constrained PB-NBV (overlap>={TAU_OVL})",
        "note": "orbit regenerated at tilt=45, standoff ~4.5m -> all in-range, no handoff needed",
        "orbit_n_az": ORBIT_N_AZ, "tau_overlap": TAU_OVL, "fix_z": fix_z, "tilt_deg": 45,
        "orbit_dist_min": round(float(odist.min()), 2), "orbit_dist_max": round(float(odist.max()), 2),
        "orbit_flight_m": round(olen, 3),
        "soft_cov_orbit": round(soft_orbit/P.N_SURF, 4), "soft_cov_final": round(soft_final/P.N_SURF, 4),
        "real_cov_orbit": round(orbit_real/len(pts), 4), "real_cov_final": round(final_real/len(pts), 4),
        "az_distribution": azs, "max_adjacent_daz_deg": max(daz) if daz else 0,
        "overlap_min": round(float(min(ovl_log)), 3) if ovl_log else 1.0,
        "overlap_mean": round(float(np.mean(ovl_log)), 3) if ovl_log else 1.0,
        "n_waypoints": len(sel), "supp_path_m": round(plen, 3),
        "total_flight_m": round(olen+plen, 3),
        "orbit_positions": orbit.tolist(), "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
