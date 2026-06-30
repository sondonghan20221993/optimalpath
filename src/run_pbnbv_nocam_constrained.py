"""
run_pbnbv_nocam_constrained.py — orbit 없는 순수 PB-NBV + overlap 제약 (ATE 안전)

run_pbnbv_nocam_soft_v2.py 와 동일한 시나리오(궤도 없음, 고정 출발점, 다중고도
tilt=45° 후보, Lambert cos¹, TAU=cos70°, U=gain/dist)에 **overlap 제약**을 추가.

추가된 제약 (ATE 안전장치):
  직전 viewpoint 가시 voxel과 후보 viewpoint 가시 voxel이
  최소 TAU_OVL(=0.30) 이상 겹쳐야 후보로 인정.
  → 연속 프레임 간 공통 표면 보장 → SLAM 추적 끊김(ATE 악화) 방지.
  → 경로가 들쑥날쑥하지 않고 매끄럽게 이어짐.

선택 순서 = 비행 순서 (greedy 재정렬 안 함 — 재정렬하면 overlap 체인이 깨짐).
출발점 가시 voxel=0(거리초과)면 첫 스텝만 제약 면제(handoff).
"""
import sys, types, json, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A
import pbnbv_paper as P
import run_pbnbv_orbit_soft as S   # soft_weight, TAU, ALPHA 재사용

ROOT = HERE.parent
OUT  = ROOT / "results" / "pbnbv_nocam_constrained_path.json"

DIST_POW = 1.0
N_AZ     = 36
N_STEPS  = 20
TAU_OVL  = 0.30                                # overlap 제약 (ATE 안전)
ALT_LEVELS = [7.0, 3.0]                         # 고도 2단계 (물체 top 위 7m, 3m)
MAX_DIST   = 10.5                               # 고도7m(tilt45 standoff≈9.9m) 수용 위해 상향
INC_LIMIT_DEG = 60                             # 입사각 임계 (70°→60° 강화, grazing 배제)
START    = np.array([-27.06, -50.51, -6.13])   # 고정 출발점


def vis_mask(c):
    return S.soft_weight(c) > 0.0


def az_of(p, t):
    return math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360


def main():
    target = P.TARGET
    pts    = P._pts_raw
    P.MAX_DIST = MAX_DIST          # soft_weight(_geom_gate)도 동일 MAX_DIST 적용
    S.TAU = math.cos(math.radians(INC_LIMIT_DEG))   # 입사각 임계 강화 (70°→60°)
    obj_z_top = pts[:, 2].min()
    z_levels  = [round(obj_z_top - dz, 2) for dz in ALT_LEVELS]

    print(f"[1] no-orbit + overlap 제약, 고도 2단계 {ALT_LEVELS}m → z={z_levels}, tilt=45°")
    print(f"    Soft Lambert cos^{S.ALPHA:.0f}, 입사각임계 {INC_LIMIT_DEG}° (TAU={S.TAU:.3f}), U=gain/dist^{DIST_POW}")
    print(f"    overlap 제약: 직전뷰 겹침 >= {TAU_OVL}  (ATE 안전), MAX_DIST={MAX_DIST}m")

    cands = A.gen_candidates_tilt45(target, z_levels, n_az=N_AZ, max_dist=MAX_DIST)
    vmask = [vis_mask(c) for c in cands]
    print(f"[2] 후보 {len(cands)}개")

    # ── 관측가능 표면 정의 (제약 수정의 핵심) ──
    # 지면 위 tilt=45 후보 '전부'를 합쳐도 W<TAU 인 voxel = 항공으로 원천 불가(밑면).
    # 커버리지는 관측가능 표면 기준으로만 측정 (지면 물체 밑면은 out-of-scope).
    obs_all = np.zeros(P.N_SURF)
    for c in cands:
        obs_all += S.soft_weight(c)
    observable = obs_all >= S.TAU
    n_obs = int(observable.sum())
    n_under = P.N_SURF - n_obs
    print(f"    관측가능 표면 {n_obs}/{P.N_SURF}  "
          f"(물리적 사각=밑면 {n_under}개: 지면 위 tilt45로 원천 불가)")

    obs_w = np.zeros(P.N_SURF)
    used  = np.zeros(len(cands), dtype=bool)
    sel, ovl_log = [], []
    cur     = START.copy()
    cur_vis = vis_mask(START)
    if int(cur_vis.sum()) == 0:
        cur_vis = None
        print(f"[3] 출발 ({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f}) 가시 voxel=0 "
              f"(거리초과) → 첫 스텝 제약 면제(handoff)")
    else:
        print(f"[3] 출발 ({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f}) "
              f"가시 voxel {int(cur_vis.sum())}개")

    for step in range(N_STEPS):
        prev = (obs_w >= S.TAU)
        if int((prev & observable).sum()) >= n_obs:
            print(f"  step{step+1}: 관측가능 표면 100% → 종료"); break
        best, bestU, bestGain, bestOvl = -1, -1e18, 0, 1.0
        for i, c in enumerate(cands):
            if used[i]:
                continue
            # overlap 제약 (cur_vis=None 이면 첫 스텝 면제)
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
        cov = ((obs_w >= S.TAU) & observable).sum() / n_obs
        print(f"  WP{step+1:02d}: U={bestU:.3f} +{bestGain}vox ovl={bestOvl:.2f} "
              f"travel={travel:.1f}m az={az_of(cands[best],target):.0f}° "
              f"z={cands[best][2]:.2f} softcov={cov*100:.1f}%")

    # 경로 = 선택 순서 그대로 (재정렬 안 함 → overlap 체인 유지)
    plen, cur2 = 0.0, START.copy()
    for wp in sel:
        plen += np.linalg.norm(wp - cur2); cur2 = wp

    A.RAYCAST_OCCLUSION = True
    live = np.ones(len(pts), dtype=bool)
    for wp in sel:
        _, vis = A.information_gain(np.asarray(wp), target, pts, live, P.FOV_DEG, MAX_DIST)
        live &= ~vis
    final_real = len(pts) - int(live.sum())
    soft_final = int((obs_w >= S.TAU).sum())
    soft_obs   = int(((obs_w >= S.TAU) & observable).sum())   # 관측가능 표면 중 커버

    azs = [round(az_of(w, target)) for w in sel]
    daz = [min(abs(azs[i+1]-azs[i]), 360-abs(azs[i+1]-azs[i])) for i in range(len(azs)-1)]
    print(f"\n[4] 결과:")
    print(f"    WP: {len(sel)}개  경로: {plen:.2f}m  az 분포: {azs}")
    print(f"    인접 az 점프 최대: {max(daz) if daz else 0}°")
    print(f"    overlap: min={min(ovl_log) if ovl_log else 1:.2f} mean={np.mean(ovl_log) if ovl_log else 1:.2f}")
    print(f"    ── 커버리지(관측가능 표면 기준) ──")
    print(f"    관측가능 표면: {soft_obs}/{n_obs} = {100*soft_obs/n_obs:.1f}%  "
          f"(밑면 {n_under}개 = 물리적 사각, out-of-scope)")
    print(f"    전체 표면 대비: soft {100*soft_final/P.N_SURF:.1f}%  /  real(raycast) {100*final_real/len(pts):.1f}%")

    waypoints = [{"idx": i+1, "pos": wp.tolist(),
                  "tilt_deg": round(A.tilt_deg_to_target(wp, target), 1),
                  "azimuth_deg": round(az_of(wp, target), 1),
                  "overlap_with_prev": round(ovl_log[i], 3)} for i, wp in enumerate(sel)]

    out = {
        "algorithm": f"no-orbit + Constrained PB-NBV (gain/dist^{DIST_POW}, "
                     f"Lambert cos1, TAU=cos70, overlap>={TAU_OVL})",
        "mode": "no_orbit_constrained",
        "selection": "soft-coverage gain / distance, overlap-filtered",
        "n_points": len(pts), "tilt_deg": 45, "alpha": S.ALPHA, "tau": round(S.TAU, 4),
        "inc_limit_deg": INC_LIMIT_DEG, "tau_overlap": TAU_OVL,
        "start_pos": START.tolist(),
        "n_surf": P.N_SURF, "n_observable": n_obs, "n_underside": n_under,
        "obs_surface_cov": round(soft_obs/n_obs, 4),
        "soft_cov_final": round(soft_final/P.N_SURF, 4),
        "real_cov_final": round(final_real/len(pts), 4),
        "az_distribution": azs, "az_spread_deg": (max(azs)-min(azs)) if azs else 0,
        "max_adjacent_daz_deg": max(daz) if daz else 0,
        "overlap_min": round(float(min(ovl_log)), 3) if ovl_log else 1.0,
        "overlap_mean": round(float(np.mean(ovl_log)), 3) if ovl_log else 1.0,
        "n_waypoints": len(sel), "path_length_m": round(plen, 3),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
