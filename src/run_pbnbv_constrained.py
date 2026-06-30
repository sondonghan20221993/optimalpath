"""
run_pbnbv_constrained.py — Overlap 제약을 건 PB-NBV (SLAM/ATE 친화 + 연속 경로)

동기:
  순수 NBV는 시점 수를 최소화 → 큰 점프 → 프레임 overlap 소실 → SLAM 추적 끊김
  → ATE 폭증. 그래서 "tracking을 유지하는 시점들 중에서만" 다음 best view를 고름.

제약 (NBV 안에 내장, 여전히 정보이득 기반):
  연속 시점 overlap(cur, cand) = |vis(cur) ∩ vis(cand)| / |vis(cur)| >= TAU_OVL
  → 텔레포트 금지 → 연속/매끄러운 경로 → 낮은 ATE.

척도: soft Lambert cos¹ 누적, W>=cos70° (각 면을 정면에 가깝게 1번은 봐야 함)
  → 위 1장으로 안 끝나므로, overlap 제약과 결합 시 물체를 한 바퀴 도는 연속 경로가 됨.

단일 고도 링(z=top-3m, tilt=45°, 36방위) → 자유도는 방위각뿐 → 연속 방위 전진 = 원형.
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
OUT  = ROOT / "results" / "pbnbv_constrained_path.json"

TAU_OVL  = 0.30      # 연속 시점 최소 overlap (ATE 친화 제약)
DIST_POW = 1.0
N_AZ     = 36
N_STEPS  = 30
START    = np.array([-27.06, -50.51, -6.13])


def vis_mask(c):
    """후보 c 가 보는 surface voxel 마스크 (soft weight>0)."""
    return S.soft_weight(c) > 0.0


def az_of(p, t):
    return math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360


def main():
    target = P.TARGET
    pts    = P._pts_raw
    fix_z  = round(pts[:, 2].min() - 3.0, 2)   # 단일 고도 링 (3m 위)

    print(f"[1] 단일 고도 링 z={fix_z}m, tilt=45°, {N_AZ}방위")
    print(f"    overlap 제약 TAU_OVL={TAU_OVL}, soft Lambert cos1, TAU=cos70°={S.TAU:.3f}")
    print(f"    선택: argmax soft_gain/dist  s.t. overlap(cur,cand)>=TAU_OVL")

    cands = A.gen_candidates_tilt45(target, [fix_z], n_az=N_AZ, max_dist=P.MAX_DIST)
    vmask = [vis_mask(c) for c in cands]
    print(f"[2] 후보 {len(cands)}개")

    obs_w = np.zeros(P.N_SURF)
    used  = np.zeros(len(cands), dtype=bool)
    sel, ovl_log = [], []
    cur = START.copy()
    cur_vis = None
    print(f"[3] Constrained NBV, 출발 ({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f}):")

    for step in range(N_STEPS):
        prev = (obs_w >= S.TAU)
        best, bestU, bestGain, bestOvl = -1, -1e18, 0, 0.0
        for i, c in enumerate(cands):
            if used[i]:
                continue
            # overlap 제약 (첫 스텝은 제약 없음)
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

    # 경로는 선택 순서 그대로 (연속 제약으로 이미 정렬됨) — Greedy 재정렬 안 함
    plen, cur2 = 0.0, START.copy()
    for wp in sel:
        plen += np.linalg.norm(wp - cur2); cur2 = wp

    # raycast 실제 점 평가
    A.RAYCAST_OCCLUSION = True
    live = np.ones(len(pts), dtype=bool)
    for wp in sel:
        _, vis = A.information_gain(wp, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    final_real = len(pts) - int(live.sum())
    soft_final = int((obs_w >= S.TAU).sum())

    azs = [round(az_of(w, target)) for w in sel]
    # 연속성 지표: 인접 WP 간 방위각 변화 최대치
    daz = [min(abs(azs[i+1]-azs[i]), 360-abs(azs[i+1]-azs[i])) for i in range(len(azs)-1)]
    print(f"\n[4] 결과:")
    print(f"    WP: {len(sel)}개  경로(선택순): {plen:.2f}m  az: {azs}")
    print(f"    연속성: 인접 방위변화 max={max(daz) if daz else 0}° (작을수록 매끄러움)")
    print(f"    overlap: min={min(ovl_log) if ovl_log else 0:.2f} mean={np.mean(ovl_log) if ovl_log else 0:.2f}")
    print(f"    soft voxel: {100*soft_final/P.N_SURF:.1f}%")
    print(f"    real(raycast): {100*final_real/len(pts):.2f}% (남은 {len(pts)-final_real}점)")

    waypoints = []
    for i, wp in enumerate(sel):
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(A.tilt_deg_to_target(wp, target), 1),
                          "azimuth_deg": round(az_of(wp, target), 1),
                          "overlap_with_prev": round(ovl_log[i], 3)})

    out = {
        "algorithm": f"Constrained PB-NBV (overlap>={TAU_OVL}, soft Lambert cos1, gain/dist)",
        "rationale": "overlap constraint keeps frame tracking -> low ATE; still info-gain driven",
        "tau_overlap": TAU_OVL, "fix_z": fix_z, "tilt_deg": 45, "tau": round(S.TAU, 4),
        "soft_cov_final": round(soft_final/P.N_SURF, 4),
        "real_cov_final": round(final_real/len(pts), 4),
        "az_distribution": azs, "max_adjacent_daz_deg": max(daz) if daz else 0,
        "overlap_min": round(float(min(ovl_log)), 3) if ovl_log else 0,
        "overlap_mean": round(float(np.mean(ovl_log)), 3) if ovl_log else 0,
        "n_waypoints": len(sel), "path_length_m": round(plen, 3),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
