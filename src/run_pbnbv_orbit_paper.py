"""
run_pbnbv_orbit_paper.py — 원형 궤도 보완 시나리오, 논문 PB-NBV 방식 (raycast 없음)

전제: 표준 원형 궤도(34개 카메라) 1바퀴 비행 후,
      논문 PB-NBV(frontier/occupied voxel + GMM ellipsoid + 0.5^r 투영)로
      남은 frontier를 보완 촬영한다. occlusion은 raycast가 아닌 0.5^r 근사.

평가: voxel 기준(논문 내부) + 실제 점 raycast 기준(공정 비교) 둘 다 보고.
"""
import sys, types, json, glob, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A
import pbnbv_paper as P   # voxel/frontier/ellipsoid machinery 재사용

ROOT = HERE.parent
META = sorted((ROOT / "real_test" / "meta").glob("*.json"))
OUT  = ROOT / "results" / "pbnbv_orbit_paper_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

N_STEPS = 20
START   = np.array([-27.06, -50.51, -6.13])


def load_orbit_positions():
    pos = []
    for m in META:
        p = json.load(open(m))["camera"]["pose"]["position"]
        pos.append([p["x"], p["y"], p["z"]])
    return np.array(pos)


def main():
    cam_pos = load_orbit_positions()
    target  = P.TARGET
    rad = np.linalg.norm(cam_pos[:, :2] - target[:2], axis=1)
    print(f"[1] 원형 궤도 {len(cam_pos)}개 카메라, 반경 {rad.mean():.1f}m(±{rad.std():.2f})")
    print(f"    논문 voxel 모델: surface voxel {P.N_SURF}개 (raycast 없음)")

    # 2) 원형 궤도가 관측한 voxel 마킹 (논문 observed_by = in_range&in_fov&front, NO raycast)
    observed = np.zeros(P.N_SURF, dtype=bool)
    for c in cam_pos:
        observed[P.observed_by(c)] = True
    n_orbit = int(observed.sum())
    print(f"[2] 원형 궤도 voxel 커버리지(논문 모델): {n_orbit}/{P.N_SURF} "
          f"({100*n_orbit/P.N_SURF:.1f}%)")
    print(f"    → 논문 모델은 가림을 0.5^r로만 근사하므로 오목부도 '관측됨'으로 마킹될 수 있음")

    # 3) PB-NBV 보완 루프 (frontier → ellipsoid 투영 F점수 + 거리 가중치)
    # Utility: U(v) = F(v) / dist(cur_pos → v)  [Bircher 2016 스타일]
    cands = P.make_candidates()
    print(f"[3] 보완 후보 {len(cands)}개 (tilt=45°, dist≤{P.MAX_DIST}m)")
    used    = np.zeros(len(cands), dtype=bool)
    path    = []
    cur_pos = cam_pos[-1].copy()   # 궤도 마지막 카메라 위치에서 출발
    print(f"[4] 논문 PB-NBV + DistWeight 보완 시점 선택:")
    print(f"    출발 위치: ({cur_pos[0]:.2f},{cur_pos[1]:.2f},{cur_pos[2]:.2f})")
    for step in range(N_STEPS):
        fro_idx = P.compute_frontier(observed)
        if len(fro_idx) == 0:
            print(f"  step{step+1}: frontier 없음 → 종료"); break
        occ_pts = P.SURF_CEN[observed];  occ_nrm = P.SURF_NRM[observed]
        fro_pts = P.SURF_CEN[fro_idx];   fro_nrm = P.SURF_NRM[fro_idx]
        occ_ell = P.fit_ellipsoids(occ_pts, occ_nrm)
        fro_ell = P.fit_ellipsoids(fro_pts, fro_nrm)
        # U(v) = F(v) / dist — 단위 이동거리당 frontier 커버 최대화 (F≥0: frontier_only)
        scores = np.array([
            -1e18 if used[i] else
            P.evaluate(c, occ_ell, fro_ell, frontier_only=True) / (np.linalg.norm(c - cur_pos) + 1e-6)
            for i, c in enumerate(cands)
        ])
        best = int(np.argmax(scores))
        newobs = P.observed_by(cands[best])
        gained = int((~observed[newobs]).sum())
        if gained == 0:
            print(f"  step{step+1}: 최고 후보 gain=0 → 종료"); break
        travel = float(np.linalg.norm(cands[best] - cur_pos))
        used[best]  = True
        observed[newobs] = True
        cur_pos = cands[best].copy()   # 위치 업데이트
        cov = observed.sum() / P.N_SURF
        az = math.degrees(math.atan2(cands[best][1]-target[1], cands[best][0]-target[0])) % 360
        path.append({"step": step+1, "pos": cands[best].tolist(),
                     "azimuth": round(az,1), "U": round(float(scores[best]),4),
                     "gained": gained, "coverage_voxel": round(cov,4),
                     "n_frontier": int(len(fro_idx))})
        print(f"  WP{step+1:02d}: U={scores[best]:.4f} +{gained}voxel "
              f"travel={travel:.1f}m frontier={len(fro_idx)} cov={cov*100:.1f}% z={cands[best][2]:.2f}")

    n_final = int(observed.sum())
    print(f"[5] 보완 후 voxel 커버리지: {n_final}/{P.N_SURF} ({100*n_final/P.N_SURF:.1f}%) "
          f"[궤도 {100*n_orbit/P.N_SURF:.1f}% → +{100*(n_final-n_orbit)/P.N_SURF:.1f}%p]")

    # 6) Greedy 재정렬 + 실제 점 raycast 평가 (공정 비교)
    sel = [np.array(p["pos"]) for p in path]
    if sel:
        order = A.greedy_path(np.array(sel), START)
        cur, plen = START.copy(), 0.0
        for wp in order:
            plen += np.linalg.norm(np.asarray(wp)-cur); cur = np.asarray(wp)
    else:
        order, plen = [], 0.0

    # 실제 점 + raycast: 궤도 단독 vs 궤도+보완
    A.RAYCAST_OCCLUSION = True
    pts = P._pts_raw
    # 궤도 단독 raycast 커버
    live = np.ones(len(pts), dtype=bool)
    for c in cam_pos:
        _, vis = A.information_gain(c, target, pts, live, P.FOV_DEG, P.MAX_DIST)
        live &= ~vis
    orbit_real = len(pts) - int(live.sum())
    # 보완 추가
    for wp in order:
        _, vis = A.information_gain(np.asarray(wp), target, pts, live, P.FOV_DEG, P.MAX_DIST)
        live &= ~vis
    final_real = len(pts) - int(live.sum())
    print(f"[6] 실제 점 raycast 평가 (공정):")
    print(f"    궤도 단독:   {orbit_real}/{len(pts)} ({100*orbit_real/len(pts):.1f}%)")
    print(f"    궤도+보완:   {final_real}/{len(pts)} ({100*final_real/len(pts):.1f}%) "
          f"[+{100*(final_real-orbit_real)/len(pts):.1f}%p]")
    print(f"    보완 WP {len(order)}개, 경로 {plen:.2f}m")

    def azf(p): return math.degrees(math.atan2(p[1]-target[1], p[0]-target[0]))%360
    waypoints = []
    for i, wp in enumerate(order):
        wp = np.asarray(wp); t = A.tilt_deg_to_target(wp, target)
        waypoints.append({"idx": i+1, "pos": wp.tolist(), "tilt_deg": round(t,1),
                          "azimuth_deg": round(azf(wp),1)})

    out = {
        "algorithm": "Orbit + supplementary PB-NBV (paper: ellipsoid 0.5^r, NO raycast) + Greedy",
        "scenario": "circular_orbit_then_paper_nbv",
        "occlusion_model": "paper_0.5^r_projection (no raycast)",
        "n_orbit_cameras": int(len(cam_pos)),
        "voxel_coverage_orbit": round(n_orbit/P.N_SURF,4),
        "voxel_coverage_final": round(n_final/P.N_SURF,4),
        "realpoint_coverage_orbit": round(orbit_real/len(pts),4),
        "realpoint_coverage_final": round(final_real/len(pts),4),
        "supplementary_count": len(order),
        "supplementary_path_length_m": round(plen,3),
        "path": path,
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT,"w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
