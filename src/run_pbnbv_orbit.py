"""
run_pbnbv_orbit.py — 원형 궤도 보완 시나리오 (논문 본문용)

전제: 드론이 표준 원형 궤도(기존 34개 카메라)를 1바퀴 비행했다고 가정.
      그 후 raycast 기준으로 '한 번도 못 본' 사각지대(오목·자기가림)를
      PB-NBV(2)+Greedy 보완 경로로 추가 촬영한다.

공동 통제조건: tilt=45° 후보, MAX_DIST=8m, raycast occlusion, FOV=89.9°
"""
import sys, types, json, glob, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A

A.RAYCAST_OCCLUSION = True   # 원형 궤도의 실제 가시성도 raycast 기준

ROOT = HERE.parent
NPZ  = ROOT / "real_test" / "real_test_pts_normals.npz"
META = sorted((ROOT / "real_test" / "meta").glob("*.json"))
OUT  = ROOT / "results" / "pbnbv_orbit_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

IG_THRESH = 5
N_SELECT  = 20
N_AZ      = 36


def quat_to_R(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def load_orbit_cameras():
    pos, fwd, fovs = [], [], []
    for m in META:
        j = json.load(open(m))
        p = j["camera"]["pose"]["position"]
        o = j["camera"]["pose"]["orientation"]
        R = quat_to_R(o["w"], o["x"], o["y"], o["z"])
        pos.append([p["x"], p["y"], p["z"]])
        fwd.append(R @ np.array([1.0, 0.0, 0.0]))   # AirSim body X = forward
        fovs.append(j["camera"]["fov"])
    return np.array(pos), np.array(fwd), float(np.mean(fovs))


def main():
    np.random.seed(42)

    # 1) 원형 궤도 카메라
    cam_pos, cam_fwd, fov_raw = load_orbit_cameras()
    fov_deg = min(fov_raw, A.MAX_FOV_DEG)
    d   = np.load(NPZ)
    pts = d["points"].astype(float)
    nrm = d["normals"].astype(float)
    target = pts.mean(axis=0)
    to_out = pts - target
    nrm[(nrm*to_out).sum(1) < 0] *= -1.0

    rad = np.linalg.norm(cam_pos[:, :2] - target[:2], axis=1)
    az  = np.degrees(np.arctan2(cam_pos[:,1]-target[1], cam_pos[:,0]-target[0])) % 360
    print(f"[1] 원형 궤도: {len(cam_pos)}개 카메라, 반경 {rad.mean():.1f}m(±{rad.std():.2f}), "
          f"방위 {az.min():.0f}~{az.max():.0f}°, FOV={fov_deg:.1f}°")
    print(f"    target=({target[0]:.2f},{target[1]:.2f},{target[2]:.2f}), 점 {len(pts):,}개")

    # 2) 원형 궤도 커버리지 (raycast 기준) → 사각지대 검출
    coverage = A.compute_coverage(pts, cam_pos, cam_fwd, fov_deg, A.MAX_DIST,
                                  pts_normals=nrm)
    underobs_mask = coverage == 0   # 한 번도 못 본 점 = 진짜 사각지대
    n_orbit_cov = int((coverage > 0).sum())
    print(f"[2] 원형 궤도 커버리지(raycast): {n_orbit_cov}/{len(pts)} "
          f"({100*n_orbit_cov/len(pts):.1f}%)")
    print(f"    사각지대(미관측): {underobs_mask.sum()}개 ({100*underobs_mask.mean():.1f}%)")
    if underobs_mask.sum() == 0:
        print("    사각지대 없음 → 보완 불필요"); return

    # 3) tilt=45° 후보 생성 (공동 통제조건)
    obj_z_top = pts[:,2].min()
    z_levels  = [round(obj_z_top - dz, 2) for dz in [1.0,2.0,3.0,4.0,5.0,6.0]]
    candidates = A.gen_candidates_tilt45(target, z_levels, n_az=N_AZ, max_dist=A.MAX_DIST)
    print(f"[3] 보완 후보 {len(candidates)}개 (tilt=45°, dist≤{A.MAX_DIST}m)")

    # 4) PB-NBV(2) + 거리 가중 보완 시점 선택 (사각지대 대상)
    # Utility U(v) = IG(v) / dist(cur_pos→v): 단위 이동거리당 정보이득 최대화
    # 근거: Bircher et al. (2016) Receding Horizon NBV, Selin et al. (2019)
    used      = np.zeros(len(candidates), dtype=bool)
    live_mask = underobs_mask.copy()
    selected_pts, sel_scores = [], []
    cur_pos = cam_pos[-1].copy()   # 궤도 마지막 카메라 위치에서 출발
    print(f"[4] PB-NBV(2)+DistWeight 보완 시점 선택 (IG>{IG_THRESH}):")
    print(f"    출발 위치: ({cur_pos[0]:.2f},{cur_pos[1]:.2f},{cur_pos[2]:.2f})")
    for step in range(min(N_SELECT, len(candidates))):
        pb = np.array([
            A.pbnbv_score_dist_weighted(candidates[i], candidates, target, pts, live_mask,
                          fov_deg, A.MAX_DIST, A.LOOKAHEAD, cur_pos, pts_normals=nrm)
            if not used[i] else -np.inf for i in range(len(candidates))], dtype=float)
        best = int(np.argmax(pb))
        ig, vis = A.information_gain(candidates[best], target, pts, live_mask,
                                     fov_deg, A.MAX_DIST, pts_normals=nrm)
        if ig <= IG_THRESH:
            print(f"  step{step+1}: IG={ig} <= {IG_THRESH} → 종료"); break
        travel = float(np.linalg.norm(candidates[best] - cur_pos))
        live_mask &= ~vis
        selected_pts.append(candidates[best]); sel_scores.append(float(pb[best]))
        used[best] = True
        cur_pos = candidates[best].copy()   # 위치 업데이트
        print(f"  WP{step+1:02d}: score={pb[best]:.3f} IG={ig} travel={travel:.1f}m 남은사각={live_mask.sum()} z={candidates[best][2]:.2f}")

    # 5) Salvage
    used_mask = np.zeros(len(candidates), dtype=bool)
    for i, c in enumerate(candidates):
        if any(np.allclose(c, s) for s in selected_pts): used_mask[i] = True
    salvage_added, live_after = A.salvage_coverage(candidates, used_mask, target, pts,
                                                   live_mask, fov_deg, A.MAX_DIST, pts_normals=nrm)
    salvage_pts = [p for p, _ in salvage_added]
    if salvage_pts:
        print(f"[5] Salvage {len(salvage_pts)}개 추가")
    live_mask = live_after

    # 6) 최종 커버리지 (궤도 + 보완)
    n_under   = int(underobs_mask.sum())
    n_filled  = n_under - int(live_mask.sum())
    final_cov = n_orbit_cov + n_filled
    print(f"[6] 보완 결과: 사각지대 {n_filled}/{n_under} 커버")
    print(f"    최종 커버리지: {final_cov}/{len(pts)} ({100*final_cov/len(pts):.1f}%) "
          f"[궤도 {100*n_orbit_cov/len(pts):.1f}% → +{100*n_filled/len(pts):.1f}%p]")

    # 7) Greedy NN 경로 (마지막 궤도 카메라에서 출발)
    all_pts    = selected_pts + salvage_pts
    all_phases = ["pbnbv"]*len(selected_pts) + ["salvage"]*len(salvage_pts)
    start = cam_pos[-1]
    path_order = A.greedy_path(np.array(all_pts), start) if all_pts else []
    path_phases = []
    for wp in path_order:
        for orig, ph in zip(all_pts, all_phases):
            if np.allclose(np.asarray(wp), orig): path_phases.append(ph); break
        else: path_phases.append("unknown")

    cur, plen = start.copy(), 0.0
    def azf(p): return math.degrees(math.atan2(p[1]-target[1], p[0]-target[0]))%360
    waypoints = []
    print(f"[7] Greedy 보완 경로 ({len(path_order)}개 WP):")
    for i, (wp, ph) in enumerate(zip(path_order, path_phases)):
        wp = np.asarray(wp); plen += np.linalg.norm(wp-cur); cur = wp
        t = A.tilt_deg_to_target(wp, target)
        waypoints.append({"idx": i+1, "pos": wp.tolist(), "tilt_deg": round(t,1),
                          "azimuth_deg": round(azf(wp),1), "phase": ph})
        tag = "[SV]" if ph=="salvage" else "    "
        print(f"  WP{i+1:02d}{tag} ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) tilt={t:.1f}° az={azf(wp):.0f}°")
    print(f"    보완 경로 길이: {plen:.2f}m")

    out = {
        "algorithm": "Orbit + supplementary NBV(2-step) + Greedy",
        "scenario": "circular_orbit_then_fill_blindspots",
        "n_orbit_cameras": int(len(cam_pos)),
        "orbit_radius_mean": round(float(rad.mean()),2),
        "n_points": int(len(pts)),
        "orbit_coverage_points": n_orbit_cov,
        "orbit_coverage_ratio": round(n_orbit_cov/len(pts),4),
        "blindspot_points": n_under,
        "blindspot_filled": n_filled,
        "final_coverage_points": final_cov,
        "final_coverage_ratio": round(final_cov/len(pts),4),
        "supplementary_path_length_m": round(plen,3),
        "pbnbv_count": len(selected_pts),
        "salvage_count": len(salvage_pts),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT,"w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
