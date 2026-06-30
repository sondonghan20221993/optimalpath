"""
run_pbnbv_orbit_fixedalt.py — 원형 궤도 보완: 고정 고도 3m + PB-NBV + U=F/dist

개선점:
  1) 고도 고정 (z = obj_z_top - 3m): 급격한 고도 변화 제거
  2) 깊이 가중치: PB-NBV ellipsoid 0.5^r → 다른 고도의 점도 거리 rank로 자동 가중
  3) 이동 가중치: U = F(v) / dist(cur_pos→v)  [Bircher 2016]
  4) 입사각 규제: theta < 70° (관측 모델 보수화, raycast 없음)
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
OUT  = ROOT / "results" / "pbnbv_orbit_fixedalt_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

INC_MAX_DEG = 70     # 입사각 규제 (논문 보수화)
FIX_ALT_M   = 3.0   # 물체 위 고정 고도 (m)
N_AZ        = 36
N_STEPS     = 20


def load_orbit_positions():
    pos = []
    for m in META:
        p = json.load(open(m))["camera"]["pose"]["position"]
        pos.append([p["x"], p["y"], p["z"]])
    return np.array(pos)


def observed_by_inc(cam_pos, inc_max_deg):
    cam_dir = P.TARGET - cam_pos; cam_dir /= np.linalg.norm(cam_dir) + 1e-9
    to   = P.SURF_CEN - cam_pos
    dist = np.linalg.norm(to, axis=1)
    in_range = (dist >= P.MIN_DIST) & (dist <= P.MAX_DIST)
    in_fov   = (to * cam_dir).sum(1) / (dist + 1e-9) >= math.cos(math.radians(P.FOV_DEG / 2))
    view = cam_pos - P.SURF_CEN
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
    cos_inc     = (P.SURF_NRM * view).sum(1)
    front_strict = cos_inc >= math.cos(math.radians(inc_max_deg))
    return np.where(in_range & in_fov & front_strict)[0]


def main():
    cam_pos = load_orbit_positions()
    target  = P.TARGET
    pts     = P._pts_raw

    # 고정 고도 계산
    obj_z_top = pts[:, 2].min()
    fix_z     = round(obj_z_top - FIX_ALT_M, 2)
    print(f"[1] 고정 고도: z={fix_z}m ({FIX_ALT_M}m above object top {obj_z_top:.3f}m)")

    # 궤도 커버리지 (규제 입사각 기준)
    observed = np.zeros(P.N_SURF, dtype=bool)
    for c in cam_pos:
        observed[observed_by_inc(c, INC_MAX_DEG)] = True
    n_orbit = int(observed.sum())
    print(f"[2] 궤도 voxel 커버 (theta<{INC_MAX_DEG}°): {n_orbit}/{P.N_SURF} ({100*n_orbit/P.N_SURF:.1f}%)")

    # 고정 고도 후보 생성 (tilt=45° 고정 × 36방위 = 36개)
    cands = A.gen_candidates_tilt45(target, [fix_z], n_az=N_AZ, max_dist=P.MAX_DIST)
    print(f"[3] 고정 고도 후보 (tilt=45°): {len(cands)}개 (z={fix_z}m, {N_AZ}방위)")

    # PB-NBV 보완 루프: U = F(v) / dist  + 깊이 0.5^r 가중 (ellipsoid 내부)
    used    = np.zeros(len(cands), dtype=bool)
    sel     = []
    cur_pos = cam_pos[-1].copy()
    print(f"[4] PB-NBV + U=F/dist (fixed alt {FIX_ALT_M}m, theta<{INC_MAX_DEG}°):")
    print(f"    출발: ({cur_pos[0]:.2f},{cur_pos[1]:.2f},{cur_pos[2]:.2f})")

    for step in range(N_STEPS):
        fro_idx = P.compute_frontier(observed)
        if len(fro_idx) == 0:
            print(f"  step{step+1}: frontier 없음 → 종료"); break
        occ_ell = P.fit_ellipsoids(P.SURF_CEN[observed],  P.SURF_NRM[observed])
        fro_ell = P.fit_ellipsoids(P.SURF_CEN[fro_idx],   P.SURF_NRM[fro_idx])
        # U(v) = F(v) / dist  — 단위 이동거리당 frontier 커버 (F≥0: frontier_only)
        scores = np.array([
            -1e18 if used[i] else
            P.evaluate(c, occ_ell, fro_ell, frontier_only=True) / (np.linalg.norm(c - cur_pos) + 1e-6)
            for i, c in enumerate(cands)
        ])
        best   = int(np.argmax(scores))
        newobs = observed_by_inc(cands[best], INC_MAX_DEG)
        gained = int((~observed[newobs]).sum())
        if gained == 0:
            print(f"  step{step+1}: gain=0 → 종료"); break
        travel = float(np.linalg.norm(cands[best] - cur_pos))
        used[best] = True; observed[newobs] = True
        cur_pos = cands[best].copy()
        sel.append(cands[best])
        cov = observed.sum() / P.N_SURF
        az  = math.degrees(math.atan2(cands[best][1]-target[1], cands[best][0]-target[0])) % 360
        print(f"  WP{step+1:02d}: U={scores[best]:.4f} +{gained}vox "
              f"travel={travel:.1f}m az={az:.0f}° cov={cov*100:.1f}%")

    # Greedy 재정렬
    if sel:
        order = A.greedy_path(np.array(sel), cam_pos[-1])
        cur, plen = cam_pos[-1].copy(), 0.0
        for wp in order:
            plen += np.linalg.norm(np.asarray(wp) - cur); cur = np.asarray(wp)
    else:
        order, plen = [], 0.0

    # 실제 점 raycast 평가
    A.RAYCAST_OCCLUSION = True
    live = np.ones(len(pts), dtype=bool)
    for c in cam_pos:
        _, vis = A.information_gain(c, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    orbit_real = len(pts) - int(live.sum())
    for wp in order:
        _, vis = A.information_gain(np.asarray(wp), target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    final_real = len(pts) - int(live.sum())

    print(f"\n[5] 최종 결과:")
    print(f"    보완 WP: {len(order)}개  경로: {plen:.2f}m  고도: 전부 z={fix_z}m")
    print(f"    실제 커버: {orbit_real}/{len(pts)} ({100*orbit_real/len(pts):.1f}%) → "
          f"{final_real}/{len(pts)} ({100*final_real/len(pts):.1f}%)")

    waypoints = []
    for i, wp in enumerate(order):
        wp = np.asarray(wp)
        t  = A.tilt_deg_to_target(wp, target)
        az = math.degrees(math.atan2(wp[1]-target[1], wp[0]-target[0])) % 360
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(t,1), "azimuth_deg": round(az,1)})
        print(f"  WP{i+1:02d}: ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) tilt={t:.1f}° az={az:.0f}°")

    out = {
        "algorithm": f"Orbit + PB-NBV fixed-alt {FIX_ALT_M}m + U=F/dist + theta<{INC_MAX_DEG}",
        "fix_z": fix_z, "fix_alt_m": FIX_ALT_M, "inc_max_deg": INC_MAX_DEG,
        "orbit_real_cov": round(orbit_real/len(pts), 4),
        "final_real_cov": round(final_real/len(pts), 4),
        "n_waypoints": len(order), "path_length_m": round(plen, 3),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
