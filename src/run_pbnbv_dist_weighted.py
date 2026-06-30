"""
run_pbnbv_dist_weighted.py — Case D: Hard Raycast + 거리 가중 PB-NBV(2) + Greedy

Case 3와 동일하지만 후보 스코어링 시 현재 위치 → 후보 거리를 나눠
이동 효율(정보이득/이동거리)을 최대화하도록 선택.
score = (IG1 + 0.5 * max(IG2)) / dist(cur_pos -> candidate)
"""
import sys, types, json, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A

A.RAYCAST_OCCLUSION = True

ROOT = HERE.parent
NPZ  = ROOT / "real_test" / "real_test_pts_normals.npz"
OUT  = ROOT / "results" / "pbnbv_dist_weighted_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

FOV_DEG   = 89.9
IG_THRESH = 5
N_SELECT  = 20
N_AZ      = 36

START = np.array([-27.06, -50.51, -6.13])   # 기존 시작점 (viz와 동일)


def main():
    np.random.seed(42)

    d   = np.load(NPZ)
    pts = d["points"].astype(float)
    print(f"[1] 점구름 {len(pts):,}개")

    underobs_mask = np.ones(len(pts), dtype=bool)
    target        = pts.mean(axis=0)
    fov_deg       = min(FOV_DEG, A.MAX_FOV_DEG)
    print(f"[2] 전체 {underobs_mask.sum()}개 미관측, 시작 위치: {START}")

    obj_z_top  = pts[:,2].min()
    z_levels   = [round(obj_z_top - dz, 2) for dz in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    candidates = A.gen_candidates_tilt45(target, z_levels, n_az=N_AZ, max_dist=A.MAX_DIST)
    print(f"[3] 후보 {len(candidates)}개 (tilt=45°, dist≤{A.MAX_DIST}m)")

    used       = np.zeros(len(candidates), dtype=bool)
    live_mask  = underobs_mask.copy()
    selected_pts, sel_scores = [], []
    cur_pos    = START.copy()   # 현재 드론 위치 추적

    print(f"[4] [PB-NBV(2)+DistWeight] 시점 선택 (IG > {IG_THRESH}):")
    for step in range(min(N_SELECT, len(candidates))):
        pb_scores = np.array([
            A.pbnbv_score_dist_weighted(
                candidates[i], candidates, target, pts, live_mask,
                fov_deg, A.MAX_DIST, A.LOOKAHEAD, cur_pos, pts_normals=None)
            if not used[i] else -np.inf
            for i in range(len(candidates))
        ], dtype=float)

        best     = int(np.argmax(pb_scores))
        pb_score = float(pb_scores[best])

        ig_raw, vis = A.information_gain(candidates[best], target, pts, live_mask,
                                         fov_deg, A.MAX_DIST, pts_normals=None)
        if ig_raw <= IG_THRESH:
            print(f"  step{step+1}: raw-IG={ig_raw} <= {IG_THRESH} → 종료")
            break

        travel = float(np.linalg.norm(candidates[best] - cur_pos))
        live_mask &= ~vis
        selected_pts.append(candidates[best])
        sel_scores.append(pb_score)
        used[best] = True
        cur_pos = candidates[best].copy()   # 위치 업데이트

        print(f"  WP{step+1:02d}: score={pb_score:.3f}  raw-IG={ig_raw}  "
              f"travel={travel:.1f}m  남은={live_mask.sum()}  z={candidates[best][2]:.2f}")

    total   = underobs_mask.sum()
    covered = total - int(live_mask.sum())
    print(f"\n[4] PB-NBV 선택: {len(selected_pts)}개 WP → {covered}/{total} ({100*covered/total:.1f}%)")

    # Salvage pass
    used_mask = np.zeros(len(candidates), dtype=bool)
    for i, c in enumerate(candidates):
        if any(np.allclose(c, s) for s in selected_pts):
            used_mask[i] = True

    salvage_added, live_after = A.salvage_coverage(
        candidates, used_mask, target, pts, live_mask,
        fov_deg, A.MAX_DIST, pts_normals=None)

    salvage_pts = [pos for pos, _ in salvage_added]
    salvage_igs = [ig for _, ig in salvage_added]
    if salvage_pts:
        print(f"[4b] [Salvage] {len(salvage_pts)}개 추가")
    live_mask = live_after
    final_covered = total - int(live_mask.sum())
    print(f"     최종 커버: {final_covered}/{total} ({100*final_covered/total:.1f}%)")

    all_pts    = selected_pts + salvage_pts
    all_phases = ["pbnbv"] * len(selected_pts) + ["salvage"] * len(salvage_pts)

    print(f"[5] [Greedy NN] {len(all_pts)}개 방문 순서 결정...")
    path_order  = A.greedy_path(np.array(all_pts), START) if all_pts else []
    path_phases = []
    for wp in path_order:
        for orig, ph in zip(all_pts, all_phases):
            if np.allclose(np.asarray(wp), orig):
                path_phases.append(ph); break
        else:
            path_phases.append("unknown")

    def az(p):
        return math.degrees(math.atan2(p[1]-target[1], p[0]-target[0])) % 360

    cur, plen = START.copy(), 0.0
    waypoints = []
    for i, (wp, ph) in enumerate(zip(path_order, path_phases)):
        wp = np.asarray(wp)
        plen += float(np.linalg.norm(wp - cur)); cur = wp
        t = A.tilt_deg_to_target(wp, target)
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(t,1), "azimuth_deg": round(az(wp),1),
                          "phase": ph})
        tag = "[SV]" if ph == "salvage" else "    "
        print(f"  WP{i+1:02d}{tag} ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) "
              f"tilt={t:.1f}° az={az(wp):.0f}°")
    print(f"  총 경로 길이: {plen:.2f}m")

    out = {
        "algorithm": "PB-NBV(2)+DistWeight + Greedy (Case D)",
        "mode": "no_camera_dist_weighted",
        "n_points": int(len(pts)),
        "fov_deg": fov_deg,
        "target": target.tolist(),
        "total_points": int(total),
        "covered_points": int(final_covered),
        "coverage_ratio": round(final_covered / total, 4),
        "path_length_m": round(plen, 3),
        "pbnbv_count": len(selected_pts),
        "salvage_count": len(salvage_pts),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
