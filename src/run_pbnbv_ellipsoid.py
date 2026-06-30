"""
run_pbnbv_ellipsoid.py — Case C: Ellipsoid-unit 0.5^r + Raycast + PB-NBV(2) + Greedy

논문의 ellipsoid 단위 0.5^r 를 점군에 직접 적용:
  1) Hard raycast → 진짜 가시 점 확정
  2) 가시 미관측 점 → KMeans 클러스터링 (≤8개)
  3) 클러스터 거리 오름차순 rank r → score = Σ count_k * 0.5^r_k

Case B(점 단위)와 Case 3(hard count) 사이의 중간 방식.
공동 통제조건: tilt=45°, MAX_DIST=8m, no-camera
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
OUT  = ROOT / "results" / "pbnbv_ellipsoid_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

FOV_DEG        = 89.9
IG_THRESH      = 0.5     # ellipsoid 단위 score 컷
N_SELECT       = 20
N_AZ           = 36
N_CLUSTERS_MAX = 8


def main():
    np.random.seed(42)

    d   = np.load(NPZ)
    pts = d["points"].astype(float)
    print(f"[1] {len(pts):,}pts  X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]"
          f"  Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]"
          f"  Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

    underobs_mask = np.ones(len(pts), dtype=bool)
    target        = pts.mean(axis=0)
    fov_deg       = min(FOV_DEG, A.MAX_FOV_DEG)
    print(f"[2] no-camera → all {underobs_mask.sum()} unobserved")
    print(f"    centroid: ({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})")

    obj_z_top  = pts[:,2].min()
    z_levels   = [round(obj_z_top - dz, 2) for dz in [1.0,2.0,3.0,4.0,5.0,6.0]]
    candidates = A.gen_candidates_tilt45(target, z_levels, n_az=N_AZ, max_dist=A.MAX_DIST)
    print(f"[3] {len(candidates)} candidates (tilt=45°, dist≤{A.MAX_DIST}m)")

    used      = np.zeros(len(candidates), dtype=bool)
    live_mask = underobs_mask.copy()
    selected_pts, sel_scores = [], []

    print(f"[4] [PB-NBV(2) Ellipsoid-unit 0.5^r] (thresh={IG_THRESH}):")
    for step in range(min(N_SELECT, len(candidates))):
        pb_scores = np.array([
            A.pbnbv_score_ellipsoid(
                candidates[i], candidates, target, pts, live_mask,
                fov_deg, A.MAX_DIST, A.LOOKAHEAD,
                n_clusters_max=N_CLUSTERS_MAX, pts_normals=None)
            if not used[i] else -np.inf
            for i in range(len(candidates))
        ], dtype=float)

        best     = int(np.argmax(pb_scores))
        pb_score = float(pb_scores[best])

        if pb_score <= IG_THRESH:
            print(f"  step{step+1}: score={pb_score:.3f} <= {IG_THRESH} → stop")
            break

        _, hard_vis = A.information_gain_ellipsoid_rank(
            candidates[best], target, pts, live_mask,
            fov_deg, A.MAX_DIST, n_clusters_max=N_CLUSTERS_MAX)
        raw_ig = int((hard_vis & live_mask).sum())

        live_mask &= ~hard_vis
        selected_pts.append(candidates[best])
        sel_scores.append(pb_score)
        used[best] = True
        print(f"  WP{step+1:02d}: ell-score={pb_score:.2f}  hard-IG={raw_ig}"
              f"  rem={live_mask.sum()}  z={candidates[best][2]:.2f}")

    total   = underobs_mask.sum()
    covered = total - int(live_mask.sum())
    print(f"\n[4] Ellipsoid selection: {len(selected_pts)} WP → {covered}/{total} ({100*covered/total:.1f}%)")

    # Salvage
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
        print(f"[4b] Salvage: {len(salvage_pts)} added")
        for k_s, (pos, ig) in enumerate(salvage_added):
            print(f"  SV{k_s+1:02d}: hard-IG={ig}  ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
    else:
        print(f"[4b] Salvage: none")

    live_mask     = live_after
    final_covered = total - int(live_mask.sum())
    print(f"     final: {final_covered}/{total} ({100*final_covered/total:.1f}%)")

    all_pts    = selected_pts + salvage_pts
    all_phases = ["ellipsoid"]*len(selected_pts) + ["salvage"]*len(salvage_pts)

    print(f"[5] Greedy NN over {len(all_pts)} WPs...")
    start      = all_pts[0] + np.array([0,0,-1.0]) if all_pts else target
    path_order = A.greedy_path(np.array(all_pts), start)

    path_phases = []
    for wp in path_order:
        wp = np.asarray(wp)
        for orig, ph in zip(all_pts, all_phases):
            if np.allclose(wp, orig):
                path_phases.append(ph); break
        else:
            path_phases.append("unknown")

    def az(p):
        return math.degrees(math.atan2(p[1]-target[1], p[0]-target[0])) % 360

    waypoints = []
    START = np.array([-27.06,-50.51,-6.13])
    cur, path_len = START.copy(), 0.0
    for i, (wp, ph) in enumerate(zip(path_order, path_phases)):
        wp = np.asarray(wp)
        path_len += np.linalg.norm(wp - cur); cur = wp
        t = A.tilt_deg_to_target(wp, target)
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(t,1), "azimuth_deg": round(az(wp),1),
                          "phase": ph})
        tag = "[SV]" if ph=="salvage" else "    "
        print(f"  WP{i+1:02d}{tag} ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})"
              f"  tilt={t:.1f}° az={az(wp):.0f}°")

    print(f"\n  total path: {path_len:.2f}m")

    out = {
        "algorithm": f"PB-NBV({A.LOOKAHEAD}) Ellipsoid-0.5^r + Greedy (no-camera)",
        "scoring": "ellipsoid_cluster_depth_rank_0.5^r_with_hard_raycast",
        "n_clusters_max": N_CLUSTERS_MAX,
        "n_points": int(len(pts)),
        "total_points": int(total),
        "covered_points": int(final_covered),
        "coverage_ratio": round(final_covered/total, 4),
        "path_length_m": round(path_len, 3),
        "pbnbv_count": len(selected_pts),
        "salvage_count": len(salvage_pts),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT,"w"), indent=2)
    print(f"\n✓ saved: {OUT}")


if __name__ == "__main__":
    main()
