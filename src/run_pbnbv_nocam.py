"""
run_pbnbv_nocam.py
기존 카메라 없이 포인트클라우드만으로 PB-NBV(2)+Greedy 경로 계획.
모든 포인트를 미관측으로 간주하고 전체 커버리지 경로를 생성한다.
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
OUT  = ROOT / "results" / "pbnbv_nocam_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── 파라미터 ──────────────────────────────────────────────────────────────────
FOV_DEG   = 89.9   # 카메라 FOV (기존 real_test 동일)
IG_THRESH = 5      # 의미 있는 최소 IG
N_SELECT  = 20     # 최대 waypoint 수
N_CANDS   = 200    # 후보 수


def main():
    np.random.seed(42)

    # 1) 포인트클라우드 로드
    d   = np.load(NPZ)
    pts = d["points"].astype(float)
    print(f"[1] 점구름 {len(pts):,}개")
    print(f"    X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
          f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

    # 2) 기존 카메라 없음 → 전체 포인트 미관측
    underobs_mask = np.ones(len(pts), dtype=bool)
    target        = pts.mean(axis=0)   # 오브젝트 centroid
    fov_deg       = min(FOV_DEG, A.MAX_FOV_DEG)
    print(f"[2] 기존 카메라 없음 → 전체 {underobs_mask.sum()}개 미관측")
    print(f"    오브젝트 centroid: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")

    # 3) 후보 시점 생성 — tilt=45° 고정, 여러 고도
    obj_z_top = pts[:,2].min()   # NED: 가장 음수 = 가장 높은 고도
    # 오브젝트 꼭대기에서 1~6m 위, 1m 간격 (horiz = vert → tilt=45°)
    z_levels = [round(obj_z_top - dz, 2) for dz in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    N_AZ     = 36   # 방위각 10° 간격

    print(f"[3] 후보 생성 — tilt=45° 고정, 고도 {len(z_levels)}레벨 × {N_AZ}방위 = {len(z_levels)*N_AZ}개")
    print(f"    z_levels={z_levels}")

    candidates = A.gen_candidates_tilt45(target, z_levels, n_az=N_AZ, max_dist=A.MAX_DIST)
    print(f"    후보 {len(candidates)}개 생성 (tilt=45°, dist≤{A.MAX_DIST}m)")

    # 4) PB-NBV(2) 시점 선택 + Greedy NN 정렬
    used      = np.zeros(len(candidates), dtype=bool)
    live_mask = underobs_mask.copy()
    selected_pts, sel_scores = [], []

    print(f"[4] [PB-NBV(2)] 시점 선택 (IG > {IG_THRESH}):")
    for step in range(min(N_SELECT, len(candidates))):
        pb_scores = np.array([
            A.pbnbv_score(candidates[i], candidates, target, pts, live_mask,
                          fov_deg, A.MAX_DIST, A.LOOKAHEAD, pts_normals=None)
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

        live_mask &= ~vis
        selected_pts.append(candidates[best])
        sel_scores.append(pb_score)
        used[best] = True
        print(f"  WP{step+1:02d}: score={pb_score:.0f}  raw-IG={ig_raw}  "
              f"남은={live_mask.sum()}  z={candidates[best][2]:.2f}")

    total   = underobs_mask.sum()
    covered = total - int(live_mask.sum())
    print(f"\n[4] PB-NBV 선택: {len(selected_pts)}개 WP → 커버 {covered}/{total} ({100*covered/total:.1f}%)")

    # Salvage pass: IG_THRESH 컷오프로 버려진 포인트 중 커버 가능한 것 추가
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
        print(f"[4b] [Salvage] {len(salvage_pts)}개 추가 (IG_THRESH 컷오프 후 남은 포인트 커버):")
        for k, (pos, ig) in enumerate(salvage_added):
            print(f"  SV{k+1:02d}: raw-IG={ig}  pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
    else:
        print(f"[4b] [Salvage] 추가 없음 (남은 {live_mask.sum()}개는 물리적으로 관측 불가)")

    live_mask = live_after
    final_covered = total - int(live_mask.sum())
    print(f"     최종 커버: {final_covered}/{total} ({100*final_covered/total:.1f}%)")

    all_pts    = selected_pts + salvage_pts
    all_phases = (["pbnbv"] * len(selected_pts)) + (["salvage"] * len(salvage_pts))

    print(f"[5] [Greedy NN] {len(all_pts)}개 시점 방문 순서 결정...")
    start = all_pts[0] + np.array([0, 0, -1.0]) if all_pts else target
    path_order = A.greedy_path(np.array(all_pts), start)

    # phase 태그 유지 (greedy 정렬 후)
    path_phases = []
    for wp in path_order:
        wp = np.asarray(wp)
        for orig, ph in zip(all_pts, all_phases):
            if np.allclose(wp, orig):
                path_phases.append(ph)
                break
        else:
            path_phases.append("unknown")
    path = path_order

    def az(p):
        return math.degrees(math.atan2(p[1]-target[1], p[0]-target[0])) % 360

    waypoints = []
    for i, (wp, ph) in enumerate(zip(path, path_phases)):
        wp = np.asarray(wp)
        t  = A.tilt_deg_to_target(wp, target)
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(t, 1), "azimuth_deg": round(az(wp), 1),
                          "phase": ph})
        tag = "[SV]" if ph == "salvage" else "    "
        print(f"  WP{i+1:02d}{tag} ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) "
              f"tilt={t:.1f}° az={az(wp):.0f}°")

    out = {
        "algorithm": f"PB-NBV({A.LOOKAHEAD}) + Greedy Path (no-camera / full-coverage)",
        "mode": "no_camera",
        "n_points": int(len(pts)),
        "fov_deg": fov_deg,
        "target": target.tolist(),
        "total_points": int(total),
        "covered_points": int(final_covered),
        "coverage_ratio": round(final_covered / total, 4),
        "pbnbv_count": len(selected_pts),
        "salvage_count": len(salvage_pts),
        "selected_scores": [float(s) for s in sel_scores],
        "salvage_igs": [int(ig) for ig in salvage_igs],
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
