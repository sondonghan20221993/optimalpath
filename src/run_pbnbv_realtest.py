"""
run_pbnbv_realtest.py
real_test 데이터(AirSim 좌표계)로 pbnbv_path.py의 PB-NBV(2)+Greedy 알고리즘을 그대로 실행.

- 점구름/법선 : real_test/real_test_pts_normals.npz (AirSim world 좌표)
- 카메라 pose : real_test/meta/*.json (position + quaternion + fov)
- 알고리즘    : pbnbv_path.py 의 함수를 import 해서 동일 로직 사용
  (open3d 는 PLY 로딩에만 쓰이므로 더미 모듈로 주입해 import 가능하게 함)
"""
import sys, types, json, glob, math
from pathlib import Path
import numpy as np

# open3d 더미 주입 (pbnbv_path 가 top-level 에서 import 하지만 함수에선 main 만 사용)
sys.modules.setdefault("open3d", types.ModuleType("open3d"))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A   # noqa: E402  동일 알고리즘 함수 재사용

# Ray-cast occlusion 활성화
USE_RAYCAST = True
A.RAYCAST_OCCLUSION = USE_RAYCAST

ROOT = HERE.parent
NPZ  = ROOT / "real_test" / "real_test_pts_normals.npz"
META = sorted((ROOT / "real_test" / "meta").glob("*.json"))
OUT  = ROOT / "results" / "pbnbv_realtest_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def quat_to_R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def load_cameras():
    pos, fwd, fovs = [], [], []
    for m in META:
        j = json.load(open(m))
        p = j["camera"]["pose"]["position"]
        o = j["camera"]["pose"]["orientation"]
        R = quat_to_R(o["w"], o["x"], o["y"], o["z"])
        pos.append([p["x"], p["y"], p["z"]])
        fwd.append(R @ np.array([1.0, 0.0, 0.0]))  # AirSim body X = forward
        fovs.append(j["camera"]["fov"])
    return np.array(pos), np.array(fwd), float(np.mean(fovs))


def main():
    np.random.seed(42)

    # 1) 카메라
    cam_positions, cam_forwards, fov_raw = load_cameras()
    fov_deg = min(fov_raw, A.MAX_FOV_DEG)
    target = (cam_positions + cam_forwards * 7.0).mean(axis=0)
    print(f"[1] 카메라 {len(cam_positions)}개, FOV(raw)={fov_raw:.1f}° -> used {fov_deg:.1f}°")
    print(f"    추정 타겟: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")

    # 2) 점구름 + 법선 (AirSim 좌표, open3d 불필요)
    d = np.load(NPZ)
    pts = d["points"].astype(float)
    pts_normals = d["normals"].astype(float)
    # 법선을 타겟 바깥쪽으로 일관화 (pbnbv_path.main 과 동일 처리)
    to_out = pts - target
    flip = (pts_normals * to_out).sum(axis=1) < 0
    pts_normals[flip] *= -1.0
    print(f"[2] 점구름 {len(pts):,}개 (법선 포함)")
    print(f"    좌표범위 X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
          f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

    # 3) coverage / 미관측
    coverage = A.compute_coverage(pts, cam_positions, cam_forwards, fov_deg,
                                  A.MAX_DIST, pts_normals=None)
    underobs_mask = A.select_underobserved_by_fraction(coverage, A.UNDEROBS_FRACTION)
    print(f"[3] coverage min={coverage.min()} p50={np.percentile(coverage,50):.0f} "
          f"max={coverage.max()} / 미관측 {underobs_mask.sum()}개 "
          f"({100*underobs_mask.mean():.1f}%)")
    under_center = pts[underobs_mask].mean(axis=0) if underobs_mask.sum() else target

    # 4) 장면에서 궤도 반경/Z 자동 유도 (AirSim NED 전용 후보 생성)
    # generate_candidates 는 MASt3R 좌표 가정(z=cam_z_mean-alt 로 더 위로 올림)이라
    # AirSim NED 에선 절대 z 레벨로 직접 후보 생성한다.
    horiz_dists = np.linalg.norm(cam_positions[:, :2] - target[:2], axis=1)
    r_mean = float(horiz_dists.mean())
    r_min  = float(horiz_dists.min())
    t_z = float(target[2])
    c_z_mean = float(cam_positions[:, 2].mean())
    # 단일 고도: 기존 카메라 평균 Z 사용 (NED: 음수=위)
    cand_z_levels = [c_z_mean]
    # 단일 궤도: 카메라와 동일 고도
    orbit_radii = [round(r_min * 0.9, 2), round(r_mean, 2), round(r_mean * 1.15, 2), round(r_mean * 1.3, 2)]

    print(f"[4] 단일 고도 궤도 후보 생성 (AirSim NED)")
    print(f"    반경: {orbit_radii}")
    print(f"    고도: z=[{c_z_mean:.2f}] (카메라 동일)")

    def _gen_candidates_ned(center, z_levels, radii, n_total):
        cands = []
        n_per = max(1, n_total // (len(z_levels) * len(radii)))
        for z in z_levels:
            for r in radii:
                for i in range(n_per):
                    th = 2 * math.pi * i / n_per
                    c = np.array([center[0] + r * math.cos(th),
                                  center[1] + r * math.sin(th), z])
                    c = A.project_to_tilt_boundary(c, target, A.MAX_TILT_DEG)
                    cands.append(c)
        return np.array(cands) if cands else np.empty((0, 3))

    candidates = _gen_candidates_ned(under_center, [c_z_mean], orbit_radii, A.N_CANDIDATES)
    print(f"    후보 총 {len(candidates)}개 (tilt>{A.MAX_TILT_DEG:.0f}° 초과분 경계로 보정)")
    if len(candidates) == 0:
        print("    !! 후보 0개 — tilt 제약을 완화하거나 반경/고도 재검토 필요")
        return
    # 유효 후보 필터링
    valid = []
    for c in candidates:
        if np.linalg.norm(cam_positions - c, axis=1).min() < 0.5:
            continue
        valid.append(c)
    valid = np.array(valid)
    print(f"    유효 후보 {len(valid)}개")

    # 5) PB-NBV 시점 선택 (live mask 갱신 + ray-cast occlusion) + IG 임계값 컷 + Greedy NN 정렬
    IG_THRESH = 2    # 이 이하 IG는 의미없는 시점으로 제외
    N_SELECT  = 15
    print(f"    Ray-cast occlusion: {'ON' if A.RAYCAST_OCCLUSION else 'OFF'}")
    used      = np.zeros(len(valid), dtype=bool)
    live_mask = underobs_mask.copy()
    selected_pts  = []
    sel_scores    = []
    print(f"    [PB-NBV(2)] 시점 선택 (lookahead IG > {IG_THRESH}):")
    for step in range(min(N_SELECT, len(valid))):
        # PB-NBV(2) 스코어 (IG1 + 0.5*max(IG2))
        pb_scores = np.array([
            A.pbnbv_score(valid[i], valid, target, pts, live_mask,
                          fov_deg, A.MAX_DIST, A.LOOKAHEAD, pts_normals=None)
            if not used[i] else -np.inf
            for i in range(len(valid))
        ], dtype=float)

        best = int(np.argmax(pb_scores))
        pb_score = float(pb_scores[best])

        # 컷오프는 순수 1-step IG로 (의미 보존)
        ig_raw, vis = A.information_gain(valid[best], target, pts, live_mask,
                                         fov_deg, A.MAX_DIST, pts_normals=None)
        if ig_raw <= IG_THRESH:
            print(f"      step{step+1}: 최대 raw-IG={ig_raw:.0f} ≤ {IG_THRESH} → 선택 종료")
            break

        live_mask &= ~vis

        selected_pts.append(valid[best])
        sel_scores.append(float(pb_score))
        used[best] = True
        print(f"      WP{step+1:02d}: PB-NBV-score={pb_score:.0f} (raw-IG={ig_raw:.0f})  남은미관측={live_mask.sum()}")

    print(f"    [Greedy NN] {len(selected_pts)}개 시점 방문 순서 결정...")
    path = A.greedy_path(np.array(selected_pts), cam_positions[-1])
    print(f"[5] PB-NBV 선택 {len(selected_pts)}개 → Greedy NN 정렬 → 경로 {len(path)} waypoint")

    def az(p):
        return math.degrees(math.atan2(p[1] - target[1], p[0] - target[0])) % 360
    waypoints = []
    for i, wp in enumerate(path):
        wp = np.asarray(wp)
        t = A.tilt_deg_to_target(wp, target)
        waypoints.append({"idx": i + 1, "pos": wp.tolist(),
                          "tilt_deg": round(t, 1), "azimuth_deg": round(az(wp), 1)})
        print(f"    WP{i+1:02d} ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) "
              f"tilt={t:.1f}° az={az(wp):.0f}°")

    out = {
        "algorithm": f"PB-NBV({A.LOOKAHEAD}) + Greedy Path (real_test/AirSim)",
        "n_cameras": int(len(cam_positions)),
        "fov_raw_deg": fov_raw, "fov_used_deg": fov_deg,
        "target": target.tolist(),
        "underobserved_ratio": float(underobs_mask.mean()),
        "underobserved_center": under_center.tolist(),
        "coverage_stats": {"min": int(coverage.min()), "max": int(coverage.max()),
                           "mean": float(coverage.mean())},
        "score_range": [float(min(sel_scores)) if sel_scores else 0, float(max(sel_scores)) if sel_scores else 0],
        "n_candidates": int(len(valid)),
        "selected_scores": [float(s) for s in sel_scores],
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
