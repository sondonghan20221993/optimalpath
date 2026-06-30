"""
run_pbnbv_orbit_soft.py — 원형 궤도 보완: Soft 입사각 규제 (다중뷰 재구성용)

배경:
  - 대상은 point cloud / 다중뷰(MVS·SfM) 재구성 맥락.
  - 공중 tilt=45° 단일 정면으로는 구조적으로 못 보는 면(법선이 수평~아래)이 존재.
  - 이런 면은 grazing 다중뷰를 융합해 복원하는 것이 현실적이고 정직한 접근.

Soft 규제 (raycast 없음, PB-NBV 기하 근사 유지):
  관측 가중치  w_cam(voxel) = max(0, cos(theta))^alpha          (Lambert 코사인)
  누적 관측    W(voxel)     = Σ_cameras w_cam(voxel)
  관측됨       W(voxel) >= TAU

파라미터 (튜닝 아님, 물리/기존조건에서 유도):
  alpha = 1      → Lambert 조도 감쇠 (광학 표준)
  TAU   = cos(70°) ≈ 0.342  → "hard 경계(70°)에서 정면 1장"과 동등한 누적 기준
                              → hard(theta<70°)와 직접 비교 가능, 임의값 제거

거리 가중치 (드론용 Bircher utility):
  U(v) = F(v) / dist(cur_pos -> v),  F = frontier_only (>=0, 수정된 evaluate)

보고: hard(정면 1장 보장) 커버 + soft(다중뷰 누적) 커버 둘 다 — 골대 안 옮김.
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
OUT  = ROOT / "results" / "pbnbv_orbit_soft_path.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── 파라미터 (물리/기존조건 유도, 커버리지 튜닝 아님) ──────────────────────────
INC_HARD_DEG = 70                         # hard 비교 기준 (정면 1장 보장)
ALPHA        = 1.0                        # Lambert 코사인 감쇠
TAU          = math.cos(math.radians(INC_HARD_DEG))  # ≈0.342, hard 경계 1장 등가
FIX_ALT_M    = 3.0
N_AZ         = 36
N_STEPS      = 20


def load_orbit_positions():
    pos = []
    for m in META:
        p = json.load(open(m))["camera"]["pose"]["position"]
        pos.append([p["x"], p["y"], p["z"]])
    return np.array(pos)


def _geom_gate(cp):
    """거리/FOV 게이트 (공통). returns (in_view_mask, cos_inc array)."""
    cam_dir = P.TARGET - cp; cam_dir /= np.linalg.norm(cam_dir) + 1e-9
    to   = P.SURF_CEN - cp
    dist = np.linalg.norm(to, axis=1)
    in_range = (dist >= P.MIN_DIST) & (dist <= P.MAX_DIST)
    in_fov   = (to * cam_dir).sum(1) / (dist + 1e-9) >= math.cos(math.radians(P.FOV_DEG / 2))
    view = cp - P.SURF_CEN
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
    cos_inc = np.clip((P.SURF_NRM * view).sum(1), 0.0, 1.0)
    return (in_range & in_fov), cos_inc


def soft_weight(cp, alpha=ALPHA):
    """Lambert 코사인 관측 가중치 w = max(0,cos_inc)^alpha (게이트 밖=0)."""
    in_view, cos_inc = _geom_gate(cp)
    return np.where(in_view, cos_inc ** alpha, 0.0)


def hard_observed(cp, inc_max=INC_HARD_DEG):
    """hard 규제: 입사각<inc_max 인 voxel 인덱스 (정면 1장 기준)."""
    in_view, cos_inc = _geom_gate(cp)
    return np.where(in_view & (cos_inc >= math.cos(math.radians(inc_max))))[0]


def main():
    cam_pos = load_orbit_positions()
    target  = P.TARGET
    pts     = P._pts_raw

    fix_z = round(pts[:, 2].min() - FIX_ALT_M, 2)
    print(f"[1] 고정 고도 z={fix_z}m, tilt=45°")
    print(f"    Soft 규제: w=cos^{ALPHA:.0f}(Lambert), TAU=cos({INC_HARD_DEG}°)={TAU:.3f}")

    # 궤도 누적 관측 (soft) + hard 관측
    obs_w   = np.zeros(P.N_SURF)
    obs_h   = np.zeros(P.N_SURF, dtype=bool)
    for c in cam_pos:
        obs_w += soft_weight(c)
        obs_h[hard_observed(c)] = True
    soft_orbit = int((obs_w >= TAU).sum())
    hard_orbit = int(obs_h.sum())
    print(f"[2] 궤도 voxel 커버: soft {soft_orbit}/{P.N_SURF} ({100*soft_orbit/P.N_SURF:.1f}%) | "
          f"hard {hard_orbit}/{P.N_SURF} ({100*hard_orbit/P.N_SURF:.1f}%)")

    # 후보: tilt=45° 고정 고도
    cands = A.gen_candidates_tilt45(target, [fix_z], n_az=N_AZ, max_dist=P.MAX_DIST)
    print(f"[3] 후보 {len(cands)}개 (tilt=45°, z={fix_z}m)")

    # 보완 루프: soft 관측 기준 frontier + U=F/dist (F>=0, frontier_only)
    used    = np.zeros(len(cands), dtype=bool)
    sel     = []
    cur_pos = cam_pos[-1].copy()
    print(f"[4] Soft PB-NBV + U=F/dist (출발 "
          f"({cur_pos[0]:.1f},{cur_pos[1]:.1f},{cur_pos[2]:.1f})):")

    for step in range(N_STEPS):
        observed = obs_w >= TAU
        fro_idx  = P.compute_frontier(observed)
        if len(fro_idx) == 0:
            print(f"  step{step+1}: frontier 없음 → 종료"); break
        occ_ell = P.fit_ellipsoids(P.SURF_CEN[observed], P.SURF_NRM[observed])
        fro_ell = P.fit_ellipsoids(P.SURF_CEN[fro_idx],  P.SURF_NRM[fro_idx])
        scores = np.array([
            -1e18 if used[i] else
            P.evaluate(c, occ_ell, fro_ell, frontier_only=True) / (np.linalg.norm(c - cur_pos) + 1e-6)
            for i, c in enumerate(cands)
        ])
        best = int(np.argmax(scores))
        # 선택 후보의 soft 기여 누적
        prev = (obs_w >= TAU).copy()
        obs_w += soft_weight(cands[best])
        gained = int(((obs_w >= TAU) & ~prev).sum())
        if gained == 0:
            print(f"  step{step+1}: gain=0 → 종료"); break
        travel = float(np.linalg.norm(cands[best] - cur_pos))
        used[best] = True
        cur_pos = cands[best].copy()
        sel.append(cands[best])
        az = math.degrees(math.atan2(cands[best][1]-target[1], cands[best][0]-target[0])) % 360
        cov = (obs_w >= TAU).sum() / P.N_SURF
        print(f"  WP{step+1:02d}: U={scores[best]:.3f} +{gained}vox travel={travel:.1f}m "
              f"az={az:.0f}° softcov={cov*100:.1f}%")

    # Greedy 재정렬 + 실제 점 raycast 평가 (공정 비교)
    A.RAYCAST_OCCLUSION = True
    if sel:
        order = A.greedy_path(np.array(sel), cam_pos[-1])
        cur, plen = cam_pos[-1].copy(), 0.0
        for wp in order:
            plen += np.linalg.norm(np.asarray(wp) - cur); cur = np.asarray(wp)
    else:
        order, plen = [], 0.0

    live = np.ones(len(pts), dtype=bool)
    for c in cam_pos:
        _, vis = A.information_gain(c, target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    orbit_real = len(pts) - int(live.sum())
    for wp in order:
        _, vis = A.information_gain(np.asarray(wp), target, pts, live, P.FOV_DEG, P.MAX_DIST); live &= ~vis
    final_real = len(pts) - int(live.sum())

    soft_final = int((obs_w >= TAU).sum())
    print(f"\n[5] 결과:")
    print(f"    보완 WP: {len(order)}개  경로: {plen:.2f}m  고도: z={fix_z}m, tilt=45°")
    print(f"    soft voxel 커버: {soft_orbit} → {soft_final} "
          f"({100*soft_final/P.N_SURF:.1f}%)")
    print(f"    실제 점(raycast) 커버: {orbit_real} → {final_real}/{len(pts)} "
          f"({100*final_real/len(pts):.1f}%)")
    print(f"    [참고] hard(theta<{INC_HARD_DEG}°, 정면1장) 궤도 커버: {100*hard_orbit/P.N_SURF:.1f}%")

    waypoints = []
    for i, wp in enumerate(order):
        wp = np.asarray(wp)
        t  = A.tilt_deg_to_target(wp, target)
        az = math.degrees(math.atan2(wp[1]-target[1], wp[0]-target[0])) % 360
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(t,1), "azimuth_deg": round(az,1)})
        print(f"  WP{i+1:02d}: ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) tilt={t:.1f}° az={az:.0f}°")

    out = {
        "algorithm": f"Orbit + Soft-regulated PB-NBV (Lambert cos^{ALPHA:.0f}, TAU=cos{INC_HARD_DEG}) + U=F/dist",
        "context": "multi-view (MVS/SfM) reconstruction",
        "fix_z": fix_z, "fix_alt_m": FIX_ALT_M, "tilt_deg": 45,
        "alpha": ALPHA, "tau": round(TAU, 4), "inc_hard_deg": INC_HARD_DEG,
        "soft_cov_orbit": round(soft_orbit/P.N_SURF, 4),
        "soft_cov_final": round(soft_final/P.N_SURF, 4),
        "hard_cov_orbit": round(hard_orbit/P.N_SURF, 4),
        "real_cov_orbit": round(orbit_real/len(pts), 4),
        "real_cov_final": round(final_real/len(pts), 4),
        "n_waypoints": len(order), "path_length_m": round(plen, 3),
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
