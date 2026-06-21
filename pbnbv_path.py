"""
pbnbv_path.py
PB-NBV(2) + Greedy Path Planning

1. Point Cloud + 기존 Camera pose로 coverage 추정
2. 미관측 영역(underobserved) 탐색
3. 후보 시점 생성 → PB-NBV(2) 스코어링 (2-step lookahead)
4. Greedy path planning으로 최종 경로 생성
5. AirSim car.py 호환 JSON + 시각화 이미지 출력
"""

import json, math, sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

for _fn in ["Malgun Gothic", "NanumGothic"]:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR       = r"C:\Users\sdh97\Desktop\blue_1_fhd_sfm(pp팍스 mast3r결과)"
PLY_PATH       = BASE_DIR + r"\pointcloud.ply"
POSES_PATH     = BASE_DIR + r"\poses.npy"
FOCALS_PATH    = BASE_DIR + r"\focals.npy"
OUT_JSON       = r"C:\Users\sdh97\Desktop\pbnbv_path.json"
OUT_IMG        = r"C:\Users\sdh97\Desktop\pbnbv_result.png"

IMG_W, IMG_H   = 1920, 1080   # FHD

# ── 알고리즘 파라미터 ─────────────────────────────────────────────────────────
N_PCD_SAMPLE   = 40_000    # point cloud 서브샘플 수
MAX_DIST       = 8.0       # 최대 가시 거리 (m)
UNDEROBS_THRESH = 3        # 이 횟수 미만이면 미관측
N_CANDIDATES   = 150       # 후보 시점 수
N_SELECT       = 10        # 최종 선택 시점 수
LOOKAHEAD      = 2         # PB-NBV lookahead 단계 수
# 후보 고도/반경: MASt3R 좌표계 기준 (카메라 Z≈4~5, 타겟 Z≈3.9)
ORBIT_ALTITUDES = [0.5, 1.0, 1.5]   # 타겟 위로 올라가는 높이 (m)
ORBIT_RADII     = [2.0, 3.5, 5.0]   # 수평 반경 (m)


# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 (MASt3R 기반)
# ════════════════════════════════════════════════════════════════════════════

def load_mast3r_data():
    """poses.npy, focals.npy 로드. 카메라 위치/방향 반환."""
    poses  = np.load(POSES_PATH)    # (N, 4, 4) camera-to-world
    focals = np.load(FOCALS_PATH)   # (N,)

    cam_positions = poses[:, :3, 3]
    cam_rotations = poses[:, :3, :3]

    # 카메라 forward 방향: OpenCV 컨벤션, camera Z = [0,0,1]
    cam_forwards = np.array([R @ np.array([0., 0., 1.]) for R in cam_rotations])

    # FOV 계산
    fov_h = 2 * np.degrees(np.arctan(IMG_W / 2 / focals.mean()))
    fov_v = 2 * np.degrees(np.arctan(IMG_H / 2 / focals.mean()))

    # 타겟 추정: 각 카메라의 3m 앞 지점 평균
    look_pts = cam_positions + cam_forwards * 3.0
    target = look_pts.mean(axis=0)

    print(f"  카메라 수: {len(poses)}, FOV: {fov_h:.1f}°(H) x {fov_v:.1f}°(V)")
    print(f"  추정 타겟: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
    return cam_positions, cam_forwards, fov_h, target


# ════════════════════════════════════════════════════════════════════════════
# 2. Coverage 추정
# ════════════════════════════════════════════════════════════════════════════

def point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist):
    """
    AirSim NED 좌표계 기준 프러스텀 가시성 체크.
    cam_dir: 카메라 광축 방향 단위벡터 (NED, +Z=아래)
    """
    v = pts - cam_pos
    dist = np.linalg.norm(v, axis=1)
    valid_dist = (dist > 0.1) & (dist < max_dist)

    v_norm = v / (dist[:, None] + 1e-8)
    cos_half = math.cos(math.radians(fov_deg / 2.0))
    dot = v_norm @ cam_dir
    in_cone = dot > cos_half

    return valid_dist & in_cone


def compute_coverage(pts, cam_positions, cam_forwards, fov_deg, max_dist):
    """
    각 포인트가 기존 카메라에서 몇 번 관측되는지 계산.
    Returns: visibility count (N,)
    """
    counts = np.zeros(len(pts), dtype=np.int32)
    for cam_pos, cam_dir in zip(cam_positions, cam_forwards):
        mask = point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist)
        counts += mask.astype(np.int32)
    return counts


# ════════════════════════════════════════════════════════════════════════════
# 3. 후보 시점 생성
# ════════════════════════════════════════════════════════════════════════════

def generate_candidates(target, cam_positions, altitudes, radii, n_total):
    """
    타겟 주변 구면 후보 시점 생성 (MASt3R 좌표계).
    카메라들의 평균 Z에서 altitude만큼 오프셋.
    Returns: (N,3)
    """
    cam_z_mean = cam_positions[:, 2].mean()
    candidates = []
    n_per = max(1, n_total // (len(altitudes) * len(radii)))
    for alt in altitudes:
        z = cam_z_mean - alt   # 카메라 평균 고도 기준으로 아래로 내려옴
        for rad in radii:
            for i in range(n_per):
                theta = 2 * math.pi * i / n_per
                x = target[0] + rad * math.cos(theta)
                y = target[1] + rad * math.sin(theta)
                candidates.append([x, y, z])
    return np.array(candidates)


# ════════════════════════════════════════════════════════════════════════════
# 4. PB-NBV(2) 스코어링
# ════════════════════════════════════════════════════════════════════════════

def information_gain(cam_pos, target, pts, underobs_mask, fov_deg,
                     max_dist, already_visible=None):
    """
    후보 시점에서 미관측 포인트 중 새로 관측 가능한 수 반환.
    카메라는 타겟을 바라보는 방향으로 설정.
    """
    to_target = target - cam_pos
    norm = np.linalg.norm(to_target)
    cam_dir = to_target / (norm + 1e-8)
    visible = point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist)
    new_vis = visible & underobs_mask
    if already_visible is not None:
        new_vis = new_vis & ~already_visible
    return int(new_vis.sum()), visible


def pbnbv_score(candidate, candidates, target, pts, underobs_mask,
                fov_deg, max_dist, lookahead):
    """
    PB-NBV(lookahead) 스코어: 1-step IG + lookahead-step 최대 IG 합산.
    """
    ig1, vis1 = information_gain(candidate, target, pts, underobs_mask, fov_deg, max_dist)
    score = float(ig1)

    if lookahead >= 2 and ig1 > 0:
        updated_mask = underobs_mask & ~vis1
        best_ig2 = 0
        sample_idx = np.random.choice(len(candidates), min(30, len(candidates)), replace=False)
        for idx in sample_idx:
            if np.allclose(candidates[idx], candidate):
                continue
            ig2, _ = information_gain(
                candidates[idx], target, pts, updated_mask,
                fov_deg, max_dist, already_visible=vis1,
            )
            best_ig2 = max(best_ig2, ig2)
        score += 0.5 * best_ig2

    return score


# ════════════════════════════════════════════════════════════════════════════
# 5. Greedy Path Planning
# ════════════════════════════════════════════════════════════════════════════

def greedy_path(selected_pts, start_pos):
    """
    Nearest-Neighbor Greedy로 시작 위치에서 모든 선택 시점 방문 순서 결정.
    Returns: 정렬된 waypoints list
    """
    remaining = list(range(len(selected_pts)))
    path = []
    cur = start_pos

    while remaining:
        dists = [np.linalg.norm(selected_pts[i] - cur) for i in remaining]
        nearest_idx = remaining[int(np.argmin(dists))]
        path.append(nearest_idx)
        cur = selected_pts[nearest_idx]
        remaining.remove(nearest_idx)

    return [selected_pts[i] for i in path]


# ════════════════════════════════════════════════════════════════════════════
# 6. 시각화
# ════════════════════════════════════════════════════════════════════════════

def visualize(pts_airsim, coverage, underobs_mask, candidates, scores,
              selected_path, airsim_cams_dict, out_path, target=None):

    fig = plt.figure(figsize=(20, 12), facecolor="#0f0f1a")
    fig.suptitle("PB-NBV(2) + Greedy Path Planning 결과", color="white",
                 fontsize=15, fontweight="bold")

    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                      left=0.05, right=0.97, top=0.93, bottom=0.05)

    ax_cov  = fig.add_subplot(gs[0, 0])   # coverage 히스토그램
    ax_top  = fig.add_subplot(gs[0, 1])   # 2D top-down
    ax_3d   = fig.add_subplot(gs[0, 2], projection="3d")  # 3D
    ax_sc   = fig.add_subplot(gs[1, 0])   # 후보 스코어
    ax_path = fig.add_subplot(gs[1, 1])   # 최종 경로 2D
    ax_p3d  = fig.add_subplot(gs[1, 2], projection="3d")  # 최종 경로 3D

    bg = "#1a1a2e"
    for ax in [ax_cov, ax_top, ax_sc, ax_path]:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    tx, ty, tz = target if target is not None else pts_airsim.mean(axis=0)

    # ── coverage 히스토그램 ─────────────────────────────────────────────────
    ax_cov.set_title("포인트별 관측 횟수 분포", color="white", fontsize=9)
    ax_cov.hist(coverage, bins=30, color="#4fc3f7", edgecolor="#0f0f1a", alpha=0.85)
    ax_cov.axvline(UNDEROBS_THRESH, color="#ef5350", linewidth=1.5,
                   linestyle="--", label=f"미관측 기준 (<{UNDEROBS_THRESH}회)")
    under_pct = 100 * underobs_mask.sum() / len(coverage)
    ax_cov.text(0.98, 0.95, f"미관측: {under_pct:.1f}%",
                transform=ax_cov.transAxes, color="#ef5350", fontsize=8,
                ha="right", va="top")
    ax_cov.set_xlabel("관측 횟수", color="#aaa", fontsize=7)
    ax_cov.set_ylabel("포인트 수", color="#aaa", fontsize=7)
    ax_cov.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white")

    # ── 2D top-down: coverage ────────────────────────────────────────────────
    ax_top.set_title("포인트 클라우드 coverage (Top-Down)", color="white", fontsize=9)
    subsample = np.random.choice(len(pts_airsim), min(5000, len(pts_airsim)), replace=False)
    pts_sub = pts_airsim[subsample]
    cov_sub = coverage[subsample]
    sc1 = ax_top.scatter(pts_sub[:, 0], pts_sub[:, 1],
                         c=cov_sub, cmap="RdYlGn", s=1, alpha=0.4,
                         vmin=0, vmax=coverage.max())
    cb1 = plt.colorbar(sc1, ax=ax_top, pad=0.01, fraction=0.04)
    cb1.set_label("관측 횟수", color="white", fontsize=6)
    cb1.ax.tick_params(labelcolor="white", labelsize=6)
    # 기존 카메라
    for pos in airsim_cams_dict.values():
        ax_top.plot(pos[0], pos[1], "w.", markersize=2, alpha=0.3)
    ax_top.plot(tx, ty, "*", color="#ffcc02", markersize=14, zorder=5)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax_top.set_ylabel("Y (m)", color="#aaa", fontsize=7)

    # ── 3D: coverage + 후보 ──────────────────────────────────────────────────
    ax_3d.set_facecolor("#0f0f1a")
    ax_3d.set_title("미관측 영역 + 후보 시점", color="white", fontsize=9)
    # 미관측 포인트
    under_pts = pts_airsim[underobs_mask]
    if len(under_pts) > 3000:
        idx = np.random.choice(len(under_pts), 3000, replace=False)
        under_pts = under_pts[idx]
    ax_3d.scatter(under_pts[:, 0], under_pts[:, 1], under_pts[:, 2],
                  c="#ef5350", s=1, alpha=0.3, label="미관측")
    # 후보 시점 (스코어로 색상)
    c_arr = np.array(candidates)
    sc_arr = np.array(scores)
    sc3 = ax_3d.scatter(c_arr[:, 0], c_arr[:, 1], c_arr[:, 2],
                        c=sc_arr, cmap="plasma", s=20, alpha=0.7, zorder=4)
    ax_3d.scatter([tx], [ty], [tz], color="#ffcc02", s=150, marker="*", zorder=6)
    ax_3d.set_xlabel("X", color="#aaa", fontsize=6, labelpad=3)
    ax_3d.set_ylabel("Y", color="#aaa", fontsize=6, labelpad=3)
    ax_3d.set_zlabel("Z", color="#aaa", fontsize=6, labelpad=3)
    ax_3d.tick_params(colors="white", labelsize=5)
    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")
    ax_3d.view_init(elev=25, azim=-60)

    # ── 후보 스코어 바 차트 ──────────────────────────────────────────────────
    ax_sc.set_title(f"PB-NBV({LOOKAHEAD}) 후보 스코어 (상위 30개)", color="white", fontsize=9)
    sorted_idx = np.argsort(scores)[::-1][:30]
    bar_scores = [scores[i] for i in sorted_idx]
    bar_colors = ["#ffcc02" if i < N_SELECT else "#4fc3f7" for i in range(len(bar_scores))]
    ax_sc.barh(range(len(bar_scores)), bar_scores[::-1], color=bar_colors[::-1], alpha=0.85)
    ax_sc.axvline(bar_scores[N_SELECT - 1] if len(bar_scores) >= N_SELECT else 0,
                  color="#ef5350", linewidth=1, linestyle="--", label="선택 기준")
    ax_sc.set_xlabel("PB-NBV 스코어 (정보 이득)", color="#aaa", fontsize=7)
    ax_sc.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white")
    ax_sc.text(0.98, 0.02, f"노랑=선택({N_SELECT}개) / 파랑=미선택",
               transform=ax_sc.transAxes, color="#aaa", fontsize=7,
               ha="right", va="bottom")

    # ── 최종 경로 2D ─────────────────────────────────────────────────────────
    ax_path.set_title("Greedy 경로 (Top-Down)", color="white", fontsize=9)
    path_arr = np.array(selected_path)
    # 기존 카메라 (회색)
    for pos in airsim_cams_dict.values():
        ax_path.plot(pos[0], pos[1], ".", color="#555", markersize=3, alpha=0.5)
    # 새 경로
    ax_path.plot(np.append(path_arr[:, 0], path_arr[0, 0]),
                 np.append(path_arr[:, 1], path_arr[0, 1]),
                 color="#00e5ff", linewidth=2, alpha=0.9, zorder=4)
    sc_path = ax_path.scatter(path_arr[:, 0], path_arr[:, 1],
                              c=path_arr[:, 2], cmap="plasma",
                              s=90, zorder=5, edgecolors="white", linewidths=0.6)
    for i, (x, y) in enumerate(zip(path_arr[:, 0], path_arr[:, 1])):
        ax_path.text(x, y + 0.4, str(i + 1), color="white", fontsize=7,
                     ha="center", va="bottom", zorder=6)
    ax_path.plot(tx, ty, "*", color="#ffcc02", markersize=16, zorder=7,
                 markeredgecolor="white", markeredgewidth=0.6)
    ax_path.text(tx, ty - 1.2, "TARGET", color="#ffcc02", fontsize=8,
                 ha="center", fontweight="bold")
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax_path.set_ylabel("Y (m)", color="#aaa", fontsize=7)
    leg_path = [
        mpatches.Patch(color="#555", label=f"기존 경로 ({len(airsim_cams_dict)}개 WP)"),
        mpatches.Patch(color="#00e5ff", label=f"PB-NBV 신규 ({len(selected_path)}개 WP)"),
    ]
    ax_path.legend(handles=leg_path, fontsize=7, facecolor=bg,
                   edgecolor="#444", labelcolor="white", loc="lower right")

    # ── 최종 경로 3D ─────────────────────────────────────────────────────────
    ax_p3d.set_facecolor("#0f0f1a")
    ax_p3d.set_title("Greedy 경로 3D", color="white", fontsize=9)
    z_vals = path_arr[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    cmap = plt.cm.plasma
    for i in range(len(path_arr) - 1):
        t_c = (z_vals[i] - z_min) / (z_max - z_min + 1e-8)
        ax_p3d.plot([path_arr[i, 0], path_arr[i+1, 0]],
                    [path_arr[i, 1], path_arr[i+1, 1]],
                    [path_arr[i, 2], path_arr[i+1, 2]],
                    color=cmap(t_c), linewidth=2.2, alpha=0.95)
    ax_p3d.scatter(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                   c=z_vals, cmap="plasma", s=60, edgecolors="white", linewidths=0.5)
    for x, y, z in zip(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]):
        ax_p3d.plot([x, x], [y, y], [z, tz], color="white", alpha=0.08, linewidth=0.5)
    ax_p3d.scatter([tx], [ty], [tz], color="#ffcc02", s=200, marker="*", zorder=6)
    ax_p3d.set_xlabel("X", color="#aaa", fontsize=6, labelpad=3)
    ax_p3d.set_ylabel("Y", color="#aaa", fontsize=6, labelpad=3)
    ax_p3d.set_zlabel("Z NED", color="#aaa", fontsize=6, labelpad=3)
    ax_p3d.tick_params(colors="white", labelsize=5)
    for pane in [ax_p3d.xaxis.pane, ax_p3d.yaxis.pane, ax_p3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")
    ax_p3d.view_init(elev=28, azim=-55)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"시각화 저장: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    # 1. MASt3R 데이터 로드
    print("[1/5] MASt3R 데이터 로드 중...")
    cam_positions, cam_forwards, fov_deg, target = load_mast3r_data()

    # 2. Point Cloud 로드 및 서브샘플링
    print("[2/5] Point Cloud 로드 중...")
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pts_all = np.asarray(pcd.points)
    cols_all = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"  원본: {len(pts_all):,}개")
    idx = np.random.choice(len(pts_all), min(N_PCD_SAMPLE, len(pts_all)), replace=False)
    pts = pts_all[idx]
    cols = cols_all[idx] if cols_all is not None else None
    print(f"  서브샘플: {len(pts):,}개")
    print(f"  좌표 범위: X={pts[:,0].min():.2f}~{pts[:,0].max():.2f}, "
          f"Y={pts[:,1].min():.2f}~{pts[:,1].max():.2f}, "
          f"Z={pts[:,2].min():.2f}~{pts[:,2].max():.2f}")

    # 3. Coverage 추정
    print(f"[3/5] Coverage 추정 중 (카메라 {len(cam_positions)}개 기준)...")
    coverage = compute_coverage(pts, cam_positions, cam_forwards, fov_deg, MAX_DIST)
    underobs_mask = coverage < UNDEROBS_THRESH
    print(f"  미관측 포인트: {underobs_mask.sum():,}개 ({100*underobs_mask.mean():.1f}%)")

    if underobs_mask.sum() > 0:
        under_center = pts[underobs_mask].mean(axis=0)
        print(f"  미관측 영역 중심: ({under_center[0]:.3f}, {under_center[1]:.3f}, {under_center[2]:.3f})")
    else:
        under_center = target

    # 4. 후보 시점 생성 + PB-NBV(2) 스코어링
    print(f"[4/5] 후보 {N_CANDIDATES}개 생성 + PB-NBV({LOOKAHEAD}) 스코어링 중...")
    candidates = generate_candidates(under_center, cam_positions, ORBIT_ALTITUDES, ORBIT_RADII, N_CANDIDATES)

    scores, valid_cands = [], []
    for cand in candidates:
        min_d = np.linalg.norm(cam_positions - cand, axis=1).min()
        if min_d < 0.5:   # 기존 카메라와 너무 가까운 후보 제거
            continue
        sc = pbnbv_score(cand, candidates, target, pts, underobs_mask,
                         fov_deg, MAX_DIST, LOOKAHEAD)
        scores.append(sc)
        valid_cands.append(cand)

    valid_cands = np.array(valid_cands)
    scores = np.array(scores)
    print(f"  유효 후보: {len(valid_cands)}개, 스코어: {scores.min():.0f}~{scores.max():.0f}")

    top_idx = np.argsort(scores)[::-1][:N_SELECT]
    selected_pts = valid_cands[top_idx]
    selected_scores = scores[top_idx]
    print(f"  선택 시점: {len(selected_pts)}개, 최고 스코어: {selected_scores[0]:.0f}")

    # 5. Greedy Path Planning
    print("[5/5] Greedy Path Planning 중...")
    start_pos = cam_positions[-1]
    final_path = greedy_path(selected_pts, start_pos)
    print(f"  최종 경로: {len(final_path)}개 웨이포인트")
    for i, wp in enumerate(final_path):
        print(f"  WP#{i+1:02d}: ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})  score={selected_scores[i]:.0f}")

    # JSON 저장
    output = {
        "schema_version": 1,
        "name": "pbnbv_greedy_path_mast3r",
        "algorithm": f"PB-NBV({LOOKAHEAD}) + Greedy Path Planning",
        "coordinate_system": "MASt3R reconstruction space",
        "n_existing_cameras": int(len(cam_positions)),
        "underobserved_ratio": float(underobs_mask.mean()),
        "underobserved_center": under_center.tolist(),
        "estimated_target": target.tolist(),
        "path_step_count": len(final_path),
        "waypoints": [
            {
                "index": i + 1,
                "position": wp.tolist(),
                "relative_to_target": (wp - target).tolist(),
                "pbnbv_score": float(selected_scores[i]),
            }
            for i, wp in enumerate(final_path)
        ],
    }
    Path(OUT_JSON).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON 저장: {OUT_JSON}")

    # 시각화 (카메라 딕셔너리 형식으로 변환)
    cam_dict = {str(i): cam_positions[i] for i in range(len(cam_positions))}
    visualize(pts, coverage, underobs_mask,
              valid_cands, scores, final_path, cam_dict, OUT_IMG, target=target)


if __name__ == "__main__":
    main()
