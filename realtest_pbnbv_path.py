"""
realtest_pbnbv_path.py

README에 명시된 PB-NBV(2) + Greedy Path Planning 파이프라인을
real_test(AirSim NED) + 실제 복원 점군(normals)에 적용.

파이프라인:
  [1] real_test 로드          → 카메라 위치·방향, FOV, 타겟
  [2] 포인트클라우드 준비     → 실제 점군(normals) 또는 합성 박스
  [3] Coverage 추정           → 포인트별 기존 관측 횟수
  [4] 후보 시점 생성          → 구면 격자 (고도 × 반경 × 방위각)
  [5] PB-NBV(2) 스코어링     → IG + 0.5 × lookahead 상위 N 선택
  [6] Greedy Path Planning    → Nearest-Neighbor 방문 순서 결정
  [7] 출력                    → JSON + 6패널 PNG + 인터랙티브 HTML

사용법:
  # 합성 박스 (self-occlusion 없음)
  python realtest_pbnbv_path.py

  # 실제 점군 + 물체 시각화
  python realtest_pbnbv_path.py \\
    --use-real-pts real_test/real_test_pts_normals.npz \\
    --pts-vis-npz  real_test/real_test_pts_vis.npz \\
    --n-select 10
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

for _fn in ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 알고리즘 파라미터 ─────────────────────────────────────────────────────────
MIN_DIST        = 4.0      # 근거리 한계 (m)
MAX_DIST        = 13.0     # 원거리 한계 (m)
UNDEROBS_THRESH = 2        # 미관측 기준 (관측 횟수 미만)
N_CANDIDATES    = 150      # 후보 시점 수
N_SELECT        = 10       # 최종 선택 웨이포인트 수
LOOKAHEAD       = 2        # PB-NBV lookahead 단계

# 후보 시점 구면 격자 파라미터
ORBIT_ALTITUDES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # 1m~7m, 1m 간격 (7개)
ORBIT_RADII     = [5.0, 6.5, 8.0]   # 수평 반경 (m)

# 합성 박스 파라미터 (--use-real-pts 미지정 시)
BOX_HALF_X    = 1.0
BOX_HALF_Y    = 1.0
BOX_HEIGHT    = 1.5
N_PTS_PER_FACE = 300


# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════════════════

def load_real_test(real_test_dir: Path, max_frames=None):
    manifest = json.loads((real_test_dir / "manifest.json").read_text(encoding="utf-8"))
    sp = manifest.get("straight_prefix_target")
    if sp:
        target = np.array(sp, dtype=float)
    else:
        tc = manifest.get("target_region_center")
        target = np.array(tc, dtype=float) if tc else None

    meta_files = sorted((real_test_dir / "meta").glob("*.json"))
    if max_frames is not None:
        meta_files = meta_files[:max_frames]

    cam_positions, fovs = [], []
    for mf in meta_files:
        d = json.loads(mf.read_text(encoding="utf-8"))
        pos = d["camera"]["pose"]["position"]
        cam_positions.append([pos["x"], pos["y"], pos["z"]])
        fovs.append(d["camera"]["fov"])

    cam_positions = np.array(cam_positions) if cam_positions else np.empty((0, 3))
    fov_deg = float(np.mean(fovs)) if fovs else 89.9

    if len(cam_positions) == 0:
        cam_forwards = np.empty((0, 3))
    elif target is not None:
        to_t = target - cam_positions
        cam_forwards = to_t / (np.linalg.norm(to_t, axis=1, keepdims=True) + 1e-8)
    else:
        center = cam_positions.mean(axis=0)
        to_c = center - cam_positions
        cam_forwards = to_c / (np.linalg.norm(to_c, axis=1, keepdims=True) + 1e-8)
        target = center

    return cam_positions, cam_forwards, fov_deg, target


def synthesize_box_pointcloud(target, hx, hy, h, n):
    tx, ty, tz = target
    top_z = tz - h
    pts = []
    for face_x in [tx - hx, tx + hx]:
        pts.append(np.column_stack([np.full(n, face_x),
            np.random.uniform(ty - hy, ty + hy, n),
            np.random.uniform(top_z, tz, n)]))
    for face_y in [ty - hy, ty + hy]:
        pts.append(np.column_stack([
            np.random.uniform(tx - hx, tx + hx, n),
            np.full(n, face_y),
            np.random.uniform(top_z, tz, n)]))
    for z_face in [top_z, tz]:
        pts.append(np.column_stack([
            np.random.uniform(tx - hx, tx + hx, n),
            np.random.uniform(ty - hy, ty + hy, n),
            np.full(n, z_face)]))
    return np.vstack(pts)


# ════════════════════════════════════════════════════════════════════════════
# 2-3. Coverage 추정
# ════════════════════════════════════════════════════════════════════════════

def point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist,
                     min_dist=MIN_DIST, normals=None):
    """[min_dist, max_dist] 밴드 + FOV 콘 안 가시성. normals 시 self-occlusion 체크."""
    v = pts - cam_pos
    dist = np.linalg.norm(v, axis=1)
    valid = (dist >= min_dist) & (dist <= max_dist)
    v_norm = v / (dist[:, None] + 1e-8)
    in_cone = (v_norm @ cam_dir) > math.cos(math.radians(fov_deg / 2.0))
    visible = valid & in_cone
    if normals is not None:
        front_face = (normals * (cam_pos - pts)).sum(axis=1) > 0
        visible = visible & front_face
    return visible


def compute_coverage(pts, cam_positions, cam_forwards, fov_deg, max_dist, normals=None):
    counts = np.zeros(len(pts), dtype=np.int32)
    for cam_pos, cam_dir in zip(cam_positions, cam_forwards):
        counts += point_in_frustum(cam_pos, cam_dir, fov_deg, pts,
                                   max_dist, normals=normals).astype(np.int32)
    return counts


# ════════════════════════════════════════════════════════════════════════════
# 4. 후보 시점 생성
# ════════════════════════════════════════════════════════════════════════════

def generate_candidates(target, altitudes, radii, n_total):
    """구면 격자: 고도 × 반경 × 방위각 균등 분할."""
    candidates = []
    n_per = max(1, n_total // (len(altitudes) * len(radii)))
    for alt in altitudes:
        z = target[2] - alt  # AirSim NED: Z음수=위
        for rad in radii:
            for i in range(n_per):
                theta = 2 * math.pi * i / n_per
                candidates.append([
                    target[0] + rad * math.cos(theta),
                    target[1] + rad * math.sin(theta),
                    z,
                ])
    return np.array(candidates)


# ════════════════════════════════════════════════════════════════════════════
# 5. PB-NBV(2) 스코어링
# ════════════════════════════════════════════════════════════════════════════

def information_gain(cam_pos, target, pts, underobs_mask, fov_deg, max_dist,
                     already_vis=None, normals=None):
    cam_dir = target - cam_pos
    cam_dir /= np.linalg.norm(cam_dir) + 1e-8
    visible = point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist, normals=normals)
    new_vis = visible & underobs_mask
    if already_vis is not None:
        new_vis = new_vis & ~already_vis
    # 거리 기반 가중치 (가까울수록 높음)
    d = np.linalg.norm(cam_pos - target)
    w = (1.0 / (d ** 2 + 1e-8)) / (1.0 / (MIN_DIST ** 2 + 1e-8))
    return float(new_vis.sum() * w), visible


def pbnbv_score(candidate, candidates, target, pts, underobs_mask,
                fov_deg, max_dist, lookahead, normals=None):
    """PB-NBV(lookahead): score = IG_1 + 0.5 × max(IG_2)"""
    ig1, vis1 = information_gain(candidate, target, pts, underobs_mask,
                                 fov_deg, max_dist, normals=normals)
    score = float(ig1)
    if lookahead >= 2 and ig1 > 0:
        updated_mask = underobs_mask & ~vis1
        sample_idx = np.random.choice(len(candidates), min(40, len(candidates)), replace=False)
        best_ig2 = 0.0
        for idx in sample_idx:
            if np.allclose(candidates[idx], candidate):
                continue
            ig2, _ = information_gain(candidates[idx], target, pts, updated_mask,
                                      fov_deg, max_dist, already_vis=vis1, normals=normals)
            best_ig2 = max(best_ig2, ig2)
        score += 0.5 * best_ig2
    return score


# ════════════════════════════════════════════════════════════════════════════
# 6. Greedy Path Planning
# ════════════════════════════════════════════════════════════════════════════

def greedy_path(selected_pts, start_pos):
    """Nearest-Neighbor Greedy: start_pos에서 출발해 방문 순서 인덱스 반환."""
    remaining = list(range(len(selected_pts)))
    order = []
    cur = np.array(start_pos)
    while remaining:
        dists = [np.linalg.norm(selected_pts[i] - cur) for i in remaining]
        nearest = remaining[int(np.argmin(dists))]
        order.append(nearest)
        cur = selected_pts[nearest]
        remaining.remove(nearest)
    return order


# ════════════════════════════════════════════════════════════════════════════
# 7a. 시각화 (PNG 6패널)
# ════════════════════════════════════════════════════════════════════════════

def visualize(pts, coverage, underobs_mask, candidates, cand_scores,
              selected_pts, selected_scores, path_order,
              cam_positions, target, out_path):

    def alt(z): return -np.asarray(z)  # NED → 고도 (양수=위)

    fig = plt.figure(figsize=(20, 12), facecolor="#0a0a14")
    fig.suptitle(f"PB-NBV({LOOKAHEAD}) + Greedy Path Planning  —  real_test",
                 color="white", fontsize=14, fontweight="bold")

    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                      left=0.05, right=0.97, top=0.93, bottom=0.05)
    ax_cov  = fig.add_subplot(gs[0, 0])
    ax_top  = fig.add_subplot(gs[0, 1])
    ax_sc   = fig.add_subplot(gs[0, 2])
    ax_path = fig.add_subplot(gs[1, 0])
    ax_3d   = fig.add_subplot(gs[1, 1], projection="3d")
    ax_p3d  = fig.add_subplot(gs[1, 2], projection="3d")

    bg = "#121220"
    for ax in [ax_cov, ax_top, ax_sc, ax_path]:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    tx, ty, tz = target
    c_arr = np.array(candidates)
    path_pts = selected_pts[path_order]

    # 1) Coverage 히스토그램
    ax_cov.set_title("포인트별 관측 횟수", color="white", fontsize=9)
    max_cnt = max(1, int(coverage.max()))
    ax_cov.hist(coverage, bins=max_cnt + 1, color="#4fc3f7", edgecolor="#0a0a14", alpha=0.85)
    ax_cov.axvline(UNDEROBS_THRESH - 0.5, color="#ef5350", lw=1.5, ls="--",
                   label=f"미관측 기준 <{UNDEROBS_THRESH}회")
    pct = 100 * underobs_mask.sum() / max(1, len(coverage))
    ax_cov.text(0.97, 0.95, f"미관측 {pct:.1f}%",
                transform=ax_cov.transAxes, color="#ef5350", fontsize=8,
                ha="right", va="top")
    ax_cov.set_xlabel("관측 횟수", color="#aaa", fontsize=7)
    ax_cov.set_ylabel("포인트 수", color="#aaa", fontsize=7)
    ax_cov.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white")

    # 2) Top-Down: 후보 시점 스코어 맵
    ax_top.set_title("후보 시점 스코어 (Top-Down)", color="white", fontsize=9)
    sc2 = ax_top.scatter(c_arr[:, 0], c_arr[:, 1], c=cand_scores, cmap="plasma",
                         s=20, alpha=0.7, zorder=3)
    cb2 = plt.colorbar(sc2, ax=ax_top, pad=0.01, fraction=0.04)
    cb2.set_label("PB-NBV score", color="white", fontsize=6)
    cb2.ax.tick_params(labelcolor="white", labelsize=6)
    # 선택된 시점 강조
    ax_top.scatter(selected_pts[:, 0], selected_pts[:, 1],
                   s=80, color="#ffcc02", edgecolors="white", lw=0.8, zorder=5,
                   label=f"선택 {len(selected_pts)}개")
    if len(cam_positions) > 0:
        ax_top.scatter(cam_positions[:, 0], cam_positions[:, 1],
                       s=10, color="white", alpha=0.3, zorder=2, label="기존 경로")
    ax_top.plot(tx, ty, "*", color="#00e5ff", ms=14, zorder=6,
                markeredgecolor="white", markeredgewidth=0.6)
    ax_top.text(tx, ty - 1.2, "TARGET", color="#00e5ff", fontsize=7, ha="center")
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax_top.set_ylabel("Y (m)", color="#aaa", fontsize=7)
    ax_top.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white",
                  loc="upper right")

    # 3) PB-NBV 스코어 분포 (상위 30개)
    ax_sc.set_title(f"PB-NBV({LOOKAHEAD}) 스코어 분포 (상위 30)", color="white", fontsize=9)
    n_sel = len(selected_pts)
    top30_idx = np.argsort(cand_scores)[::-1][:30]
    top30 = cand_scores[top30_idx]
    bar_colors = ["#ffcc02" if i < n_sel else "#4fc3f7" for i in range(len(top30))]
    ax_sc.barh(range(len(top30)), top30[::-1], color=bar_colors[::-1], alpha=0.85)
    if len(top30) >= n_sel:
        ax_sc.axvline(top30[n_sel - 1], color="#ef5350", lw=1.2, ls="--",
                      label=f"선택 기준 (상위 {n_sel}개)")
    ax_sc.text(0.97, 0.03, f"노랑={n_sel}개 선택 / 파랑=미선택",
               transform=ax_sc.transAxes, color="white", fontsize=7, ha="right", va="bottom")
    ax_sc.set_xlabel("Score", color="#aaa", fontsize=7)
    ax_sc.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white")

    # 4) Greedy 경로 Top-Down + 카메라 방향
    loop = np.vstack([path_pts, path_pts[0]])
    total_dist = sum(np.linalg.norm(loop[i+1] - loop[i]) for i in range(len(loop)-1))
    ax_path.set_title(f"Greedy Path  ({total_dist:.1f} m)  — 화살표=촬영방향", color="white", fontsize=9)
    if len(cam_positions) > 0:
        ax_path.scatter(cam_positions[:, 0], cam_positions[:, 1],
                        s=10, color="#555", alpha=0.4, zorder=2)
    ax_path.plot(loop[:, 0], loop[:, 1], color="#00e5ff", lw=2, alpha=0.9, zorder=4)
    arrow_len = 1.5  # 화살표 길이 (m)
    for i, wp in enumerate(path_pts):
        ax_path.plot(wp[0], wp[1], "o", color="#00e5ff", ms=8,
                     markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        ax_path.text(wp[0], wp[1] + 0.55, str(i + 1), color="white",
                     fontsize=7, ha="center", va="bottom", zorder=6)
        # 카메라 → 타겟 방향 화살표 (Top-Down: XY 평면)
        d2 = np.array([tx - wp[0], ty - wp[1]])
        d2 = d2 / (np.linalg.norm(d2) + 1e-8) * arrow_len
        ax_path.annotate("", xy=(wp[0] + d2[0], wp[1] + d2[1]), xytext=(wp[0], wp[1]),
                         arrowprops=dict(arrowstyle="->", color="#00ffcc",
                                         lw=1.2, mutation_scale=10), zorder=7)
    ax_path.plot(tx, ty, "*", color="#ffcc02", ms=14, zorder=8,
                 markeredgecolor="white", markeredgewidth=0.6)
    ax_path.text(tx, ty - 1.2, "TARGET", color="#ffcc02", fontsize=7, ha="center")
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax_path.set_ylabel("Y (m)", color="#aaa", fontsize=7)

    # 5) 3D: 미관측 점군 + 후보
    ax_3d.set_facecolor("#0a0a14")
    ax_3d.set_title("미관측 영역 + 후보 (3D)", color="white", fontsize=9)
    under_pts = pts[underobs_mask]
    if len(under_pts) > 3000:
        idx = np.random.choice(len(under_pts), 3000, replace=False)
        under_pts = under_pts[idx]
    ax_3d.scatter(under_pts[:, 0], under_pts[:, 1], alt(under_pts[:, 2]),
                  c="#ef5350", s=1, alpha=0.3)
    ax_3d.scatter(c_arr[:, 0], c_arr[:, 1], alt(c_arr[:, 2]),
                  c=cand_scores, cmap="plasma", s=15, alpha=0.6, zorder=4)
    ax_3d.scatter([tx], [ty], alt([tz]), color="#ffcc02", s=100, marker="*", zorder=6)
    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")
    ax_3d.set_xlabel("X", color="#aaa", fontsize=6, labelpad=2)
    ax_3d.set_ylabel("Y", color="#aaa", fontsize=6, labelpad=2)
    ax_3d.set_zlabel("Alt (m)", color="#aaa", fontsize=6, labelpad=2)
    ax_3d.tick_params(colors="white", labelsize=5)
    ax_3d.view_init(elev=25, azim=-60)

    # 6) Greedy 경로 3D + 카메라 방향
    ax_p3d.set_facecolor("#0a0a14")
    ax_p3d.set_title("Greedy Path 3D  — 화살표=촬영방향", color="white", fontsize=9)
    z_a = alt(path_pts[:, 2])
    cmap_p = plt.cm.plasma
    z_min, z_max = z_a.min(), z_a.max()
    for i in range(len(path_pts) - 1):
        t_c = (z_a[i] - z_min) / (z_max - z_min + 1e-8)
        ax_p3d.plot([path_pts[i, 0], path_pts[i+1, 0]],
                    [path_pts[i, 1], path_pts[i+1, 1]],
                    [z_a[i], z_a[i+1]],
                    color=cmap_p(t_c), lw=2.2, alpha=0.95)
    ax_p3d.scatter(path_pts[:, 0], path_pts[:, 1], z_a,
                   c=z_a, cmap="plasma", s=55, edgecolors="white", lw=0.5, zorder=5)
    for i, (x, y, za) in enumerate(zip(path_pts[:, 0], path_pts[:, 1], z_a)):
        ax_p3d.text(x, y, za + 0.35, str(i + 1), color="white", fontsize=6)
        # 카메라 방향 화살표 (3D, alt 공간)
        wp = path_pts[i]
        d3 = np.array([tx - wp[0], ty - wp[1], -tz + wp[2]])  # alt 공간: z 반전
        d3 = d3 / (np.linalg.norm(d3) + 1e-8) * arrow_len
        ax_p3d.quiver(x, y, za, d3[0], d3[1], d3[2],
                      color="#00ffcc", length=1.0, arrow_length_ratio=0.35,
                      linewidth=1.2, alpha=0.85)
    ax_p3d.scatter([tx], [ty], alt([tz]), color="#ffcc02", s=150, marker="*", zorder=6)
    for pane in [ax_p3d.xaxis.pane, ax_p3d.yaxis.pane, ax_p3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")
    ax_p3d.set_xlabel("X", color="#aaa", fontsize=6, labelpad=2)
    ax_p3d.set_ylabel("Y", color="#aaa", fontsize=6, labelpad=2)
    ax_p3d.set_zlabel("Alt (m)", color="#aaa", fontsize=6, labelpad=2)
    ax_p3d.tick_params(colors="white", labelsize=5)
    ax_p3d.view_init(elev=28, azim=-55)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"PNG 저장: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 7b. 인터랙티브 HTML (Plotly)
# ════════════════════════════════════════════════════════════════════════════

def visualize_html(pts_vis_npz, selected_pts, selected_scores, path_order,
                   cam_positions, target, out_path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [!] plotly 미설치 → HTML 건너뜀")
        return

    def alt(z): return -np.asarray(z)

    fig = go.Figure()

    # 기존 경로
    if len(cam_positions) > 0:
        cl = np.vstack([cam_positions, cam_positions[0]])
        fig.add_trace(go.Scatter3d(
            x=cl[:, 0], y=cl[:, 1], z=alt(cl[:, 2]),
            mode="lines+markers",
            line=dict(color="#55aaff", width=3),
            marker=dict(size=5, color="#55aaff"),
            name=f"기존 경로 ({len(cam_positions)}개)"))

    # Greedy 경로 선
    path_pts = selected_pts[path_order]
    sc_ord = selected_scores[path_order]
    loop = np.vstack([path_pts, path_pts[0]])
    total_dist = sum(np.linalg.norm(loop[i+1] - loop[i]) for i in range(len(loop)-1))
    fig.add_trace(go.Scatter3d(
        x=loop[:, 0], y=loop[:, 1], z=alt(loop[:, 2]),
        mode="lines",
        line=dict(color="#ff6b35", width=3, dash="dash"),
        name=f"Greedy path ({total_dist:.1f}m)",
        hoverinfo="skip"))

    # 카메라 방향 콘 (Plotly Cone)
    dirs = np.array([target - wp for wp in path_pts])
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    fig.add_trace(go.Cone(
        x=path_pts[:, 0], y=path_pts[:, 1], z=alt(path_pts[:, 2]),
        u=dirs[:, 0], v=dirs[:, 1], w=-dirs[:, 2],  # alt 공간: w 반전
        sizemode="absolute", sizeref=1.5,
        colorscale=[[0, "#00ffcc"], [1, "#00ffcc"]],
        showscale=False, opacity=0.7,
        name="촬영 방향",
        hoverinfo="skip"))

    # 선택 시점 (방문 순서 번호)
    ranks = np.arange(1, len(path_pts) + 1)
    fig.add_trace(go.Scatter3d(
        x=path_pts[:, 0], y=path_pts[:, 1], z=alt(path_pts[:, 2]),
        mode="markers+text",
        text=[f"#{r}" for r in ranks],
        textposition="top center",
        textfont=dict(color="white", size=10),
        marker=dict(
            size=10, color=ranks, colorscale="Plasma_r",
            showscale=True, cmin=1, cmax=len(ranks),
            colorbar=dict(title=dict(text="방문순서", font=dict(color="white")), thickness=12,
                          tickfont=dict(color="white")),
            line=dict(color="white", width=0.8)),
        name=f"PB-NBV 선택 ({len(path_pts)}개)",
        customdata=sc_ord,
        hovertemplate=(
            "방문순서=%{text}<br>"
            "x=%{x:.2f} y=%{y:.2f} alt=%{z:.2f}m<br>"
            "score=%{customdata:.2f}<extra></extra>")))

    # 물체 점군 (시각화용)
    if pts_vis_npz:
        try:
            obj_data = np.load(pts_vis_npz)
            obj_pts = obj_data["points"]
            if "colors" in obj_data:
                c = obj_data["colors"]
                obj_colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                              for r, g, b in c]
            else:
                obj_colors = "#00e676"
            fig.add_trace(go.Scatter3d(
                x=obj_pts[:, 0], y=obj_pts[:, 1], z=alt(obj_pts[:, 2]),
                mode="markers",
                marker=dict(size=3, color=obj_colors, opacity=0.9, line=dict(width=0)),
                name=f"물체 ({len(obj_pts):,}pts)"))
        except Exception as e:
            print(f"  [!] 물체 점군 로드 실패: {e}")

    # 타겟
    fig.add_trace(go.Scatter3d(
        x=[target[0]], y=[target[1]], z=alt([target[2]]),
        mode="markers+text",
        marker=dict(size=12, color="#ffcc02", symbol="diamond",
                    line=dict(color="white", width=1)),
        text=["TARGET"], textposition="top center",
        textfont=dict(color="#ffcc02", size=13),
        name="Target",
        hovertemplate="TARGET<br>x=%{x:.2f} y=%{y:.2f} alt=%{z:.2f}m<extra></extra>"))

    fig.update_layout(
        title=dict(
            text=(f"PB-NBV({LOOKAHEAD}) + Greedy Path Planning — real_test<br>"
                  "<sup>drag to rotate · scroll to zoom · double-click to reset</sup>"),
            font=dict(color="white", size=15), x=0.5),
        scene=dict(
            xaxis=dict(title=dict(text="X (m)", font=dict(color="white")), backgroundcolor="#0d0d1e",
                       gridcolor="#333", zerolinecolor="#555",
                       tickfont=dict(color="white")),
            yaxis=dict(title=dict(text="Y (m)", font=dict(color="white")), backgroundcolor="#0d0d1e",
                       gridcolor="#333", zerolinecolor="#555",
                       tickfont=dict(color="white")),
            zaxis=dict(title=dict(text="Altitude (m)", font=dict(color="white")), backgroundcolor="#0d0d1e",
                       gridcolor="#333", zerolinecolor="#555",
                       tickfont=dict(color="white")),
            bgcolor="#0d0d1e",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            aspectmode="data"),
        paper_bgcolor="#0a0a14", plot_bgcolor="#0a0a14",
        legend=dict(font=dict(color="white"), bgcolor="#1a1a30",
                    bordercolor="#444", x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=80, b=0), height=750)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    win_path = str(out_path).replace('/mnt/c/', 'C:\\').replace('/', '\\')
    print(f"HTML 저장: {out_path}")
    print(f"Windows: {win_path}")


# ════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="PB-NBV(2) + Greedy Path Planning (real_test)")
    ap.add_argument("--real-test-dir", default="real_test")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="사용할 최대 프레임 수")
    ap.add_argument("--use-real-pts", default=None,
                    help="실제 점군 .npz (points+normals). 미지정시 합성 박스")
    ap.add_argument("--pts-vis-npz", default=None,
                    help="HTML 물체 시각화용 .npz (points+colors)")
    ap.add_argument("--n-select", type=int, default=N_SELECT,
                    help=f"최종 선택 웨이포인트 수 (기본 {N_SELECT})")
    ap.add_argument("--n-candidates", type=int, default=N_CANDIDATES,
                    help=f"후보 시점 수 (기본 {N_CANDIDATES})")
    ap.add_argument("--output-dir", default="results/realtest_pbnbv_path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rt = Path(args.real_test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] 데이터 로드 ─────────────────────────────────────────────────────
    print("[1/6] real_test 로드...")
    cam_positions, cam_forwards, fov_deg, target = load_real_test(rt, args.max_frames)
    print(f"  프레임 {len(cam_positions)}개  FOV {fov_deg:.1f}°  "
          f"타겟 ({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})")

    # ── [2] 포인트클라우드 ──────────────────────────────────────────────────
    print("[2/6] 포인트클라우드 준비...")
    normals = None
    if args.use_real_pts:
        data = np.load(args.use_real_pts)
        pts = data["points"].astype(np.float64)
        normals = data["normals"].astype(np.float64)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (nlen + 1e-8)
        print(f"  실제 점군 {len(pts):,}개  (self-occlusion 활성)")
    else:
        pts = synthesize_box_pointcloud(target, BOX_HALF_X, BOX_HALF_Y, BOX_HEIGHT,
                                        N_PTS_PER_FACE)
        print(f"  합성 박스 점군 {len(pts):,}개")

    # ── [3] Coverage 추정 ───────────────────────────────────────────────────
    print("[3/6] Coverage 추정...")
    if len(cam_positions) > 0:
        coverage = compute_coverage(pts, cam_positions, cam_forwards, fov_deg,
                                    MAX_DIST, normals=normals)
    else:
        coverage = np.zeros(len(pts), dtype=np.int32)

    underobs_mask = coverage < UNDEROBS_THRESH
    if len(cam_positions) == 0 or underobs_mask.sum() == 0:
        underobs_mask = np.ones(len(pts), dtype=bool)
    pct = 100 * underobs_mask.sum() / max(1, len(pts))
    print(f"  미관측 {underobs_mask.sum():,}개 ({pct:.1f}%)")

    # ── [4] 후보 시점 생성 ──────────────────────────────────────────────────
    print(f"[4/6] 후보 {args.n_candidates}개 생성 (구면 격자)...")
    candidates = generate_candidates(target, ORBIT_ALTITUDES, ORBIT_RADII, args.n_candidates)
    # 기존 카메라와 너무 가까운 후보 제거
    if len(cam_positions) > 0:
        keep = np.array([
            np.linalg.norm(cam_positions - c, axis=1).min() >= 0.3
            for c in candidates
        ])
        candidates = candidates[keep]
    print(f"  유효 후보 {len(candidates)}개")
    print(f"  고도 범위: {-candidates[:,2].max():.1f} ~ {-candidates[:,2].min():.1f}m  "
          f"수평 반경: {ORBIT_RADII[0]}~{ORBIT_RADII[-1]}m")

    # ── [5] PB-NBV(2) 스코어링 → 상위 N 선택 ───────────────────────────────
    print(f"[5/6] PB-NBV({LOOKAHEAD}) 스코어링 ({len(candidates)}개)...")
    cand_scores = np.array([
        pbnbv_score(c, candidates, target, pts, underobs_mask,
                    fov_deg, MAX_DIST, LOOKAHEAD, normals=normals)
        for c in candidates
    ])
    print(f"  스코어 범위: {cand_scores.min():.2f} ~ {cand_scores.max():.2f}")

    top_idx = np.argsort(cand_scores)[::-1][:args.n_select]
    selected_pts = candidates[top_idx]
    selected_scores = cand_scores[top_idx]
    print(f"  선택 {len(selected_pts)}개: {selected_scores[0]:.2f} ~ {selected_scores[-1]:.2f}")

    # ── [6] Greedy Path Planning ────────────────────────────────────────────
    print("[6/6] Greedy Path Planning...")
    if len(cam_positions) > 0:
        start_pos = cam_positions[-1]
        print(f"  출발점: 마지막 기존 카메라 {start_pos}")
    else:
        start_pos = np.array([target[0] - 8.0, target[1], target[2] - 5.0])
        print(f"  출발점 (기본 진입점): ({start_pos[0]:.1f},{start_pos[1]:.1f},{start_pos[2]:.1f})")

    path_order = greedy_path(selected_pts, start_pos)
    path_pts = selected_pts[path_order]
    total_dist = sum(np.linalg.norm(path_pts[i+1] - path_pts[i])
                     for i in range(len(path_pts) - 1))
    print(f"  총 이동거리: {total_dist:.1f}m")
    for i, oi in enumerate(path_order):
        wp = selected_pts[oi]
        rad = math.hypot(wp[0] - target[0], wp[1] - target[1])
        az = math.degrees(math.atan2(wp[1] - target[1], wp[0] - target[0]))
        print(f"  WP#{i+1:02d}: ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f})  "
              f"고도{abs(wp[2]):.1f}m  반경{rad:.1f}m  방위{az:+.0f}°  "
              f"score={selected_scores[oi]:.2f}")

    # ── 출력 ────────────────────────────────────────────────────────────────
    out_json = out_dir / "realtest_pbnbv_path.json"
    result = {
        "schema_version": 1,
        "name": "realtest_pbnbv_path",
        "algorithm": f"PB-NBV({LOOKAHEAD}) + Greedy Path Planning",
        "coordinate_system": "AirSim NED (Z<0=above ground)",
        "parameters": {
            "lookahead": LOOKAHEAD,
            "n_candidates": int(len(candidates)),
            "n_select": int(len(selected_pts)),
            "orbit_altitudes_m": ORBIT_ALTITUDES,
            "orbit_radii_m": ORBIT_RADII,
            "max_dist_m": MAX_DIST,
            "min_dist_m": MIN_DIST,
            "underobs_thresh": UNDEROBS_THRESH,
        },
        "n_existing_cameras": int(len(cam_positions)),
        "n_pts": int(len(pts)),
        "underobserved_ratio": float(underobs_mask.mean()),
        "target_position": target.tolist(),
        "total_path_distance_m": float(total_dist),
        "path_step_count": int(len(path_pts)),
        "waypoints": [
            {
                "index": i + 1,
                "position": path_pts[i].tolist(),
                "relative_to_target": (path_pts[i] - target).tolist(),
                "altitude_m": float(abs(path_pts[i][2])),
                "radius_m": float(math.hypot(path_pts[i][0] - target[0],
                                             path_pts[i][1] - target[1])),
                "azimuth_deg": float(math.degrees(math.atan2(
                    path_pts[i][1] - target[1], path_pts[i][0] - target[0]))),
                "pbnbv_score": float(selected_scores[path_order[i]]),
            }
            for i in range(len(path_pts))
        ],
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON: {out_json}")

    out_png = out_dir / "realtest_pbnbv_path.png"
    visualize(pts, coverage, underobs_mask, candidates, cand_scores,
              selected_pts, selected_scores, path_order,
              cam_positions, target, str(out_png))

    out_html = out_dir / "realtest_pbnbv_path.html"
    visualize_html(args.pts_vis_npz, selected_pts, selected_scores, path_order,
                   cam_positions, target, str(out_html))

    print("\n완료.")


if __name__ == "__main__":
    main()
