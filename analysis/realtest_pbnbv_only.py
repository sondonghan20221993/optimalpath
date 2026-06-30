"""
realtest_pbnbv_only.py

PB-NBV(2) "지점 생성"만 단독 실행합니다. (Greedy 경로 연결 없음)
선택된 시점들을 선으로 잇지 않고 점 분포 그대로 시각화하여
PB-NBV가 객관적으로 어떤 지점을 뽑는지 판단할 수 있게 합니다.

사용법:
  python realtest_pbnbv_only.py
  python realtest_pbnbv_only.py --n-select 12
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

for _fn in ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 알고리즘 파라미터 (realtest_optimal_path.py와 동일) ──────────────────────
N_PTS_PER_FACE   = 300
MIN_DIST         = 4.0     # 근거리 한계
MAX_DIST         = 13.0    # 원거리 한계
UNDEROBS_THRESH  = 2
N_CANDIDATES     = 200
N_SELECT         = 12
LOOKAHEAD        = 2

CAND_ALTITUDES = [4.0, 5.5, 7.0, 8.5]
CAND_RADII     = [5.0, 6.5, 8.0]

BOX_HALF_X = 1.0
BOX_HALF_Y = 1.0
BOX_HEIGHT = 1.5


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def load_real_test(real_test_dir: Path, max_frames: int = None):
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


def synthesize_box_pointcloud(target, half_x, half_y, height, n_per_face):
    tx, ty, tz = target
    top_z = tz - height
    pts = []
    for face_x in [tx - half_x, tx + half_x]:
        ys = np.random.uniform(ty - half_y, ty + half_y, n_per_face)
        zs = np.random.uniform(top_z, tz, n_per_face)
        pts.append(np.column_stack([np.full(n_per_face, face_x), ys, zs]))
    for face_y in [ty - half_y, ty + half_y]:
        xs = np.random.uniform(tx - half_x, tx + half_x, n_per_face)
        zs = np.random.uniform(top_z, tz, n_per_face)
        pts.append(np.column_stack([xs, np.full(n_per_face, face_y), zs]))
    xs = np.random.uniform(tx - half_x, tx + half_x, n_per_face)
    ys = np.random.uniform(ty - half_y, ty + half_y, n_per_face)
    pts.append(np.column_stack([xs, ys, np.full(n_per_face, top_z)]))
    xs = np.random.uniform(tx - half_x, tx + half_x, n_per_face)
    ys = np.random.uniform(ty - half_y, ty + half_y, n_per_face)
    pts.append(np.column_stack([xs, ys, np.full(n_per_face, tz)]))
    return np.vstack(pts)


def point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist, min_dist=MIN_DIST,
                     normals=None):
    """카메라 가시 밴드 [min_dist, max_dist] + FOV 콘 안에 있는 점만 True.
    normals 제공 시 self-occlusion 체크 추가 (법선이 카메라 방향을 향하는 면만 가시)."""
    v = pts - cam_pos
    dist = np.linalg.norm(v, axis=1)
    valid = (dist >= min_dist) & (dist <= max_dist)
    v_norm = v / (dist[:, None] + 1e-8)
    cos_half = math.cos(math.radians(fov_deg / 2.0))
    in_cone = (v_norm @ cam_dir) > cos_half
    visible = valid & in_cone
    if normals is not None:
        # dot(normal, cam_pos - point) > 0 → 카메라가 면 앞쪽 (self-occlusion 제거)
        cam_to_pt = cam_pos - pts   # (N, 3)
        front_face = (normals * cam_to_pt).sum(axis=1) > 0
        visible = visible & front_face
    return visible


def point_distance_weights(cam_pos, target, max_dist, min_dist=MIN_DIST):
    """카메라→타겟 중심 거리 기반 가중치 (가까울수록 높음, 방향 편향 없음)."""
    d = np.linalg.norm(cam_pos - target)
    w = 1.0 / (d ** 2 + 1e-8)
    w_max = 1.0 / (min_dist ** 2 + 1e-8)
    return w / (w_max + 1e-8)


def compute_coverage(pts, cam_positions, cam_forwards, fov_deg, max_dist,
                     min_dist=MIN_DIST, normals=None):
    counts = np.zeros(len(pts), dtype=np.int32)
    for cam_pos, cam_dir in zip(cam_positions, cam_forwards):
        counts += point_in_frustum(cam_pos, cam_dir, fov_deg, pts,
                                   max_dist, min_dist, normals).astype(np.int32)
    return counts


def generate_candidates(target, altitudes, radii, n_total):
    """기존 방식: 동심원(ring) 위에만 후보 생성."""
    candidates = []
    n_per = max(1, n_total // (len(altitudes) * len(radii)))
    for alt in altitudes:
        z = target[2] - alt
        for rad in radii:
            for i in range(n_per):
                theta = 2 * math.pi * i / n_per
                x = target[0] + rad * math.cos(theta)
                y = target[1] + rad * math.sin(theta)
                candidates.append([x, y, z])
    return np.array(candidates)


def generate_candidates_uniform(target, n_total, r_min=MIN_DIST, r_max=MAX_DIST,
                                alt_min=1.0, alt_max=7.0):
    """
    방위각을 균일하게 나눠서 후보 생성 (한쪽 쏠림 방지).
    각 방위각 슬라이스마다 고도·반경을 무작위로 샘플.
    """
    tx, ty, tz = target
    n_az = 36                        # 방위각 36등분 (10°씩)
    n_per_az = math.ceil(n_total / n_az)
    out = []
    for i in range(n_az):
        az_min = 2 * math.pi * i / n_az
        az_max = 2 * math.pi * (i + 1) / n_az
        for _ in range(n_per_az * 6):
            if len(out) >= n_total:
                break
            az = np.random.uniform(az_min, az_max)
            alt = np.random.uniform(alt_min, alt_max)
            # 수평 반경: 3D 거리 밴드와 고도를 고려
            r3d = np.random.uniform(r_min, r_max)
            dz = alt
            r_horiz_sq = r3d**2 - dz**2
            if r_horiz_sq <= 0:
                continue
            r_horiz = math.sqrt(r_horiz_sq)
            x = tx + r_horiz * math.cos(az)
            y = ty + r_horiz * math.sin(az)
            z = tz - alt
            out.append([x, y, z])
    return np.array(out[:n_total])


def information_gain(cam_pos, target, pts, underobs_mask, fov_deg, max_dist,
                     already_vis=None, normals=None):
    cam_dir = target - cam_pos
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-8)
    visible = point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist, normals=normals)
    new_vis = visible & underobs_mask
    if already_vis is not None:
        new_vis = new_vis & ~already_vis
    w = point_distance_weights(cam_pos, target, max_dist)
    score = float(new_vis.sum() * w)
    return score, visible


def pbnbv_score(candidate, candidates, target, pts, underobs_mask, fov_deg, max_dist,
                lookahead, normals=None):
    ig1, vis1 = information_gain(candidate, target, pts, underobs_mask, fov_deg, max_dist,
                                 normals=normals)
    score = float(ig1)
    if lookahead >= 2 and ig1 > 0:
        updated_mask = underobs_mask & ~vis1
        sample_idx = np.random.choice(len(candidates), min(40, len(candidates)), replace=False)
        best_ig2 = 0
        for idx in sample_idx:
            if np.allclose(candidates[idx], candidate):
                continue
            ig2, _ = information_gain(candidates[idx], target, pts, updated_mask,
                                      fov_deg, max_dist, already_vis=vis1, normals=normals)
            best_ig2 = max(best_ig2, ig2)
        score += 0.5 * best_ig2
    return score


def sequential_pbnbv(candidates, target, pts, underobs_mask, fov_deg, max_dist, n_select,
                     normals=None, search_radius=None):
    """순차적 PB-NBV.
    search_radius 지정 시: 현재 위치에서 반경 내 후보만 평가 (위치 기반 DFS).
    None 이면 기존 전역 선택.
    """
    remaining_mask = underobs_mask.copy()
    selected = []
    selected_scores = []
    selected_set = set()
    current_pos = None  # 현재 드론 위치

    for step in range(n_select):
        if remaining_mask.sum() == 0:
            print(f"  [커버 완료] 전체 재시작...")
            remaining_mask = underobs_mask.copy()

        # 현재 위치 기반이면 반경 내 후보만, 아니면 전체
        if search_radius is not None and current_pos is not None:
            dists_from_cur = np.linalg.norm(candidates - current_pos, axis=1)
            local_mask = dists_from_cur <= search_radius
            # 반경 내 후보가 없으면 반경 확장
            if local_mask.sum() == 0:
                local_mask = np.ones(len(candidates), dtype=bool)
        else:
            local_mask = np.ones(len(candidates), dtype=bool)

        best_score, best_cand, best_vis, best_i = -1, None, None, -1
        for i, (cand, in_local) in enumerate(zip(candidates, local_mask)):
            if not in_local or i in selected_set:
                continue
            ig, vis = information_gain(cand, target, pts, remaining_mask, fov_deg, max_dist,
                                       normals=normals)
            if ig > best_score:
                best_score = ig
                best_cand = cand
                best_vis = vis
                best_i = i

        if best_cand is None:
            print(f"  [!] 선택 가능한 후보 없음 → 중단")
            break

        if best_score <= 0:
            print(f"  [커버 한계] 재시작...")
            remaining_mask = underobs_mask.copy()
            continue

        selected.append(best_cand)
        selected_scores.append(best_score)
        selected_set.add(best_i)
        current_pos = best_cand  # 현재 위치 갱신
        remaining_mask = remaining_mask & ~best_vis

        covered = (~remaining_mask & underobs_mask).sum()
        total = underobs_mask.sum()
        travel = f"  이동{np.linalg.norm(best_cand - selected[-2]):.1f}m" if len(selected) > 1 else ""
        print(f"  step {step+1:02d}: score={best_score:.1f}  "
              f"coverage {covered}/{total} ({100*covered/total:.1f}%){travel}")

    return np.array(selected), np.array(selected_scores)


def local_dfs_pbnbv(start_pos, target, pts, underobs_mask, fov_deg, max_dist, n_select,
                    normals=None, step_size=3.0, n_az=24, alt_offsets=(-1.5, 0, 1.5),
                    alt_min=1.0, alt_max=10.0):
    """
    원형 후보 없는 위치 기반 DFS:
    현재 위치에서 n_az 방향 × alt_offsets 고도 조합으로 로컬 후보를 동적 생성,
    info_gain 최고 방향으로 step_size 이동 반복.
    """
    remaining_mask = underobs_mask.copy()
    selected = [np.array(start_pos)]
    selected_scores = [0.0]
    cur = np.array(start_pos, dtype=float)

    # 첫 위치 커버리지 반영
    _, vis0 = information_gain(cur, target, pts, remaining_mask, fov_deg, max_dist, normals=normals)
    remaining_mask = remaining_mask & ~vis0
    covered = (~remaining_mask & underobs_mask).sum()
    total = underobs_mask.sum()
    print(f"  start : coverage {covered}/{total} ({100*covered/total:.1f}%)")

    visited = [np.array(start_pos)]  # 방문 이력

    for step in range(n_select - 1):
        if remaining_mask.sum() == 0:
            print(f"  [커버 완료 {step+1}스텝] 재시작...")
            remaining_mask = underobs_mask.copy()

        # 현재 위치에서 로컬 후보 생성 (이미 방문한 위치 제외)
        local_cands = []
        for az_i in range(n_az):
            az = 2 * math.pi * az_i / n_az
            dx = step_size * math.cos(az)
            dy = step_size * math.sin(az)
            for dz in alt_offsets:
                cand = cur + np.array([dx, dy, dz])
                alt_val = abs(cand[2] - target[2])
                if not (alt_min <= alt_val <= alt_max):
                    continue
                # 이미 방문한 위치와 너무 가까우면 제외
                too_close = any(np.linalg.norm(cand - v) < step_size * 0.7 for v in visited)
                if not too_close:
                    local_cands.append(cand)

        if not local_cands:
            print(f"  [!] 주변에 미방문 후보 없음 → 중단")
            break

        best_score, best_cand, best_vis = -1, None, None
        for cand in local_cands:
            ig, vis = information_gain(cand, target, pts, remaining_mask, fov_deg, max_dist,
                                       normals=normals)
            if ig > best_score:
                best_score, best_cand, best_vis = ig, cand, vis

        if best_score <= 0:
            print(f"  [커버 한계] 재시작...")
            remaining_mask = underobs_mask.copy()
            continue

        selected.append(best_cand)
        selected_scores.append(best_score)
        cur = best_cand
        visited.append(best_cand)
        remaining_mask = remaining_mask & ~best_vis

        covered = (~remaining_mask & underobs_mask).sum()
        print(f"  step {step+1:02d}: score={best_score:.1f}  "
              f"coverage {covered}/{total} ({100*covered/total:.1f}%)  "
              f"위치({cur[0]:.1f},{cur[1]:.1f},고도{abs(cur[2]):.1f}m)")

    return np.array(selected), np.array(selected_scores)


# ── 시각화 (경로 선 없이 선택 지점만) ────────────────────────────────────────
def visualize_points_only(pts, candidates, scores, selected_pts, selected_scores,
                          existing_cams, target, out_path, show_candidates=True):
    fig = plt.figure(figsize=(20, 9), facecolor="#0a0a14")

    # 2열: 왼쪽 Top-Down 크게, 오른쪽 위 3D + 아래 점수
    gs = plt.GridSpec(2, 2, figure=fig, width_ratios=[1.4, 1],
                      wspace=0.25, hspace=0.35,
                      left=0.05, right=0.97, top=0.93, bottom=0.06)
    ax_top = fig.add_subplot(gs[:, 0])       # 왼쪽 전체
    ax_3d  = fig.add_subplot(gs[0, 1], projection="3d")  # 오른쪽 위
    ax_sc  = fig.add_subplot(gs[1, 1])       # 오른쪽 아래

    bg = "#121220"
    for ax in [ax_top, ax_sc]:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    tx, ty, tz = target
    sel = np.array(selected_pts)

    # ── Top-Down ──────────────────────────────────────────────────────────────
    ax_top.set_facecolor("#0d0d1e")
    ax_top.set_title("PB-NBV Viewpoint Selection  (Top-Down)",
                     color="white", fontsize=12, fontweight="bold", pad=10)

    # 기존 real_test 경로
    if len(existing_cams) > 0:
        ax_top.plot(np.append(existing_cams[:, 0], existing_cams[0, 0]),
                    np.append(existing_cams[:, 1], existing_cams[0, 1]),
                    color="#55aaff", linewidth=2.0, alpha=0.7, zorder=3, linestyle="--")
        ax_top.scatter(existing_cams[:, 0], existing_cams[:, 1],
                       color="#55aaff", s=55, alpha=1.0, zorder=4, edgecolors="white",
                       linewidths=0.5, label=f"real_test path  ({len(existing_cams)} frames)")

    # 순위 기반 색상 (1위=노랑, 꼴찌=보라)
    ranks = np.argsort(np.argsort(-selected_scores)) + 1  # 1=최고점
    sc = ax_top.scatter(sel[:, 0], sel[:, 1], c=ranks, cmap="plasma_r",
                        s=220, edgecolors="white", linewidths=1.0, zorder=6,
                        vmin=1, vmax=len(sel),
                        label=f"PB-NBV selected  ({len(sel)})")
    cb = plt.colorbar(sc, ax=ax_top, pad=0.02, fraction=0.035)
    cb.set_label("Rank  (1=best)", color="white", fontsize=9)
    cb.ax.tick_params(labelcolor="white", labelsize=8)
    cb.ax.yaxis.label.set_color("white")

    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (m)", color="#bbb", fontsize=9)
    ax_top.set_ylabel("Y (m)", color="#bbb", fontsize=9)
    ax_top.legend(fontsize=9, facecolor="#1a1a30", edgecolor="#444",
                  labelcolor="white", loc="upper right")
    ax_top.tick_params(colors="white")
    for sp in ax_top.spines.values():
        sp.set_edgecolor("#333")

    # ── 3D ───────────────────────────────────────────────────────────────────
    ax_3d.set_facecolor("#0a0a14")
    ax_3d.set_title("3D View", color="white", fontsize=10, pad=6)
    if len(existing_cams) > 0:
        ax_3d.scatter(existing_cams[:, 0], existing_cams[:, 1], existing_cams[:, 2],
                      color="#55aaff", s=30, alpha=0.9, zorder=4)
    ax_3d.scatter(sel[:, 0], sel[:, 1], sel[:, 2],
                  c=ranks, cmap="plasma_r",
                  s=60, edgecolors="white", linewidths=0.4, zorder=5,
                  vmin=1, vmax=len(sel))
    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")
    ax_3d.set_xlabel("X", color="#aaa", fontsize=7)
    ax_3d.set_ylabel("Y", color="#aaa", fontsize=7)
    ax_3d.set_zlabel("Z", color="#aaa", fontsize=7)
    ax_3d.tick_params(colors="white", labelsize=6)
    ax_3d.view_init(elev=28, azim=-55)

    # ── 점수 분포 ─────────────────────────────────────────────────────────────
    ax_sc.set_title("Score Distribution", color="white", fontsize=10)
    order = np.argsort(scores)[::-1]
    ranked = scores[order]
    bar_colors = ["#ffcc02" if i < len(sel) else "#334" for i in range(len(ranked))]
    ax_sc.bar(range(len(ranked)), ranked, color=bar_colors, width=1.0)
    ax_sc.axvline(len(sel) - 1, color="#ef5350", linewidth=1.5, linestyle="--",
                  label=f"cutoff (top {len(sel)})")
    ax_sc.set_xlabel("Rank", color="#bbb", fontsize=8)
    ax_sc.set_ylabel("Score", color="#bbb", fontsize=8)
    uniq = np.unique(np.round(scores, 1))
    disc = "discriminative" if len(uniq) > 2 else "all tied"
    ax_sc.text(0.98, 0.95,
               f"unique: {len(uniq)}  |  {scores.min():.0f} ~ {scores.max():.0f}  |  {disc}",
               transform=ax_sc.transAxes, color="white", fontsize=8,
               ha="right", va="top", bbox=dict(facecolor="#1a1a30", edgecolor="#444"))
    ax_sc.legend(fontsize=8, facecolor=bg, edgecolor="#444", labelcolor="white")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"시각화 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PB-NBV(2) 지점 생성만 단독 실행")
    parser.add_argument("--real-test-dir", default="real_test")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="사용할 최대 프레임 수 (예: 17 = 1바퀴만)")
    parser.add_argument("--output", default="results/realtest_pbnbv_ring/realtest_pbnbv_points.json")
    parser.add_argument("--output-img", default="results/realtest_pbnbv_ring/realtest_pbnbv_points.png")
    parser.add_argument("--n-select", type=int, default=N_SELECT)
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--candidate-mode", choices=["ring", "uniform"], default="ring",
                        help="ring=동심원(기존), uniform=공간 균일 무작위(공정 검증)")
    parser.add_argument("--only-selected", action="store_true",
                        help="회색 후보·기존 카메라를 빼고 선택된 점만 표시")
    parser.add_argument("--batch", action="store_true",
                        help="커버 업데이트 없이 전체 후보 점수 매겨 상위 N개 선택 (시각화용)")
    parser.add_argument("--min-sep", type=float, default=2.0,
                        help="배치 모드: 선택된 시점 간 최소 거리 m (기본 2.0)")
    parser.add_argument("--search-radius", type=float, default=None,
                        help="순차 DFS: 현재 위치에서 다음 후보 탐색 반경 m (예: 4.0)")
    parser.add_argument("--local-dfs", action="store_true",
                        help="원형 후보 없는 로컬 방향 탐색 DFS")
    parser.add_argument("--step-size", type=float, default=3.0,
                        help="로컬 DFS 한 스텝 이동 거리 m (기본 3.0)")
    parser.add_argument("--start-pos", default=None,
                        help="로컬 DFS 출발점 'x,y,z' NED (미지정시 타겟 북쪽 8m 고도 5m)")
    parser.add_argument("--use-real-pts", default=None,
                        help="실제 복원 점군 .npz 경로 (points+normals). 합성 박스 대신 사용")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    rt = Path(args.real_test_dir)

    print("[1/4] real_test 로드...")
    cam_positions, cam_forwards, fov_deg, target = load_real_test(rt, args.max_frames)
    print(f"  프레임 {len(cam_positions)}개, FOV {fov_deg:.1f}, "
          f"타겟 ({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})")
    if len(cam_positions) == 0:
        cam_positions = np.empty((0, 3))

    print("[2/4] 포인트클라우드 준비...")
    normals = None
    if args.use_real_pts:
        data = np.load(args.use_real_pts)
        pts = data["points"].astype(np.float64)
        normals = data["normals"].astype(np.float64)
        # 법선 단위벡터 정규화
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (nlen + 1e-8)
        print(f"  실제 점군 로드: {len(pts):,}개 (normals 포함, self-occlusion 활성)")
    else:
        pts = synthesize_box_pointcloud(target, BOX_HALF_X, BOX_HALF_Y, BOX_HEIGHT, N_PTS_PER_FACE)
        print(f"  합성 박스 점군: {len(pts):,}개")

    print("[3/4] coverage + 후보 PB-NBV 스코어링...")
    if len(cam_positions) > 0:
        coverage = compute_coverage(pts, cam_positions, cam_forwards, fov_deg, MAX_DIST,
                                    normals=normals)
    else:
        coverage = np.zeros(len(pts), dtype=np.int32)
    underobs_mask = coverage < UNDEROBS_THRESH
    if len(cam_positions) == 0:
        print("  기존 경로 없음 → 전체 포인트 미관측")
        underobs_mask = np.ones(len(pts), dtype=bool)
    elif underobs_mask.sum() == 0:
        print("  [!] 미관측 없음 → 전체 포인트 대상")
        underobs_mask = np.ones(len(pts), dtype=bool)

    if args.candidate_mode == "uniform":
        print("  후보 생성: 공간 균일 무작위 (uniform)")
        candidates = generate_candidates_uniform(target, args.n_candidates)
    else:
        print("  후보 생성: 동심원 (ring)")
        candidates = generate_candidates(target, CAND_ALTITUDES, CAND_RADII, args.n_candidates)
    valid = []
    for cand in candidates:
        if len(cam_positions) > 0 and np.linalg.norm(cam_positions - cand, axis=1).min() < 0.3:
            continue
        valid.append(cand)
    valid = np.array(valid)
    print(f"  유효 후보 {len(valid)}개")

    if args.batch:
        print("[4/4] 배치 PB-NBV (커버 업데이트 없이 전체 스코어 → 공간 다양성 선택)...")
        all_scores = np.array([
            information_gain(c, target, pts, underobs_mask, fov_deg, MAX_DIST,
                             normals=normals)[0]
            for c in valid
        ])
        # 공간 다양성: 선택된 점 주변 min_sep 이내 후보 제거
        min_sep = args.min_sep
        order = np.argsort(-all_scores)
        sel_idx, excluded = [], set()
        for i in order:
            if i in excluded:
                continue
            sel_idx.append(i)
            if len(sel_idx) >= args.n_select:
                break
            # 이 후보 주변 min_sep 이내 모두 제거
            dists = np.linalg.norm(valid - valid[i], axis=1)
            for j in np.where(dists < min_sep)[0]:
                excluded.add(j)
        sel_pts = valid[sel_idx]
        sel_sc = all_scores[sel_idx]
    else:
        if args.local_dfs:
            print(f"[4/4] 로컬 방향 DFS (step={args.step_size}m, 원형 후보 없음)...")
            if args.start_pos:
                sx, sy, sz = map(float, args.start_pos.split(","))
                start_pos = np.array([sx, sy, sz])
                print(f"  출발점 (지정): {start_pos}")
            else:
                # 타겟 북쪽(X-방향) 8m, 고도 5m (사전 스캔 없이 고정 진입점)
                start_pos = np.array([target[0] - 8.0, target[1], target[2] - 5.0])
                print(f"  출발점 (기본 진입점): {start_pos}")
            sel_pts, sel_sc = local_dfs_pbnbv(
                start_pos, target, pts, underobs_mask, fov_deg, MAX_DIST, args.n_select,
                normals=normals, step_size=args.step_size
            )
        else:
            mode_str = f"반경 {args.search_radius}m 위치기반" if args.search_radius else "전역"
            print(f"[4/4] 순차적 PB-NBV ({mode_str} DFS)...")
            sel_pts, sel_sc = sequential_pbnbv(
                valid, target, pts, underobs_mask, fov_deg, MAX_DIST, args.n_select,
                normals=normals, search_radius=args.search_radius
            )
    scores = sel_sc
    uniq = np.unique(np.round(scores, 1))
    for i, (wp, sc) in enumerate(zip(sel_pts, sel_sc)):
        alt = abs(wp[2])
        rad = math.hypot(wp[0]-target[0], wp[1]-target[1])
        az = math.degrees(math.atan2(wp[1]-target[1], wp[0]-target[0]))
        print(f"  #{i+1:02d} ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) "
              f"고도{alt:.1f} 반경{rad:.1f} 방위{az:+.0f}도 score={sc:.0f}")

    out = {
        "schema_version": 1,
        "name": "realtest_pbnbv_points_only",
        "algorithm": f"PB-NBV({LOOKAHEAD}) viewpoint selection (no path ordering)",
        "source_data": str(rt),
        "target_position_airsim_ned": target.tolist(),
        "n_candidates": int(len(valid)),
        "n_unique_scores": int(len(uniq)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "selected_count": int(len(sel_pts)),
        "selected_points": [
            {
                "rank": i + 1,
                "position": [float(wp[0]), float(wp[1]), float(wp[2])],
                "altitude_m": float(abs(wp[2])),
                "radius_m": float(math.hypot(wp[0]-target[0], wp[1]-target[1])),
                "azimuth_deg": float(math.degrees(math.atan2(wp[1]-target[1], wp[0]-target[0]))),
                "pbnbv_score": float(sc),
            }
            for i, (wp, sc) in enumerate(zip(sel_pts, sel_sc))
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_img).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON 저장: {args.output}")

    visualize_points_only(pts, valid, scores, sel_pts, sel_sc,
                          cam_positions, target, args.output_img,
                          show_candidates=not args.only_selected)
    print("완료.")


if __name__ == "__main__":
    main()
