"""
realtest_optimal_path.py

real_test/ 폴더의 AirSim 비행 데이터(34프레임)를 분석하여
PB-NBV(2) + Greedy Path Planning으로 최적 추가 경로를 생성합니다.

포인트클라우드가 없는 경우 타겟 박스 주변에 합성 포인트클라우드를 생성합니다.

사용법:
  python realtest_optimal_path.py
  python realtest_optimal_path.py --real-test-dir real_test --output realtest_optimal.json
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

for _fn in ["Malgun Gothic", "NanumGothic"]:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 기본 경로 설정 ────────────────────────────────────────────────────────────
DEFAULT_REAL_TEST_DIR = "real_test"
DEFAULT_OUTPUT_JSON   = "results/realtest_optimal_path/realtest_optimal.json"
DEFAULT_OUTPUT_IMG    = "results/realtest_optimal_path/realtest_result.png"

# ── 알고리즘 파라미터 ─────────────────────────────────────────────────────────
N_PTS_PER_FACE   = 300     # 박스 면당 합성 포인트 수 (6면 × N = 총 포인트 수)
MAX_DIST         = 10.0    # 최대 가시 거리 (m)
UNDEROBS_THRESH  = 2       # 이 횟수 미만 = 미관측
N_CANDIDATES     = 200     # 후보 시점 수
N_SELECT         = 12      # 최종 선택 웨이포인트 수
LOOKAHEAD        = 2       # PB-NBV lookahead 단계

# 후보 고도 및 반경 (AirSim NED 기준, 타겟 위로 올라가는 높이)
CAND_ALTITUDES = [4.0, 5.5, 7.0, 8.5]   # m (기존 비행 고도 ~5m 기준)
CAND_RADII     = [5.0, 6.5, 8.0]         # m (기존 반경 ~6.6m 기준)

# 타겟 박스 크기 추정 (AirSim 잔디밭 빨간 박스, 단위 m)
BOX_HALF_X = 1.0   # X 방향 반폭
BOX_HALF_Y = 1.0   # Y 방향 반폭
BOX_HEIGHT  = 1.5  # 높이 (NED 상방)


# ════════════════════════════════════════════════════════════════════════════
# 1. real_test 데이터 로드
# ════════════════════════════════════════════════════════════════════════════

def load_real_test(real_test_dir: Path):
    """meta/*.json → 카메라 위치, 전방벡터, FOV, 타겟 반환"""
    manifest = json.loads((real_test_dir / "manifest.json").read_text(encoding="utf-8"))

    # 타겟 좌표
    sp = manifest.get("straight_prefix_target")
    if sp:
        target = np.array(sp, dtype=float)
    else:
        tc = manifest.get("target_region_center")
        target = np.array(tc, dtype=float) if tc else None

    meta_files = sorted((real_test_dir / "meta").glob("*.json"))
    if not meta_files:
        raise FileNotFoundError(f"meta JSON 없음: {real_test_dir / 'meta'}")

    cam_positions = []
    fovs = []

    for mf in meta_files:
        d = json.loads(mf.read_text(encoding="utf-8"))
        cam = d["camera"]
        pos = cam["pose"]["position"]
        cam_positions.append([pos["x"], pos["y"], pos["z"]])
        fovs.append(cam["fov"])

    cam_positions = np.array(cam_positions)
    fov_deg = float(np.mean(fovs))

    # 카메라 전방벡터 = 타겟 방향
    if target is not None:
        to_target = target - cam_positions
        norms = np.linalg.norm(to_target, axis=1, keepdims=True)
        cam_forwards = to_target / (norms + 1e-8)
    else:
        # manifest에 타겟 없으면 카메라 평균 위치 기준 추정
        center = cam_positions.mean(axis=0)
        to_center = center - cam_positions
        norms = np.linalg.norm(to_center, axis=1, keepdims=True)
        cam_forwards = to_center / (norms + 1e-8)
        target = center

    print(f"  로드 완료: {len(cam_positions)}개 프레임, FOV={fov_deg:.1f}°")
    print(f"  타겟: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
    print(f"  고도 범위: {abs(cam_positions[:,2].min()):.1f}~{abs(cam_positions[:,2].max()):.1f}m")
    radii = np.linalg.norm(cam_positions[:,:2] - target[:2], axis=1)
    print(f"  수평 반경: {radii.min():.1f}~{radii.max():.1f}m  평균 {radii.mean():.1f}m")

    return cam_positions, cam_forwards, fov_deg, target


# ════════════════════════════════════════════════════════════════════════════
# 2. 합성 포인트클라우드 생성 (타겟 박스 표면)
# ════════════════════════════════════════════════════════════════════════════

def synthesize_box_pointcloud(target, half_x, half_y, height, n_per_face):
    """
    타겟 위치 중심 박스 표면에 균일 포인트를 생성합니다.
    AirSim NED 기준: target[2]=바닥, 박스 상단 = target[2] - height
    """
    tx, ty, tz = target
    top_z = tz - height  # NED에서 위 = z 음수

    pts = []

    # ±X 면 (y, z 샘플)
    for face_x in [tx - half_x, tx + half_x]:
        ys = np.random.uniform(ty - half_y, ty + half_y, n_per_face)
        zs = np.random.uniform(top_z, tz, n_per_face)
        pts.append(np.column_stack([np.full(n_per_face, face_x), ys, zs]))

    # ±Y 면 (x, z 샘플)
    for face_y in [ty - half_y, ty + half_y]:
        xs = np.random.uniform(tx - half_x, tx + half_x, n_per_face)
        zs = np.random.uniform(top_z, tz, n_per_face)
        pts.append(np.column_stack([xs, np.full(n_per_face, face_y), zs]))

    # 상단 면
    xs = np.random.uniform(tx - half_x, tx + half_x, n_per_face)
    ys = np.random.uniform(ty - half_y, ty + half_y, n_per_face)
    pts.append(np.column_stack([xs, ys, np.full(n_per_face, top_z)]))

    # 바닥 면
    xs = np.random.uniform(tx - half_x, tx + half_x, n_per_face)
    ys = np.random.uniform(ty - half_y, ty + half_y, n_per_face)
    pts.append(np.column_stack([xs, ys, np.full(n_per_face, tz)]))

    return np.vstack(pts)


# ════════════════════════════════════════════════════════════════════════════
# 3. Coverage 추정
# ════════════════════════════════════════════════════════════════════════════

def point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist):
    v = pts - cam_pos
    dist = np.linalg.norm(v, axis=1)
    valid = (dist > 0.05) & (dist < max_dist)
    v_norm = v / (dist[:, None] + 1e-8)
    cos_half = math.cos(math.radians(fov_deg / 2.0))
    in_cone = (v_norm @ cam_dir) > cos_half
    return valid & in_cone


def compute_coverage(pts, cam_positions, cam_forwards, fov_deg, max_dist):
    counts = np.zeros(len(pts), dtype=np.int32)
    for cam_pos, cam_dir in zip(cam_positions, cam_forwards):
        counts += point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist).astype(np.int32)
    return counts


# ════════════════════════════════════════════════════════════════════════════
# 4. 후보 시점 생성
# ════════════════════════════════════════════════════════════════════════════

def generate_candidates(target, altitudes, radii, n_total):
    candidates = []
    n_per = max(1, n_total // (len(altitudes) * len(radii)))
    for alt in altitudes:
        z = target[2] - alt  # NED: 위 = target_z - alt
        for rad in radii:
            for i in range(n_per):
                theta = 2 * math.pi * i / n_per
                x = target[0] + rad * math.cos(theta)
                y = target[1] + rad * math.sin(theta)
                candidates.append([x, y, z])
    return np.array(candidates)


# ════════════════════════════════════════════════════════════════════════════
# 5. PB-NBV(2) 스코어링
# ════════════════════════════════════════════════════════════════════════════

def information_gain(cam_pos, target, pts, underobs_mask, fov_deg, max_dist, already_vis=None):
    cam_dir = target - cam_pos
    cam_dir = cam_dir / (np.linalg.norm(cam_dir) + 1e-8)
    visible = point_in_frustum(cam_pos, cam_dir, fov_deg, pts, max_dist)
    new_vis = visible & underobs_mask
    if already_vis is not None:
        new_vis = new_vis & ~already_vis
    return int(new_vis.sum()), visible


def pbnbv_score(candidate, candidates, target, pts, underobs_mask, fov_deg, max_dist, lookahead):
    ig1, vis1 = information_gain(candidate, target, pts, underobs_mask, fov_deg, max_dist)
    score = float(ig1)

    if lookahead >= 2 and ig1 > 0:
        updated_mask = underobs_mask & ~vis1
        sample_idx = np.random.choice(len(candidates), min(40, len(candidates)), replace=False)
        best_ig2 = 0
        for idx in sample_idx:
            if np.allclose(candidates[idx], candidate):
                continue
            ig2, _ = information_gain(candidates[idx], target, pts, updated_mask,
                                      fov_deg, max_dist, already_vis=vis1)
            best_ig2 = max(best_ig2, ig2)
        score += 0.5 * best_ig2

    return score


# ════════════════════════════════════════════════════════════════════════════
# 6. Greedy Path Planning
# ════════════════════════════════════════════════════════════════════════════

def greedy_path(selected_pts, start_pos):
    remaining = list(range(len(selected_pts)))
    path = []
    cur = start_pos
    while remaining:
        dists = [np.linalg.norm(selected_pts[i] - cur) for i in remaining]
        nearest = remaining[int(np.argmin(dists))]
        path.append(nearest)
        cur = selected_pts[nearest]
        remaining.remove(nearest)
    return [selected_pts[i] for i in path]


# ════════════════════════════════════════════════════════════════════════════
# 7. 시각화
# ════════════════════════════════════════════════════════════════════════════

def visualize(pts, coverage, underobs_mask, candidates, scores,
              existing_cams, selected_path, target, out_path):
    fig = plt.figure(figsize=(22, 13), facecolor="#0f0f1a")
    fig.suptitle("real_test 기반  PB-NBV(2) + Greedy 최적 경로 생성 결과",
                 color="white", fontsize=14, fontweight="bold")

    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33,
                      left=0.05, right=0.97, top=0.93, bottom=0.05)

    ax_cov  = fig.add_subplot(gs[0, 0])
    ax_top  = fig.add_subplot(gs[0, 1])
    ax_sc   = fig.add_subplot(gs[0, 2])
    ax_path = fig.add_subplot(gs[1, 0])
    ax_3d   = fig.add_subplot(gs[1, 1], projection="3d")
    ax_alt  = fig.add_subplot(gs[1, 2])

    bg = "#1a1a2e"
    for ax in [ax_cov, ax_top, ax_sc, ax_path, ax_alt]:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    tx, ty, tz = target
    path_arr = np.array(selected_path)

    # ── 1. Coverage 히스토그램 ───────────────────────────────────────────────
    ax_cov.set_title("포인트별 관측 횟수 (기존 34프레임)", color="white", fontsize=9)
    ax_cov.hist(coverage, bins=30, color="#4fc3f7", edgecolor="#0f0f1a", alpha=0.85)
    ax_cov.axvline(UNDEROBS_THRESH, color="#ef5350", linewidth=1.5,
                   linestyle="--", label=f"미관측 기준 (<{UNDEROBS_THRESH}회)")
    pct = 100 * underobs_mask.sum() / len(coverage)
    ax_cov.text(0.97, 0.95, f"미관측: {underobs_mask.sum()}개 ({pct:.1f}%)",
                transform=ax_cov.transAxes, color="#ef5350", fontsize=8,
                ha="right", va="top")
    ax_cov.set_xlabel("관측 횟수", color="#aaa", fontsize=7)
    ax_cov.set_ylabel("포인트 수", color="#aaa", fontsize=7)
    ax_cov.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white")

    # ── 2. Top-Down Coverage ─────────────────────────────────────────────────
    ax_top.set_title("포인트클라우드 coverage (Top-Down)", color="white", fontsize=9)
    sub = np.random.choice(len(pts), min(4000, len(pts)), replace=False)
    sc1 = ax_top.scatter(pts[sub, 0], pts[sub, 1],
                         c=coverage[sub], cmap="RdYlGn", s=2, alpha=0.5,
                         vmin=0, vmax=coverage.max())
    cb1 = plt.colorbar(sc1, ax=ax_top, pad=0.01, fraction=0.04)
    cb1.set_label("관측 횟수", color="white", fontsize=6)
    cb1.ax.tick_params(labelcolor="white", labelsize=6)
    ax_top.scatter(existing_cams[:, 0], existing_cams[:, 1],
                   color="#aaaaaa", s=10, alpha=0.5, label="기존 카메라(34개)", zorder=4)
    ax_top.plot(tx, ty, "*", color="#ffcc02", markersize=14, zorder=5)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax_top.set_ylabel("Y (m)", color="#aaa", fontsize=7)
    ax_top.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white")

    # ── 3. 후보 스코어 ──────────────────────────────────────────────────────
    ax_sc.set_title(f"PB-NBV({LOOKAHEAD}) 후보 스코어 (상위 30개)", color="white", fontsize=9)
    si = np.argsort(scores)[::-1][:30]
    bar_s = [scores[i] for i in si]
    bar_c = ["#ffcc02" if i < N_SELECT else "#4fc3f7" for i in range(len(bar_s))]
    ax_sc.barh(range(len(bar_s)), bar_s[::-1], color=bar_c[::-1], alpha=0.85)
    thresh = bar_s[N_SELECT - 1] if len(bar_s) >= N_SELECT else 0
    ax_sc.axvline(thresh, color="#ef5350", linewidth=1, linestyle="--")
    ax_sc.set_xlabel("PB-NBV 스코어 (정보 이득)", color="#aaa", fontsize=7)
    ax_sc.text(0.97, 0.02, f"노랑=선택({N_SELECT}) / 파랑=미선택",
               transform=ax_sc.transAxes, color="#aaa", fontsize=7, ha="right", va="bottom")

    # ── 4. 최종 경로 2D ─────────────────────────────────────────────────────
    ax_path.set_title("최적 경로 (Top-Down)", color="white", fontsize=9)
    ax_path.scatter(existing_cams[:, 0], existing_cams[:, 1],
                    color="#555", s=15, alpha=0.6, zorder=2, label=f"기존 ({len(existing_cams)}개)")
    ax_path.plot(np.append(path_arr[:, 0], path_arr[0, 0]),
                 np.append(path_arr[:, 1], path_arr[0, 1]),
                 color="#00e5ff", linewidth=2.2, alpha=0.9, zorder=4)
    sc_p = ax_path.scatter(path_arr[:, 0], path_arr[:, 1],
                           c=path_arr[:, 2], cmap="plasma",
                           s=100, zorder=5, edgecolors="white", linewidths=0.7)
    for i, (x, y) in enumerate(zip(path_arr[:, 0], path_arr[:, 1])):
        ax_path.text(x, y + 0.4, str(i + 1), color="white", fontsize=7.5,
                     ha="center", va="bottom", zorder=6)
    ax_path.plot(tx, ty, "*", color="#ffcc02", markersize=18, zorder=7,
                 markeredgecolor="white", markeredgewidth=0.7)
    ax_path.text(tx, ty - 1.3, "TARGET", color="#ffcc02", fontsize=8,
                 ha="center", fontweight="bold")
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax_path.set_ylabel("Y (m)", color="#aaa", fontsize=7)
    ax_path.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white",
                   loc="lower right")

    # ── 5. 3D 경로 ──────────────────────────────────────────────────────────
    ax_3d.set_facecolor("#0f0f1a")
    ax_3d.set_title("3D 비교 (기존 vs 최적)", color="white", fontsize=9, pad=8)
    ax_3d.scatter(existing_cams[:, 0], existing_cams[:, 1], existing_cams[:, 2],
                  color="#888888", s=18, alpha=0.5, label="기존 경로")
    z_min, z_max = path_arr[:, 2].min(), path_arr[:, 2].max()
    cmap = plt.cm.plasma
    for i in range(len(path_arr) - 1):
        t_c = (path_arr[i, 2] - z_min) / (z_max - z_min + 1e-8)
        ax_3d.plot([path_arr[i, 0], path_arr[i+1, 0]],
                   [path_arr[i, 1], path_arr[i+1, 1]],
                   [path_arr[i, 2], path_arr[i+1, 2]],
                   color=cmap(t_c), linewidth=2.2, alpha=0.95)
    ax_3d.scatter(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                  c=path_arr[:, 2], cmap="plasma", s=60,
                  edgecolors="white", linewidths=0.5, zorder=5)
    ax_3d.scatter([tx], [ty], [tz], color="#ffcc02", s=200, marker="*", zorder=6)
    ax_3d.set_xlabel("X", color="#aaa", fontsize=6, labelpad=3)
    ax_3d.set_ylabel("Y", color="#aaa", fontsize=6, labelpad=3)
    ax_3d.set_zlabel("Z NED", color="#aaa", fontsize=6, labelpad=3)
    ax_3d.tick_params(colors="white", labelsize=5)
    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")
    ax_3d.view_init(elev=28, azim=-55)
    patches = [
        mpatches.Patch(color="#888888", label="기존 경로 (34 WP)"),
        mpatches.Patch(color=cmap(0.8),  label=f"신규 최적 ({len(path_arr)} WP)"),
        mpatches.Patch(color="#ffcc02",  label="타겟"),
    ]
    ax_3d.legend(handles=patches, fontsize=7, facecolor="#1a1a2e",
                 edgecolor="#444", labelcolor="white", loc="upper left")

    # ── 6. 고도 프로파일 ─────────────────────────────────────────────────────
    ax_alt.set_title("웨이포인트별 고도 프로파일", color="white", fontsize=9)
    wp_idx = np.arange(1, len(path_arr) + 1)
    altitudes = [abs(z) for z in path_arr[:, 2]]
    ex_alts = [abs(z) for z in existing_cams[:, 2]]
    ax_alt.plot(wp_idx, altitudes, color="#00e5ff", linewidth=2,
                marker="o", markersize=6, label=f"신규 경로 (avg {np.mean(altitudes):.1f}m)")
    ax_alt.axhline(np.mean(ex_alts), color="#888888", linewidth=1.2, linestyle="--",
                   label=f"기존 평균 고도 ({np.mean(ex_alts):.1f}m)")
    for i, (xi, yi) in enumerate(zip(wp_idx, altitudes)):
        ax_alt.text(xi, yi + 0.1, f"{yi:.1f}", color="white", fontsize=7, ha="center")
    ax_alt.set_xlabel("웨이포인트 번호", color="#aaa", fontsize=7)
    ax_alt.set_ylabel("고도 (m)", color="#aaa", fontsize=7)
    ax_alt.legend(fontsize=7, facecolor=bg, edgecolor="#444", labelcolor="white")
    ax_alt.set_xticks(wp_idx)

    # 통계 텍스트
    total_dist = sum(np.linalg.norm(path_arr[i+1] - path_arr[i])
                     for i in range(len(path_arr)-1))
    stats = (f"신규 WP: {len(path_arr)}개  |  총 비행거리: {total_dist:.1f}m  |  "
             f"미관측 해소율: {pct:.1f}%  |  고도: {min(altitudes):.1f}~{max(altitudes):.1f}m")
    fig.text(0.5, 0.005, stats, ha="center", color="#aaa", fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"시각화 저장: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="real_test 데이터 기반 PB-NBV(2) 최적 경로 생성"
    )
    parser.add_argument("--real-test-dir", default=DEFAULT_REAL_TEST_DIR,
                        help="real_test 폴더 경로")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSON,
                        help="출력 JSON 경로")
    parser.add_argument("--output-img", default=DEFAULT_OUTPUT_IMG,
                        help="출력 시각화 이미지 경로")
    parser.add_argument("--n-select", type=int, default=N_SELECT,
                        help="최종 선택 웨이포인트 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    real_test_dir = Path(args.real_test_dir)

    # 1. 데이터 로드
    print("[1/5] real_test 데이터 로드 중...")
    cam_positions, cam_forwards, fov_deg, target = load_real_test(real_test_dir)

    # 2. 합성 포인트클라우드 생성
    print("[2/5] 타겟 박스 합성 포인트클라우드 생성 중...")
    pts = synthesize_box_pointcloud(target, BOX_HALF_X, BOX_HALF_Y, BOX_HEIGHT,
                                    N_PTS_PER_FACE)
    print(f"  포인트 수: {len(pts):,}개  (박스 {BOX_HALF_X*2:.1f}×{BOX_HALF_Y*2:.1f}×{BOX_HEIGHT:.1f}m)")

    # 3. Coverage 추정
    print(f"[3/5] Coverage 추정 중 (기존 카메라 {len(cam_positions)}개 기준)...")
    coverage = compute_coverage(pts, cam_positions, cam_forwards, fov_deg, MAX_DIST)
    underobs_mask = coverage < UNDEROBS_THRESH
    pct = 100 * underobs_mask.mean()
    print(f"  미관측 포인트: {underobs_mask.sum():,}개 ({pct:.1f}%)")

    if underobs_mask.sum() == 0:
        print("  [!] 미관측 영역 없음 → 전체 포인트 대상으로 진행")
        underobs_mask = np.ones(len(pts), dtype=bool)

    # 4. 후보 시점 생성 + PB-NBV(2) 스코어링
    print(f"[4/5] 후보 {N_CANDIDATES}개 생성 + PB-NBV({LOOKAHEAD}) 스코어링 중...")
    candidates = generate_candidates(target, CAND_ALTITUDES, CAND_RADII, N_CANDIDATES)

    scores, valid_cands = [], []
    for cand in candidates:
        min_d = np.linalg.norm(cam_positions - cand, axis=1).min()
        if min_d < 0.3:
            continue
        sc = pbnbv_score(cand, candidates, target, pts, underobs_mask,
                         fov_deg, MAX_DIST, LOOKAHEAD)
        scores.append(sc)
        valid_cands.append(cand)

    valid_cands = np.array(valid_cands)
    scores = np.array(scores)
    print(f"  유효 후보: {len(valid_cands)}개, 스코어: {scores.min():.0f}~{scores.max():.0f}")

    n_sel = min(args.n_select, len(valid_cands))
    top_idx = np.argsort(scores)[::-1][:n_sel]
    selected_pts = valid_cands[top_idx]
    selected_scores = scores[top_idx]
    print(f"  선택 시점: {n_sel}개, 최고 스코어: {selected_scores[0]:.0f}")

    # 5. Greedy Path Planning
    print("[5/5] Greedy Path Planning 중...")
    start_pos = cam_positions[-1]  # 마지막 촬영 위치에서 시작
    final_path = greedy_path(selected_pts, start_pos)

    total_dist = sum(np.linalg.norm(np.array(final_path[i+1]) - np.array(final_path[i]))
                     for i in range(len(final_path) - 1))
    print(f"  최종 경로: {len(final_path)}개 WP, 총 이동거리: {total_dist:.1f}m")
    for i, wp in enumerate(final_path):
        alt = abs(wp[2])
        rad = math.sqrt((wp[0]-target[0])**2 + (wp[1]-target[1])**2)
        print(f"  WP#{i+1:02d}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})"
              f"  고도={alt:.1f}m  반경={rad:.1f}m  score={selected_scores[i]:.0f}")

    # JSON 저장
    output_data = {
        "schema_version": 1,
        "name": "realtest_optimal_path",
        "algorithm": f"PB-NBV({LOOKAHEAD}) + Greedy Path Planning",
        "source_data": str(real_test_dir),
        "n_existing_frames": int(len(cam_positions)),
        "target_position_airsim_ned": target.tolist(),
        "underobserved_ratio": float(underobs_mask.mean()),
        "path_step_count": len(final_path),
        "total_distance_m": float(total_dist),
        "waypoints": [
            {
                "index": i + 1,
                "position": [float(wp[0]), float(wp[1]), float(wp[2])],
                "relative_to_target": [
                    float(wp[0] - target[0]),
                    float(wp[1] - target[1]),
                    float(wp[2] - target[2]),
                ],
                "altitude_m": float(abs(wp[2])),
                "radius_m": float(math.sqrt((wp[0]-target[0])**2 + (wp[1]-target[1])**2)),
                "pbnbv_score": float(selected_scores[i]),
            }
            for i, wp in enumerate(final_path)
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_img).parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nJSON 저장: {out_path}")

    # 시각화
    visualize(pts, coverage, underobs_mask, valid_cands, scores,
              cam_positions, final_path, target, args.output_img)

    print("\n완료.")
    print(f"  경로 JSON : {args.output}")
    print(f"  시각화    : {args.output_img}")
    print(f"\ncar.py 실행 예시:")
    print(f"  python car.py --mode orbit_then_recommended \\")
    print(f"      --recommended-json {args.output} \\")
    print(f"      --target-x {target[0]:.2f} --target-y {target[1]:.2f} --target-z {target[2]:.2f}")


if __name__ == "__main__":
    main()
