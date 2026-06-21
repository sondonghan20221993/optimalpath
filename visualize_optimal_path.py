"""
visualize_optimal_path.py
blue_1.mp4 탐지 결과 + 최적 경로 시각화
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Windows 한글 폰트 설정
_korean_fonts = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Nanum Gothic"]
for _fn in _korean_fonts:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

VIDEO_PATH    = r"C:\Users\sdh97\Desktop\6_16주차장\blue_1.mp4"
JSON_PATH     = r"C:\Users\sdh97\Desktop\blue1_optimal_path.json"
EXISTING_PATH = r"C:\Users\sdh97\Desktop\path\optimal\manifest.json"
OUT_PATH      = r"C:\Users\sdh97\Desktop\blue1_result_visual.png"
SAMPLE_EVERY  = 15

BLUE_LOWER = np.array([95,  80,  60])
BLUE_UPPER = np.array([135, 255, 255])
MIN_AREA   = 200


# ── 탐지 ─────────────────────────────────────────────────────────────────────

def detect_blue(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0, mask
    lg   = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(lg)
    if area < MIN_AREA:
        return None, 0.0, mask
    M = cv2.moments(lg)
    if M["m00"] == 0:
        return None, 0.0, mask
    return (M["m10"] / M["m00"], M["m01"] / M["m00"]), float(area), mask


def run_detection(video_path, sample_every):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_idx, areas, centers = [], [], []
    best_frame, best_area, best_mask = None, 0, None

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % sample_every == 0:
            center, area, mask = detect_blue(frame)
            frames_idx.append(fi)
            areas.append(area)
            centers.append(center)
            if area > best_area:
                best_area  = area
                best_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                best_mask  = mask
        fi += 1
    cap.release()
    return frames_idx, areas, centers, best_frame, best_mask, W, H


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("탐지 재실행 중...")
    frames_idx, areas, centers, best_frame, best_mask, W, H = run_detection(
        VIDEO_PATH, SAMPLE_EVERY
    )
    detected = sum(1 for a in areas if a > 0)
    print(f"탐지 완료: {detected}/{len(areas)} 샘플")

    print("JSON 로드 중...")
    data   = json.loads(Path(JSON_PATH).read_text(encoding="utf-8"))
    target = data["target_region_center"]
    wps    = data["waypoints"]
    tx, ty = target[0], target[1]

    # 기존 경로 로드 (path/optimal/manifest.json)
    ex_data = json.loads(Path(EXISTING_PATH).read_text(encoding="utf-8"))
    # waypoints 형식: [[x, y, z], ...]
    ex_wps_raw = ex_data["waypoints"]
    ex_x = [p[0] for p in ex_wps_raw]
    ex_y = [p[1] for p in ex_wps_raw]
    ex_z = [p[2] for p in ex_wps_raw]
    print(f"기존 경로 웨이포인트: {len(ex_wps_raw)}개")

    # ── 레이아웃 ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f1a")
    fig.suptitle("blue_1.mp4  ·  파란 차량 탐지 & 최적 경로 시각화",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = GridSpec(3, 4, figure=fig,
                  hspace=0.45, wspace=0.35,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_best  = fig.add_subplot(gs[0, 0])   # 최고 탐지 프레임
    ax_mask  = fig.add_subplot(gs[0, 1])   # HSV 마스크
    ax_area  = fig.add_subplot(gs[1, :2])  # 탐지 면적 타임라인
    ax_2d    = fig.add_subplot(gs[0:2, 2:])# 2D 경로 평면도
    ax_3d    = fig.add_subplot(gs[2, :],   projection="3d")  # 3D 경로

    bg = "#1a1a2e"
    for ax in [ax_best, ax_mask, ax_area, ax_2d]:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # ── 1. 최고 탐지 프레임 ──────────────────────────────────────────────────
    ax_best.set_title("최고 탐지 프레임", color="white", fontsize=10)
    if best_frame is not None:
        ax_best.imshow(best_frame)
        # 탐지 위치 표시
        best_fi = frames_idx[int(np.argmax(areas))]
        best_cx = centers[int(np.argmax(areas))]
        if best_cx:
            ax_best.plot(best_cx[0], best_cx[1], "r+", markersize=16, markeredgewidth=2)
            ax_best.plot(best_cx[0], best_cx[1], "ro", markersize=8, fillstyle="none", markeredgewidth=2)
        ax_best.set_title(f"최고 탐지 (프레임 #{best_fi}, {int(max(areas))}px²)",
                          color="white", fontsize=9)
    ax_best.axis("off")

    # ── 2. HSV 마스크 ────────────────────────────────────────────────────────
    ax_mask.set_title("HSV 파란색 마스크", color="white", fontsize=10)
    if best_mask is not None:
        ax_mask.imshow(best_mask, cmap="Blues")
    ax_mask.axis("off")

    # ── 3. 탐지 면적 타임라인 ────────────────────────────────────────────────
    ax_area.set_title("프레임별 파란 차량 탐지 면적", color="white", fontsize=10)
    det_x = [frames_idx[i] for i, a in enumerate(areas) if a > 0]
    det_y = [a for a in areas if a > 0]
    nd_x  = [frames_idx[i] for i, a in enumerate(areas) if a == 0]

    ax_area.fill_between(det_x, det_y, alpha=0.3, color="#4fc3f7")
    ax_area.plot(det_x, det_y, color="#4fc3f7", linewidth=1.2, label=f"탐지 ({detected}프레임)")
    if nd_x:
        ax_area.scatter(nd_x, [0] * len(nd_x), marker="|", color="#ef5350",
                        s=40, alpha=0.5, label=f"미탐지 ({len(nd_x)}프레임)")
    ax_area.axhline(np.mean(det_y) if det_y else 0, color="#ffcc02",
                    linestyle="--", linewidth=0.8, label=f"평균 {int(np.mean(det_y) if det_y else 0)}px²")
    ax_area.set_xlabel("프레임 번호", color="#aaa", fontsize=8)
    ax_area.set_ylabel("탐지 면적 (px²)", color="#aaa", fontsize=8)
    ax_area.legend(fontsize=7.5, facecolor="#1a1a2e", edgecolor="#444",
                   labelcolor="white", loc="upper right")
    ax_area.set_xlim(frames_idx[0], frames_idx[-1])

    # ── 4. 2D 경로 평면도 (Top-Down) ─────────────────────────────────────────
    ax_2d.set_title("경로 비교 평면도 (Top-Down View)", color="white", fontsize=10)

    wp_x = [wp["position"][0] for wp in wps]
    wp_y = [wp["position"][1] for wp in wps]
    wp_z = [wp["position"][2] for wp in wps]
    z_min, z_max = min(wp_z), max(wp_z)

    # ── 기존 경로 (흰색 점선) ────────────────────────────────────────────────
    ax_2d.plot(ex_x + [ex_x[0]], ex_y + [ex_y[0]],
               color="#aaaaaa", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)
    ax_2d.scatter(ex_x, ex_y, color="#cccccc", s=18, zorder=3,
                  edgecolors="none", alpha=0.7)

    # ── 신규 생성 경로 (plasma 색상) ─────────────────────────────────────────
    cmap = plt.cm.plasma
    for i in range(len(wps) - 1):
        t = (wp_z[i] - z_min) / (z_max - z_min + 1e-8)
        ax_2d.plot([wp_x[i], wp_x[i+1]], [wp_y[i], wp_y[i+1]],
                   color=cmap(t), linewidth=2.0, alpha=0.9, zorder=4)

    sc = ax_2d.scatter(wp_x, wp_y, c=wp_z, cmap="plasma",
                       s=90, zorder=5, edgecolors="white", linewidths=0.6)
    for i, (x, y) in enumerate(zip(wp_x, wp_y)):
        ax_2d.text(x, y + 0.3, str(i + 1), color="white", fontsize=6.5,
                   ha="center", va="bottom", zorder=6)

    # 타겟
    ax_2d.plot(tx, ty, marker="*", color="#ffcc02", markersize=18,
               zorder=7, markeredgecolor="white", markeredgewidth=0.8)
    ax_2d.text(tx, ty - 0.8, "TARGET", color="#ffcc02", fontsize=8,
               ha="center", fontweight="bold", zorder=8)

    # 원형 참고선
    for r in [7.0, 9.8]:
        circle = plt.Circle((tx, ty), r, fill=False,
                             color="#ffffff18", linestyle="--", linewidth=0.7)
        ax_2d.add_patch(circle)
        ax_2d.text(tx + r * 0.72, ty + r * 0.72, f"r={r}m",
                   color="#555", fontsize=7)

    cb = plt.colorbar(sc, ax=ax_2d, pad=0.01, fraction=0.03)
    cb.set_label("신규 경로 고도 Z (NED)", color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    # 2D 범례
    leg2d = [
        mpatches.Patch(color="#aaaaaa", label=f"기존 경로 ({len(ex_wps_raw)}개 WP)"),
        mpatches.Patch(color=cmap(1.0),  label=f"신규 저고도 6m ({sum(1 for z in wp_z if z > -6)//1}개)"),
        mpatches.Patch(color=cmap(0.0),  label=f"신규 고고도 9.6m ({sum(1 for z in wp_z if z <= -6)//1}개)"),
    ]
    ax_2d.legend(handles=leg2d, fontsize=7.5, facecolor="#1a1a2e",
                 edgecolor="#444", labelcolor="white", loc="lower right")

    ax_2d.set_xlabel("X (m)", color="#aaa", fontsize=8)
    ax_2d.set_ylabel("Y (m)", color="#aaa", fontsize=8)
    ax_2d.set_aspect("equal")

    # ── 5. 3D 경로 ───────────────────────────────────────────────────────────
    ax_3d.set_facecolor("#0f0f1a")
    ax_3d.set_title("3D 비행 경로 비교 (기존 vs 신규)", color="white", fontsize=10, pad=10)

    ground_z = target[2]
    margin   = 16

    # ── 기존 경로 (흰색/회색) ────────────────────────────────────────────────
    for i in range(len(ex_x) - 1):
        ax_3d.plot([ex_x[i], ex_x[i+1]],
                   [ex_y[i], ex_y[i+1]],
                   [ex_z[i], ex_z[i+1]],
                   color="#888888", linewidth=1.2, alpha=0.5)
    ax_3d.scatter(ex_x, ex_y, ex_z, color="#aaaaaa", s=20,
                  depthshade=True, edgecolors="none", alpha=0.6)

    # ── 신규 경로 (plasma 색상) ──────────────────────────────────────────────
    for i in range(len(wps) - 1):
        t = (wp_z[i] - z_min) / (z_max - z_min + 1e-8)
        ax_3d.plot([wp_x[i], wp_x[i+1]],
                   [wp_y[i], wp_y[i+1]],
                   [wp_z[i], wp_z[i+1]],
                   color=cmap(t), linewidth=2.2, alpha=0.95)

    ax_3d.scatter(wp_x, wp_y, wp_z, c=wp_z, cmap="plasma",
                  s=70, depthshade=True, edgecolors="white", linewidths=0.5)

    # 수직선 (신규 경로만)
    for x, y, z in zip(wp_x, wp_y, wp_z):
        ax_3d.plot([x, x], [y, y], [z, ground_z],
                   color="white", alpha=0.08, linewidth=0.5)

    # 타겟
    ax_3d.scatter([tx], [ty], [ground_z], color="#ffcc02",
                  s=220, marker="*", zorder=5, depthshade=False)

    # 지면 평면 (반투명)
    gx = np.array([[tx - margin, tx + margin],
                   [tx - margin, tx + margin]])
    gy = np.array([[ty - margin, ty - margin],
                   [ty + margin, ty + margin]])
    gz = np.full_like(gx, ground_z)
    ax_3d.plot_surface(gx, gy, gz, alpha=0.06, color="#4fc3f7")

    ax_3d.set_xlabel("X (m)", color="#aaa", fontsize=7, labelpad=5)
    ax_3d.set_ylabel("Y (m)", color="#aaa", fontsize=7, labelpad=5)
    ax_3d.set_zlabel("Z NED (m)", color="#aaa", fontsize=7, labelpad=5)
    ax_3d.tick_params(colors="white", labelsize=6)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor("#333")
    ax_3d.yaxis.pane.set_edgecolor("#333")
    ax_3d.zaxis.pane.set_edgecolor("#333")
    ax_3d.view_init(elev=28, azim=-55)

    # 범례
    patches = [
        mpatches.Patch(color="#888888", label=f"기존 경로  ({len(ex_wps_raw)}개 WP, orbit+recommended)"),
        mpatches.Patch(color=cmap(1.0),  label=f"신규 저고도 6m  (반경 7m)"),
        mpatches.Patch(color=cmap(0.0),  label=f"신규 고고도 9.6m  (반경 9.8m)"),
        mpatches.Patch(color="#ffcc02",  label=f"타겟  ({tx:.1f}, {ty:.1f})"),
    ]
    ax_3d.legend(handles=patches, loc="upper left", fontsize=8,
                 facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    # ── 통계 텍스트 ──────────────────────────────────────────────────────────
    stats = (
        f"탐지율: {detected}/{len(areas)} ({100*detected/len(areas):.0f}%)  "
        f"|  최대 면적: {int(max(areas)) if areas else 0}px²  "
        f"|  평균 면적: {int(np.mean(det_y)) if det_y else 0}px²  "
        f"|  신규 WP: {len(wps)}개  |  기존 WP: {len(ex_wps_raw)}개  "
        f"|  신규 고도: {abs(z_max):.1f}~{abs(z_min):.1f}m"
    )
    fig.text(0.5, 0.005, stats, ha="center", color="#aaa", fontsize=8)

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"시각화 저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
