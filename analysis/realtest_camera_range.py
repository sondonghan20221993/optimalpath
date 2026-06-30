"""
realtest_camera_range.py

카메라 '촬영 가능 거리 밴드' 모델을 시각적으로 검증한다.
한 시점(viewpoint)에서:
  - 근거리 데드존 (거리 < MIN_DIST) : 너무 가까워서 못 찍음 (빨강)
  - 촬영 가능 밴드 (MIN_DIST ~ MAX_DIST, FOV 안) : 찍힘 (초록)
  - 원거리/화각 밖 : 못 찍음 (회색)
타겟 박스 점들을 위 3가지로 분류해 개수를 센다.

사용법:
  python realtest_camera_range.py
  python realtest_camera_range.py --view-radius 6.6 --view-alt 5.0 --min-dist 3 --max-dist 12
"""

import argparse, json, math, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

TARGET = np.array([-33.67, -50.83, 0.18])
BOX_HALF_X, BOX_HALF_Y, BOX_HEIGHT = 1.0, 1.0, 1.5
N_PER_FACE = 300
FOV_DEG = 89.9


def synth_box(target, hx, hy, h, n):
    tx, ty, tz = target; top = tz - h; pts = []
    for fx in [tx-hx, tx+hx]:
        pts.append(np.column_stack([np.full(n, fx),
            np.random.uniform(ty-hy, ty+hy, n), np.random.uniform(top, tz, n)]))
    for fy in [ty-hy, ty+hy]:
        pts.append(np.column_stack([np.random.uniform(tx-hx, tx+hx, n),
            np.full(n, fy), np.random.uniform(top, tz, n)]))
    for z in [top, tz]:
        pts.append(np.column_stack([np.random.uniform(tx-hx, tx+hx, n),
            np.random.uniform(ty-hy, ty+hy, n), np.full(n, z)]))
    return np.vstack(pts)


def classify(cam_pos, cam_dir, pts, fov_deg, min_d, max_d):
    """각 점을 0=데드존(가까움) 1=밴드(촬영) 2=원거리/화각밖 으로 분류."""
    v = pts - cam_pos
    dist = np.linalg.norm(v, axis=1)
    vn = v / (dist[:, None] + 1e-8)
    in_cone = (vn @ cam_dir) > math.cos(math.radians(fov_deg/2))
    label = np.full(len(pts), 2, dtype=int)        # 기본: 못 찍음
    too_close = dist < min_d
    band = (dist >= min_d) & (dist <= max_d) & in_cone
    label[band] = 1
    label[too_close] = 0                           # 데드존이 우선
    return label, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view-radius", type=float, default=6.6, help="시점 수평 반경 m")
    ap.add_argument("--view-alt", type=float, default=5.0, help="시점 고도 m")
    ap.add_argument("--view-azim", type=float, default=0.0, help="시점 방위각 deg")
    ap.add_argument("--min-dist", type=float, default=3.0)
    ap.add_argument("--max-dist", type=float, default=12.0)
    ap.add_argument("--output-img", default="results/camera_range/camera_range.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    pts = synth_box(TARGET, BOX_HALF_X, BOX_HALF_Y, BOX_HEIGHT, N_PER_FACE)

    az = math.radians(args.view_azim)
    cam = np.array([TARGET[0] + args.view_radius*math.cos(az),
                    TARGET[1] + args.view_radius*math.sin(az),
                    TARGET[2] - args.view_alt])
    cam_dir = TARGET - cam; cam_dir /= np.linalg.norm(cam_dir)
    cam_to_target = np.linalg.norm(cam - TARGET)

    label, dist = classify(cam, cam_dir, pts, FOV_DEG, args.min_dist, args.max_dist)
    n_close = int((label == 0).sum())
    n_band  = int((label == 1).sum())
    n_out   = int((label == 2).sum())

    print(f"시점: 반경 {args.view_radius}m, 고도 {args.view_alt}m, 방위 {args.view_azim:.0f}도")
    print(f"  카메라->타겟 거리: {cam_to_target:.2f} m")
    print(f"  밴드: [{args.min_dist}, {args.max_dist}] m, FOV {FOV_DEG} deg")
    print(f"  데드존(너무 가까움): {n_close}개")
    print(f"  촬영 가능:          {n_band}개")
    print(f"  원거리/화각 밖:     {n_out}개")
    print(f"  점-카메라 거리 범위: {dist.min():.2f} ~ {dist.max():.2f} m")

    # ── 시각화 (Top-Down) ──
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#0f0f1a")
    ax.set_facecolor("#15151f")
    tx, ty = TARGET[0], TARGET[1]

    # 밴드 링(데드존 경계, 원거리 경계)을 카메라 기준 원으로 표시
    ax.add_patch(Circle((cam[0], cam[1]), args.min_dist, fill=True,
                        color="#ef5350", alpha=0.12, zorder=1))
    ax.add_patch(Circle((cam[0], cam[1]), args.min_dist, fill=False,
                        color="#ef5350", ls="--", lw=1.3, zorder=2,
                        label=f"dead-zone (< {args.min_dist}m, too close)"))
    ax.add_patch(Circle((cam[0], cam[1]), args.max_dist, fill=False,
                        color="#4fc3f7", ls="--", lw=1.3, zorder=2,
                        label=f"far limit ({args.max_dist}m)"))

    # FOV 콘 (두 경계선)
    base = math.degrees(math.atan2(cam_dir[1], cam_dir[0]))
    half = FOV_DEG/2
    ax.add_patch(Wedge((cam[0], cam[1]), args.max_dist, base-half, base+half,
                       width=args.max_dist-args.min_dist, color="#ffcc02",
                       alpha=0.08, zorder=1))

    colors = {0: "#ef5350", 1: "#26c281", 2: "#666666"}
    names  = {0: f"too close: {n_close}", 1: f"capturable: {n_band}",
              2: f"not captured: {n_out}"}
    for lab in [2, 0, 1]:
        m = label == lab
        ax.scatter(pts[m, 0], pts[m, 1], s=14, color=colors[lab],
                   alpha=0.85, label=names[lab], zorder=4)

    ax.plot(cam[0], cam[1], "^", color="white", markersize=14, zorder=6)
    ax.text(cam[0], cam[1]+0.4, "CAMERA", color="white", fontsize=9, ha="center")
    ax.plot([cam[0], tx], [cam[1], ty], color="#888", lw=0.8, ls=":", zorder=3)
    ax.plot(tx, ty, "*", color="#ffcc02", markersize=20, zorder=7,
            markeredgecolor="white")
    ax.text(tx, ty-0.6, "TARGET", color="#ffcc02", fontsize=9, ha="center")

    ax.set_title(f"camera capture-range model  (cam->target {cam_to_target:.1f}m)",
                 color="white", fontsize=12)
    ax.set_xlabel("X (m)", color="#aaa"); ax.set_ylabel("Y (m)", color="#aaa")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9,
              loc="upper right")

    Path(args.output_img).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_img, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n시각화 저장: {args.output_img}")


if __name__ == "__main__":
    main()
