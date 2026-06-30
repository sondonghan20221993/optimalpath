"""
viz_realtest.py
pbnbv_realtest_path.json + real_test_pts_normals.npz 로 결과 시각화 이미지 생성
"""
import sys, json, math, types
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.lines import Line2D

for _fn in ["Malgun Gothic", "NanumGothic"]:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT   = HERE.parent
NPZ    = ROOT / "real_test" / "real_test_pts_normals.npz"
META   = sorted((ROOT / "real_test" / "meta").glob("*.json"))
JSON   = ROOT / "results" / "pbnbv_realtest_path.json"
OUT    = ROOT / "results" / "pbnbv_realtest_viz.png"

BG   = "#0f0f1a"
DARK = "#1a1a2e"
C_PT_UNDER  = "#ef5350"
C_PT_COVER  = "#4fc3f7"
C_CAM       = "#aaaaaa"
C_WP        = "#ffd600"
C_PATH      = "#76ff03"
C_TARGET    = "#ff6d00"

# ── 데이터 로드 ─────────────────────────────────────────────────────────────
d       = np.load(NPZ)
pts     = d["points"].astype(float)

result  = json.load(open(JSON))
target  = np.array(result["target"])
wps_raw = [np.array(w["pos"]) for w in result["waypoints"]]
scores  = result["selected_scores"]

cam_pos, cam_fwd = [], []
import json as _json
for m in META:
    j = _json.load(open(m))
    p = j["camera"]["pose"]["position"]
    o = j["camera"]["pose"]["orientation"]
    import types as _t
    w, x, y, z = o["w"], o["x"], o["y"], o["z"]
    n = math.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])
    cam_pos.append([p["x"], p["y"], p["z"]])
    cam_fwd.append(R @ np.array([1., 0., 0.]))
cam_pos = np.array(cam_pos)
cam_fwd = np.array(cam_fwd)

fov_deg = min(float(result["fov_raw_deg"]), A.MAX_FOV_DEG)

# coverage & underobs (raycast ON)
A.RAYCAST_OCCLUSION = True
coverage    = A.compute_coverage(pts, cam_pos, cam_fwd, fov_deg, A.MAX_DIST, pts_normals=None)
under_mask  = A.select_underobserved_by_fraction(coverage, A.UNDEROBS_FRACTION)

# coverage reduction simulation
live = under_mask.copy()
cov_left = [int(live.sum())]
for wp in wps_raw:
    to_t   = target - wp
    cam_dir = to_t / (np.linalg.norm(to_t) + 1e-8)
    _, vis = A.information_gain(wp, target, pts, live, fov_deg, A.MAX_DIST, pts_normals=None)
    live &= ~vis
    cov_left.append(int(live.sum()))

# ── 레이아웃 ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14), facecolor=BG)
fig.suptitle("PB-NBV(2) + Greedy Path  |  real_test 결과",
             color="white", fontsize=16, fontweight="bold", y=0.97)

gs = plt.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                  left=0.05, right=0.97, top=0.93, bottom=0.05)
ax_top  = fig.add_subplot(gs[0, 0])
ax_3d   = fig.add_subplot(gs[0, 1], projection="3d")
ax_sc   = fig.add_subplot(gs[0, 2])
ax_path = fig.add_subplot(gs[1, 0])
ax_cov  = fig.add_subplot(gs[1, 1])
ax_tilt = fig.add_subplot(gs[1, 2])

for ax in [ax_top, ax_sc, ax_path, ax_cov, ax_tilt]:
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
ax_3d.set_facecolor(BG)

# ── ① Top-down: 포인트클라우드 + 카메라 + 타겟 ───────────────────────────────
ax_top.set_title("Point Cloud + Cameras (Top-Down)", color="white", fontsize=10)
np.random.seed(0)
sub = np.random.choice(len(pts), min(4000, len(pts)), replace=False)
ax_top.scatter(pts[sub, 0], pts[sub, 1],
               c=["#ef5350" if under_mask[i] else "#4fc3f7" for i in sub],
               s=2, alpha=0.5)
ax_top.scatter(cam_pos[:, 0], cam_pos[:, 1], c=C_CAM, s=15, zorder=4, alpha=0.6)
ax_top.plot(target[0], target[1], "*", color=C_TARGET, ms=14, zorder=6)
ax_top.set_aspect("equal")
ax_top.set_xlabel("X (m)", color="#aaa", fontsize=8)
ax_top.set_ylabel("Y (m)", color="#aaa", fontsize=8)
ax_top.legend(handles=[
    mpatches.Patch(color=C_PT_UNDER, label=f"미관측 ({under_mask.sum()}개)"),
    mpatches.Patch(color=C_PT_COVER, label="관측됨"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_CAM, ms=6, label="기존 카메라"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor=C_TARGET, ms=10, label="타겟"),
], fontsize=7, facecolor=DARK, edgecolor="#444", labelcolor="white")

# ── ② 3D: 점구름 + 선택 시점 + 경로 ─────────────────────────────────────────
ax_3d.set_title("3D: 미관측 영역 + 선택 시점 + 경로", color="white", fontsize=10)
under_pts = pts[under_mask]
over_pts  = pts[~under_mask]
sub2 = np.random.choice(len(over_pts), min(1500, len(over_pts)), replace=False)
ax_3d.scatter(over_pts[sub2, 0], over_pts[sub2, 1], over_pts[sub2, 2],
              c=C_PT_COVER, s=1, alpha=0.2)
ax_3d.scatter(under_pts[:, 0], under_pts[:, 1], under_pts[:, 2],
              c=C_PT_UNDER, s=3, alpha=0.6)
wps = np.array(wps_raw)
for i, wp in enumerate(wps):
    ax_3d.scatter(*wp, c=C_WP, s=60, zorder=5)
    ax_3d.text(wp[0], wp[1], wp[2]+0.15, f"WP{i+1:02d}", color="white", fontsize=6)
for i in range(len(wps)-1):
    ax_3d.plot([wps[i,0], wps[i+1,0]], [wps[i,1], wps[i+1,1]], [wps[i,2], wps[i+1,2]],
               color=C_PATH, lw=1.5, alpha=0.8)
ax_3d.scatter(*target, c=C_TARGET, s=100, marker="*", zorder=6)
ax_3d.set_xlabel("X", color="#aaa", fontsize=7); ax_3d.set_ylabel("Y", color="#aaa", fontsize=7)
ax_3d.set_zlabel("Z", color="#aaa", fontsize=7)
ax_3d.tick_params(colors="white", labelsize=6)
ax_3d.xaxis.pane.fill = False; ax_3d.yaxis.pane.fill = False; ax_3d.zaxis.pane.fill = False

# ── ③ PB-NBV score 막대 ──────────────────────────────────────────────────────
ax_sc.set_title("PB-NBV(2) 스코어 (선택 순서)", color="white", fontsize=10)
x_idx = np.arange(1, len(scores)+1)
bars = ax_sc.bar(x_idx, scores, color=C_WP, edgecolor=BG, alpha=0.9)
for bar, s in zip(bars, scores):
    ax_sc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
               f"{s:.0f}", ha="center", va="bottom", color="white", fontsize=8)
ax_sc.set_xlabel("선택 순서", color="#aaa", fontsize=8)
ax_sc.set_ylabel("PB-NBV(2) score", color="#aaa", fontsize=8)
ax_sc.set_xticks(x_idx)
ax_sc.set_xticklabels([f"WP{i:02d}" for i in x_idx], rotation=45, fontsize=7)

# ── ④ Greedy 경로 Top-Down ───────────────────────────────────────────────────
ax_path.set_title("Greedy NN 경로 (Top-Down)", color="white", fontsize=10)
sub3 = np.random.choice(len(pts), min(3000, len(pts)), replace=False)
ax_path.scatter(pts[sub3, 0], pts[sub3, 1],
                c=["#ef5350" if under_mask[i] else "#2a3a55" for i in sub3],
                s=2, alpha=0.4)
ax_path.scatter(cam_pos[:, 0], cam_pos[:, 1], c=C_CAM, s=10, zorder=3, alpha=0.4)
ax_path.plot(wps[:, 0], wps[:, 1], "-", color=C_PATH, lw=2, zorder=4, alpha=0.85)
for i, wp in enumerate(wps):
    ax_path.scatter(wp[0], wp[1], c=C_WP, s=80, zorder=5)
    ax_path.text(wp[0]+0.05, wp[1]+0.05, f"WP{i+1:02d}", color="white", fontsize=7, zorder=6)
ax_path.plot(target[0], target[1], "*", color=C_TARGET, ms=14, zorder=6)
ax_path.set_aspect("equal")
ax_path.set_xlabel("X (m)", color="#aaa", fontsize=8)
ax_path.set_ylabel("Y (m)", color="#aaa", fontsize=8)

# ── ⑤ 커버리지 감소 곡선 ─────────────────────────────────────────────────────
ax_cov.set_title("미관측 포인트 감소 (WP 누적)", color="white", fontsize=10)
steps = list(range(len(cov_left)))
ax_cov.plot(steps, cov_left, "-o", color="#76ff03", lw=2, ms=6)
for i, v in enumerate(cov_left):
    ax_cov.text(i, v+3, str(v), ha="center", va="bottom", color="white", fontsize=7)
ax_cov.fill_between(steps, cov_left, alpha=0.15, color="#76ff03")
ax_cov.set_xlabel("누적 WP 수", color="#aaa", fontsize=8)
ax_cov.set_ylabel("미관측 포인트 수", color="#aaa", fontsize=8)
ax_cov.set_xticks(steps)
ax_cov.set_xticklabels([f"WP{i:02d}" if i > 0 else "시작" for i in steps], rotation=45, fontsize=7)
start = cov_left[0]
end   = cov_left[-1]
ax_cov.text(0.97, 0.95, f"커버: {start-end}/{start} ({100*(start-end)/start:.1f}%)",
            transform=ax_cov.transAxes, color="#76ff03", fontsize=9, ha="right", va="top")

# ── ⑥ tilt 각도 분포 ─────────────────────────────────────────────────────────
ax_tilt.set_title("WP 틸트각 (타겟 기준)", color="white", fontsize=10)
tilts  = [w["tilt_deg"] for w in result["waypoints"]]
azs    = [w["azimuth_deg"] for w in result["waypoints"]]
colors = [plt.cm.plasma(t/50) for t in tilts]
bars2  = ax_tilt.bar(range(1, len(tilts)+1), tilts, color=colors, edgecolor=BG, alpha=0.9)
ax_tilt.axhline(A.MAX_TILT_DEG, color="#ef5350", lw=1.5, ls="--", label=f"한계 {A.MAX_TILT_DEG:.0f}°")
for bar, t, az in zip(bars2, tilts, azs):
    ax_tilt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{t:.1f}°", ha="center", va="bottom", color="white", fontsize=7)
    ax_tilt.text(bar.get_x()+bar.get_width()/2, 1,
                 f"{az:.0f}°", ha="center", va="bottom", color="#aaa", fontsize=6)
ax_tilt.set_xlabel("WP (숫자 아래: azimuth°)", color="#aaa", fontsize=8)
ax_tilt.set_ylabel("틸트각 (°)", color="#aaa", fontsize=8)
ax_tilt.set_xticks(range(1, len(tilts)+1))
ax_tilt.set_xticklabels([f"WP{i:02d}" for i in range(1, len(tilts)+1)], rotation=45, fontsize=7)
ax_tilt.set_ylim(0, 50)
ax_tilt.legend(fontsize=8, facecolor=DARK, edgecolor="#444", labelcolor="white")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ 저장: {OUT}")
