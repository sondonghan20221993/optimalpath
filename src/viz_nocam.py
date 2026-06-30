"""viz_nocam.py — no-camera 시나리오 결과 시각화"""
import sys, types, json, math
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
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.lines import Line2D

ROOT  = HERE.parent
NPZ   = ROOT / "real_test" / "real_test_pts_normals.npz"
JSON  = ROOT / "results" / "pbnbv_nocam_path.json"
OUT   = ROOT / "results" / "pbnbv_nocam_viz.png"

BG, DARK   = "#0f0f1a", "#1a1a2e"
C_PT       = "#4fc3f7"
C_WP       = "#ffd600"
C_WP_SV    = "#ff4081"   # salvage waypoint
C_PATH     = "#76ff03"
C_TGT      = "#ff6d00"

pts    = np.load(NPZ)["points"].astype(float)
res    = json.load(open(JSON))
target = np.array(res["target"])
wps    = np.array([w["pos"] for w in res["waypoints"]])
phases = [w.get("phase", "pbnbv") for w in res["waypoints"]]
scores = res["selected_scores"]
tilts  = [w["tilt_deg"] for w in res["waypoints"]]
azs    = [w["azimuth_deg"] for w in res["waypoints"]]

# coverage reduction
A.RAYCAST_OCCLUSION = True
fov_deg  = res["fov_deg"]
live     = np.ones(len(pts), dtype=bool)
cov_left = [int(live.sum())]
for wp in wps:
    _, vis = A.information_gain(wp, target, pts, live, fov_deg, A.MAX_DIST, pts_normals=None)
    live &= ~vis
    cov_left.append(int(live.sum()))

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor=BG)
n_pb = res.get("pbnbv_count", len(scores))
n_sv = res.get("salvage_count", 0)
fig.suptitle(f"PB-NBV(2) + Greedy Path  |  No-Camera  |  PB-NBV:{n_pb}wp + Salvage:{n_sv}wp  →  {res['coverage_ratio']*100:.1f}% coverage",
             color="white", fontsize=15, fontweight="bold", y=0.97)

gs = plt.GridSpec(2, 3, fig, hspace=0.38, wspace=0.30,
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
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
ax_3d.set_facecolor(BG)

np.random.seed(0)
sub = np.random.choice(len(pts), min(4000, len(pts)), replace=False)

# ① Top-down: point cloud
ax_top.set_title("Point Cloud (Top-Down)", color="white", fontsize=10)
ax_top.scatter(pts[sub,0], pts[sub,1], c=C_PT, s=2, alpha=0.5)
ax_top.plot(target[0], target[1], "*", color=C_TGT, ms=14, zorder=5)
ax_top.set_aspect("equal")
ax_top.set_xlabel("X (m)", color="#aaa", fontsize=8)
ax_top.set_ylabel("Y (m)", color="#aaa", fontsize=8)
ax_top.legend(handles=[
    mpatches.Patch(color=C_PT, label=f"Point cloud ({len(pts):,} pts)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor=C_TGT, ms=10, label="Centroid"),
], fontsize=7, facecolor=DARK, edgecolor="#444", labelcolor="white")

# ② 3D
ax_3d.set_title("3D: Point Cloud + Selected WPs + Path", color="white", fontsize=10)
ax_3d.scatter(pts[sub,0], pts[sub,1], pts[sub,2], c=C_PT, s=1, alpha=0.25)
for i, wp in enumerate(wps):
    c = C_WP_SV if phases[i] == "salvage" else C_WP
    ax_3d.scatter(*wp, c=c, s=60, zorder=5)
    ax_3d.text(wp[0], wp[1], wp[2]+0.15, f"WP{i+1:02d}", color="white", fontsize=6)
for i in range(len(wps)-1):
    ax_3d.plot([wps[i,0],wps[i+1,0]], [wps[i,1],wps[i+1,1]], [wps[i,2],wps[i+1,2]],
               color=C_PATH, lw=1.5, alpha=0.8)
ax_3d.scatter(*target, c=C_TGT, s=100, marker="*", zorder=6)
ax_3d.set_xlabel("X", color="#aaa", fontsize=7)
ax_3d.set_ylabel("Y", color="#aaa", fontsize=7)
ax_3d.set_zlabel("Z", color="#aaa", fontsize=7)
ax_3d.tick_params(colors="white", labelsize=6)
ax_3d.xaxis.pane.fill = ax_3d.yaxis.pane.fill = ax_3d.zaxis.pane.fill = False

# ③ PB-NBV score + salvage IG bar
ax_sc.set_title("PB-NBV(2) Score  +  Salvage IG", color="white", fontsize=10)
sv_igs = res.get("salvage_igs", [])
all_scores = list(scores) + [float(ig) for ig in sv_igs]
all_colors = [C_WP]*len(scores) + [C_WP_SV]*len(sv_igs)
x_idx = np.arange(1, len(all_scores)+1)
bars = ax_sc.bar(x_idx, all_scores, color=all_colors, edgecolor=BG, alpha=0.9)
for bar, s in zip(bars, all_scores):
    ax_sc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
               f"{s:.0f}", ha="center", va="bottom", color="white", fontsize=8)
ax_sc.axvline(len(scores)+0.5, color="#555", lw=1.5, ls="--")
ax_sc.text(len(scores)+0.6, max(all_scores)*0.85, "Salvage", color=C_WP_SV, fontsize=8)
ax_sc.set_xlabel("Selection step", color="#aaa", fontsize=8)
ax_sc.set_ylabel("Score / IG", color="#aaa", fontsize=8)
ax_sc.set_xticks(x_idx)
ax_sc.set_xticklabels([f"WP{i:02d}" for i in x_idx], rotation=45, fontsize=7)

# ④ Greedy path top-down
ax_path.set_title("Greedy NN Path (Top-Down)", color="white", fontsize=10)
ax_path.scatter(pts[sub,0], pts[sub,1], c="#2a3a55", s=2, alpha=0.4)
ax_path.plot(wps[:,0], wps[:,1], "-", color=C_PATH, lw=2, zorder=4, alpha=0.85)
for i, wp in enumerate(wps):
    c = C_WP_SV if phases[i] == "salvage" else C_WP
    ax_path.scatter(wp[0], wp[1], c=c, s=80, zorder=5)
    tag = "*" if phases[i] == "salvage" else ""
    ax_path.text(wp[0]+0.05, wp[1]+0.05, f"WP{i+1:02d}{tag}", color="white", fontsize=7, zorder=6)
ax_path.plot(target[0], target[1], "*", color=C_TGT, ms=14, zorder=6)
ax_path.set_aspect("equal")
ax_path.set_xlabel("X (m)", color="#aaa", fontsize=8)
ax_path.set_ylabel("Y (m)", color="#aaa", fontsize=8)

# ⑤ Coverage reduction
ax_cov.set_title("Uncovered Points Remaining", color="white", fontsize=10)
steps = list(range(len(cov_left)))
ax_cov.plot(steps, cov_left, "-o", color=C_PATH, lw=2, ms=6)
for i, v in enumerate(cov_left):
    ax_cov.text(i, v + len(pts)*0.01, str(v), ha="center", va="bottom",
                color="white", fontsize=7)
ax_cov.fill_between(steps, cov_left, alpha=0.15, color=C_PATH)
ax_cov.set_xlabel("Cumulative WPs", color="#aaa", fontsize=8)
ax_cov.set_ylabel("Uncovered points", color="#aaa", fontsize=8)
ax_cov.set_xticks(steps)
ax_cov.set_xticklabels(["Start"]+[f"WP{i:02d}" for i in range(1,len(steps))], rotation=45, fontsize=7)
# salvage 구간 음영
sv_start = n_pb + 1
if n_sv > 0 and sv_start < len(steps):
    ax_cov.axvspan(sv_start - 0.5, len(steps) - 0.5, alpha=0.08, color=C_WP_SV)
    ax_cov.text(sv_start - 0.3, max(cov_left)*0.5, "Salvage", color=C_WP_SV, fontsize=8, alpha=0.8)
total = res["total_points"]
covered = res["covered_points"]
ax_cov.text(0.97, 0.95, f"Coverage: {covered}/{total} ({100*covered/total:.1f}%)",
            transform=ax_cov.transAxes, color=C_PATH, fontsize=9, ha="right", va="top")

# ⑥ Tilt distribution
ax_tilt.set_title("WP Tilt Angle (to Centroid)", color="white", fontsize=10)
colors_t = [plt.cm.plasma(t/50) for t in tilts]
bars2 = ax_tilt.bar(range(1, len(tilts)+1), tilts, color=colors_t, edgecolor=BG, alpha=0.9)
ax_tilt.axhline(A.MAX_TILT_DEG, color="#ef5350", lw=1.5, ls="--",
                label=f"Limit {A.MAX_TILT_DEG:.0f} deg")
for bar, t, az in zip(bars2, tilts, azs):
    ax_tilt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{t:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    ax_tilt.text(bar.get_x()+bar.get_width()/2, 1,
                 f"az{az:.0f}", ha="center", va="bottom", color="#aaa", fontsize=6)
ax_tilt.set_xlabel("WP (az = azimuth deg)", color="#aaa", fontsize=8)
ax_tilt.set_ylabel("Tilt (deg)", color="#aaa", fontsize=8)
ax_tilt.set_xticks(range(1, len(tilts)+1))
ax_tilt.set_xticklabels([f"WP{i:02d}" for i in range(1, len(tilts)+1)], rotation=45, fontsize=7)
ax_tilt.set_ylim(0, 52)
ax_tilt.legend(fontsize=8, facecolor=DARK, edgecolor="#444", labelcolor="white")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ {OUT}")
