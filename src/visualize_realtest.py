"""
visualize_realtest.py
pbnbv_realtest_path.json 결과를 3D + 평면 2종 이미지로 저장
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
RES    = ROOT / "results" / "pbnbv_realtest_path.json"
NPZ    = ROOT / "real_test" / "real_test_pts_normals.npz"
OUT    = ROOT / "results" / "pbnbv_realtest_viz.png"

r    = json.load(open(RES))
tgt  = np.array(r["target"])
wps  = [np.array(w["pos"]) for w in r["waypoints"]]
azs  = [w["azimuth_deg"] for w in r["waypoints"]]
tilts= [w["tilt_deg"]    for w in r["waypoints"]]
scrs = r["selected_scores"]

d    = np.load(NPZ)
pts  = d["points"]

# 서브샘플 (시각화용)
idx  = np.random.default_rng(0).choice(len(pts), min(1200, len(pts)), replace=False)
sp   = pts[idx]

wp_arr = np.array(wps)

fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("#0e1117")

# ── 1) 3D View ──────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
ax3.set_facecolor("#0e1117")

ax3.scatter(sp[:,0], sp[:,1], sp[:,2], s=2, c="#5bc8f5", alpha=0.25, linewidths=0)
ax3.scatter(*tgt, s=120, c="gold", marker="*", zorder=5, label="Target")

# 경로 화살표
for i in range(len(wps)-1):
    a, b = wps[i], wps[i+1]
    ax3.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], c="#ff6b6b", lw=1.5, alpha=0.8)
    ax3.quiver(a[0], a[1], a[2],
               (b[0]-a[0])*0.4, (b[1]-a[1])*0.4, (b[2]-a[2])*0.4,
               color="#ff6b6b", arrow_length_ratio=0.4, linewidth=1.2)

# waypoint 산점도 (스코어로 색상)
sc_arr = np.array(scrs)
sc_norm = (sc_arr - sc_arr.min()) / (sc_arr.max() - sc_arr.min() + 1e-9)
colors = plt.cm.RdYlGn(sc_norm)
ax3.scatter(wp_arr[:,0], wp_arr[:,1], wp_arr[:,2],
            s=80, c=colors, edgecolors="white", linewidths=0.5, zorder=6)
for i, wp in enumerate(wps):
    ax3.text(wp[0]+0.1, wp[1]+0.1, wp[2]+0.1, f"{i+1}", color="white",
             fontsize=7.5, fontweight="bold")

ax3.set_xlabel("X", color="white", fontsize=9)
ax3.set_ylabel("Y", color="white", fontsize=9)
ax3.set_zlabel("Z (NED)", color="white", fontsize=9)
ax3.tick_params(colors="white", labelsize=7)
for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#333")
ax3.set_title("PB-NBV(2) + Greedy Path  [3D View]",
              color="white", fontsize=11, pad=10)

# ── 2) 평면도 (XY) + 보조 패널 ──────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("#12151c")

ax2.scatter(sp[:,0], sp[:,1], s=3, c="#5bc8f5", alpha=0.25, linewidths=0)
ax2.scatter(*tgt[:2], s=160, c="gold", marker="*", zorder=5, label="Target")

for i in range(len(wps)-1):
    a, b = wps[i], wps[i+1]
    ax2.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]),
                 arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.5))

ax2.scatter(wp_arr[:,0], wp_arr[:,1],
            s=90, c=colors, edgecolors="white", linewidths=0.5, zorder=6)
for i, wp in enumerate(wps):
    ax2.text(wp[0]+0.05, wp[1]+0.05, f"{i+1}",
             color="white", fontsize=8, fontweight="bold")

# 방위각 선
for wp in wps:
    ax2.plot([tgt[0], wp[0]], [tgt[1], wp[1]],
             color="#444", lw=0.6, linestyle="--", alpha=0.5)

ax2.set_xlabel("X  (m)", color="white", fontsize=9)
ax2.set_ylabel("Y  (m)", color="white", fontsize=9)
ax2.tick_params(colors="white", labelsize=7)
ax2.spines[:].set_color("#333")
ax2.set_aspect("equal")
ax2.set_title("Top View (XY)", color="white", fontsize=11, pad=8)

# 텍스트 테이블
tbl_x = 0.72
lines = [
    f"{'WP':>3}  {'Az':>6}  {'Tilt':>6}  {'Score':>6}",
    "─" * 30,
]
for i, w in enumerate(r["waypoints"]):
    lines.append(f"{i+1:>3}  {w['azimuth_deg']:>6.1f}°  {w['tilt_deg']:>5.1f}°  {scrs[i]:>6.0f}")
lines += [
    "─" * 30,
    f"미관측 비율: {r['underobserved_ratio']*100:.1f}%",
    f"점구름: {len(pts):,}개",
    f"FOV: {r['fov_used_deg']:.1f}°",
]
fig.text(tbl_x, 0.88, "\n".join(lines),
         transform=fig.transFigure,
         fontsize=7.5, color="white",
         fontfamily="monospace",
         va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", fc="#1c2030", ec="#445", lw=0.8))

sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                             norm=plt.Normalize(sc_arr.min(), sc_arr.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
cbar.set_label("PB-NBV Score", color="white", fontsize=8)
cbar.ax.tick_params(colors="white", labelsize=7)

plt.tight_layout(rect=[0, 0, 0.70, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"저장: {OUT}")
