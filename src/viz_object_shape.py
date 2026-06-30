"""viz_object_shape.py — 정제된 점군의 실제 형상 진단 (3뷰 + 밀도)"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
p = np.load(ROOT/"real_test"/"real_test_pts_normals.npz")["points"].astype(float)
OUT = ROOT/"results"/"object_shape.png"

BG, DARK = "#0f0f1a", "#1a1a2e"
c = p.mean(0)

fig = plt.figure(figsize=(20, 6), facecolor=BG)
fig.suptitle(f"Real object shape (SOR-cleaned, {len(p)} pts)  |  body 1.55 x 1.10 x 0.34 m  (concave U-slab)",
             color="white", fontsize=13, fontweight="bold", y=1.02)

views = [
    ("Top-down  X-Y  (the U-shape)", 0, 1, "X (m)", "Y (m)"),
    ("Front  X-Z  (height 0.36m)",   0, 2, "X (m)", "Z (m)"),
    ("Side  Y-Z  (flat slab)",       1, 2, "Y (m)", "Z (m)"),
]
for i, (title, a, b, xl, yl) in enumerate(views):
    ax = fig.add_subplot(1, 4, i+1)
    ax.set_facecolor(DARK); ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(title, color="white", fontsize=10)
    ax.scatter(p[:, a], p[:, b], c=p[:, 2], cmap="viridis", s=4, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel(xl, color="#aaa", fontsize=8); ax.set_ylabel(yl, color="#aaa", fontsize=8)

# 4th: XY density heatmap
ax = fig.add_subplot(1, 4, 4)
ax.set_facecolor(DARK); ax.tick_params(colors="white", labelsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#444")
ax.set_title("XY point density (perimeter dense = U-frame)", color="white", fontsize=10)
H, xe, ye = np.histogram2d(p[:,0], p[:,1], bins=20)
im = ax.imshow(H.T, origin="lower", extent=[xe[0],xe[-1],ye[0],ye[-1]],
               cmap="inferno", aspect="equal")
ax.set_xlabel("X (m)", color="#aaa", fontsize=8); ax.set_ylabel("Y (m)", color="#aaa", fontsize=8)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelcolor="white", labelsize=7)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ {OUT}")
