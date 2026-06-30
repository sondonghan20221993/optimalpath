"""viz_orbit_compare.py — 원형 궤도 보완: 논문방식 vs 규제 vs raycast 비교"""
import sys, types, json, glob, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = HERE.parent
NPZ  = ROOT / "real_test" / "real_test_pts_normals.npz"
META = sorted((ROOT / "real_test" / "meta").glob("*.json"))
OUT  = ROOT / "results" / "orbit_compare.png"

BG, DARK = "#0f0f1a", "#1a1a2e"
C_PT, C_BLIND, C_TGT, C_CAM = "#2a3a55", "#ff1744", "#ff6d00", "#7e57c2"


def quatR(w, x, y, z):
    n = math.sqrt(w*w+x*x+y*y+z*z); w,x,y,z = w/n,x/n,y/n,z/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

d = np.load(NPZ); pts = d["points"].astype(float); nrm = d["normals"].astype(float)
target = pts.mean(0); nrm[(nrm*(pts-target)).sum(1) < 0] *= -1
cam_pos, cam_fwd = [], []
for m in META:
    j = json.load(open(m)); p = j["camera"]["pose"]["position"]; o = j["camera"]["pose"]["orientation"]
    cam_pos.append([p["x"],p["y"],p["z"]]); cam_fwd.append(quatR(o["w"],o["x"],o["y"],o["z"])@np.array([1.,0,0]))
cam_pos, cam_fwd = np.array(cam_pos), np.array(cam_fwd)

A.RAYCAST_OCCLUSION = True
cov = A.compute_coverage(pts, cam_pos, cam_fwd, 89.9, A.MAX_DIST, pts_normals=nrm)
blind = cov == 0
print(f"궤도 raycast 커버 {100*(cov>0).mean():.1f}%, 사각지대 {blind.sum()}개")

# 결과 로드
paper      = json.load(open(ROOT/"results/pbnbv_orbit_paper_path.json"))      # θ90, 0 WP
strict     = json.load(open(ROOT/"results/pbnbv_orbit_paper_strict.json"))    # sweep list
# 규제로 real 100% 처음 도달하는 임계 선택 (없으면 가장 강한 규제)
_recovered = [r for r in strict if r["real_cov_final"] >= 0.999]
strict_best = (max(_recovered, key=lambda r: r["inc_max_deg"]) if _recovered
               else min(strict, key=lambda r: r["inc_max_deg"]))
STRICT_TH   = strict_best["inc_max_deg"]

paper_wp    = np.array([w["pos"] for w in paper["waypoints"]]) if paper["waypoints"] else np.empty((0,3))
strict_wp   = np.array(strict_best["waypoints"]) if strict_best["waypoints"] else np.empty((0,3))

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 13), facecolor=BG)
fig.suptitle("Circular Orbit + Supplementary NBV  |  Paper(0.5^r) vs Regulated vs Ray-cast",
             color="white", fontsize=14, fontweight="bold", y=0.97)
gs = gridspec.GridSpec(2, 3, fig, hspace=0.28, wspace=0.22,
                       left=0.04, right=0.98, top=0.91, bottom=0.06)

np.random.seed(0)
sub = np.random.choice(len(pts), min(2500, len(pts)), replace=False)

def base(ax, title, col):
    ax.set_facecolor(DARK); ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(title, color=col, fontsize=10, fontweight="bold")
    ax.scatter(pts[sub,0], pts[sub,1], c=C_PT, s=2, alpha=0.35)
    ax.plot(target[0], target[1], "*", color=C_TGT, ms=15, zorder=6)
    ax.set_aspect("equal"); ax.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax.set_ylabel("Y (m)", color="#aaa", fontsize=7)

# (a) 궤도 + 사각지대
ax = fig.add_subplot(gs[0,0]); base(ax, "(a) Orbit + real blind spots (raycast)", "white")
ax.scatter(cam_pos[:,0], cam_pos[:,1], c=C_CAM, s=22, zorder=4, label="orbit cam (34)")
ax.scatter(pts[blind,0], pts[blind,1], c=C_BLIND, s=10, zorder=5, label=f"blind ({blind.sum()})")
ax.legend(fontsize=7, facecolor=DARK, edgecolor="#444", labelcolor="white", loc="upper right")
ax.text(0.03,0.97, "orbit covers 95.0%\n(raycast, real pts)", transform=ax.transAxes,
        color="white", fontsize=8, va="top", fontweight="bold")

# (b) 논문 방식 θ90
ax = fig.add_subplot(gs[0,1]); base(ax, "(b) Paper PB-NBV (0.5^r, theta<90)", "#40c4ff")
ax.scatter(cam_pos[:,0], cam_pos[:,1], c=C_CAM, s=15, alpha=0.4, zorder=3)
ax.scatter(pts[blind,0], pts[blind,1], c=C_BLIND, s=8, alpha=0.5, zorder=4)
ax.text(0.03,0.97, "voxel claims 100%\nfrontier = 0\n=> 0 supplement WP\nreal STILL 95%",
        transform=ax.transAxes, color="#40c4ff", fontsize=9, va="top", fontweight="bold")

# (c) 규제 PB-NBV + 거리 가중치 (ours)
ax = fig.add_subplot(gs[0,2])
base(ax, f"(c) Regulated PB-NBV + U=F/dist (theta<{STRICT_TH:.0f})", "#69f0ae")
ax.scatter(cam_pos[:,0], cam_pos[:,1], c=C_CAM, s=15, alpha=0.4, zorder=3)
ax.scatter(pts[blind,0], pts[blind,1], c=C_BLIND, s=8, alpha=0.4, zorder=4)
if len(strict_wp):
    ax.plot(strict_wp[:,0], strict_wp[:,1], "-o", color="#69f0ae", lw=2, ms=9,
            zorder=6, mec="white", mew=0.6)
    for k,w in enumerate(strict_wp):
        ax.text(w[0]+0.05, w[1]+0.05, f"{k+1}", color="white", fontsize=8, zorder=7)
n_s = len(strict_wp); rc = 100*strict_best["real_cov_final"]
ax.text(0.03,0.97, f"PB-NBV (no raycast)\n+ incidence reg theta<{STRICT_TH:.0f}\n"
        f"+ U=F/dist weighting\n{n_s} WP -> real {rc:.1f}%",
        transform=ax.transAxes, color="#69f0ae", fontsize=9, va="top", fontweight="bold")

# (d) 규제 sweep
ax = fig.add_subplot(gs[1,0]); ax.set_facecolor(DARK); ax.tick_params(colors="white", labelsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#444")
ax.set_title("(d) Regulation: incidence-angle cap sweep", color="white", fontsize=10)
ths = [r["inc_max_deg"] for r in strict]
vcov = [100*r["voxel_cov_orbit"] for r in strict]
rcov = [100*r["real_cov_final"] for r in strict]
nsup = [r["supplementary"] for r in strict]
ax.plot(ths, vcov, "-o", color="#40c4ff", lw=2, label="voxel cov (orbit, claimed)")
ax.plot(ths, rcov, "-s", color="#69f0ae", lw=2, label="real cov (final, raycast)")
ax2 = ax.twinx(); ax2.bar(ths, nsup, width=4, color="#ff9100", alpha=0.4, label="supplement WP")
ax2.set_ylabel("# supplement WP", color="#ff9100", fontsize=8); ax2.tick_params(colors="#ff9100", labelsize=7)
ax.set_xlabel("incidence cap theta_max (deg)", color="#aaa", fontsize=8)
ax.set_ylabel("coverage (%)", color="#aaa", fontsize=8)
ax.invert_xaxis()
ax.legend(fontsize=7, facecolor=DARK, edgecolor="#444", labelcolor="white", loc="lower left")
ax.text(0.5,0.06,"stricter -> finds more gaps;\nreal cov reaches 100% at theta<=60",
        transform=ax.transAxes, color="#ffab40", fontsize=8, ha="center", fontweight="bold")

# (e) voxel-claim vs real bar
ax = fig.add_subplot(gs[1,1]); ax.set_facecolor(DARK); ax.tick_params(colors="white", labelsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#444")
ax.set_title("(e) Claimed voxel cov vs REAL cov", color="white", fontsize=10)
labels = ["Paper\ntheta<90", f"Regulated+DistW\ntheta<{STRICT_TH:.0f}\n(ours)"]
claimed = [100, 100*strict_best["voxel_cov_final"]]
real    = [95.0, 100*strict_best["real_cov_final"]]
x = np.arange(len(labels)); w = 0.35
ax.bar(x-w/2, claimed, w, color="#40c4ff", label="claimed (voxel)")
ax.bar(x+w/2, real, w, color="#69f0ae", label="REAL (raycast)")
for i,(c,r) in enumerate(zip(claimed, real)):
    ax.text(i-w/2, c+0.3, f"{c:.0f}", ha="center", color="white", fontsize=8)
    ax.text(i+w/2, r+0.3, f"{r:.1f}", ha="center", color="white", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(90, 103)
ax.set_ylabel("coverage (%)", color="#aaa", fontsize=8)
ax.legend(fontsize=8, facecolor=DARK, edgecolor="#444", labelcolor="white", loc="lower right")
ax.text(0.5,0.42,"Paper theta<90:\n100% claimed but\n95% real = 5%p BLIND",
        transform=ax.transAxes, color="#ff5252", fontsize=8, ha="center", va="center", fontweight="bold")

# (f) 텍스트 결론
ax = fig.add_subplot(gs[1,2]); ax.set_facecolor(DARK); ax.axis("off")
n_reg = strict_best["supplementary"]; rc = 100*strict_best["real_cov_final"]
lines = [
    ("Key findings (SOR-cleaned data)", "white", 11, "bold"),
    ("", "white", 8, "normal"),
    ("1) Paper 0.5^r (theta<90): occluded regions", "#40c4ff", 9, "normal"),
    ("   marked as 'observed' -> frontier=0,", "#ccc", 8, "normal"),
    ("   0 supplement WP. Real coverage 95%.", "#ff5252", 9, "bold"),
    ("", "white", 8, "normal"),
    (f"2) Ours: PB-NBV + incidence reg (theta<{STRICT_TH:.0f})", "#69f0ae", 9, "normal"),
    (f"   + U=F/dist (Bircher 2016 utility)", "#ccc", 8, "normal"),
    (f"   -> {n_reg} WP, real {rc:.1f}% (no raycast)", "#69f0ae", 9, "bold"),
    ("", "white", 8, "normal"),
    ("   Incidence regulation: grazing-angle views", "#aaa", 8, "normal"),
    ("   marked 'unobserved' -> frontier survives", "#aaa", 8, "normal"),
    ("   -> supplement WP selected.", "#aaa", 8, "normal"),
    ("", "white", 8, "normal"),
    ("   U=F/dist: prefers closer blind-spot WPs", "#aaa", 8, "normal"),
    ("   -> shorter travel, efficient coverage.", "#aaa", 8, "normal"),
    ("", "white", 8, "normal"),
    ("=> PB-NBV extended for drone inspection:", "white", 9, "bold"),
    ("   regulation + travel-cost utility.", "white", 9, "bold"),
]
y = 0.97; dy = 1.0/(len(lines)+1)
for t,c,s,w_ in lines:
    ax.text(0.02, y, t, color=c, fontsize=s, va="top", fontweight=w_, transform=ax.transAxes)
    y -= dy

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ {OUT}")
