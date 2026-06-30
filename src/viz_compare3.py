"""viz_compare3.py — Case1/2/3 비교 시각화 (공동통제)"""
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
from matplotlib.lines import Line2D

ROOT = HERE.parent
NPZ  = ROOT / "real_test" / "real_test_pts_normals.npz"
OUT  = ROOT / "results" / "compare3_cases.png"

BG, DARK = "#0f0f1a", "#1a1a2e"
C_PT   = "#2a3a55"
C_TGT  = "#ff6d00"
COLORS = ["#ffd600", "#40c4ff", "#69f0ae"]   # case1/2/3
START  = np.array([-27.06, -50.51, -6.13])

pts    = np.load(NPZ)["points"].astype(float)
target = pts.mean(axis=0)
A.RAYCAST_OCCLUSION = True

def load_wps(path, key):
    j = json.load(open(path))
    if key == "path":
        return [np.array(p["pos"]) for p in j["path"]]
    return [np.array(w["pos"]) for w in j["waypoints"]]

cases = [
    ("Case 1: Paper sequential NBV",       ROOT/"results/pbnbv_paper/pbnbv_paper.json",  "path"),
    ("Case 2: Hybrid (paper-F + Greedy)",  ROOT/"results/pbnbv_hybrid/pbnbv_hybrid.json","waypoints"),
    ("Case 3: Current (IG + lookahead)",   ROOT/"results/pbnbv_nocam_path.json",         "waypoints"),
]

def path_len(seq, start):
    cur, tot = np.array(start), 0.0
    for p in seq:
        tot += np.linalg.norm(np.array(p)-cur); cur = np.array(p)
    return tot

def point_cov(seq):
    live = np.ones(len(pts), dtype=bool); curve=[0]
    for wp in seq:
        _, vis = A.information_gain(np.array(wp), target, pts, live, 89.9, A.MAX_DIST)
        live &= ~vis; curve.append(len(pts)-int(live.sum()))
    return curve, 100*(len(pts)-int(live.sum()))/len(pts)

data = []
for name, p, key in cases:
    wps = load_wps(p, key)
    curve, cov = point_cov(wps)
    plen = path_len(wps, START)
    data.append({"name": name, "wps": wps, "curve": curve, "cov": cov, "plen": plen})

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12), facecolor=BG)
fig.suptitle("PB-NBV Case Comparison  |  real_test 2438pts, tilt=45deg, MAX_DIST=8m, raycast eval",
             color="white", fontsize=14, fontweight="bold", y=0.97)
gs = plt.GridSpec(2, 3, fig, hspace=0.30, wspace=0.25,
                  left=0.05, right=0.97, top=0.92, bottom=0.07)

np.random.seed(0)
sub = np.random.choice(len(pts), min(3000, len(pts)), replace=False)

# 상단: 3개 경로 top-down
for i, d in enumerate(data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(d["name"], color="white", fontsize=10)
    ax.scatter(pts[sub,0], pts[sub,1], c=C_PT, s=2, alpha=0.4)
    w = np.array(d["wps"])
    ax.plot(w[:,0], w[:,1], "-", color=COLORS[i], lw=2, zorder=4, alpha=0.85)
    for k, wp in enumerate(w):
        ax.scatter(wp[0], wp[1], c=COLORS[i], s=90, zorder=5, edgecolor="white", linewidth=0.5)
        ax.text(wp[0]+0.05, wp[1]+0.05, f"{k+1}", color="white", fontsize=8, zorder=6)
    ax.scatter(START[0], START[1], marker="s", c="white", s=60, zorder=5)
    ax.text(START[0], START[1]-0.3, "start", color="white", fontsize=7, ha="center")
    ax.plot(target[0], target[1], "*", color=C_TGT, ms=16, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax.set_ylabel("Y (m)", color="#aaa", fontsize=7)
    ax.text(0.03, 0.97, f"{len(w)} WP\ncov {d['cov']:.1f}%\npath {d['plen']:.1f}m",
            transform=ax.transAxes, color=COLORS[i], fontsize=9, va="top", fontweight="bold")

# 하단 좌: coverage 곡선
ax_cov = fig.add_subplot(gs[1, 0])
ax_cov.set_facecolor(DARK); ax_cov.tick_params(colors="white", labelsize=8)
for sp in ax_cov.spines.values(): sp.set_edgecolor("#444")
ax_cov.set_title("Coverage vs # Waypoints (real points)", color="white", fontsize=10)
for i, d in enumerate(data):
    cov_pct = [100*c/len(pts) for c in d["curve"]]
    ax_cov.plot(range(len(cov_pct)), cov_pct, "-o", color=COLORS[i], lw=2, ms=5,
                label=d["name"].split(":")[0])
ax_cov.axhline(100, color="#555", ls="--", lw=1)
ax_cov.set_xlabel("# Waypoints", color="#aaa", fontsize=8)
ax_cov.set_ylabel("Coverage (%)", color="#aaa", fontsize=8)
ax_cov.legend(fontsize=8, facecolor=DARK, edgecolor="#444", labelcolor="white", loc="lower right")
ax_cov.set_ylim(0, 105)

# 하단 중: WP수 / 커버 / 거리 막대
ax_bar = fig.add_subplot(gs[1, 1])
ax_bar.set_facecolor(DARK); ax_bar.tick_params(colors="white", labelsize=8)
for sp in ax_bar.spines.values(): sp.set_edgecolor("#444")
ax_bar.set_title("Summary Metrics", color="white", fontsize=10)
labels = ["WP count", "Coverage %", "Path (m)"]
x = np.arange(len(labels)); width = 0.25
for i, d in enumerate(data):
    vals = [len(d["wps"]), d["cov"], d["plen"]]
    bars = ax_bar.bar(x + (i-1)*width, vals, width, color=COLORS[i],
                      label=d["name"].split(":")[0], edgecolor=BG)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{v:.0f}" if v>10 else f"{v:.1f}", ha="center", va="bottom",
                    color="white", fontsize=7)
ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels, fontsize=8)
ax_bar.legend(fontsize=8, facecolor=DARK, edgecolor="#444", labelcolor="white")

# 하단 우: 텍스트 요약
ax_txt = fig.add_subplot(gs[1, 2])
ax_txt.set_facecolor(DARK); ax_txt.axis("off")
for sp in ax_txt.spines.values(): sp.set_edgecolor("#444")
lines = [
    ("Key findings", "white", 12, "bold"),
    ("", "white", 9, "normal"),
    ("Case 1 vs 2: SAME 3 viewpoints,", COLORS[0], 10, "normal"),
    ("  SAME 90.9% coverage,", "#ccc", 9, "normal"),
    (f"  but Greedy reorder cuts path", COLORS[1], 10, "normal"),
    (f"  {data[0]['plen']:.1f}m -> {data[1]['plen']:.1f}m (-5.3%)", "#ccc", 9, "normal"),
    ("", "white", 9, "normal"),
    ("-> Greedy stage = same view,", "#aaa", 9, "normal"),
    ("   shorter path (2-stage merit)", "#aaa", 9, "normal"),
    ("", "white", 9, "normal"),
    ("Case 3: raycast+salvage ->", COLORS[2], 10, "normal"),
    ("  100% coverage but 7 WP,", "#ccc", 9, "normal"),
    ("  simplified (count-based) score", "#ccc", 9, "normal"),
    ("", "white", 9, "normal"),
    ("Paper-faithful = Case 1/2.", "white", 10, "bold"),
    ("2-stage description = Case 2.", "white", 10, "bold"),
]
y = 0.95
for txt, col, sz, w in lines:
    ax_txt.text(0.03, y, txt, color=col, fontsize=sz, va="top",
                fontweight=w, transform=ax_txt.transAxes)
    y -= 0.062

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ {OUT}")
