"""viz_compare4.py — A/B 포함 4-Case 비교 시각화"""
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
import matplotlib.gridspec as gridspec

ROOT = HERE.parent
NPZ  = ROOT / "real_test" / "real_test_pts_normals.npz"
OUT  = ROOT / "results" / "compare4_cases.png"

BG, DARK = "#0f0f1a", "#1a1a2e"
C_PT  = "#2a3a55"
C_TGT = "#ff6d00"
# Case1=yellow, Case2=cyan, Case3=green, CaseB=magenta
COLORS = ["#ffd600", "#40c4ff", "#69f0ae", "#ff4081"]
START  = np.array([-27.06, -50.51, -6.13])

pts    = np.load(NPZ)["points"].astype(float)
target = pts.mean(axis=0)
A.RAYCAST_OCCLUSION = True


def load_wps(path, key):
    j = json.load(open(path))
    return [np.array(p["pos"]) for p in j[key]]


cases = [
    ("Case A-1: Paper NBV sequential",
     ROOT / "results/pbnbv_paper/pbnbv_paper.json", "path"),
    ("Case A-2: Hybrid (paper-F + Greedy)",
     ROOT / "results/pbnbv_hybrid/pbnbv_hybrid.json", "waypoints"),
    ("Case 3: Hard Raycast + PB-NBV(2)",
     ROOT / "results/pbnbv_nocam_path.json", "waypoints"),
    ("Case B: Soft 0.5^r Depth-Rank",
     ROOT / "results/pbnbv_softrank_path.json", "waypoints"),
]


def path_len(seq):
    cur, tot = START.copy(), 0.0
    for p in seq:
        tot += np.linalg.norm(np.asarray(p) - cur); cur = np.asarray(p)
    return tot


def point_cov(seq):
    live = np.ones(len(pts), dtype=bool)
    curve = [0]
    for wp in seq:
        _, vis = A.information_gain(np.array(wp), target, pts, live, 89.9, A.MAX_DIST)
        live &= ~vis
        curve.append(len(pts) - int(live.sum()))
    return curve, 100 * (len(pts) - int(live.sum())) / len(pts)


data = []
for name, p, key in cases:
    wps = load_wps(p, key)
    curve, cov = point_cov(wps)
    pl = path_len(wps)
    data.append({"name": name, "wps": wps, "curve": curve, "cov": cov, "plen": pl})

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 14), facecolor=BG)
fig.suptitle(
    "PB-NBV 4-Case Comparison  |  2438pts · tilt=45° · MAX_DIST=8m · no-camera · raycast eval",
    color="white", fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, fig,
                       hspace=0.38, wspace=0.25,
                       left=0.04, right=0.97, top=0.93, bottom=0.05)

np.random.seed(0)
sub = np.random.choice(len(pts), min(3000, len(pts)), replace=False)

# ─ 상단: 4개 경로 top-down ───────────────────────────────────────────────────
for i, d in enumerate(data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(d["name"], color="white", fontsize=9, fontweight="bold")

    ax.scatter(pts[sub, 0], pts[sub, 1], c=C_PT, s=2, alpha=0.4)
    w = np.array(d["wps"])
    ax.plot(w[:, 0], w[:, 1], "-", color=COLORS[i], lw=2, zorder=4, alpha=0.85)
    for k, wp in enumerate(w):
        ax.scatter(wp[0], wp[1], c=COLORS[i], s=80, zorder=5,
                   edgecolor="white", linewidth=0.5)
        ax.text(wp[0] + 0.05, wp[1] + 0.05, f"{k+1}",
                color="white", fontsize=8, zorder=6)
    ax.scatter(START[0], START[1], marker="s", c="white", s=55, zorder=5)
    ax.text(START[0], START[1] - 0.35, "start",
            color="white", fontsize=7, ha="center")
    ax.plot(target[0], target[1], "*", color=C_TGT, ms=14, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax.set_ylabel("Y (m)", color="#aaa", fontsize=7)
    ax.text(0.03, 0.97,
            f"{len(w)} WP\ncov {d['cov']:.1f}%\npath {d['plen']:.1f}m",
            transform=ax.transAxes, color=COLORS[i],
            fontsize=9, va="top", fontweight="bold")

# ─ 중단 좌: Coverage 곡선 ────────────────────────────────────────────────────
ax_cov = fig.add_subplot(gs[1, 0:2])
ax_cov.set_facecolor(DARK)
ax_cov.tick_params(colors="white", labelsize=8)
for sp in ax_cov.spines.values(): sp.set_edgecolor("#444")
ax_cov.set_title("Coverage vs # Waypoints (real-point + raycast)", color="white", fontsize=10)
for i, d in enumerate(data):
    pct = [100 * c / len(pts) for c in d["curve"]]
    ax_cov.plot(range(len(pct)), pct, "-o", color=COLORS[i], lw=2, ms=5,
                label=d["name"].replace("\n", " "))
ax_cov.axhline(100, color="#555", ls="--", lw=1)
ax_cov.set_xlabel("# Waypoints", color="#aaa", fontsize=8)
ax_cov.set_ylabel("Coverage (%)", color="#aaa", fontsize=8)
ax_cov.legend(fontsize=8, facecolor=DARK, edgecolor="#444", labelcolor="white",
              loc="lower right")
ax_cov.set_ylim(0, 108)

# ─ 중단 우: Summary bar chart ────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 2:4])
ax_bar.set_facecolor(DARK)
ax_bar.tick_params(colors="white", labelsize=8)
for sp in ax_bar.spines.values(): sp.set_edgecolor("#444")
ax_bar.set_title("Summary Metrics", color="white", fontsize=10)

labels = ["WP count", "Coverage %", "Path (m)"]
x = np.arange(len(labels))
width = 0.20
for i, d in enumerate(data):
    vals = [len(d["wps"]), d["cov"], d["plen"]]
    bars = ax_bar.bar(x + (i - 1.5) * width, vals, width,
                      color=COLORS[i], label=d["name"].replace("\n", " "), edgecolor=BG)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.0f}" if v > 10 else f"{v:.1f}",
                    ha="center", va="bottom", color="white", fontsize=7)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=8)
ax_bar.legend(fontsize=7, facecolor=DARK, edgecolor="#444", labelcolor="white")

# ─ 하단: 분석 텍스트 ─────────────────────────────────────────────────────────
ax_txt = fig.add_subplot(gs[2, :])
ax_txt.set_facecolor(DARK)
ax_txt.axis("off")
for sp in ax_txt.spines.values(): sp.set_edgecolor("#444")

cA1, cA2, c3, cB = data[0], data[1], data[2], data[3]
lines = [
    ("── Case Analysis Summary ──────────────────────────────────────────────────────────────────────────", "white", 9, "normal"),
    (f"  Case A-1 (Paper NBV sequential):  {len(cA1['wps'])} WP | cov {cA1['cov']:.1f}% | path {cA1['plen']:.1f}m"
     "  ->  Paper-faithful. 0.5^r ellipsoid weight, frontier-occupied split, GMM clustering.",
     COLORS[0], 9, "normal"),
    (f"  Case A-2 (Hybrid paper-F + Greedy):  {len(cA2['wps'])} WP | cov {cA2['cov']:.1f}% | path {cA2['plen']:.1f}m"
     f"  ->  Same 3 viewpoints as A-1, Greedy reorder saves {cA1['plen']-cA2['plen']:.1f}m.",
     COLORS[1], 9, "normal"),
    (f"  Case 3  (Hard Raycast + PB-NBV(2)):  {len(c3['wps'])} WP | cov {c3['cov']:.1f}% | path {c3['plen']:.1f}m"
     "  ->  Hard raycast occlusion (vis=1/0). Point-level IG. Circular path, 100% coverage.",
     COLORS[2], 9, "normal"),
    (f"  Case B  (Soft 0.5^r Depth-Rank):   {len(cB['wps'])} WP | cov {cB['cov']:.1f}% | path {cB['plen']:.1f}m"
     "  ->  Paper 0.5^r extended to per-point level. Occluded pts contribute 0.5,0.25... -> more WPs.",
     COLORS[3], 9, "normal"),
    ("", "white", 8, "normal"),
    ("── Key Comparisons ───────────────────────────────────────────────────────────────────────────────", "white", 9, "normal"),
    (f"  A-1 vs A-2: Same 3 viewpoints. Greedy reorder -> {cA1['plen']-cA2['plen']:.1f}m shorter path. Coverage identical. "
     "Validates 2-stage (select + order) design.",
     "#cccccc", 9, "normal"),
    (f"  A   vs 3 : Raycast raises coverage 90.9% -> 100%. WP: 3 -> 7. Precise self-occlusion handling.",
     "#cccccc", 9, "normal"),
    (f"  3   vs B : Soft 0.5^r: WP 7 -> 14, path {c3['plen']:.1f}m -> {cB['plen']:.1f}m (+{cB['plen']-c3['plen']:.1f}m). "
     "Soft weight keeps adding WPs for partially-occluded regions.",
     "#cccccc", 9, "normal"),
    ("", "white", 8, "normal"),
    (f"  CONCLUSION: Case 3 best balances coverage(100%), path efficiency, and circular shape. "
     f"Case B inherits paper's 0.5^r concept but over-generates WPs ({len(cB['wps'])} vs {len(c3['wps'])}). "
     "A-1/2 proves paper faithfulness but leaves concave regions uncovered.",
     "white", 10, "bold"),
]

y = 0.97
dy = 1.0 / (len(lines) + 1)
for txt, col, sz, w in lines:
    ax_txt.text(0.01, y, txt, color=col, fontsize=sz, va="top",
                fontweight=w, transform=ax_txt.transAxes, wrap=False)
    y -= dy

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ {OUT}")
