"""viz_compare5.py — 5-Case 비교 (A1, A2, 3, B, C)"""
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
OUT  = ROOT / "results" / "compare5_cases.png"

BG, DARK = "#0f0f1a", "#1a1a2e"
C_PT  = "#2a3a55"
C_TGT = "#ff6d00"
# A1=gold, A2=cyan, 3=green, B=magenta, C=orange
COLORS = ["#ffd600", "#40c4ff", "#69f0ae", "#ff4081", "#ff9100"]
START  = np.array([-27.06, -50.51, -6.13])

pts    = np.load(NPZ)["points"].astype(float)
target = pts.mean(axis=0)
A.RAYCAST_OCCLUSION = True


def load_wps(path, key):
    j = json.load(open(path))
    return [np.array(p["pos"]) for p in j[key]]


cases = [
    ("A-1: Paper NBV",        ROOT/"results/pbnbv_paper/pbnbv_paper.json",   "path"),
    ("A-2: Hybrid+Greedy",    ROOT/"results/pbnbv_hybrid/pbnbv_hybrid.json", "waypoints"),
    ("3: Hard Raycast",        ROOT/"results/pbnbv_nocam_path.json",          "waypoints"),
    ("B: Soft 0.5^r/point",   ROOT/"results/pbnbv_softrank_path.json",       "waypoints"),
    ("C: Ellipsoid 0.5^r",    ROOT/"results/pbnbv_ellipsoid_path.json",      "waypoints"),
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
    return curve, 100*(len(pts)-int(live.sum()))/len(pts)


data = []
for name, p, key in cases:
    wps   = load_wps(p, key)
    curve, cov = point_cov(wps)
    pl    = path_len(wps)
    data.append({"name": name, "wps": wps, "curve": curve, "cov": cov, "plen": pl})

# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 16), facecolor=BG)
fig.suptitle(
    f"PB-NBV 5-Case Comparison  |  {len(pts)}pts (SOR-cleaned)  tilt=45deg  MAX_DIST=8m  no-camera  raycast-eval",
    color="white", fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 5, fig,
                       hspace=0.38, wspace=0.22,
                       left=0.03, right=0.98, top=0.93, bottom=0.04)

np.random.seed(0)
sub = np.random.choice(len(pts), min(3000, len(pts)), replace=False)

# ── row 0: top-down paths ────────────────────────────────────────────────────
for i, d in enumerate(data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(d["name"], color=COLORS[i], fontsize=10, fontweight="bold")

    ax.scatter(pts[sub,0], pts[sub,1], c=C_PT, s=2, alpha=0.4)
    w = np.array(d["wps"])
    ax.plot(w[:,0], w[:,1], "-", color=COLORS[i], lw=2, zorder=4, alpha=0.85)
    for k, wp in enumerate(w):
        ax.scatter(wp[0], wp[1], c=COLORS[i], s=80, zorder=5,
                   edgecolor="white", linewidth=0.5)
        ax.text(wp[0]+0.05, wp[1]+0.05, f"{k+1}",
                color="white", fontsize=8, zorder=6)
    ax.scatter(START[0], START[1], marker="s", c="white", s=55, zorder=5)
    ax.plot(target[0], target[1], "*", color=C_TGT, ms=14, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#aaa", fontsize=7)
    ax.set_ylabel("Y (m)", color="#aaa", fontsize=7)
    ax.text(0.03, 0.97,
            f"{len(w)} WP\ncov {d['cov']:.1f}%\n{d['plen']:.1f}m",
            transform=ax.transAxes, color=COLORS[i],
            fontsize=9, va="top", fontweight="bold")

# ── row 1 left: coverage curve ───────────────────────────────────────────────
ax_cov = fig.add_subplot(gs[1, 0:3])
ax_cov.set_facecolor(DARK)
ax_cov.tick_params(colors="white", labelsize=8)
for sp in ax_cov.spines.values(): sp.set_edgecolor("#444")
ax_cov.set_title("Coverage vs # Waypoints (real-point + raycast)", color="white", fontsize=10)
for i, d in enumerate(data):
    pct = [100*c/len(pts) for c in d["curve"]]
    ax_cov.plot(range(len(pct)), pct, "-o", color=COLORS[i], lw=2, ms=5,
                label=d["name"])
ax_cov.axhline(100, color="#555", ls="--", lw=1)
ax_cov.set_xlabel("# Waypoints", color="#aaa", fontsize=8)
ax_cov.set_ylabel("Coverage (%)", color="#aaa", fontsize=8)
ax_cov.legend(fontsize=8, facecolor=DARK, edgecolor="#444", labelcolor="white",
              loc="lower right")
ax_cov.set_ylim(0, 108)

# ── row 1 right: bar chart ────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 3:5])
ax_bar.set_facecolor(DARK)
ax_bar.tick_params(colors="white", labelsize=8)
for sp in ax_bar.spines.values(): sp.set_edgecolor("#444")
ax_bar.set_title("Summary Metrics", color="white", fontsize=10)

metrics = ["WP count", "Coverage %", "Path (m)"]
x = np.arange(len(metrics))
width = 0.16
for i, d in enumerate(data):
    vals = [len(d["wps"]), d["cov"], d["plen"]]
    bars = ax_bar.bar(x + (i-2)*width, vals, width,
                      color=COLORS[i], label=d["name"], edgecolor=BG)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{v:.0f}" if v>10 else f"{v:.1f}",
                    ha="center", va="bottom", color="white", fontsize=6)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics, fontsize=8)
ax_bar.legend(fontsize=7, facecolor=DARK, edgecolor="#444", labelcolor="white",
              loc="upper left")

# ── row 2: text analysis ──────────────────────────────────────────────────────
ax_txt = fig.add_subplot(gs[2, :])
ax_txt.set_facecolor(DARK)
ax_txt.axis("off")

d_a1, d_a2, d_3, d_b, d_c = data

def row(txt, col="#cccccc", sz=9, bold=False):
    return (txt, col, sz, "bold" if bold else "normal")

lines = [
    row("5-Case Analysis  |  Scoring method  /  Occlusion model  /  Key result", "white", 10, True),
    row(""),
    row(f"  Case A-1  Paper NBV       : {len(d_a1['wps'])} WP  {d_a1['cov']:.1f}%  {d_a1['plen']:.1f}m"
        "   Ellipsoid F-score, frontier/occupied GMM, 0.5^r per cluster. No hard raycast.",
        COLORS[0]),
    row(f"  Case A-2  Hybrid+Greedy   : {len(d_a2['wps'])} WP  {d_a2['cov']:.1f}%  {d_a2['plen']:.1f}m"
        f"   Same 3 viewpoints as A-1, Greedy reorder saves {d_a1['plen']-d_a2['plen']:.1f}m.",
        COLORS[1]),
    row(f"  Case 3    Hard Raycast    : {len(d_3['wps'])} WP  {d_3['cov']:.1f}%  {d_3['plen']:.1f}m"
        "   Hard occlusion (vis=1/0). Point-level IG count. Circular orbit.",
        COLORS[2]),
    row(f"  Case B    Soft 0.5^r/pt   : {len(d_b['wps'])} WP  {d_b['cov']:.1f}%  {d_b['plen']:.1f}m"
        "   Paper 0.5^r extended per-point within direction bin. Soft occlusion.",
        COLORS[3]),
    row(f"  Case C    Ellipsoid 0.5^r : {len(d_c['wps'])} WP  {d_c['cov']:.1f}%  {d_c['plen']:.1f}m"
        "   Hard raycast + KMeans cluster -> 0.5^r per cluster depth rank.",
        COLORS[4]),
    row(""),
    row(f"  KEY: After SOR cleaning (real object 1.2x1.1x0.36m flat slab), ALL 5 cases reach 100% coverage.",
        "white", 10, True),
    row(f"       fewest WP : Case 3 (Hard Raycast) = {len(d_3['wps'])} WP, {d_3['plen']:.1f}m"
        f"      shortest path : Case B (Soft 0.5^r) = {d_b['plen']:.1f}m ({len(d_b['wps'])} WP)",
        "white", 10, True),
    row(f"  WHY changed: prev runs anchored candidate altitude to a NOISE point (obj_top -3.85m vs real -0.38m),"
        f" placing cameras ~3.5m too high. Cleaned data -> correct low/close viewpoints -> far fewer WP needed.",
        "#aaaaaa"),
    row(f"  Case 3 (raycast) needs only {len(d_3['wps'])} WP: small flat object has little self-occlusion when"
        f" viewed from correct altitude. Case C (ellipsoid) spreads across 4 z-levels -> longest path ({d_c['plen']:.1f}m).",
        "#aaaaaa"),
]

y = 0.97
dy = 1.0 / (len(lines) + 1)
for txt, col, sz, w in lines:
    ax_txt.text(0.01, y, txt, color=col, fontsize=sz, va="top",
                fontweight=w, transform=ax_txt.transAxes)
    y -= dy

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ {OUT}")
