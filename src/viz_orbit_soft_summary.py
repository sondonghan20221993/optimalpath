"""
viz_orbit_soft_summary.py — Orbit + Soft NBV 전체 시나리오 요약 이미지

4-panel:
  (a) 방법 진화 타임라인 (테이블+화살표)
  (b) 최종 경로 top-down (orbit + 3 WP)
  (c) 커버리지 비교 bar (orbit vs orbit+softNBV, 3지표)
  (d) 파라미터 정당화 요약
"""
import sys, types, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P

ROOT = HERE.parent
META = sorted((ROOT / "real_test" / "meta").glob("*.json"))
OUT  = ROOT / "results" / "orbit_soft_summary.png"

def load_orbit_pos():
    pos = []
    for m in META:
        p = json.load(open(m))["camera"]["pose"]["position"]
        pos.append([p["x"], p["y"], p["z"]])
    return np.array(pos)

def main():
    cam_pos = load_orbit_pos()
    pts     = P._pts_raw
    target  = P.TARGET
    WPS     = np.array([
        [-32.563, -53.449, -3.38],
        [-33.095, -53.591, -3.38],
        [-33.643, -53.639, -3.38],
    ])

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    COL = {'bg': '#0d1117', 'ax': '#161b22', 'txt': '#e6edf3',
           'orbit': '#58a6ff', 'wp': '#f78166', 'tgt': '#ffa657',
           'soft': '#3fb950', 'hard': '#d29922', 'real': '#58a6ff',
           'grid': '#21262d'}

    # ── (a) 방법 진화 타임라인 ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(COL['ax']); ax_a.set_aspect('equal')
    ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10)
    ax_a.axis('off')
    ax_a.set_title('(a) Development History & Bug Fixes', color=COL['txt'], fontsize=12, pad=6)

    steps = [
        (5.0, 8.5, 'Circular Orbit Only\n(34 cameras)', '#30363d', COL['hard'],
         'hard: 95.0%  real: 99.1%'),
        (5.0, 6.2, 'Paper PB-NBV Supplement\n(tilt=35deg BUG)', '#30363d', '#f85149',
         '→ actual tilt was 35deg (gen_candidates_fixed_alt bug found)'),
        (5.0, 4.2, 'Fix tilt=45deg + F<0 Bug Fix\n(frontier_only=True)', '#30363d', '#f85149',
         '→ evaluate() frontier_only=True, U=F/dist now correct (dist-U corr: +0.85 → -0.14)'),
        (5.0, 2.2, 'Soft Lambert Regulation\n(alpha=1, TAU=cos70deg)', '#1a3a1a', COL['soft'],
         '3 WP · 7.92 m · real 100.0% ✓'),
    ]
    for i, (x, y, title, bg, outline, detail) in enumerate(steps):
        rect = mpatches.FancyBboxPatch((x-3.5, y-0.7), 7.0, 1.4,
                                        boxstyle="round,pad=0.1",
                                        facecolor=bg, edgecolor=outline, linewidth=1.5)
        ax_a.add_patch(rect)
        ax_a.text(x, y+0.15, title, ha='center', va='center',
                  color=COL['txt'], fontsize=9, fontweight='bold')
        ax_a.text(x, y-0.35, detail, ha='center', va='center',
                  color='#8b949e', fontsize=7.5)
        if i < len(steps)-1:
            ax_a.annotate('', xy=(x, steps[i+1][1]+0.7),
                          xytext=(x, y-0.7),
                          arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))

    # ── (b) 최종 경로 top-down ───────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(COL['ax'])
    ax_b.set_title('(b) Final Path Top-Down (Orbit + Soft NBV, 3 WP)', color=COL['txt'], fontsize=12, pad=6)

    ax_b.scatter(pts[:, 0], pts[:, 1], c='#6e7681', s=3, alpha=0.5, zorder=1)
    ax_b.scatter(*target[:2], marker='*', s=200, c=COL['tgt'], zorder=5, label='Target')

    # orbit ring
    ox, oy = cam_pos[:, 0], cam_pos[:, 1]
    ax_b.scatter(ox, oy, c=COL['orbit'], s=18, alpha=0.6, zorder=3)
    theta_ring = np.linspace(0, 2*np.pi, 200)
    r_orbit = np.linalg.norm(cam_pos[:, :2] - target[:2], axis=1).mean()
    ax_b.plot(target[0] + r_orbit*np.cos(theta_ring),
              target[1] + r_orbit*np.sin(theta_ring),
              '--', color=COL['orbit'], alpha=0.3, lw=1)

    # WPs
    ax_b.scatter(WPS[:, 0], WPS[:, 1], c=COL['wp'], s=120, marker='D',
                 zorder=5, label=f'Soft NBV WP ({len(WPS)} pts)')
    # path from last orbit cam → WPs
    start = cam_pos[-1, :2]
    ax_b.annotate('', xy=WPS[0, :2], xytext=start,
                  arrowprops=dict(arrowstyle='->', color=COL['wp'], lw=2))
    for i in range(len(WPS)-1):
        ax_b.annotate('', xy=WPS[i+1, :2], xytext=WPS[i, :2],
                      arrowprops=dict(arrowstyle='->', color=COL['wp'], lw=2))
    for i, wp in enumerate(WPS):
        az = math.degrees(math.atan2(wp[1]-target[1], wp[0]-target[0])) % 360
        ax_b.text(wp[0]-0.3, wp[1]+0.4, f'WP{i+1}\naz={az:.0f}°',
                  color=COL['wp'], fontsize=7.5, ha='center')

    ax_b.set_xlabel('X (m)', color=COL['txt'], fontsize=9)
    ax_b.set_ylabel('Y (m)', color=COL['txt'], fontsize=9)
    ax_b.tick_params(colors=COL['txt'], labelsize=8)
    for sp in ax_b.spines.values(): sp.set_color(COL['grid'])
    ax_b.grid(True, color=COL['grid'], alpha=0.4, lw=0.5)
    ax_b.set_aspect('equal')
    leg = ax_b.legend(fontsize=8, facecolor='#21262d', edgecolor=COL['grid'],
                      labelcolor=COL['txt'], loc='upper right')

    # z annotation
    ax_b.text(0.02, 0.02,
              f'z={-3.38}m (tilt=45deg, 3m above top)\nPath 7.92m',
              transform=ax_b.transAxes, color='#8b949e', fontsize=8,
              va='bottom', ha='left')

    # ── (c) 커버리지 비교 bar ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(COL['ax'])
    ax_c.set_title('(c) Coverage Comparison (3 Metrics)', color=COL['txt'], fontsize=12, pad=6)

    metrics = ['hard\n(theta<70deg, single frontal)', 'soft\n(Lambert cos^1, TAU=cos70deg)', 'real\n(raycast groundtruth)']
    orbit_vals   = [95.0, 97.8, 99.1]
    soft_nbv_vals = [95.0, 99.4, 100.0]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax_c.bar(x - w/2, orbit_vals, w, label='Orbit Only', color=COL['orbit'], alpha=0.7)
    bars2 = ax_c.bar(x + w/2, soft_nbv_vals, w, label='Orbit + Soft NBV (3WP)', color=COL['soft'], alpha=0.9)

    for bar, val in zip(bars1, orbit_vals):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                  f'{val:.1f}%', ha='center', va='bottom', color=COL['txt'], fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, soft_nbv_vals):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                  f'{val:.1f}%', ha='center', va='bottom', color=COL['txt'], fontsize=9, fontweight='bold')

    ax_c.set_ylim(88, 103)
    ax_c.set_xticks(x); ax_c.set_xticklabels(metrics, color=COL['txt'], fontsize=9)
    ax_c.set_ylabel('Coverage (%)', color=COL['txt'], fontsize=9)
    ax_c.text(0.02, 0.02, 'Honest dual reporting: soft covers multi-view fusion\nhard = single frontal guarantee (theta<70deg)',
              transform=ax_c.transAxes, color='#8b949e', fontsize=7.5, va='bottom')
    ax_c.tick_params(colors=COL['txt'], labelsize=8)
    for sp in ax_c.spines.values(): sp.set_color(COL['grid'])
    ax_c.grid(True, axis='y', color=COL['grid'], alpha=0.4, lw=0.5)
    ax_c.axhline(100, color='#f85149', lw=1, ls='--', alpha=0.5)

    # delta annotation
    deltas = [v2-v1 for v1,v2 in zip(orbit_vals, soft_nbv_vals)]
    for xi, d in zip(x, deltas):
        col = COL['soft'] if d > 0 else '#8b949e'
        sign = '+' if d >= 0 else ''
        ax_c.text(xi, 89.5, f'+{d:.1f}%p' if d > 0 else f'{d:.1f}%p', ha='center', color=col,
                  fontsize=9, fontweight='bold')

    leg2 = ax_c.legend(fontsize=9, facecolor='#21262d', edgecolor=COL['grid'],
                       labelcolor=COL['txt'], loc='lower right')

    # ── (d) 파라미터 정당화 ─────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(COL['ax'])
    ax_d.set_title('(d) Defense Points & Parameter Justification', color=COL['txt'], fontsize=12, pad=6)
    ax_d.axis('off')

    lines = [
        ('[ Method Selection ]', '#ffa657', 10, True),
        ('', COL['txt'], 9, False),
        ('  Point cloud + MVS/SfM reconstruction context', COL['txt'], 9, False),
        ('  -> Soft multi-view fusion is physically appropriate', '#8b949e', 8.5, False),
        ('  40/85 frontier voxels structurally unobservable', COL['txt'], 9, False),
        ('  from tilt=45deg (min incidence angle 71.3deg > 70deg)', '#8b949e', 8.5, False),
        ('', COL['txt'], 9, False),
        ('[ Parameters (NOT tuned to coverage) ]', '#ffa657', 10, True),
        ('', COL['txt'], 9, False),
        ('  alpha = 1.0   Lambert cosine (irradiance standard)', COL['soft'], 9, False),
        ('  TAU   = cos(70deg) = 0.342', COL['soft'], 9, False),
        ('         equiv. to hard theta<70deg single frontal view', '#8b949e', 8.5, False),
        ('', COL['txt'], 9, False),
        ('[ Bug Fixed: F<0 reversed U=F/dist preference ]', '#ffa657', 10, True),
        ('', COL['txt'], 9, False),
        ('  frontier_only=True  =>  F = Sigma(frontier*W) >= 0', COL['soft'], 9, False),
        ('  Before:  dist<->U corr = +0.853  (preferred FARTHER)', '#f85149', 8.5, False),
        ('  After:   dist<->U corr = -0.143  (prefers CLOSER) OK', COL['soft'], 8.5, False),
        ('', COL['txt'], 9, False),
        ('[ Honest Dual Reporting ]', '#ffa657', 10, True),
        ('  hard / soft / real(raycast) all reported separately', COL['txt'], 9, False),
    ]

    y = 9.5
    for text, color, size, bold in lines:
        if text == '':
            y -= 0.3
            continue
        ax_d.text(0.03, y/10, text, transform=ax_d.transAxes,
                  color=color, fontsize=size,
                  fontweight='bold' if bold else 'normal',
                  va='top', fontfamily='monospace' if text.startswith(' ') else 'sans-serif')
        y -= 0.42

    # ── 제목 ─────────────────────────────────────────────────────────────
    fig.suptitle('Orbit + Soft-Regulated PB-NBV (Lambert cos^1, TAU=cos70deg) — Final Result Summary',
                 color=COL['txt'], fontsize=14, fontweight='bold')

    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=COL['bg'])
    print(f'✓ 저장: {OUT}')

if __name__ == '__main__':
    main()
