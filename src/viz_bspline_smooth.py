"""
viz_bspline_smooth.py — 2층 구조 시각화: PB-NBV waypoint(불변) + B-spline 평활 궤적
좌  : 직선 지그재그 경로 vs B-spline 매끄러운 궤적 (같은 WP)
우  : 2층 구조 설명 + 수치
"""
import sys, types, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P

ROOT = HERE.parent
JS   = ROOT / "results" / "bspline_smoothed_path.json"
OUT  = ROOT / "results" / "bspline_smoothed_path.png"


def main():
    d      = json.load(open(JS))
    pts    = P._pts_raw
    t      = P.TARGET
    ctrl   = np.array(d["ctrl_points"])       # start + WP들
    smooth = np.array(d["smooth_trajectory"])
    start  = ctrl[0]
    wps    = ctrl[1:]

    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d', 'bg': '#0d1117',
         'pc':  '#6e7681', 'raw': '#f85149', 'smooth': '#3fb950',
         'start': '#ffa657', 'tgt': '#f778ba', 'mut': '#8b949e'}

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.20,
                           width_ratios=[1.55, 1],
                           left=0.05, right=0.97, top=0.88, bottom=0.08)

    # ── 좌: 경로 비교 ──
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) Same NBV waypoints — straight (zigzag) vs B-spline (smooth)',
                 color=C['txt'], fontsize=13, pad=8)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=12, alpha=0.8, zorder=2,
               label='GT slab (FBX)')
    ax.scatter(*t[:2], marker='*', s=400, c=C['tgt'], zorder=9, label='target')

    # 직선 지그재그 (층1 원본)
    ax.plot(ctrl[:, 0], ctrl[:, 1], '--', color=C['raw'], lw=1.6, alpha=0.7,
            zorder=3, label=f'straight ({d["raw_path_length_m"]}m, turn≤{d["raw_max_turn_deg"]:.0f}°)')
    # B-spline 매끄러운 궤적 (층2)
    ax.plot(smooth[:, 0], smooth[:, 1], '-', color=C['smooth'], lw=2.6,
            zorder=5, label=f'B-spline ({d["smooth_path_length_m"]}m, C2 continuous)')

    ax.scatter(*start[:2], marker='s', s=180, c=C['start'], zorder=7,
               edgecolors='white', linewidths=0.8, label='start')
    ax.scatter(wps[:, 0], wps[:, 1], c=C['smooth'], s=210, marker='D', zorder=8,
               edgecolors='white', linewidths=0.9, label='NBV waypoint (fixed)')
    for i, w in enumerate(wps):
        ax.text(w[0], w[1]+0.18, f'WP{i+1}', color='white', fontsize=11,
                ha='center', va='bottom', fontweight='bold', zorder=10)

    ax.set_xlabel('X (m)', color=C['txt'], fontsize=10)
    ax.set_ylabel('Y (m)', color=C['txt'], fontsize=10)
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── 우: 2층 구조 설명 ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C['ax'])
    ax2.axis('off')
    lines = [
        ('[ 2-LAYER STRUCTURE ]', C['tgt'], 12.5, True),
        ('Layer1  viewpoint selection', C['txt'], 10.5, True),
        ('   = PB-NBV info-gain (UNTOUCHED)', C['smooth'], 9.5, False),
        ('   WP positions fixed -> no attack point', C['mut'], 9, False),
        ('Layer2  trajectory smoothing', C['txt'], 10.5, True),
        ('   = cubic B-spline through WPs', C['smooth'], 9.5, False),
        ('   safety/ATE, not science', C['mut'], 9, False),
        ('', C['txt'], 5, False),
        ('[ EFFECT ]', C['tgt'], 12.5, True),
        (f'turn angle : {d["raw_max_turn_deg"]:.0f}° -> ~2°/step', C['txt'], 10, False),
        (f'path len   : {d["raw_path_length_m"]}m -> {d["smooth_path_length_m"]}m', C['txt'], 10, False),
        (f'WP offpath : {d["wp_offpath_max_m"]:.3f}m (~0, fixed)', C['smooth'], 10, True),
        ('C2 continuous -> dynamically feasible', C['txt'], 9.5, False),
        ('', C['txt'], 5, False),
        ('[ LITERATURE ]', C['mut'], 12, True),
        ('Vasquez-Gomez: distance term in NBV', C['mut'], 9, False),
        ('  utility reduces back-and-forth motion', C['mut'], 8.5, False),
        ('Smooth CPP (MPC): B-spline continuity', C['mut'], 9, False),
        ('  -> smoothness + dynamic feasibility', C['mut'], 8.5, False),
        ('', C['txt'], 5, False),
        ('Coverage 100% unchanged (same WPs).', C['smooth'], 10, True),
    ]
    y = 0.98
    for text, color, size, bold in lines:
        if text == '':
            y -= 0.025; continue
        ax2.text(0.03, y, text, transform=ax2.transAxes, color=color,
                 fontsize=size, fontweight='bold' if bold else 'normal',
                 va='top', family='monospace')
        y -= 0.048

    fig.suptitle('PB-NBV (Layer1, fixed WPs) + B-spline Trajectory Smoothing (Layer2) — '
                 'safe flight, original intent intact',
                 color=C['txt'], fontsize=13.5, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')


if __name__ == '__main__':
    main()
