"""
viz_nocam_constrained.py — orbit 없는 PB-NBV + overlap 제약 결과 시각화
좌  : top-down 경로 (overlap 라벨)
우상: step별 overlap 막대 (ATE 안전)
우하: 결과 + 제약 서술
"""
import sys, types, json, math
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
JS   = ROOT / "results" / "pbnbv_nocam_constrained_path.json"
OUT  = ROOT / "results" / "nocam_constrained_path.png"


def main():
    d   = json.load(open(JS))
    pts = P._pts_raw
    t   = P.TARGET
    wps = np.array([w["pos"] for w in d["waypoints"]])
    azs = [w["azimuth_deg"] for w in d["waypoints"]]
    ovl = [w["overlap_with_prev"] for w in d["waypoints"]]
    start = np.array(d["start_pos"])

    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d', 'bg': '#0d1117',
         'pc':  '#6e7681', 'path': '#3fb950', 'start': '#ffa657',
         'tgt': '#f778ba', 'warn': '#f85149', 'mut': '#8b949e'}

    fig = plt.figure(figsize=(19, 9))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.20,
                           width_ratios=[1.4, 1], height_ratios=[1, 1],
                           left=0.05, right=0.97, top=0.89, bottom=0.08)

    # ── 좌: 경로 ──
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) No-orbit PB-NBV + overlap constraint — FBX GT',
                 color=C['txt'], fontsize=13, pad=8)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=12, alpha=0.8, zorder=2,
               label='GT slab (FBX)')
    ax.scatter(*t[:2], marker='*', s=400, c=C['tgt'], zorder=8, label='target')
    ax.scatter(*start[:2], marker='s', s=180, c=C['start'], zorder=6,
               edgecolors='white', linewidths=0.8, label='start (fixed)')

    ax.annotate('', xy=wps[0, :2], xytext=start[:2],
                arrowprops=dict(arrowstyle='->', color=C['start'], lw=1.8,
                                ls='--', alpha=0.8))
    ax.text((start[0]+wps[0,0])/2, (start[1]+wps[0,1])/2, 'handoff',
            color=C['start'], fontsize=8, style='italic', ha='center')
    for i in range(len(wps)-1):
        ax.annotate('', xy=wps[i+1, :2], xytext=wps[i, :2],
                    arrowprops=dict(arrowstyle='->', color=C['path'], lw=2.4))
        mx, my = (wps[i, 0]+wps[i+1, 0])/2, (wps[i, 1]+wps[i+1, 1])/2
        ax.text(mx, my, f'ovl {ovl[i+1]:.2f}', color='#9be9a8', fontsize=8.5,
                ha='center', va='center', zorder=9,
                bbox=dict(boxstyle='round,pad=0.15', fc='#13261a', ec='none'))
    ax.scatter(wps[:, 0], wps[:, 1], c=C['path'], s=200, marker='D', zorder=7,
               edgecolors='white', linewidths=0.8, label='NBV waypoint')
    for i, w in enumerate(wps):
        ax.text(w[0], w[1]+0.18, f'WP{i+1}', color='white', fontsize=10,
                ha='center', va='bottom', fontweight='bold', zorder=10)
        ax.text(w[0], w[1]-0.22, f'az={round(azs[i])}° z={w[2]:.1f}',
                color=C['path'], fontsize=8.5, ha='center', va='top')

    ax.set_xlabel('X (m)', color=C['txt'], fontsize=10)
    ax.set_ylabel('Y (m)', color=C['txt'], fontsize=10)
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── 우상: overlap 막대 ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C['ax'])
    ax2.set_title('(b) Step overlap vs constraint (ATE safety)',
                  color=C['txt'], fontsize=12, pad=6)
    steps = list(range(1, len(ovl)+1))
    bars = ax2.bar(steps, ovl, color=C['path'], alpha=0.85, width=0.6)
    for b, v in zip(bars, ovl):
        ax2.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}',
                 ha='center', va='bottom', color=C['txt'], fontsize=10, fontweight='bold')
    ax2.axhline(d['tau_overlap'], color=C['warn'], ls='--', lw=1.5,
                label=f"constraint = {d['tau_overlap']}")
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel('WP #', color=C['txt'], fontsize=9)
    ax2.set_ylabel('overlap w/ prev', color=C['txt'], fontsize=9)
    ax2.set_xticks(steps)
    ax2.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, axis='y', color=C['grid'], alpha=0.3, lw=0.5)
    ax2.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
               labelcolor=C['txt'], loc='lower right')

    # ── 우하: 결과 + 제약 서술 ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(C['ax'])
    ax3.axis('off')
    lines = [
        ('[ CONSTRAINTS ]', C['tgt'], 12.5, True),
        ('tilt = 45 deg (fixed)', C['txt'], 9.5, False),
        ('MAX_DIST = 8 m, FOV = 89.9 deg', C['txt'], 9.5, False),
        ('Lambert cos^1, TAU = cos70 = 0.342', C['txt'], 9.5, False),
        (f'overlap >= {d["tau_overlap"]}  <-- ATE safety', C['path'], 9.5, True),
        ('U = soft_gain / dist^1.0', C['txt'], 9.5, False),
        ('', C['txt'], 5, False),
        ('[ RESULT ]', C['tgt'], 12.5, True),
        (f'WP {d["n_waypoints"]}  |  path {d["path_length_m"]:.2f} m', C['txt'], 10.5, False),
        (f'overlap min {d["overlap_min"]} / mean {d["overlap_mean"]}', C['txt'], 10, False),
        (f'max az jump {d["max_adjacent_daz_deg"]} deg '
         f'(was 250 w/o constraint)', C['mut'], 9.5, False),
        (f'real coverage {100*d["real_cov_final"]:.1f}%', C['path'], 11, True),
        ('', C['txt'], 5, False),
        ('[ vs NO-CONSTRAINT v2 ]', C['warn'], 12, True),
        ('v2          : 11.79 m, overlap unbounded', C['mut'], 9.5, False),
        (f'constrained : {d["path_length_m"]:.2f} m, overlap >= {d["overlap_min"]}',
         C['path'], 9.5, True),
        ('-> tracking-safe, still info-gain driven', C['txt'], 9.5, False),
    ]
    y = 0.98
    for text, color, size, bold in lines:
        if text == '':
            y -= 0.028; continue
        ax3.text(0.03, y, text, transform=ax3.transAxes, color=color,
                 fontsize=size, fontweight='bold' if bold else 'normal',
                 va='top', family='monospace')
        y -= 0.057

    fig.suptitle('No-orbit Constrained PB-NBV on FBX GT — overlap-filtered, ATE-safe, 100% real coverage',
                 color=C['txt'], fontsize=14, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')


if __name__ == '__main__':
    main()
