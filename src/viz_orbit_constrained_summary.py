"""
viz_orbit_constrained_summary.py — 궤도끝 constrained PB-NBV 결과 요약 (설명용 1장)

좌  : 경로 top-down (원형 궤도 + 끝점 handoff + constrained 보완)
우상: 보완 step별 overlap (ATE 친화)
우하: 결과 수치 + 방어 포인트 + 정직한 약점 (텍스트)
"""
import sys, types, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P

ROOT = HERE.parent
JS   = ROOT / "results" / "pbnbv_orbit_constrained_path.json"
OUT  = ROOT / "results" / "orbit_constrained_summary.png"


def main():
    d   = json.load(open(JS))
    pts = P._pts_raw
    t   = P.TARGET
    orbit = np.array(d["orbit_positions"])
    wps = np.array([w["pos"] for w in d["waypoints"]])
    ovl = [w["overlap_with_prev"] for w in d["waypoints"]]
    orbit_end = orbit[-1]

    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d', 'bg': '#0d1117',
         'pc': '#6e7681', 'orbit': '#58a6ff', 'path': '#3fb950',
         'end': '#ffa657', 'tgt': '#f778ba', 'warn': '#f85149', 'mut': '#8b949e'}

    fig = plt.figure(figsize=(19, 10))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.20,
                           width_ratios=[1.35, 1], height_ratios=[1, 1],
                           left=0.05, right=0.975, top=0.90, bottom=0.06)

    # ── 좌: 경로 (두 행 차지) ──
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) Orbit (34 cams) → continue from orbit-end → constrained NBV supplement',
                 color=C['txt'], fontsize=12.5, pad=8)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=3, alpha=0.4, zorder=1)
    ax.scatter(*t[:2], marker='*', s=320, c=C['tgt'], zorder=8, label='target')
    ax.scatter(orbit[:, 0], orbit[:, 1], c=C['orbit'], s=22, alpha=0.6, zorder=2,
               label=f'orbit ({len(orbit)} cams)')
    ro = np.linalg.norm(orbit[:, :2]-t[:2], axis=1).mean()
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(t[0]+ro*np.cos(th), t[1]+ro*np.sin(th), '--', color=C['orbit'],
            alpha=0.3, lw=1, zorder=2)
    ax.scatter(*orbit_end[:2], marker='s', s=160, c=C['end'], zorder=7,
               edgecolors='white', linewidths=0.8, label='orbit end (start)')
    ax.annotate('orbit end az360°', orbit_end[:2], textcoords='offset points',
                xytext=(8, 6), color=C['end'], fontsize=8.5, fontweight='bold')
    ax.annotate('', xy=wps[0, :2], xytext=orbit_end[:2],
                arrowprops=dict(arrowstyle='->', color=C['end'], lw=1.8, ls='--', alpha=0.8))
    ax.text((orbit_end[0]+wps[0, 0])/2, (orbit_end[1]+wps[0, 1])/2, 'handoff',
            color=C['end'], fontsize=8, style='italic', ha='center')
    for i in range(len(wps)-1):
        ax.annotate('', xy=wps[i+1, :2], xytext=wps[i, :2],
                    arrowprops=dict(arrowstyle='->', color=C['path'], lw=2.2))
        mx, my = (wps[i, 0]+wps[i+1, 0])/2, (wps[i, 1]+wps[i+1, 1])/2
        ax.text(mx, my, f'{ovl[i+1]:.2f}', color='#9be9a8', fontsize=7.5,
                ha='center', va='center', zorder=9)
    ax.scatter(wps[:, 0], wps[:, 1], c=C['path'], s=130, marker='D', zorder=6,
               edgecolors='white', linewidths=0.6, label='supplement WP')
    for i, w in enumerate(wps):
        ax.text(w[0], w[1]+0.3, f'{i+1}', color='white', fontsize=9,
                ha='center', va='center', fontweight='bold', zorder=10)
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=8.5, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── 우상: overlap ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C['ax'])
    ax2.set_title('(b) Supplement Overlap per Step (ATE-friendliness)',
                  color=C['txt'], fontsize=12, pad=6)
    steps = list(range(1, len(ovl)+1))
    bars = ax2.bar(steps, ovl, color=C['path'], alpha=0.85, width=0.65)
    for b, v in zip(bars, ovl):
        ax2.text(b.get_x()+b.get_width()/2, v+0.012, f'{v:.2f}',
                 ha='center', va='bottom', color=C['txt'], fontsize=8.5)
    ax2.axhline(d['tau_overlap'], color=C['warn'], ls='--', lw=1.5,
                label=f"constraint={d['tau_overlap']}")
    ax2.set_ylim(0, 1.14)
    ax2.set_xlabel('Supplement WP #', color=C['txt'], fontsize=9)
    ax2.set_ylabel('overlap w/ prev', color=C['txt'], fontsize=9)
    ax2.set_xticks(steps)
    ax2.tick_params(colors=C['txt'], labelsize=8.5)
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, axis='y', color=C['grid'], alpha=0.3, lw=0.5)
    ax2.legend(fontsize=8.5, facecolor='#21262d', edgecolor=C['grid'],
               labelcolor=C['txt'], loc='lower right')

    # ── 우하: 결과 + 방어 포인트 텍스트 ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(C['ax'])
    ax3.axis('off')
    for sp in ax3.spines.values(): sp.set_color(C['grid'])
    lines = [
        ('[ RESULT ]', C['tgt'], 12, True),
        (f"Orbit 34 cams:  soft {100*d['soft_cov_orbit']:.1f}% / real {100*d['real_cov_orbit']:.1f}%",
         C['txt'], 10.5, False),
        (f"+ Constrained NBV:  {d['n_waypoints']} WP, {d['path_length_m']} m, overlap>={d['tau_overlap']}",
         C['txt'], 10.5, False),
        (f"=> soft 100% / real raycast {100*d['real_cov_final']:.1f}%  (0 pts left)",
         C['path'], 10.5, True),
        ('', C['txt'], 6, False),
        ('[ DEFENSE POINTS ]', C['tgt'], 12, True),
        ('OK  Starts from orbit end (cam_pos[-1]) — continuous flight,',
         C['path'], 10, False),
        ('      no arbitrary restart point', C['mut'], 9, False),
        ('OK  Tracking-safe — supplement overlap 0.92-1.0 (low ATE)',
         C['path'], 10, False),
        ('OK  Still PB-NBV — viewpoints by info-gain (gain/dist);',
         C['path'], 10, False),
        ('      overlap is only a filter, shape NOT forced -> no attack point',
         C['mut'], 9, False),
        ('', C['txt'], 6, False),
        ('[ HONEST LIMITATION ]', C['warn'], 12, True),
        ('!   Orbit last camera is beyond MAX_DIST (sees 0 voxels)',
         C['warn'], 10, False),
        ('      -> first re-entry = handoff (constraint waived).',
         C['txt'], 9.5, False),
        ('      Orbit design leaves tracking range at its end;',
         C['mut'], 9, False),
        ('      state this explicitly in the report.', C['mut'], 9, False),
    ]
    y = 0.97
    for text, color, size, bold in lines:
        if text == '':
            y -= 0.035; continue
        ax3.text(0.02, y, text, transform=ax3.transAxes, color=color,
                 fontsize=size, fontweight='bold' if bold else 'normal',
                 va='top', family='monospace' if text.startswith(('OK', '!', '  ', '=')) else 'sans-serif')
        y -= 0.058

    fig.suptitle('Orbit-end Continuation + Constrained PB-NBV — continuous, tracking-safe, info-gain driven',
                 color=C['txt'], fontsize=14.5, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')


if __name__ == '__main__':
    main()
