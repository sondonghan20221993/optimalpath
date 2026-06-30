"""
viz_orbit_constrained.py — 궤도 끝점에서 이어가는 constrained PB-NBV 시각화

(a) top-down: 원형 궤도(34대) + 끝점 handoff + 보완 WP 경로(overlap 라벨)
(b) 보완 step별 overlap (ATE 친화)
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
OUT  = ROOT / "results" / "orbit_constrained_path.png"


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
         'end': '#ffa657', 'tgt': '#f778ba'}

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22,
                           left=0.06, right=0.97, top=0.88, bottom=0.10,
                           width_ratios=[1.3, 1])

    # ── (a) ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) Orbit (34 cams) → continue from orbit-end → constrained NBV supplement',
                 color=C['txt'], fontsize=12.5, pad=8)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=3, alpha=0.4, zorder=1)
    ax.scatter(*t[:2], marker='*', s=320, c=C['tgt'], zorder=8, label='target')
    # 궤도
    ax.scatter(orbit[:, 0], orbit[:, 1], c=C['orbit'], s=22, alpha=0.6, zorder=2,
               label=f'orbit ({len(orbit)} cams)')
    ro = np.linalg.norm(orbit[:, :2]-t[:2], axis=1).mean()
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(t[0]+ro*np.cos(th), t[1]+ro*np.sin(th), '--', color=C['orbit'],
            alpha=0.3, lw=1, zorder=2)
    # 궤도 끝점
    ax.scatter(*orbit_end[:2], marker='s', s=160, c=C['end'], zorder=7,
               edgecolors='white', linewidths=0.8, label='orbit end (start)')
    ax.annotate('orbit end\naz360°', orbit_end[:2], textcoords='offset points',
                xytext=(8, 6), color=C['end'], fontsize=8.5, fontweight='bold')
    # handoff 화살표 (점선)
    ax.annotate('', xy=wps[0, :2], xytext=orbit_end[:2],
                arrowprops=dict(arrowstyle='->', color=C['end'], lw=1.8, ls='--', alpha=0.8))
    ax.text((orbit_end[0]+wps[0,0])/2, (orbit_end[1]+wps[0,1])/2, 'handoff',
            color=C['end'], fontsize=8, style='italic', ha='center')
    # 보완 경로
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
    ax.text(0.03, 0.04,
            f"orbit soft {100*d['soft_cov_orbit']:.1f}% / real {100*d['real_cov_orbit']:.1f}%\n"
            f"+ {d['n_waypoints']} WP ({d['path_length_m']} m), overlap>={d['tau_overlap']}\n"
            f"→ soft {100*d['soft_cov_final']:.1f}% / real {100*d['real_cov_final']:.1f}%  "
            f"(ovl min {d['overlap_min']})",
            transform=ax.transAxes, color=C['path'], fontsize=9.5, va='bottom',
            fontweight='bold',
            bbox=dict(boxstyle='round', fc='#161b22', ec=C['path'], alpha=0.85))
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=8.5, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── (b) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C['ax'])
    ax2.set_title('(b) Supplement Frame Overlap per Step (ATE-friendliness)',
                  color=C['txt'], fontsize=12.5, pad=8)
    steps = list(range(1, len(ovl)+1))
    bars = ax2.bar(steps, ovl, color=C['path'], alpha=0.85, width=0.65)
    for b, v in zip(bars, ovl):
        ax2.text(b.get_x()+b.get_width()/2, v+0.012, f'{v:.2f}',
                 ha='center', va='bottom', color=C['txt'], fontsize=9)
    ax2.axhline(d['tau_overlap'], color='#f85149', ls='--', lw=1.5,
                label=f"constraint={d['tau_overlap']}")
    ax2.text(len(ovl)*0.5, d['tau_overlap']+0.03, 'all supplement steps keep tracking',
             color='#f85149', fontsize=9, ha='center')
    ax2.set_ylim(0, 1.12)
    ax2.set_xlabel('Supplement WP #', color=C['txt'])
    ax2.set_ylabel('overlap with previous view', color=C['txt'])
    ax2.set_xticks(steps)
    ax2.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, axis='y', color=C['grid'], alpha=0.3, lw=0.5)
    ax2.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
               labelcolor=C['txt'], loc='lower right')

    fig.suptitle('Orbit-end Continuation + Constrained PB-NBV — continuous, tracking-safe, info-gain driven',
                 color=C['txt'], fontsize=14, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')
    print(f"  궤도 real {100*d['real_cov_orbit']:.1f}% → +{d['n_waypoints']}WP → {100*d['real_cov_final']:.1f}%, "
          f"보완 {d['path_length_m']}m, ovl min {d['overlap_min']}")


if __name__ == '__main__':
    main()
