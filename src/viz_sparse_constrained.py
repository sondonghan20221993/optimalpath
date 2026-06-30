"""
viz_sparse_constrained.py — 성긴 tilt=45° 궤도 + constrained PB-NBV 보완 (NBV 가치 입증)

좌  : 경로 (성긴 4대 궤도 + NBV 보완이 사각지대를 채움)
우상: 커버리지 진행 (궤도 단독 → +NBV)
우하: overlap 막대 + 핵심 메시지
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
JS   = ROOT / "results" / "pbnbv_sparse_constrained_path.json"
OUT  = ROOT / "results" / "sparse_constrained_summary.png"


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
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.20,
                           width_ratios=[1.35, 1], height_ratios=[1, 1],
                           left=0.05, right=0.975, top=0.90, bottom=0.07)

    # ── 좌: 경로 ──
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) Sparse tilt=45° orbit (4 cams) + constrained NBV fills the gaps',
                 color=C['txt'], fontsize=12.5, pad=8)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=3, alpha=0.4, zorder=1)
    ax.scatter(*t[:2], marker='*', s=320, c=C['tgt'], zorder=8, label='target')
    ro = np.linalg.norm(orbit[:, :2]-t[:2], axis=1).mean()
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(t[0]+ro*np.cos(th), t[1]+ro*np.sin(th), '--', color=C['orbit'],
            alpha=0.3, lw=1, zorder=2)
    # 궤도 4대 (연결선 = 성긴 비행)
    ax.plot(np.append(orbit[:, 0], orbit[0, 0]), np.append(orbit[:, 1], orbit[0, 1]),
            '-', color=C['orbit'], alpha=0.4, lw=1.4, zorder=2)
    ax.scatter(orbit[:, 0], orbit[:, 1], c=C['orbit'], s=180, marker='o', zorder=4,
               edgecolors='white', linewidths=0.7, label='orbit (4 cams, sparse)')
    for c in orbit:
        ax.text(c[0], c[1]-0.5, f'{round(math.degrees(math.atan2(c[1]-t[1],c[0]-t[0]))%360)}°',
                color=C['orbit'], fontsize=8, ha='center')
    ax.scatter(*orbit_end[:2], marker='s', s=160, c=C['end'], zorder=7,
               edgecolors='white', linewidths=0.8, label='orbit end (NBV start)')
    # 보완 경로
    ax.annotate('', xy=wps[0, :2], xytext=orbit_end[:2],
                arrowprops=dict(arrowstyle='->', color=C['path'], lw=2.0, alpha=0.9))
    for i in range(len(wps)-1):
        ax.annotate('', xy=wps[i+1, :2], xytext=wps[i, :2],
                    arrowprops=dict(arrowstyle='->', color=C['path'], lw=2.2))
        mx, my = (wps[i, 0]+wps[i+1, 0])/2, (wps[i, 1]+wps[i+1, 1])/2
        ax.text(mx, my, f'{ovl[i+1]:.2f}', color='#9be9a8', fontsize=7,
                ha='center', va='center', zorder=9)
    ax.scatter(wps[:, 0], wps[:, 1], c=C['path'], s=120, marker='D', zorder=6,
               edgecolors='white', linewidths=0.6, label='NBV supplement WP')
    for i, w in enumerate(wps):
        ax.text(w[0], w[1]+0.28, f'{i+1}', color='white', fontsize=8.5,
                ha='center', va='center', fontweight='bold', zorder=10)
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=8.5, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── 우상: 커버리지 진행 ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C['ax'])
    ax2.set_title('(b) Coverage: sparse orbit alone → + NBV supplement',
                  color=C['txt'], fontsize=12, pad=6)
    labels = ['Orbit\nsoft', '+NBV\nsoft', 'Orbit\nreal', '+NBV\nreal']
    vals = [100*d['soft_cov_orbit'], 100*d['soft_cov_final'],
            100*d['real_cov_orbit'], 100*d['real_cov_final']]
    cols = [C['orbit'], C['path'], C['orbit'], C['path']]
    bars = ax2.bar(labels, vals, color=cols, alpha=0.88)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.1f}%',
                 ha='center', va='bottom', color=C['txt'], fontsize=10, fontweight='bold')
    ax2.set_ylim(97.5, 100.6)
    ax2.set_ylabel('coverage (%)', color=C['txt'], fontsize=9)
    ax2.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, axis='y', color=C['grid'], alpha=0.3, lw=0.5)

    # ── 우하: overlap + 메시지 ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(C['ax'])
    ax3.set_title('(c) NBV supplement overlap (ATE-safe) + takeaways',
                  color=C['txt'], fontsize=12, pad=6)
    steps = list(range(1, len(ovl)+1))
    bars = ax3.bar(steps, ovl, color=C['path'], alpha=0.8, width=0.7)
    ax3.axhline(d['tau_overlap'], color=C['warn'], ls='--', lw=1.3)
    ax3.set_ylim(0, 1.35)
    ax3.set_xlabel('NBV WP #', color=C['txt'], fontsize=9)
    ax3.set_ylabel('overlap', color=C['txt'], fontsize=9)
    ax3.set_xticks(steps)
    ax3.tick_params(colors=C['txt'], labelsize=8)
    for sp in ax3.spines.values(): sp.set_color(C['grid'])
    ax3.grid(True, axis='y', color=C['grid'], alpha=0.25, lw=0.5)
    msg = (f"sparse orbit {100*d['real_cov_orbit']:.1f}% real -> NBV adds "
           f"{d['n_waypoints']} WP -> {100*d['real_cov_final']:.1f}% (0 left)\n"
           f"overlap {d['overlap_min']}-1.0 (ATE-safe) | consistent tilt=45°, "
           f"{d['orbit_dist_min']}m | still info-gain driven")
    ax3.text(0.5, 1.18, msg, transform=ax3.transAxes, color=C['mut'],
             fontsize=8.5, ha='center', va='bottom')

    fig.suptitle('Sparse Orbit + Constrained PB-NBV — NBV completes coverage where a sparse orbit leaves gaps',
                 color=C['txt'], fontsize=14.5, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')
    print(f"  궤도4대 real {100*d['real_cov_orbit']:.1f}% → +{d['n_waypoints']}WP → "
          f"{100*d['real_cov_final']:.1f}%, 총 {d['total_flight_m']}m")


if __name__ == '__main__':
    main()
