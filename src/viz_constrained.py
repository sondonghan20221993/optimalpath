"""
viz_constrained.py — Overlap 제약 PB-NBV 결과 시각화 (ATE 친화 연속 경로)

(a) top-down 경로: WP 순서 화살표 + overlap 라벨
(b) step별 overlap (ATE 친화 지표) + 누적 soft 커버 곡선
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
JS   = ROOT / "results" / "pbnbv_constrained_path.json"
OUT  = ROOT / "results" / "constrained_path.png"

START = np.array([-27.06, -50.51, -6.13])


def main():
    d   = json.load(open(JS))
    pts = P._pts_raw
    t   = P.TARGET
    wps = np.array([w["pos"] for w in d["waypoints"]])
    ovl = [w["overlap_with_prev"] for w in d["waypoints"]]
    azs = d["az_distribution"]

    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d', 'bg': '#0d1117',
         'pc': '#6e7681', 'path': '#3fb950', 'start': '#ffa657', 'tgt': '#f778ba'}

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22,
                           left=0.06, right=0.97, top=0.88, bottom=0.10,
                           width_ratios=[1.25, 1])

    # ── (a) 경로 top-down ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) Constrained PB-NBV Path (overlap-gated, info-gain driven)',
                 color=C['txt'], fontsize=13, pad=8)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=3, alpha=0.4, zorder=1)
    ax.scatter(*t[:2], marker='*', s=320, c=C['tgt'], zorder=6, label='target')
    # 궤도 링(참고)
    r = np.linalg.norm(wps[:, :2] - t[:2], axis=1).mean()
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(t[0]+r*np.cos(th), t[1]+r*np.sin(th), '--', color=C['path'],
            alpha=0.25, lw=1, zorder=2)
    # 시작 화살표
    ax.scatter(*START[:2], marker='s', s=120, c=C['start'], zorder=5,
               edgecolors='white', linewidths=0.6, label='start')
    ax.annotate('', xy=wps[0, :2], xytext=START[:2],
                arrowprops=dict(arrowstyle='->', color=C['start'], lw=2, alpha=0.8))
    # WP 순서 화살표 + overlap 라벨
    for i in range(len(wps)-1):
        mx, my = (wps[i, 0]+wps[i+1, 0])/2, (wps[i, 1]+wps[i+1, 1])/2
        ax.annotate('', xy=wps[i+1, :2], xytext=wps[i, :2],
                    arrowprops=dict(arrowstyle='->', color=C['path'], lw=2.2))
        ax.text(mx, my, f'{ovl[i+1]:.2f}', color='#9be9a8', fontsize=7.5,
                ha='center', va='center', zorder=7)
    ax.scatter(wps[:, 0], wps[:, 1], c=C['path'], s=130, marker='D', zorder=5,
               edgecolors='white', linewidths=0.6, label='waypoint')
    for i, w in enumerate(wps):
        ax.text(w[0], w[1]+0.32, f'{i+1}', color='white', fontsize=9,
                ha='center', va='center', fontweight='bold', zorder=8)
    ax.text(0.03, 0.04,
            f"{d['n_waypoints']} WP  |  {d['path_length_m']} m\n"
            f"overlap min {d['overlap_min']} / mean {d['overlap_mean']}\n"
            f"real cov {100*d['real_cov_final']:.1f}%  |  soft {100*d['soft_cov_final']:.1f}%",
            transform=ax.transAxes, color=C['path'], fontsize=10,
            va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', fc='#161b22', ec=C['path'], alpha=0.8))
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── (b) overlap + 커버 ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(C['ax'])
    ax2.set_title('(b) Frame Overlap per Step  (ATE-friendliness)',
                  color=C['txt'], fontsize=13, pad=8)
    steps = list(range(1, len(ovl)+1))
    bars = ax2.bar(steps, ovl, color=C['path'], alpha=0.85, width=0.7)
    for b, v in zip(bars, ovl):
        ax2.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.2f}',
                 ha='center', va='bottom', color=C['txt'], fontsize=8)
    ax2.axhline(d['tau_overlap'], color='#f85149', ls='--', lw=1.5,
                label=f"constraint TAU_OVL={d['tau_overlap']}")
    ax2.text(len(ovl)*0.5, d['tau_overlap']+0.02,
             'all steps well above the tracking floor',
             color='#f85149', fontsize=9, ha='center')
    ax2.set_ylim(0, 1.12)
    ax2.set_xlabel('Waypoint #', color=C['txt'])
    ax2.set_ylabel('overlap with previous view', color=C['txt'])
    ax2.set_xticks(steps)
    ax2.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, axis='y', color=C['grid'], alpha=0.3, lw=0.5)
    ax2.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
               labelcolor=C['txt'], loc='lower right')

    fig.suptitle('Constrained PB-NBV — overlap-gated viewpoints keep SLAM tracking (low ATE), '
                 'still information-gain driven',
                 color=C['txt'], fontsize=14, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')
    print(f"  {d['n_waypoints']} WP, {d['path_length_m']}m, overlap min {d['overlap_min']} "
          f"mean {d['overlap_mean']}, real {100*d['real_cov_final']:.1f}%")


if __name__ == '__main__':
    main()
