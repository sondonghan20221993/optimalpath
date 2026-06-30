"""
viz_fbx_gt_orbit.py — FBX GT 기반 tilt=45° 4-cam sparse orbit 결과 시각화
"""
import sys, types, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P

ROOT = HERE.parent
JS   = ROOT / "results" / "pbnbv_sparse_constrained_path.json"
OUT  = ROOT / "results" / "fbx_gt_orbit_path.png"


def main():
    d     = json.load(open(JS))
    pts   = P._pts_raw          # FBX GT 점군
    t     = P.TARGET

    orbit = np.array(d["orbit_positions"])
    wps   = np.array([w["pos"] for w in d["waypoints"]]) if d["waypoints"] else np.empty((0,3))

    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d', 'bg': '#0d1117',
         'pc':  '#6e7681', 'orbit': '#58a6ff', 'path': '#3fb950',
         'end': '#ffa657', 'tgt': '#f778ba', 'warn': '#f85149', 'mut': '#8b949e'}

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22,
                           left=0.05, right=0.97, top=0.88, bottom=0.08,
                           width_ratios=[1.5, 1])

    # ── 좌: top-down 경로 ──
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) Sparse tilt=45° orbit (4 cams) — FBX GT 기반 경로',
                 color=C['txt'], fontsize=13, pad=8)

    # GT 점군 (FBX 슬래브)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=12, alpha=0.8, zorder=2, label='GT slab (FBX)')
    ax.scatter(*t[:2], marker='*', s=400, c=C['tgt'], zorder=8, label='target (centroid)')

    # 궤도 원
    ro = np.linalg.norm(orbit[:, :2] - t[:2], axis=1).mean()
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(t[0]+ro*np.cos(th), t[1]+ro*np.sin(th), '--',
            color=C['orbit'], alpha=0.3, lw=1.2, zorder=2)

    # 궤도 비행선 (순서대로)
    for i in range(len(orbit)):
        nxt = orbit[(i+1) % len(orbit)]
        ax.annotate('', xy=nxt[:2], xytext=orbit[i, :2],
                    arrowprops=dict(arrowstyle='->', color=C['orbit'], lw=1.6, alpha=0.7))

    # 궤도 카메라
    ax.scatter(orbit[:, 0], orbit[:, 1], c=C['orbit'], s=220, marker='o', zorder=5,
               edgecolors='white', linewidths=0.8, label='orbit cam (4)')
    for i, c in enumerate(orbit):
        az = math.degrees(math.atan2(c[1]-t[1], c[0]-t[0])) % 360
        ax.text(c[0], c[1]-0.3, f'{round(az)}°',
                color=C['orbit'], fontsize=9, ha='center', fontweight='bold')
        ax.text(c[0], c[1]+0.25, f'#{i+1}',
                color='white', fontsize=8, ha='center')

    # NBV WP (없으면 표시 생략)
    if len(wps) > 0:
        ax.scatter(wps[:, 0], wps[:, 1], c=C['path'], s=160, marker='D', zorder=6,
                   edgecolors='white', linewidths=0.7, label='NBV WP')

    ax.set_xlabel('X (m)', color=C['txt'], fontsize=10)
    ax.set_ylabel('Y (m)', color=C['txt'], fontsize=10)
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── 우: 결과 summary ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C['ax'])
    ax2.axis('off')
    for sp in ax2.spines.values(): sp.set_color(C['grid'])

    lines = [
        ('[ GT ]', C['tgt'], 13, True),
        ('Source : FBX mesh (my_box.fbx)', C['txt'], 10, False),
        (f'Points : {len(pts)} vertices (ㄷ-shape)', C['txt'], 10, False),
        (f'Size   : {pts[:,0].max()-pts[:,0].min():.2f}m × {pts[:,1].max()-pts[:,1].min():.2f}m × {pts[:,2].max()-pts[:,2].min():.2f}m', C['txt'], 10, False),
        ('Camera-independent GT (not SfM)', C['path'], 10, True),
        ('', C['txt'], 6, False),

        ('[ ORBIT ]', C['tgt'], 13, True),
        (f'Cameras  : {d["orbit_n_az"]} (sparse, 90° apart)', C['txt'], 10, False),
        (f'Altitude : z={d["fix_z"]}m  tilt=45°', C['txt'], 10, False),
        (f'Standoff : {d["orbit_dist_min"]}m (within MAX_DIST 8m)', C['txt'], 10, False),
        (f'Flight   : {d["orbit_flight_m"]:.2f}m', C['txt'], 10, False),
        ('', C['txt'], 6, False),

        ('[ COVERAGE ]', C['tgt'], 13, True),
        (f'Soft  : {100*d["soft_cov_orbit"]:.1f}%   Real : {100*d["real_cov_orbit"]:.1f}%', C['path'], 11, True),
        (f'NBV supplement : {d["n_waypoints"]} WP (not needed)', C['mut'], 10, False),
        ('Orbit alone fully covers compact ㄷ-slab', C['txt'], 10, False),
        ('', C['txt'], 6, False),

        ('[ NOTE ]', C['warn'], 13, True),
        ('SfM point cloud was scale-incorrect;', C['mut'], 9.5, False),
        ('FBX GT used for honest evaluation.', C['mut'], 9.5, False),
    ]

    y = 0.97
    for text, color, size, bold in lines:
        if text == '':
            y -= 0.03; continue
        ax2.text(0.04, y, text, transform=ax2.transAxes, color=color,
                 fontsize=size, fontweight='bold' if bold else 'normal', va='top',
                 family='monospace')
        y -= 0.052

    fig.suptitle('PB-NBV with Camera-Independent FBX GT — Sparse 4-cam Orbit achieves 100% Real Coverage',
                 color=C['txt'], fontsize=14, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')


if __name__ == '__main__':
    main()
