"""
viz_nocam_v2_fbx.py — orbit 없이 순수 PB-NBV (from scratch), FBX GT 기반 시각화
좌  : top-down 경로 (출발 → NBV가 선택한 viewpoint들)
우  : 결과 summary + 로직 설명
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
JS   = ROOT / "results" / "pbnbv_nocam_soft_v2_path.json"
OUT  = ROOT / "results" / "nocam_v2_fbx_path.png"


def main():
    d   = json.load(open(JS))
    pts = P._pts_raw
    t   = P.TARGET
    wps = np.array([w["pos"] for w in d["waypoints"]])
    azs = [w["azimuth_deg"] for w in d["waypoints"]]
    # 출발점: WP1에서 살짝 떨어진 시작 (스크립트 START 재현)
    start = np.array([-27.06, -50.51, -6.13])

    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d', 'bg': '#0d1117',
         'pc':  '#6e7681', 'path': '#3fb950', 'start': '#ffa657',
         'tgt': '#f778ba', 'warn': '#f85149', 'mut': '#8b949e'}

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.22,
                           left=0.05, right=0.97, top=0.88, bottom=0.08,
                           width_ratios=[1.5, 1])

    # ── 좌: 경로 ──
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(C['ax'])
    ax.set_title('(a) Pure PB-NBV from scratch (no orbit) — FBX GT',
                 color=C['txt'], fontsize=13, pad=8)
    ax.scatter(pts[:, 0], pts[:, 1], c=C['pc'], s=12, alpha=0.8, zorder=2,
               label='GT slab (FBX)')
    ax.scatter(*t[:2], marker='*', s=400, c=C['tgt'], zorder=8,
               label='target (centroid)')

    # 출발점
    ax.scatter(*start[:2], marker='s', s=180, c=C['start'], zorder=6,
               edgecolors='white', linewidths=0.8, label='start')
    # 출발 → WP1
    ax.annotate('', xy=wps[0, :2], xytext=start[:2],
                arrowprops=dict(arrowstyle='->', color=C['path'], lw=2.0, alpha=0.9))
    # WP 간 경로
    for i in range(len(wps)-1):
        ax.annotate('', xy=wps[i+1, :2], xytext=wps[i, :2],
                    arrowprops=dict(arrowstyle='->', color=C['path'], lw=2.4))
    # WP 마커
    ax.scatter(wps[:, 0], wps[:, 1], c=C['path'], s=200, marker='D', zorder=7,
               edgecolors='white', linewidths=0.8, label='NBV waypoint')
    for i, w in enumerate(wps):
        az = azs[i]
        ax.text(w[0], w[1]+0.18, f'WP{i+1}', color='white', fontsize=10,
                ha='center', va='bottom', fontweight='bold', zorder=10)
        ax.text(w[0], w[1]-0.22, f'az={round(az)}°  z={w[2]:.1f}',
                color=C['path'], fontsize=8.5, ha='center', va='top')

    ax.set_xlabel('X (m)', color=C['txt'], fontsize=10)
    ax.set_ylabel('Y (m)', color=C['txt'], fontsize=10)
    ax.tick_params(colors=C['txt'], labelsize=9)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.legend(fontsize=9, facecolor='#21262d', edgecolor=C['grid'],
              labelcolor=C['txt'], loc='upper right')
    ax.set_aspect('equal')

    # ── 우: summary ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C['ax'])
    ax2.axis('off')
    lines = [
        ('[ LOGIC ]', C['tgt'], 13, True),
        ('No orbit. Pure greedy PB-NBV.', C['txt'], 10, False),
        ('Candidates: 180 (6 alt x 36 az), tilt=45', C['txt'], 9.5, False),
        ('Each step pick max U:', C['txt'], 10, False),
        ('   U = soft_gain / dist^1.0', C['path'], 10, True),
        ('   gain = newly covered voxels (Lambert)', C['mut'], 9, False),
        ('Stop when gain = 0.', C['txt'], 10, False),
        ('', C['txt'], 6, False),

        ('[ RESULT ]', C['tgt'], 13, True),
        (f'Waypoints : {d["n_waypoints"]}', C['txt'], 11, False),
        (f'Path      : {d["path_length_m"]:.2f} m', C['txt'], 11, False),
        (f'Az spread : {d["az_spread_deg"]}deg (true 3D coverage)', C['txt'], 10, False),
        (f'Soft cov  : {100*d["soft_cov_final"]:.1f}%', C['txt'], 11, False),
        (f'Real cov  : {100*d["real_cov_final"]:.1f}%', C['path'], 12, True),
        ('', C['txt'], 6, False),

        ('[ vs ORBIT ]', C['tgt'], 13, True),
        ('Orbit (4 cam) : 13.49 m, real 100%', C['mut'], 10, False),
        (f'NBV (no orbit): {d["path_length_m"]:.2f} m, real 100%', C['path'], 10, True),
        ('-> NBV reaches 100% with less flight,', C['txt'], 9.5, False),
        ('   selecting only 3 informative views.', C['txt'], 9.5, False),
        ('', C['txt'], 6, False),

        ('[ GT ]', C['warn'], 13, True),
        ('FBX mesh GT (camera-independent).', C['mut'], 9.5, False),
        (f'{P._pts_raw.shape[0]} verts, '
         f'{pts[:,0].max()-pts[:,0].min():.2f}x'
         f'{pts[:,1].max()-pts[:,1].min():.2f}x'
         f'{pts[:,2].max()-pts[:,2].min():.2f} m', C['mut'], 9.5, False),
    ]
    y = 0.97
    for text, color, size, bold in lines:
        if text == '':
            y -= 0.03; continue
        ax2.text(0.04, y, text, transform=ax2.transAxes, color=color,
                 fontsize=size, fontweight='bold' if bold else 'normal',
                 va='top', family='monospace')
        y -= 0.05

    fig.suptitle('Pure PB-NBV (no orbit) on FBX GT — 3 informative views, 11.79 m, 100% real coverage',
                 color=C['txt'], fontsize=14, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')


if __name__ == '__main__':
    main()
