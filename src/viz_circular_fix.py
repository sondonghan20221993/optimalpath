"""
viz_circular_fix.py — 경로가 한쪽에 뭉치는 원인(F-score 아티팩트)과 수정 결과

좌: 현재 (U=F/dist) — F-score 투영 아티팩트로 az=270° 한 점에 붕괴
우: 수정 (U=real_softgain/dist) — 선택기준을 실제 누적관측 gain으로 일치 → arc로 분산
하단: 후보별 F-score vs 실제 soft-gain 비교 (az=270° F만 10배 뻥튀기 입증)
"""
import sys, types, json, math, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P
import pbnbv_path as A
import run_pbnbv_orbit_soft as S

ROOT = HERE.parent
SC = Path("/tmp/claude-1000/-home-sdh2983/f06e13ff-33e6-4ae8-849c-9a1318701cab/scratchpad/circ.pkl")
OUT = ROOT / "results" / "circular_fix.png"


def az_of(p, t):
    return math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360


def draw_path(ax, pts, cam, target, path, title, color, C):
    ax.set_facecolor(C['ax'])
    ax.set_title(title, color=C['txt'], fontsize=12, pad=6)
    ax.scatter(pts[:, 0], pts[:, 1], c='#6e7681', s=3, alpha=0.4, zorder=1)
    ax.scatter(*target[:2], marker='*', s=200, c=C['tgt'], zorder=6)
    # orbit ring
    r = np.linalg.norm(cam[:, :2]-target[:2], axis=1).mean()
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(target[0]+r*np.cos(th), target[1]+r*np.sin(th), '--',
            color=C['orbit'], alpha=0.3, lw=1)
    ax.scatter(cam[:, 0], cam[:, 1], c=C['orbit'], s=14, alpha=0.5, zorder=2)
    path = np.array(path)
    start = cam[-1, :2]
    ax.annotate('', xy=path[0, :2], xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    for i in range(len(path)-1):
        ax.annotate('', xy=path[i+1, :2], xytext=path[i, :2],
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.scatter(path[:, 0], path[:, 1], c=color, s=110, marker='D',
               zorder=5, edgecolors='white', linewidths=0.5)
    for i, w in enumerate(path):
        ax.text(w[0], w[1]+0.35, f'{i+1}', color='white', fontsize=8,
                ha='center', va='center', fontweight='bold', zorder=7)
    azs = [az_of(w, target) for w in path]
    ax.text(0.03, 0.03, f'{len(path)} WP\naz: {[round(a) for a in azs]}',
            transform=ax.transAxes, color=color, fontsize=9, va='bottom',
            fontweight='bold')
    ax.set_xlabel('X (m)', color=C['txt'], fontsize=9)
    ax.set_ylabel('Y (m)', color=C['txt'], fontsize=9)
    ax.tick_params(colors=C['txt'], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax.set_aspect('equal')


def main():
    d = pickle.load(open(SC, 'rb'))
    pts = np.array(d['pts']); cam = np.array(d['orbit']); target = np.array(d['target'])
    pathF = d['pathF']; pathG = d['pathG1']

    fig = plt.figure(figsize=(17, 13))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22,
                           height_ratios=[1.25, 1],
                           left=0.07, right=0.96, top=0.92, bottom=0.08)
    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d',
         'orbit': '#58a6ff', 'tgt': '#ffa657', 'bad': '#f85149', 'good': '#3fb950'}

    # 상단 두 경로
    ax1 = fig.add_subplot(gs[0, 0])
    draw_path(ax1, pts, cam, target, pathF,
              '(a) BEFORE: U = F_score / dist  → collapses to one spot',
              C['bad'], C)
    ax2 = fig.add_subplot(gs[0, 1])
    draw_path(ax2, pts, cam, target, pathG,
              '(b) AFTER: U = real_soft_gain / dist  → spreads into an arc',
              C['good'], C)

    # 하단: 후보별 F-score vs 실제 gain (az=270 뻥튀기 입증)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(C['ax'])
    ax3.set_title('(c) Root Cause: ellipsoid F-score inflates az=270° by ~10x vs its actual coverage gain',
                  color=C['txt'], fontsize=12, pad=6)

    fix_z = round(pts[:, 2].min()-3.0, 2)
    cands = A.gen_candidates_tilt45(target, [fix_z], n_az=36, max_dist=P.MAX_DIST)
    obs_w = np.zeros(P.N_SURF)
    for c in cam:
        obs_w += S.soft_weight(c)
    observed = obs_w >= S.TAU
    fro = P.compute_frontier(observed)
    oe = P.fit_ellipsoids(P.SURF_CEN[observed], P.SURF_NRM[observed])
    fe = P.fit_ellipsoids(P.SURF_CEN[fro], P.SURF_NRM[fro])
    azs, Fs, gains = [], [], []
    for c in cands:
        azs.append(az_of(c, target))
        Fs.append(P.evaluate(c, oe, fe, frontier_only=True))
        nw = obs_w + S.soft_weight(c)
        gains.append(int(((nw >= S.TAU) & ~observed).sum()))
    order = np.argsort(azs)
    azs = np.array(azs)[order]; Fs = np.array(Fs)[order]; gains = np.array(gains)[order]

    ax3b = ax3.twinx()
    l1, = ax3.plot(azs, Fs, '-o', color=C['bad'], lw=2, markersize=5,
                   label='ellipsoid F-score (selection proxy)')
    l2, = ax3b.plot(azs, gains, '-s', color=C['good'], lw=2, markersize=5,
                    label='actual new soft-coverage gain')
    ax3.axvline(270, color=C['bad'], ls='--', alpha=0.5)
    ax3.annotate('F-score spikes to 249\n(artifact — projection of\ndistant frontier ellipsoids)',
                 xy=(270, 249), xytext=(150, 180),
                 color=C['bad'], fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=C['bad'], lw=1.2))
    ax3.text(270, 30, 'but real gain here\nis only 22 (= az 240/300)',
             color=C['good'], fontsize=8.5, ha='center')

    ax3.set_xlabel('candidate azimuth (deg)', color=C['txt'], fontsize=10)
    ax3.set_ylabel('F-score', color=C['bad'], fontsize=10)
    ax3b.set_ylabel('actual soft-gain (voxels)', color=C['good'], fontsize=10)
    ax3.tick_params(colors=C['txt'], labelsize=9); ax3.tick_params(axis='y', colors=C['bad'])
    ax3b.tick_params(axis='y', colors=C['good'], labelsize=9)
    ax3.set_xticks(range(0, 361, 30))
    for sp in ax3.spines.values():
        sp.set_color(C['grid'])
    for sp in ax3b.spines.values():
        sp.set_color(C['grid'])
    ax3.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
    ax3.legend(handles=[l1, l2], fontsize=9, facecolor='#21262d',
               edgecolor=C['grid'], labelcolor=C['txt'], loc='upper left')

    fig.suptitle('Why the Path Clustered on One Side — and the Fix (consistent selection metric)',
                 color=C['txt'], fontsize=14, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f'✓ 저장: {OUT}')


if __name__ == '__main__':
    main()
