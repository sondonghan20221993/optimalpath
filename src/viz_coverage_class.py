"""
viz_coverage_class.py — 표면 voxel을 3분류로 색칠해 "왜 사각이 남는지" 직접 설명
  초록 = 커버됨(관측가능 & soft>=TAU)
  주황 = 미커버(관측가능하나 아직 안 본 voxel)
  빨강 = 물리적 사각(밑면, 지면 위 tilt45로 원천 불가)
좌: top-down(XY), 우: 측면(XZ) — 밑면 사각이 왜 안 보이는지 측면에서 드러남
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
import run_pbnbv_orbit_soft as S

ROOT = HERE.parent
JS   = ROOT / "results" / "pbnbv_nocam_constrained_path.json"
OUT  = ROOT / "results" / "coverage_class.png"


def main():
    d = json.load(open(JS))
    P.MAX_DIST = 10.5
    S.TAU = math.cos(math.radians(d.get("inc_limit_deg", 70)))   # run과 동일 입사각 임계
    t = P.TARGET
    cen, nrm = P.SURF_CEN, P.SURF_NRM
    allv = [np.array(d["start_pos"])] + [np.array(w["pos"]) for w in d["waypoints"]]
    wps = np.array([w["pos"] for w in d["waypoints"]])

    # 관측가능 마스크 (전체 후보 합산)
    obj_top = P._pts_raw[:, 2].min()
    import pbnbv_path as A
    cands = A.gen_candidates_tilt45(t, [round(obj_top-7, 2), round(obj_top-3, 2)],
                                    n_az=36, max_dist=10.5)
    obs_all = np.zeros(P.N_SURF)
    for c in cands: obs_all += S.soft_weight(c)
    observable = obs_all >= S.TAU

    # 실제 경로 누적 커버
    obs_w = np.zeros(P.N_SURF)
    for c in allv: obs_w += S.soft_weight(c)
    covered = obs_w >= S.TAU

    cls_cov   = covered & observable          # 초록
    cls_uncov = (~covered) & observable       # 주황
    cls_under = ~observable                    # 빨강 (밑면)

    C = {'txt': '#e6edf3', 'ax': '#161b22', 'grid': '#21262d', 'bg': '#0d1117',
         'cov': '#3fb950', 'uncov': '#ffa657', 'under': '#f85149',
         'path': '#58a6ff', 'tgt': '#f778ba', 'mut': '#8b949e'}

    fig = plt.figure(figsize=(18, 8.5))
    fig.patch.set_facecolor(C['bg'])
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.16,
                           left=0.05, right=0.97, top=0.87, bottom=0.09)

    def draw(ax, ix, iy, xl, yl, title, show_path=True):
        ax.set_facecolor(C['ax'])
        ax.set_title(title, color=C['txt'], fontsize=12.5, pad=8)
        ax.scatter(cen[cls_cov, ix], cen[cls_cov, iy], c=C['cov'], s=30, alpha=0.9,
                   label=f'covered ({cls_cov.sum()})', zorder=4)
        ax.scatter(cen[cls_uncov, ix], cen[cls_uncov, iy], c=C['uncov'], s=40, alpha=0.95,
                   label=f'uncovered-observable ({cls_uncov.sum()})', zorder=5,
                   edgecolors='white', linewidths=0.4)
        ax.scatter(cen[cls_under, ix], cen[cls_under, iy], c=C['under'], s=40, alpha=0.95,
                   label=f'underside / physical occlusion ({cls_under.sum()})', zorder=5,
                   marker='v', edgecolors='white', linewidths=0.4)
        ax.scatter(t[ix], t[iy], marker='*', s=300, c=C['tgt'], zorder=8)
        if show_path:
            ax.scatter(wps[:, ix], wps[:, iy], c=C['path'], s=150, marker='D',
                       zorder=7, edgecolors='white', linewidths=0.7, label='NBV WP')
            for i, w in enumerate(wps):
                ax.text(w[ix], w[iy], f'{i+1}', color='white', fontsize=8,
                        ha='center', va='center', fontweight='bold', zorder=9)
        ax.set_xlabel(xl, color=C['txt'], fontsize=10)
        ax.set_ylabel(yl, color=C['txt'], fontsize=10)
        ax.tick_params(colors=C['txt'], labelsize=9)
        for sp in ax.spines.values(): sp.set_color(C['grid'])
        ax.grid(True, color=C['grid'], alpha=0.3, lw=0.5)
        ax.legend(fontsize=8.5, facecolor='#21262d', edgecolor=C['grid'],
                  labelcolor=C['txt'], loc='upper right')
        ax.set_aspect('equal')

    ax1 = fig.add_subplot(gs[0])
    draw(ax1, 0, 1, 'X (m)', 'Y (m)', '(a) Top-down (XY) — WP positions')
    ax2 = fig.add_subplot(gs[1])
    draw(ax2, 0, 2, 'X (m)', 'Z (m)  (NED: down=+)',
         '(b) Side (XZ) — red undersides face the ground', show_path=False)
    ax2.invert_yaxis()  # NED: 위가 음수 → 물체 위가 위로 오게

    n_obs = int(observable.sum())
    msg = (f"Observable surface {int(cls_cov.sum())}/{n_obs} = "
           f"{100*cls_cov.sum()/n_obs:.1f}% covered   |   "
           f"{int(cls_under.sum())} undersides physically occluded "
           f"(need underground flight, out-of-scope at tilt=45 aerial)")
    fig.suptitle('Why the gap is NOT the right side — it is the downward-facing undersides\n' + msg,
                 color=C['txt'], fontsize=13, fontweight='bold')
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    print(f'✓ 저장: {OUT}')
    print(f'  covered {cls_cov.sum()} / uncovered-obs {cls_uncov.sum()} / underside {cls_under.sum()}')


if __name__ == '__main__':
    main()
