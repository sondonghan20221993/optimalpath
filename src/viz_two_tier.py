"""
viz_two_tier.py — 2단계 복원 전략 시각화
  Stage1: 고도 7m 광역 궤도 → 지형(terrain) + 물체 '개략' 복원
  Stage2: 고도 3m 근접 궤도 → 물체 '세부' 복원
지형 = 물체 주변 지면 패치(z=0). 물체 = FBX 8000점.
"""
import sys, types, math
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
import pbnbv_path as A

ROOT = HERE.parent
OUT  = ROOT / "results" / "two_tier_strategy.png"
MAX_DIST = 12.0          # 7m 광역(standoff~10m + 지형 가장자리) 수용
COS = math.cos(math.radians(60))
P.MAX_DIST = MAX_DIST


def main():
    t = P.TARGET
    obj = P._pts_raw
    obj_n = np.load(ROOT/'real_test'/'airsim_gt_pts.npz')['normals'].astype(float)
    top = obj[:, 2].min()

    # ── 지형(지면) 패치: 물체 주변 6×6m, z=0(지면), 법선 위(=NED -z) ──
    gx = np.linspace(t[0]-3.5, t[0]+3.5, 70)
    gy = np.linspace(t[1]-3.5, t[1]+3.5, 70)
    GX, GY = np.meshgrid(gx, gy)
    ground = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
    # 물체가 점유한 영역의 지면점은 제거(겹침 방지)
    keep = ~((ground[:,0]>obj[:,0].min()-0.05)&(ground[:,0]<obj[:,0].max()+0.05)&
             (ground[:,1]>obj[:,1].min()-0.05)&(ground[:,1]<obj[:,1].max()+0.05))
    ground = ground[keep]
    ground_n = np.tile([0,0,-1.0], (len(ground),1))   # 지면 법선 = 위

    # 전체 장면
    scene = np.vstack([obj, ground])
    scene_n = np.vstack([obj_n, ground_n])
    is_obj = np.zeros(len(scene), bool); is_obj[:len(obj)] = True

    az = lambda p: math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360

    def reconstruct(alt, n_az, pts, nrm):
        z = round(top - alt, 2)
        orb = A.gen_candidates_tilt45(t, [z], n_az=n_az, max_dist=MAX_DIST)
        orb = np.array(sorted(orb, key=az))
        A.RAYCAST_OCCLUSION = True
        seen = np.zeros(len(pts), int)
        for c in orb:
            live = np.ones(len(pts), bool)
            _, vis = A.information_gain(c, t, pts, live, P.FOV_DEG, MAX_DIST)
            v = c - pts; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
            seen += (vis & ((nrm*v).sum(1) >= COS)).astype(int)
        return orb, seen > 0

    # Stage1: 7m 광역 (지형+물체)
    orb7, rec7 = reconstruct(7.0, 24, scene, scene_n)
    # Stage2: 3m 근접 (물체만 평가, 같은 장면이되 물체 세부)
    orb3, rec3 = reconstruct(3.0, 36, scene, scene_n)

    C = {'txt': '#e6edf3', 'ax': '#0d1117', 'bg': '#0d1117',
         'obj': '#3fb950', 'gnd': '#7d8590', 'miss': '#f85149',
         'cam7': '#58a6ff', 'cam3': '#d2a8ff'}

    X, Y, Z = scene[:,0], scene[:,1], -scene[:,2]
    dx, dy = np.ptp(X), np.ptp(Y); dz = 0.36

    fig = plt.figure(figsize=(18, 8.5), facecolor=C['bg'])

    def panel(pos, orb, rec, alt, n_az, title, camc):
        ax = fig.add_subplot(pos, projection='3d', facecolor=C['ax'])
        # 지형: 복원/미복원
        g = ~is_obj
        ax.scatter(X[g & rec], Y[g & rec], Z[g & rec], c=C['gnd'], s=1.5, alpha=0.5)
        ax.scatter(X[g & ~rec], Y[g & ~rec], Z[g & ~rec], c=C['miss'], s=1.5, alpha=0.4)
        # 물체: 복원/미복원
        o = is_obj
        ax.scatter(X[o & rec], Y[o & rec], Z[o & rec], c=C['obj'], s=3)
        ax.scatter(X[o & ~rec], Y[o & ~rec], Z[o & ~rec], c=C['miss'], s=3)
        # 카메라 + 시선
        ocz = -orb[:,2]
        ax.scatter(orb[:,0], orb[:,1], ocz, c=camc, s=22, marker='D')
        for c in orb:
            d = (t - c); d = d/np.linalg.norm(d)*1.0
            ax.plot([c[0], c[0]+d[0]], [c[1], c[1]+d[1]], [-c[2], -(c[2]+d[2])],
                    c='#ffa657', lw=0.6, alpha=0.7)
        ax.set_box_aspect((dx, dy, max(dz, (ocz.max()))))
        ro = np.linalg.norm(orb[0]-t)
        oc = 100*rec[o].sum()/o.sum(); gc = 100*rec[g].sum()/g.sum()
        ax.set_title(f'{title}\n{n_az} cams @ {alt:.0f}m (standoff {ro:.1f}m)\n'
                     f'object {oc:.0f}%  |  terrain {gc:.0f}%',
                     color=C['txt'], fontsize=11)
        ax.view_init(elev=22, azim=-58)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis): a.pane.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=5)
        return oc, gc

    o7, g7 = panel(121, orb7, rec7, 7.0, 24, 'Stage 1 — 7m wide: terrain overview', C['cam7'])
    o3, g3 = panel(122, orb3, rec3, 3.0, 36, 'Stage 2 — 3m close: object detail', C['cam3'])

    fig.suptitle('Two-tier strategy — Stage1 (7m) maps terrain & object coarsely, '
                 'Stage2 (3m) reconstructs object in detail\n'
                 'green=object reconstructed, gray=terrain reconstructed, red=occluded',
                 color=C['txt'], fontsize=13)
    fig.savefig(OUT, dpi=140, facecolor=C['bg'], bbox_inches='tight')
    print(f'✓ {OUT}')
    print(f'  7m: object {o7:.1f}% / terrain {g7:.1f}%')
    print(f'  3m: object {o3:.1f}% / terrain {g3:.1f}%')


if __name__ == '__main__':
    main()
