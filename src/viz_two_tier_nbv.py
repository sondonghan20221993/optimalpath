"""
viz_two_tier_nbv.py — 2단계 복원 전략을 PB-NBV(정보획득 시점선택)로
  Stage1: 7m 후보 링에서 NBV greedy → 지형+물체 개략
  Stage2: 3m 후보 링에서 NBV greedy → 물체 세부
NBV utility U = (새로 복원되는 점 수) / dist^1.  복원 = raycast가시 ∩ 입사각<60°.
orbit(균등 36대)과 WP수·복원율 비교.
"""
import sys, types, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P
import pbnbv_path as A

ROOT = HERE.parent
OUT  = ROOT / "results" / "two_tier_nbv.png"
MAX_DIST = 12.0
COS = math.cos(math.radians(60))
P.MAX_DIST = MAX_DIST


def build_scene():
    t = P.TARGET
    obj = P._pts_raw
    obj_n = np.load(ROOT/'real_test'/'airsim_gt_pts.npz')['normals'].astype(float)
    gx = np.linspace(t[0]-3.5, t[0]+3.5, 70)
    gy = np.linspace(t[1]-3.5, t[1]+3.5, 70)
    GX, GY = np.meshgrid(gx, gy)
    g = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
    keep = ~((g[:,0]>obj[:,0].min()-0.05)&(g[:,0]<obj[:,0].max()+0.05)&
             (g[:,1]>obj[:,1].min()-0.05)&(g[:,1]<obj[:,1].max()+0.05))
    g = g[keep]; g_n = np.tile([0,0,-1.0],(len(g),1))
    scene = np.vstack([obj, g]); scene_n = np.vstack([obj_n, g_n])
    is_obj = np.zeros(len(scene), bool); is_obj[:len(obj)] = True
    return scene, scene_n, is_obj


def good_masks(cands, pts, nrm):
    """각 후보가 양질 복원하는 점 mask 리스트."""
    A.RAYCAST_OCCLUSION = True
    out = []
    for c in cands:
        live = np.ones(len(pts), bool)
        _, vis = A.information_gain(c, P.TARGET, pts, live, P.FOV_DEG, MAX_DIST)
        v = c - pts; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
        out.append(vis & ((nrm*v).sum(1) >= COS))
    return out


def nbv_greedy(cands, masks, start, target=None, max_wp=40):
    """U=new/dist 그리디. gain은 target 마스크 내 새 복원점만 계산.
       복원 누적 mask(전체)와 선택 WP 반환."""
    n = len(masks[0]); rec = np.zeros(n, bool)
    if target is None: target = np.ones(n, bool)
    used = np.zeros(len(cands), bool); sel = []
    cur = start.copy()
    for _ in range(max_wp):
        best, bU, bgain = -1, -1, 0
        for i, c in enumerate(cands):
            if used[i]: continue
            gain = int((masks[i] & ~rec & target).sum())   # target 내 신규만
            if gain == 0: continue
            d = np.linalg.norm(c - cur) + 1e-6
            U = gain / d
            if U > bU: bU, best, bgain = U, i, gain
        if best < 0 or bgain == 0: break
        rec |= masks[best]; used[best] = True
        cur = cands[best].copy(); sel.append(cands[best])
    return rec, np.array(sel)


def main():
    t = P.TARGET
    scene, scene_n, is_obj = build_scene()
    top = P._pts_raw[:, 2].min()
    az = lambda p: math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360
    start = np.array([t[0]+9, t[1], top-7])     # 광역 시작점

    def run(alt):
        cands = A.gen_candidates_tilt45(t, [round(top-alt,2)], n_az=36, max_dist=MAX_DIST)
        cands = np.array(sorted(cands, key=az))
        masks = good_masks(cands, scene, scene_n)
        return cands, masks

    # Stage1 NBV @7m — 지형 광역 타깃 (지형+물체 전체)
    c7, m7 = run(7.0)
    rec7, wp7 = nbv_greedy(c7, m7, start, target=None)
    # Stage2 NBV @3m — 물체만 타깃 (세부 채움), Stage1 복원 위에서 이어서
    c3, m3 = run(3.0)
    start3 = wp7[-1] if len(wp7) else start
    # Stage2는 물체 복원 gain만 본다 → 적은 WP로 효율적
    rec3_acc = rec7.copy()
    _, wp3 = nbv_greedy(c3, m3, start3, target=is_obj)
    for c in wp3:                       # wp3 누적을 rec7 위에 반영
        i = int(np.argmin(np.linalg.norm(c3 - c, axis=1)))
        rec3_acc |= m3[i]
    rec_final = rec3_acc

    # orbit 비교 (균등 36대 @3m)
    orb = np.array(sorted(A.gen_candidates_tilt45(t,[round(top-3,2)],n_az=36,max_dist=MAX_DIST), key=az))
    om = good_masks(orb, scene, scene_n)
    rec_orb = np.zeros(len(scene), bool)
    for mm in om: rec_orb |= mm

    o = is_obj; g = ~is_obj
    def pct(rec, sel): return 100*rec[o].sum()/o.sum(), 100*rec[g].sum()/g.sum()
    o7p, g7p = pct(rec7, wp7)
    of, gf  = pct(rec_final, None)
    oop, gop = pct(rec_orb, orb)

    C = {'txt':'#e6edf3','ax':'#0d1117','bg':'#0d1117','obj':'#3fb950',
         'gnd':'#7d8590','miss':'#f85149','cam7':'#58a6ff','cam3':'#d2a8ff'}
    X,Y,Z = scene[:,0], scene[:,1], -scene[:,2]
    dx,dy = np.ptp(X), np.ptp(Y)

    fig = plt.figure(figsize=(19,8.5), facecolor=C['bg'])

    def panel(pos, rec, wps, camc, alt, ttl, op, gp, nwp):
        ax = fig.add_subplot(pos, projection='3d', facecolor=C['ax'])
        ax.scatter(X[g&rec],Y[g&rec],Z[g&rec],c=C['gnd'],s=1.5,alpha=0.5)
        ax.scatter(X[g&~rec],Y[g&~rec],Z[g&~rec],c=C['miss'],s=1.2,alpha=0.35)
        ax.scatter(X[o&rec],Y[o&rec],Z[o&rec],c=C['obj'],s=3)
        ax.scatter(X[o&~rec],Y[o&~rec],Z[o&~rec],c=C['miss'],s=3)
        if len(wps):
            ax.scatter(wps[:,0],wps[:,1],-wps[:,2],c=camc,s=30,marker='D')
            for c in wps:
                d=(t-c); d=d/np.linalg.norm(d)*1.0
                ax.plot([c[0],c[0]+d[0]],[c[1],c[1]+d[1]],[-c[2],-(c[2]+d[2])],c='#ffa657',lw=0.6,alpha=0.7)
            zt = max(0.36, -wps[:,2].max())
        else: zt = 0.36
        ax.set_box_aspect((dx,dy,zt))
        ax.set_title(f'{ttl}  ({nwp} WP)\nobject {op:.0f}%  |  terrain {gp:.0f}%',color=C['txt'],fontsize=11)
        ax.view_init(elev=22,azim=-58)
        for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.pane.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e',labelsize=5)

    panel(131, rec7, wp7, C['cam7'], 7, 'NBV Stage1 @7m (terrain)', o7p, g7p, len(wp7))
    panel(132, rec_final, np.vstack([wp7,wp3]) if len(wp3) else wp7, C['cam3'], 3,
          'NBV Stage1+2 (7m→3m)', of, gf, len(wp7)+len(wp3))
    panel(133, rec_orb, orb, C['cam3'], 3, 'Orbit 36 @3m (baseline)', oop, gop, len(orb))

    fig.suptitle('Two-tier reconstruction with PB-NBV — '
                 f'NBV {len(wp7)+len(wp3)} WP reaches object {of:.0f}%/terrain {gf:.0f}%  '
                 f'vs  orbit {len(orb)} cams object {oop:.0f}%/terrain {gop:.0f}%',
                 color=C['txt'], fontsize=12.5)
    fig.savefig(OUT, dpi=140, facecolor=C['bg'], bbox_inches='tight')
    print(f'✓ {OUT}')
    print(f'  NBV 7m: {len(wp7)} WP -> object {o7p:.1f}% / terrain {g7p:.1f}%')
    print(f'  NBV 7m+3m: {len(wp7)+len(wp3)} WP -> object {of:.1f}% / terrain {gf:.1f}%')
    print(f'  Orbit 36 @3m: object {oop:.1f}% / terrain {gop:.1f}%')


if __name__ == '__main__':
    main()
