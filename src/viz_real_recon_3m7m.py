"""
viz_real_recon_3m7m.py — 새 실측 비행(real_test_new: 3m 17컷 + 7m 17컷)으로 물체 복원 평가·시각화.
3m 비행 / 7m 비행 / 결합 각각의 물체·지형 coverage.
복원 = raycast 가시 ∩ 입사각<60°.  (7m standoff~12.4m → MAX_DIST=13)
출력: results/real37_recon.png, real37_object.png, real37_object.html
"""
import sys, types, json, glob, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P
import pbnbv_path as A

ROOT = HERE.parent
NEW  = ROOT / 'real_test_new'
MAX_DIST = 13.0
COS = math.cos(math.radians(60))
P.MAX_DIST = MAX_DIST


def quat_forward(q):
    w, x, y, z = q
    return np.array([1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)])


def load_poses(d):
    pos, quat = [], []
    for f in sorted(glob.glob(str(d/'meta'/'*.json'))):
        j = json.load(open(f)); p = j['camera']['pose']['position']; o = j['camera']['pose']['orientation']
        pos.append([p['x'], p['y'], p['z']]); quat.append([o['w'], o['x'], o['y'], o['z']])
    return np.array(pos), np.array(quat)


def build_scene():
    t = P.TARGET
    obj = P._pts_raw
    obj_n = np.load(ROOT/'real_test'/'airsim_gt_pts.npz')['normals'].astype(float)
    gx = np.linspace(t[0]-3.5, t[0]+3.5, 70); gy = np.linspace(t[1]-3.5, t[1]+3.5, 70)
    GX, GY = np.meshgrid(gx, gy)
    g = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
    keep = ~((g[:,0]>obj[:,0].min()-0.05)&(g[:,0]<obj[:,0].max()+0.05)&
             (g[:,1]>obj[:,1].min()-0.05)&(g[:,1]<obj[:,1].max()+0.05))
    g = g[keep]; g_n = np.tile([0,0,-1.0],(len(g),1))
    scene = np.vstack([obj, g]); scene_n = np.vstack([obj_n, g_n])
    is_obj = np.zeros(len(scene), bool); is_obj[:len(obj)] = True
    return scene, scene_n, is_obj


def recon_mask(cams, scene, scene_n):
    A.RAYCAST_OCCLUSION = True
    rec = np.zeros(len(scene), bool)
    for c in cams:
        live = np.ones(len(scene), bool)
        _, vis = A.information_gain(c, P.TARGET, scene, live, P.FOV_DEG, MAX_DIST)
        v = c - scene; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
        rec |= vis & ((scene_n*v).sum(1) >= COS)
    return rec


def xyz(a):
    a = np.atleast_2d(a); return a[:,0], a[:,1], -a[:,2]


def make_html(scene, is_obj, rec, cams, title, out, cam_color='#58a6ff'):
    """장면+카메라+시선+경로 포함 인터랙티브 HTML (recon_tier_* 스타일)."""
    o, g = is_obj, ~is_obj; t = P.TARGET
    no_r, ng_r = int(rec[o].sum()), int(rec[g].sum()); no, ng = int(o.sum()), int(g.sum())
    op, gp = 100*no_r/no, 100*ng_r/ng
    fig = go.Figure()
    ox,oy,oz = xyz(scene[o&rec]); fig.add_trace(go.Scatter3d(x=ox,y=oy,z=oz,mode='markers',
        marker=dict(size=3.2,color='#3fb950'), name=f'물체 복원 ({no_r})'))
    if (o&~rec).any():
        mx,my,mz = xyz(scene[o&~rec]); fig.add_trace(go.Scatter3d(x=mx,y=my,z=mz,mode='markers',
            marker=dict(size=3.2,color='#f85149'), name=f'물체 미복원 ({no-no_r})'))
    gx2,gy2,gz2 = xyz(scene[g&rec]); fig.add_trace(go.Scatter3d(x=gx2,y=gy2,z=gz2,mode='markers',
        marker=dict(size=1.6,color='#7d8590',opacity=0.5), name=f'지형 복원 ({ng_r})'))
    if (g&~rec).any():
        ux,uy,uz = xyz(scene[g&~rec]); fig.add_trace(go.Scatter3d(x=ux,y=uy,z=uz,mode='markers',
            marker=dict(size=1.6,color='#f85149',opacity=0.35), name=f'지형 미복원 ({ng-ng_r})'))
    cx,cy,cz = xyz(cams); fig.add_trace(go.Scatter3d(x=cx,y=cy,z=cz,mode='markers+lines',
        marker=dict(size=4,color=cam_color,symbol='diamond',line=dict(color='white',width=1)),
        line=dict(color=cam_color,width=2), name=f'카메라 ({len(cams)})'))
    dirs=t-cams; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)+1e-9; End=cams+dirs
    sx,sy,sz=xyz(cams); ex,ey,ez=xyz(End); lx,ly,lz=[],[],[]
    for i in range(len(cams)): lx+=[sx[i],ex[i],None]; ly+=[sy[i],ey[i],None]; lz+=[sz[i],ez[i],None]
    fig.add_trace(go.Scatter3d(x=lx,y=ly,z=lz,mode='lines',line=dict(color='#ffa657',width=2),name='view dir'))
    tx,ty,tz=xyz(t); fig.add_trace(go.Scatter3d(x=tx,y=ty,z=tz,mode='markers',
        marker=dict(size=6,color='#f778ba',symbol='diamond'),name='target'))
    fig.update_layout(
        title=dict(text=f'{title}<br><sub>{len(cams)} shots · 물체 {op:.1f}% ({no_r}/{no}) · '
            f'지형 {gp:.1f}% · raycast∩입사각<60° · up=-Z(NED)</sub>', font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='up (m)',
                   xaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   yaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   zaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   aspectmode='data'),
        legend=dict(bgcolor='#161b22',bordercolor='#21262d',borderwidth=1))
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f'✓ {out.name}  {len(cams)}컷 물체{op:.1f}% 지형{gp:.1f}%')


def main():
    pos3, q3 = load_poses(NEW/'real_test_3m')
    pos7, q7 = load_poses(NEW/'real_test_7m')
    posA = np.vstack([pos3, pos7])
    scene, scene_n, is_obj = build_scene()
    o, g = is_obj, ~is_obj
    t = P.TARGET

    # orientation vs look-at 검증
    def chk(pos, q):
        ds=[]
        for c, qq in zip(pos, q):
            fw=quat_forward(qq); fw/=np.linalg.norm(fw)+1e-9
            la=t-c; la/=np.linalg.norm(la)+1e-9; ds.append(float(fw@la))
        return np.mean(ds)
    print(f"[검증] look-at 일치 코사인: 3m={chk(pos3,q3):.2f}, 7m={chk(pos7,q7):.2f}")
    print(f"  고도: 3m={(-pos3[:,2]).mean():.2f}m  7m={(-pos7[:,2]).mean():.2f}m")

    pct = lambda r: (100*r[o].sum()/o.sum(), 100*r[g].sum()/g.sum())
    rec3 = recon_mask(pos3, scene, scene_n)
    rec7 = recon_mask(pos7, scene, scene_n)
    recA = recon_mask(posA, scene, scene_n)
    o3,g3 = pct(rec3); o7,g7 = pct(rec7); oA,gA = pct(recA)
    print('── 새 실측 비행 복원율 (raycast ∩ 입사각<60°) ──')
    print(f'  3m 단독 (17컷): 물체 {o3:5.1f}% | 지형 {g3:5.1f}%')
    print(f'  7m 단독 (17컷): 물체 {o7:5.1f}% | 지형 {g7:5.1f}%')
    print(f'  3m+7m   (34컷): 물체 {oA:5.1f}% | 지형 {gA:5.1f}%')
    print(f'  7m가 3m 대비 물체 추가분: {oA-o3:+.1f}%p')

    C = {'txt':'#e6edf3','ax':'#0d1117','bg':'#0d1117','obj':'#3fb950',
         'gnd':'#7d8590','miss':'#f85149','c3':'#d2a8ff','c7':'#58a6ff'}
    X,Y,Z = scene[:,0], scene[:,1], -scene[:,2]
    dx,dy = np.ptp(X), np.ptp(Y)

    # ── 비행 맥락 3패널 ──
    fig = plt.figure(figsize=(19,8.5), facecolor=C['bg'])
    def panel(posn, cams, rec, camc, ttl, op, gp):
        ax = fig.add_subplot(posn, projection='3d', facecolor=C['ax'])
        ax.scatter(X[g&rec],Y[g&rec],Z[g&rec],c=C['gnd'],s=1.4,alpha=0.5)
        ax.scatter(X[g&~rec],Y[g&~rec],Z[g&~rec],c=C['miss'],s=1.2,alpha=0.3)
        ax.scatter(X[o&rec],Y[o&rec],Z[o&rec],c=C['obj'],s=3)
        ax.scatter(X[o&~rec],Y[o&~rec],Z[o&~rec],c=C['miss'],s=3)
        cx,cy,cz = xyz(cams)
        ax.scatter(cx,cy,cz,c=camc,s=26,marker='D',edgecolors='white',linewidths=0.3)
        for c in cams:
            d=(t-c); d=d/np.linalg.norm(d)*1.0
            ax.plot([c[0],c[0]+d[0]],[c[1],c[1]+d[1]],[-c[2],-(c[2]+d[2])],c='#ffa657',lw=0.5,alpha=0.6)
        ax.set_box_aspect((dx,dy,max(0.36,cz.max())))
        ax.set_title(f'{ttl}\nobject {op:.0f}%  |  terrain {gp:.0f}%',color=C['txt'],fontsize=11)
        ax.view_init(elev=22,azim=-58)
        for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.pane.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e',labelsize=5)
    panel(131, pos3, rec3, C['c3'], f'Real 3m ({len(pos3)} shots)', o3, g3)
    panel(132, pos7, rec7, C['c7'], f'Real 7m ({len(pos7)} shots)', o7, g7)
    panel(133, posA, recA, C['c7'], f'Real 3m+7m ({len(posA)} shots)', oA, gA)
    fig.suptitle('Reconstruction from the ACTUAL flights (real 3m + 7m, 17+17 frames) — '
                 f'combined object {oA:.0f}% / terrain {gA:.0f}%  (green=reconstructed, red=missed/underside)',
                 color=C['txt'], fontsize=12.5)
    fig.savefig(ROOT/'results'/'real37_recon.png', dpi=140, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {ROOT/'results'/'real37_recon.png'}")

    # ── 물체 close-up (결합) ──
    obj = P._pts_raw; ro = recA[o]
    OX, OY, OZ = obj[:,0], obj[:,1], -obj[:,2]
    figc = plt.figure(figsize=(15,7), facecolor=C['bg'])
    def closeup(posn, elev, azim, ttl):
        ax = figc.add_subplot(posn, projection='3d', facecolor=C['ax'])
        ax.scatter(OX[ro],OY[ro],OZ[ro],c=C['obj'],s=9,label=f'reconstructed ({int(ro.sum())})')
        ax.scatter(OX[~ro],OY[~ro],OZ[~ro],c=C['miss'],s=9,label=f'missed/underside ({int((~ro).sum())})')
        ax.set_box_aspect((np.ptp(OX),np.ptp(OY),np.ptp(OZ) or 0.36))
        ax.view_init(elev=elev,azim=azim); ax.set_title(ttl,color=C['txt'],fontsize=11)
        for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.pane.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e',labelsize=6)
        ax.legend(fontsize=8,facecolor='#21262d',edgecolor='#21262d',labelcolor=C['txt'],loc='upper right')
    closeup(121,  35, -55, 'Object close-up (top oblique)')
    closeup(122, -35, -55, 'Object close-up (underside) - red = ground-facing bottom')
    figc.suptitle(f'Object reconstruction from real 3m+7m flights (34 frames) — '
                  f'{oA:.1f}% reconstructed; remaining ~{100-oA:.0f}% = downward undersides',
                  color=C['txt'], fontsize=12.5)
    figc.savefig(ROOT/'results'/'real37_object.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {ROOT/'results'/'real37_object.png'}")

    # ── 물체만 인터랙티브 HTML ──
    bx,by,bz = xyz(obj[ro]); rx,ry,rz = xyz(obj[~ro])
    figo = go.Figure()
    figo.add_trace(go.Scatter3d(x=bx,y=by,z=bz,mode='markers',
        marker=dict(size=2.6,color='#3fb950'), name=f'복원됨 ({int(ro.sum())})'))
    figo.add_trace(go.Scatter3d(x=rx,y=ry,z=rz,mode='markers',
        marker=dict(size=2.6,color='#f85149'), name=f'미복원/밑면 ({int((~ro).sum())})'))
    figo.update_layout(
        title=dict(text=f'실측 3m+7m 비행 — 물체 복원 클로즈업 (34프레임)<br>'
            f'<sub>물체 {oA:.1f}% ({int(ro.sum())}/{int(o.sum())}) · 3m단독 {o3:.0f}% · 7m단독 {o7:.0f}% · 빨강=밑면</sub>',
            font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='up (m)',
                   xaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   yaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   zaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   aspectmode='data', camera=dict(eye=dict(x=1.4,y=1.4,z=1.0))),
        legend=dict(bgcolor='#161b22',bordercolor='#21262d',borderwidth=1))
    figo.write_html(str(ROOT/'results'/'real37_object.html'), include_plotlyjs='cdn')
    print(f"✓ {ROOT/'results'/'real37_object.html'}")

    # ── 비행별 인터랙티브 HTML (카메라·경로 포함) ──
    make_html(scene, is_obj, rec3, pos3, '실측 3m 비행 — 물체 세부 복원',
              ROOT/'results'/'real37_3m.html', cam_color='#d2a8ff')
    make_html(scene, is_obj, rec7, pos7, '실측 7m 비행 — 지형 광역 복원',
              ROOT/'results'/'real37_7m.html', cam_color='#58a6ff')
    make_html(scene, is_obj, recA, posA, '실측 3m+7m 결합 — 물체+지형',
              ROOT/'results'/'real37_all.html', cam_color='#58a6ff')


if __name__ == '__main__':
    main()
