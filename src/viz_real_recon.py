"""
viz_real_recon.py — 실제 AirSim 비행(real_test/meta 34프레임)으로 물체 복원 평가·시각화.
실측 카메라 위치(비균등·나선, 고도 4.3m+6m)를 그대로 사용.
복원 = raycast 가시 ∩ 입사각<60°. Loop1(4.3m)/Loop2(6m)/전체 비교.
출력: results/real_recon.png  +  results/real_recon.html
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
MAX_DIST = 12.0
COS = math.cos(math.radians(60))
P.MAX_DIST = MAX_DIST


def quat_forward(q):
    """AirSim Hamilton quaternion (w,x,y,z)로 body +X(전방)를 월드로 회전."""
    w, x, y, z = q
    # R @ [1,0,0]  (rotation matrix 첫 열)
    return np.array([1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)])


def load_real():
    rows = []
    for f in sorted(glob.glob(str(ROOT/'real_test'/'meta'/'*.json'))):
        d = json.load(open(f))
        p = d['camera']['pose']['position']; o = d['camera']['pose']['orientation']
        rows.append((int(d['frame_id']),
                     np.array([p['x'], p['y'], p['z']]),
                     np.array([o['w'], o['x'], o['y'], o['z']])))
    rows.sort()
    pos = np.array([r[1] for r in rows]); quat = np.array([r[2] for r in rows])
    return pos, quat


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
    """실측 카메라(위치, look-at target)로 복원되는 점 누적 mask."""
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


def main():
    pos, quat = load_real()
    scene, scene_n, is_obj = build_scene()
    o, g = is_obj, ~is_obj
    t = P.TARGET

    # orientation(실측) vs look-at 일치도 검증
    dots = []
    for c, q in zip(pos, quat):
        fwd = quat_forward(q); fwd /= np.linalg.norm(fwd)+1e-9
        la = t - c; la /= np.linalg.norm(la)+1e-9
        dots.append(float(fwd @ la))
    dots = np.array(dots)
    print(f"[검증] 실측 광축 vs look-at(물체) 코사인: 평균 {dots.mean():.2f}, "
          f"최소 {dots.min():.2f}  → {math.degrees(math.acos(np.clip(dots.mean(),-1,1))):.1f}° 평균 오차")

    # Loop 분리 (고도 4.3m vs 6m): alt = -z
    alt = -pos[:, 2]
    loop1 = pos[alt < 5.0]      # ~4.3m
    loop2 = pos[alt >= 5.0]     # ~5.6~6.1m
    print(f"  Loop1(저고도 {(-loop1[:,2]).mean():.1f}m): {len(loop1)}프레임 | "
          f"Loop2(고고도 {(-loop2[:,2]).mean():.1f}m): {len(loop2)}프레임")

    pct = lambda r: (100*r[o].sum()/o.sum(), 100*r[g].sum()/g.sum())
    rec1 = recon_mask(loop1, scene, scene_n)
    rec2 = recon_mask(loop2, scene, scene_n)
    recA = recon_mask(pos,   scene, scene_n)
    o1,gg1 = pct(rec1); o2,gg2 = pct(rec2); oA,gA = pct(recA)
    print('── 실측 비행 복원율 (raycast ∩ 입사각<60°) ──')
    print(f'  Loop1 4.3m  ({len(loop1)}컷): 물체 {o1:5.1f}% | 지형 {gg1:5.1f}%')
    print(f'  Loop2 6m    ({len(loop2)}컷): 물체 {o2:5.1f}% | 지형 {gg2:5.1f}%')
    print(f'  전체 34컷           : 물체 {oA:5.1f}% | 지형 {gA:5.1f}%')

    # ── PNG (3패널 oblique) ──
    C = {'txt':'#e6edf3','ax':'#0d1117','bg':'#0d1117','obj':'#3fb950',
         'gnd':'#7d8590','miss':'#f85149','c1':'#58a6ff','c2':'#d2a8ff'}
    X,Y,Z = scene[:,0], scene[:,1], -scene[:,2]
    dx,dy = np.ptp(X), np.ptp(Y)
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
        zt = max(0.36, cz.max())
        ax.set_box_aspect((dx,dy,zt))
        ax.set_title(f'{ttl}\nobject {op:.0f}%  |  terrain {gp:.0f}%',color=C['txt'],fontsize=11)
        ax.view_init(elev=22,azim=-58)
        for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.pane.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e',labelsize=5)

    panel(131, loop1, rec1, C['c1'], f'Real Loop1 @4.3m ({len(loop1)} shots)', o1, gg1)
    panel(132, loop2, rec2, C['c2'], f'Real Loop2 @6m ({len(loop2)} shots)', o2, gg2)
    panel(133, pos,   recA, C['c2'], f'Real all ({len(pos)} shots)', oA, gA)
    fig.suptitle('Reconstruction from the ACTUAL AirSim flight (real_test, 34 frames) — '
                 f'non-uniform spiral orbit reaches object {oA:.0f}% / terrain {gA:.0f}%  '
                 '(green=reconstructed, red=missed/underside)', color=C['txt'], fontsize=12.5)
    fig.savefig(ROOT/'results'/'real_recon.png', dpi=140, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {ROOT/'results'/'real_recon.png'}")

    # ── 물체 close-up (전체 34컷 복원) : 윗면 oblique + 밑면 ──
    obj = P._pts_raw
    ro = recA[o]                       # 물체 점들의 복원 여부
    OX, OY, OZ = obj[:,0], obj[:,1], -obj[:,2]
    figc = plt.figure(figsize=(15,7), facecolor=C['bg'])
    def closeup(posn, elev, azim, ttl):
        ax = figc.add_subplot(posn, projection='3d', facecolor=C['ax'])
        ax.scatter(OX[ro],OY[ro],OZ[ro],c=C['obj'],s=9,label=f'reconstructed ({int(ro.sum())})')
        ax.scatter(OX[~ro],OY[~ro],OZ[~ro],c=C['miss'],s=9,label=f'missed/underside ({int((~ro).sum())})')
        ax.set_box_aspect((np.ptp(OX),np.ptp(OY),np.ptp(OZ) or 0.36))
        ax.view_init(elev=elev,azim=azim)
        ax.set_title(ttl,color=C['txt'],fontsize=11)
        for a in (ax.xaxis,ax.yaxis,ax.zaxis): a.pane.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e',labelsize=6)
        ax.legend(fontsize=8,facecolor='#21262d',edgecolor='#21262d',labelcolor=C['txt'],loc='upper right')
    closeup(121,  35, -55, 'Object close-up (top oblique)')
    closeup(122, -35, -55, 'Object close-up (underside) - red = ground-facing bottom')
    figc.suptitle(f'Object reconstruction from real flight (34 frames) — '
                  f'{oA:.1f}% reconstructed; remaining ~{100-oA:.0f}% = downward undersides',
                  color=C['txt'], fontsize=12.5)
    figc.savefig(ROOT/'results'/'real_recon_object.png', dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {ROOT/'results'/'real_recon_object.png'}")

    # ── HTML (전체 34컷 인터랙티브) ──
    no_r, ng_r = int(recA[o].sum()), int(recA[g].sum())
    no, ng = int(o.sum()), int(g.sum())
    figh = go.Figure()
    ox,oy,oz = xyz(scene[o&recA]); figh.add_trace(go.Scatter3d(x=ox,y=oy,z=oz,mode='markers',
        marker=dict(size=3.2,color='#3fb950'), name=f'물체 복원 ({no_r})'))
    mx,my,mz = xyz(scene[o&~recA]); figh.add_trace(go.Scatter3d(x=mx,y=my,z=mz,mode='markers',
        marker=dict(size=3.2,color='#f85149'), name=f'물체 미복원 ({no-no_r})'))
    gx2,gy2,gz2 = xyz(scene[g&recA]); figh.add_trace(go.Scatter3d(x=gx2,y=gy2,z=gz2,mode='markers',
        marker=dict(size=1.6,color='#7d8590',opacity=0.5), name=f'지형 복원 ({ng_r})'))
    cx,cy,cz = xyz(pos); figh.add_trace(go.Scatter3d(x=cx,y=cy,z=cz,mode='markers+lines',
        marker=dict(size=4,color='#d2a8ff',symbol='diamond',line=dict(color='white',width=1)),
        line=dict(color='#d2a8ff',width=2), name=f'실측 카메라 ({len(pos)})'))
    dirs = t-pos; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)+1e-9; End=pos+dirs
    sx,sy,sz=xyz(pos); ex,ey,ez=xyz(End); lx,ly,lz=[],[],[]
    for i in range(len(pos)): lx+=[sx[i],ex[i],None]; ly+=[sy[i],ey[i],None]; lz+=[sz[i],ez[i],None]
    figh.add_trace(go.Scatter3d(x=lx,y=ly,z=lz,mode='lines',line=dict(color='#ffa657',width=2),name='view dir'))
    tx,ty,tz=xyz(t); figh.add_trace(go.Scatter3d(x=tx,y=ty,z=tz,mode='markers',
        marker=dict(size=6,color='#f778ba',symbol='diamond'),name='target'))
    figh.update_layout(
        title=dict(text=f'실제 AirSim 비행 복원 (real_test 34프레임)<br>'
            f'<sub>물체 {oA:.1f}% ({no_r}/{no}) · 지형 {gA:.1f}% | 비균등·나선 4.3m+6m | up=-Z(NED)</sub>',
            font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='up (m)',
                   xaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   yaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   zaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   aspectmode='data'),
        legend=dict(bgcolor='#161b22',bordercolor='#21262d',borderwidth=1))
    figh.write_html(str(ROOT/'results'/'real_recon.html'), include_plotlyjs='cdn')
    print(f"✓ {ROOT/'results'/'real_recon.html'}")

    # ── 물체만 꽉 채운 인터랙티브 HTML (카메라 제외 → 물체에 자동 줌) ──
    obj = P._pts_raw; ro = recA[o]
    bx, by, bz = xyz(obj[ro]); rx, ry, rz = xyz(obj[~ro])
    figo = go.Figure()
    figo.add_trace(go.Scatter3d(x=bx,y=by,z=bz,mode='markers',
        marker=dict(size=2.6,color='#3fb950'), name=f'복원됨 ({int(ro.sum())})'))
    figo.add_trace(go.Scatter3d(x=rx,y=ry,z=rz,mode='markers',
        marker=dict(size=2.6,color='#f85149'), name=f'미복원/밑면 ({int((~ro).sum())})'))
    figo.update_layout(
        title=dict(text=f'실측 비행 — 물체 복원 클로즈업 (34프레임)<br>'
            f'<sub>물체 {oA:.1f}% ({int(ro.sum())}/{int(o.sum())}) 복원 · 빨강=지면향 밑면 · up=-Z</sub>',
            font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='up (m)',
                   xaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   yaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   zaxis=dict(backgroundcolor='#161b22',color='#e6edf3',gridcolor='#21262d'),
                   aspectmode='data',
                   camera=dict(eye=dict(x=1.4, y=1.4, z=1.0))),
        legend=dict(bgcolor='#161b22',bordercolor='#21262d',borderwidth=1))
    figo.write_html(str(ROOT/'results'/'real_recon_object.html'), include_plotlyjs='cdn')
    print(f"✓ {ROOT/'results'/'real_recon_object.html'}")


if __name__ == '__main__':
    main()
