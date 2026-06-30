"""
viz_two_tier_nbv_html.py — 2단계 PB-NBV 복원을 인터랙티브 3D HTML(panel별)로.
viz_two_tier_nbv.png의 3개 패널을 각각 회전/줌 가능한 Plotly HTML로 출력.
  recon_tier_7m.html      : NBV Stage1 @7m (지형+물체 개략)
  recon_tier_7m3m.html    : NBV Stage1+2 (7m→3m, 물체 세부)
  recon_tier_orbit.html   : Orbit 36 @3m (baseline)
복원 = raycast 가시 ∩ 입사각<60°. 색: 물체 복원=초록/미복원=빨강,
지형 복원=회색/미복원=옅은빨강. 카메라 다이아 + 시선 화살표 + 경로선.
"""
import sys, types, math
from pathlib import Path
import numpy as np
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
    A.RAYCAST_OCCLUSION = True
    out = []
    for c in cands:
        live = np.ones(len(pts), bool)
        _, vis = A.information_gain(c, P.TARGET, pts, live, P.FOV_DEG, MAX_DIST)
        v = c - pts; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
        out.append(vis & ((nrm*v).sum(1) >= COS))
    return out


def nbv_greedy(cands, masks, start, target=None, max_wp=60):
    n = len(masks[0]); rec = np.zeros(n, bool)
    if target is None: target = np.ones(n, bool)
    used = np.zeros(len(cands), bool); sel = []
    cur = start.copy()
    for _ in range(max_wp):
        best, bU, bgain = -1, -1, 0
        for i, c in enumerate(cands):
            if used[i]: continue
            gain = int((masks[i] & ~rec & target).sum())
            if gain == 0: continue
            d = np.linalg.norm(c - cur) + 1e-6
            U = gain / d
            if U > bU: bU, best, bgain = U, i, gain
        if best < 0 or bgain == 0: break
        rec |= masks[best]; used[best] = True
        cur = cands[best].copy(); sel.append(cands[best])
    return rec, np.array(sel)


def xyz(a):
    a = np.atleast_2d(a)
    return a[:, 0], a[:, 1], -a[:, 2]


def make_html(scene, is_obj, rec, wps, title, out, path_loop=False, cam_color='#58a6ff'):
    o, g = is_obj, ~is_obj
    t = P.TARGET
    no_rec, ng_rec = int(rec[o].sum()), int(rec[g].sum())
    no, ng = int(o.sum()), int(g.sum())
    op, gp = 100*no_rec/no, 100*ng_rec/ng

    fig = go.Figure()
    # 물체
    ox, oy, oz = xyz(scene[o & rec]); fig.add_trace(go.Scatter3d(
        x=ox, y=oy, z=oz, mode='markers', marker=dict(size=3.2, color='#3fb950'),
        name=f'물체 복원 ({no_rec})'))
    if (o & ~rec).any():
        mx, my, mz = xyz(scene[o & ~rec]); fig.add_trace(go.Scatter3d(
            x=mx, y=my, z=mz, mode='markers', marker=dict(size=3.2, color='#f85149'),
            name=f'물체 미복원 ({no-no_rec})'))
    # 지형
    gx, gy, gz = xyz(scene[g & rec]); fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode='markers', marker=dict(size=1.6, color='#7d8590', opacity=0.5),
        name=f'지형 복원 ({ng_rec})'))
    if (g & ~rec).any():
        ux, uy, uz = xyz(scene[g & ~rec]); fig.add_trace(go.Scatter3d(
            x=ux, y=uy, z=uz, mode='markers', marker=dict(size=1.6, color='#f85149', opacity=0.35),
            name=f'지형 미복원 ({ng-ng_rec})'))
    # 카메라
    if len(wps):
        cx, cy, cz = xyz(wps); fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode='markers',
            marker=dict(size=4.5, color=cam_color, symbol='diamond', line=dict(color='white', width=1)),
            name=f'cameras ({len(wps)})'))
        # 시선 화살표
        dirs = t - wps; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)+1e-9
        End = wps + dirs*1.0
        sx, sy, sz = xyz(wps); ex, ey, ez = xyz(End)
        lx, ly, lz = [], [], []
        for i in range(len(wps)):
            lx += [sx[i], ex[i], None]; ly += [sy[i], ey[i], None]; lz += [sz[i], ez[i], None]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
            line=dict(color='#ffa657', width=3), name='view dir'))
        fig.add_trace(go.Cone(x=ex, y=ey, z=ez, u=ex-sx, v=ey-sy, w=ez-sz,
            sizemode='absolute', sizeref=0.35, anchor='tip', showscale=False,
            colorscale=[[0,'#ffa657'],[1,'#ffa657']], hoverinfo='skip'))
        # 경로선: 그리디 선택 순서가 아니라 실제 비행 궤도 순서로 정렬해 표시
        #   (고도가 높은 링 먼저 = z 작을수록 위 NED, 각 링 안에서는 방위각 순)
        seq = sorted(range(len(wps)), key=lambda i: (round(float(wps[i][2]), 2),
                     math.atan2(wps[i][1]-t[1], wps[i][0]-t[0])))
        ordered = wps[seq]
        rr = np.vstack([ordered, ordered[0]]) if path_loop else ordered
        px, py, pz = xyz(rr)
        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines',
            line=dict(color=cam_color, width=2), name='flight path'))
    # 타겟
    tx, ty, tz = xyz(t)
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='markers',
        marker=dict(size=6, color='#f778ba', symbol='diamond'), name='target'))

    fig.update_layout(
        title=dict(text=f'{title}<br><sub>{len(wps)} WP | 물체 {op:.0f}% ({no_rec}/{no}) · '
                        f'지형 {gp:.0f}% ({ng_rec}/{ng}) | raycast∩입사각<60° | up=-Z(NED)</sub>',
                   font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='up (m)',
                   xaxis=dict(backgroundcolor='#161b22', color='#e6edf3', gridcolor='#21262d'),
                   yaxis=dict(backgroundcolor='#161b22', color='#e6edf3', gridcolor='#21262d'),
                   zaxis=dict(backgroundcolor='#161b22', color='#e6edf3', gridcolor='#21262d'),
                   aspectmode='data'),
        legend=dict(bgcolor='#161b22', bordercolor='#21262d', borderwidth=1))
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f'✓ {out.name}  {len(wps)}WP 물체{op:.1f}% 지형{gp:.1f}%')


def main():
    t = P.TARGET
    scene, scene_n, is_obj = build_scene()
    top = P._pts_raw[:, 2].min()
    az = lambda p: math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360
    start = np.array([t[0]+9, t[1], top-7])

    def run(alt):
        cands = A.gen_candidates_tilt45(t, [round(top-alt,2)], n_az=36, max_dist=MAX_DIST)
        cands = np.array(sorted(cands, key=az))
        return cands, good_masks(cands, scene, scene_n)

    c7, m7 = run(7.0)
    c3, m3 = run(3.0)

    # ── 고도별 단독 coverage (그 링을 한 바퀴 다 돌았을 때의 합집합) ──
    def union(masks):
        u = np.zeros(len(scene), bool)
        for mm in masks: u |= mm
        return u
    o, g = is_obj, ~is_obj
    pct = lambda r: (100*r[o].sum()/o.sum(), 100*r[g].sum()/g.sum())
    rec_7only = union(m7)
    rec_3only = union(m3)
    rec_both  = rec_7only | rec_3only
    o7, gg7 = pct(rec_7only); o3, gg3 = pct(rec_3only); ob, gb = pct(rec_both)
    print('── 고도별 단독 coverage (링 한 바퀴 합집합) ──')
    print(f'  7m only   : 물체 {o7:5.1f}%  | 지형 {gg7:5.1f}%')
    print(f'  3m only   : 물체 {o3:5.1f}%  | 지형 {gg3:5.1f}%')
    print(f'  7m + 3m   : 물체 {ob:5.1f}%  | 지형 {gb:5.1f}%')
    print(f'  3m가 7m 대비 물체 추가 복원분: +{ob-o7:.1f}%p')

    # Stage1 @7m (NBV)
    rec7, wp7 = nbv_greedy(c7, m7, start, target=None)
    make_html(scene, is_obj, rec7, wp7, 'PB-NBV Stage1 @7m — 지형 광역 복원',
              ROOT/'results'/'recon_tier_7m.html', cam_color='#58a6ff')

    # Stage2 @3m 단독 (물체 세부) — 3m 링만 비행했을 때
    rec3, wp3o = nbv_greedy(c3, m3, start, target=None)
    make_html(scene, is_obj, rec3, wp3o, 'PB-NBV Stage2 @3m 단독 — 물체 세부 복원',
              ROOT/'results'/'recon_tier_3m.html', cam_color='#d2a8ff')

    # Stage1+2 누적 (7m → 3m, 3m는 물체 타깃)
    start3 = wp7[-1] if len(wp7) else start
    _, wp3 = nbv_greedy(c3, m3, start3, target=is_obj)
    rec_final = rec7.copy()
    for c in wp3:
        i = int(np.argmin(np.linalg.norm(c3 - c, axis=1)))
        rec_final |= m3[i]
    wp_all = np.vstack([wp7, wp3]) if len(wp3) else wp7
    make_html(scene, is_obj, rec_final, wp_all, 'PB-NBV Stage1+2 (7m→3m) — 물체 세부 복원',
              ROOT/'results'/'recon_tier_7m3m.html', cam_color='#d2a8ff')

    # Orbit baseline 36 @3m
    orb = np.array(sorted(A.gen_candidates_tilt45(t,[round(top-3,2)],n_az=36,max_dist=MAX_DIST), key=az))
    om = good_masks(orb, scene, scene_n)
    rec_orb = union(om)
    make_html(scene, is_obj, rec_orb, orb, 'Orbit 36 @3m (baseline)',
              ROOT/'results'/'recon_tier_orbit.html', path_loop=True, cam_color='#d2a8ff')


if __name__ == '__main__':
    main()
