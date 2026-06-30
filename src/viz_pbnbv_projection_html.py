"""
viz_pbnbv_projection_html.py — projection 기반(실제 PB-NBV) 검증을 인터랙티브 HTML로.
객체(관측가능/밑면), 균등·비균등 비행 포즈(토글), 결론(추가 시점 0개) 표시.
출력: results/pbnbv_projection.html
"""
import sys, glob, json, math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_projection_uniform as M   # build_ellipsoids, project_visible, 상수 재사용
import pbnbv_path as A

ROOT = M.ROOT; t = M.t; obj = M.obj; top = M.top
az = M.az


def load(d):
    ps = []
    for f in sorted(glob.glob(str(d/'meta'/'*.json'))):
        p = json.load(open(f))['camera']['pose']['position']; ps.append([p['x'], p['y'], p['z']])
    return np.array(ps)


def xyz(a):
    a = np.atleast_2d(a); return a[:, 0], a[:, 1], -a[:, 2]


def cam_trace(cams, color, name, show):
    """고도별 링으로 분리: 같은 고도 안에서만 방위각 순으로 연결(원형 가로지르기 방지)."""
    tr = []
    alt = np.round(-cams[:, 2]).astype(int)          # 고도(m) 라벨
    tiers = sorted(set(alt.tolist()))
    shades = {tiers[i]: c for i, c in enumerate(
        ([color] if len(tiers) == 1 else _shade(color, len(tiers))))}
    first = True
    for a in tiers:
        ring = cams[alt == a]
        ang = np.array([math.atan2(p[1]-t[1], p[0]-t[0]) for p in ring])
        order = np.argsort(ang)                       # 방위각 순
        ring = ring[order]
        rx, ry, rz = xyz(ring)
        # 링은 닫아서 한 바퀴(처음점 다시) — 같은 고도 안에서만
        cx = np.append(rx, rx[0]); cy = np.append(ry, ry[0]); cz = np.append(rz, rz[0])
        col = shades[a]
        tr.append(go.Scatter3d(x=cx, y=cy, z=cz, mode='markers+lines',
                  marker=dict(size=4, color=col, symbol='diamond', line=dict(color='white', width=1)),
                  line=dict(color=col, width=2, dash='solid'),
                  name=f'{name} · {a}m ring', visible=show, legendgroup=name,
                  showlegend=True))
        # 시선
        d = t - ring; d = d/(np.linalg.norm(d, axis=1, keepdims=True)+1e-9); E = ring + d
        sx, sy, sz = xyz(ring); ex, ey, ez = xyz(E)
        lx, ly, lz = [], [], []
        for i in range(len(ring)): lx += [sx[i], ex[i], None]; ly += [sy[i], ey[i], None]; lz += [sz[i], ez[i], None]
        tr.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color=col, width=1.2),
                  name=name+' view', visible=show, legendgroup=name, showlegend=False, opacity=0.45))
        first = False
    return tr


def _shade(hexc, n):
    """기준색을 밝기 다르게 n단계."""
    hexc = hexc.lstrip('#'); base = np.array([int(hexc[i:i+2], 16) for i in (0, 2, 4)], float)
    out = []
    for k in range(n):
        f = 0.65 + 0.55*k/max(1, n-1)                # 어두→밝
        c = np.clip(base*f, 0, 255).astype(int)
        out.append('#%02x%02x%02x' % tuple(c))
    return out


def main():
    cen, nrm, rad, pts_idx, npts = M.build_ellipsoids(400)
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=M.MAX_DIST)
                       for a in [3., 5., 7.]])
    dense_vis = [M.project_visible(c, cen, nrm, rad) for c in dense]
    observable = np.zeros(len(cen), bool)
    for m in dense_vis: observable |= m

    def pmask(ellmask):
        m = np.zeros(len(obj), bool)
        for k in np.where(ellmask)[0]: m[pts_idx[k]] = True
        return m
    obs_pts = pmask(observable)

    U = np.vstack([load(M.UNI/'real_test_3m_uniform'), load(M.UNI/'real_test_7m_uniform')])
    N = np.vstack([load(M.NEW/'real_test_3m'), load(M.NEW/'real_test_7m')])

    def cov(cams):
        s = np.zeros(len(cen), bool)
        for c in cams: s |= M.project_visible(c, cen, nrm, rad)
        s &= observable
        return s, 100*npts[s].sum()/npts[observable].sum()
    seenU, covU = cov(U); seenN, covN = cov(N)

    fig = go.Figure()
    # 객체: 관측가능(녹) / 밑면·불가(회)
    gx, gy, gz = xyz(obj[obs_pts])
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode='markers',
        marker=dict(size=2.6, color='#3fb950'), name=f'관측가능 표면 ({int(obs_pts.sum())})'))
    ux, uy, uz = xyz(obj[~obs_pts])
    fig.add_trace(go.Scatter3d(x=ux, y=uy, z=uz, mode='markers',
        marker=dict(size=2.2, color='#6e7681', opacity=0.45), name=f'밑면·관측불가 ({int((~obs_pts).sum())})'))
    # 비행 포즈
    for tr in cam_trace(U, '#58a6ff', '균등(uniform) 포즈', True): fig.add_trace(tr)
    for tr in cam_trace(N, '#f778ba', '비균등 포즈', 'legendonly'): fig.add_trace(tr)
    tx, ty, tz = xyz(t)
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='markers',
        marker=dict(size=6, color='#f778ba', symbol='diamond'), name='target'))

    note = ("<b>실제 PB-NBV (projection 기반, raycast 아님)</b><br>"
            "표면을 ellipsoid 단위로 투영, 가림은 2D 투영겹침으로 처리<br>"
            "→ raycast 해상도 순환성 제거<br><br>"
            "<b>추가 시점 (실측 궤도 이후):</b><br>"
            f"  균등  : {covU:.0f}% 도달 → <b>+0개</b><br>"
            f"  비균등: {covN:.0f}% 도달 → <b>+0개</b><br>"
            "  (K=90·200·400·800 모두 +0, 강건)<br><br>"
            "<b>대조 — raycast(점 72방위):</b> +20~22개<br>"
            "  = 17방위 비행 ≠ 72방위 정의 해상도격차 인공물")
    fig.add_annotation(text=note, xref='paper', yref='paper', x=0.01, y=0.99,
                       align='left', showarrow=False, font=dict(color='#e6edf3', size=11.5),
                       bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d', borderwidth=1, borderpad=8)

    fig.update_layout(
        title=dict(text='실제 PB-NBV (projection) 검증 — 실측 궤도 이후 <b>추가 경로 0개</b>'
                        '<br><sub>객체=ㄷ자 슬래브 · 범례에서 균등/비균등 포즈 토글 · up=−Z(NED)</sub>',
                   font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(
                   xaxis=dict(visible=False, showbackground=False, showgrid=False,
                              showticklabels=False, title='', zeroline=False),
                   yaxis=dict(visible=False, showbackground=False, showgrid=False,
                              showticklabels=False, title='', zeroline=False),
                   zaxis=dict(visible=False, showbackground=False, showgrid=False,
                              showticklabels=False, title='', zeroline=False),
                   aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        legend=dict(bgcolor='#161b22', bordercolor='#21262d', borderwidth=1, x=0.99, xanchor='right'))
    out = ROOT/'results'/'pbnbv_projection.html'
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f"✓ {out}  (균등 {covU:.0f}%/+0, 비균등 {covN:.0f}%/+0)")


if __name__ == '__main__':
    main()
