"""
viz_pbnbv_projection_5m_html.py — 5m 단독 버전 projection PB-NBV HTML.
※ 5m 실측 비행은 없음(캡처는 3m·7m). 5m 고도 균등 링(17컷, tilt45)을 '합성'해 동일 분석.
출력: results/pbnbv_projection_5m.html
"""
import sys, math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_projection_uniform as M
import pbnbv_path as A
from viz_pbnbv_projection_html import load, xyz

ROOT = M.ROOT; t = M.t; obj = M.obj; top = M.top


def smooth_ring(cams, color, name, show):
    """카메라 마커 + 같은 반지름·고도의 매끄러운 원(120점) + 시선."""
    R = np.linalg.norm(cams[:, :2] - t[:2], axis=1).mean()
    zc = cams[:, 2].mean()
    th = np.linspace(0, 2*np.pi, 120)
    cx = t[0] + R*np.cos(th); cy = t[1] + R*np.sin(th); cz = np.full_like(th, -zc)
    tr = [go.Scatter3d(x=cx, y=cy, z=cz, mode='lines',
                       line=dict(color=color, width=5), name=name, visible=show, legendgroup=name)]
    ang = np.array([math.atan2(p[1]-t[1], p[0]-t[0]) for p in cams])
    ring = cams[np.argsort(ang)]
    mx, my, mz = xyz(ring)
    tr.append(go.Scatter3d(x=mx, y=my, z=mz, mode='markers',
              marker=dict(size=5, color='white', symbol='circle',
                          line=dict(color=color, width=2)),
              name=name+' shots', visible=show, legendgroup=name, showlegend=False))
    d = t - ring; d = d/(np.linalg.norm(d, axis=1, keepdims=True)+1e-9); E = ring + d
    sx, sy, sz = xyz(ring); ex, ey, ez = xyz(E)
    lx, ly, lz = [], [], []
    for i in range(len(ring)): lx += [sx[i], ex[i], None]; ly += [sy[i], ey[i], None]; lz += [sz[i], ez[i], None]
    tr.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color=color, width=1.4),
              name=name+' view', visible=show, legendgroup=name, showlegend=False, opacity=0.4))
    return tr


def main():
    cen, nrm, rad, pts_idx, npts = M.build_ellipsoids(400)
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=M.MAX_DIST)
                       for a in [3., 5., 7.]])
    dense_vis = [M.project_visible(c, cen, nrm, rad) for c in dense]
    observable = np.zeros(len(cen), bool)
    for m in dense_vis: observable |= m

    def pmask(em):
        m = np.zeros(len(obj), bool)
        for k in np.where(em)[0]: m[pts_idx[k]] = True
        return m
    obs_pts = pmask(observable)

    # 5m 균등 링 (17컷, tilt45) — 합성
    ring5 = A.gen_candidates_tilt45(t, [round(top-5, 2)], n_az=17, max_dist=M.MAX_DIST)

    def cov_add(cams):
        s = np.zeros(len(cen), bool)
        for c in cams: s |= M.project_visible(c, cen, nrm, rad)
        s &= observable
        c0 = 100*npts[s].sum()/npts[observable].sum()
        # PB-NBV 추가
        used = np.zeros(len(dense), bool); n = 0
        while True:
            rem = observable & ~s
            if rem.sum() == 0: break
            best, bg = -1, 0
            for i in range(len(dense)):
                if used[i]: continue
                g = npts[dense_vis[i] & rem].sum()
                if g > bg: bg, best = g, i
            if best < 0 or bg == 0: break
            used[best] = True; s |= dense_vis[best] & observable; n += 1
        return c0, n, s
    c5, add5, seen5 = cov_add(ring5)
    print(f"5m 균등 링(17): 관측가능 {c5:.1f}% → PB-NBV 추가 {add5}개  (standoff {np.linalg.norm(ring5[0]-t):.1f}m)")

    fig = go.Figure()
    gx, gy, gz = xyz(obj[obs_pts])
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode='markers',
        marker=dict(size=2.6, color='#3fb950'), name=f'관측가능 표면 ({int(obs_pts.sum())})'))
    ux, uy, uz = xyz(obj[~obs_pts])
    fig.add_trace(go.Scatter3d(x=ux, y=uy, z=uz, mode='markers',
        marker=dict(size=2.2, color='#6e7681', opacity=0.45), name=f'밑면·관측불가 ({int((~obs_pts).sum())})'))
    for tr in smooth_ring(ring5, '#d2a8ff', '5m 링(합성)', True): fig.add_trace(tr)
    tx, ty, tz = xyz(t)
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='markers',
        marker=dict(size=6, color='#f778ba', symbol='diamond'), name='target'))

    note = ("<b>5m 단독 (균등 링 17컷) — projection PB-NBV</b><br>"
            "※ 5m 실측 없음 → 5m 고도 균등 링 <b>합성</b> (tilt45)<br>"
            f"  standoff ≈ {np.linalg.norm(ring5[0]-t):.1f} m<br><br>"
            f"  5m 링: {c5:.0f}% 도달 → <b>+{add5}개</b><br><br>"
            f"  관측가능 = 물체 공중관측면 {int(obs_pts.sum())}/{len(obj)} (밑면 제외)")
    fig.add_annotation(text=note, xref='paper', yref='paper', x=0.01, y=0.99,
                       align='left', showarrow=False, font=dict(color='#e6edf3', size=11.5),
                       bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d', borderwidth=1, borderpad=8)

    fig.update_layout(
        title=dict(text=f'5m 단독 검증 (projection PB-NBV, 합성 링) — 관측가능면 <b>{c5:.0f}%, 추가 {add5}개</b>'
                        '<br><sub>5m 실측 없음·합성 링 · 객체=ㄷ자 슬래브 · up=−Z(NED)</sub>',
                   font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(
            xaxis=dict(visible=True, showbackground=True, backgroundcolor='#161b22',
                       showgrid=True, gridcolor='#30363d', showticklabels=False, title='', zeroline=False),
            yaxis=dict(visible=True, showbackground=True, backgroundcolor='#161b22',
                       showgrid=True, gridcolor='#30363d', showticklabels=False, title='', zeroline=False),
            zaxis=dict(visible=True, showbackground=True, backgroundcolor='#161b22',
                       showgrid=True, gridcolor='#30363d', showticklabels=False, title='', zeroline=False),
            aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        legend=dict(bgcolor='#161b22', bordercolor='#21262d', borderwidth=1, x=0.99, xanchor='right'))
    out = ROOT/'results'/'pbnbv_projection_5m.html'
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f"✓ {out}")


if __name__ == '__main__':
    main()
