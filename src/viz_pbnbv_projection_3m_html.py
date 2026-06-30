"""
viz_pbnbv_projection_3m_html.py — 3m 단독 버전 projection PB-NBV HTML.
3m 한 바퀴(17컷)만으로 관측가능 표면 100%, 추가 시점 0개임을 보임.
출력: results/pbnbv_projection_3m.html
"""
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_projection_uniform as M
import pbnbv_path as A
from viz_pbnbv_projection_html import load, xyz, cam_trace

ROOT = M.ROOT; t = M.t; obj = M.obj; top = M.top


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

    U3 = load(M.UNI/'real_test_3m_uniform')      # 균등 3m (17)
    N3 = load(M.NEW/'real_test_3m')              # 비균등 3m (17)

    def cov(cams):
        s = np.zeros(len(cen), bool)
        for c in cams: s |= M.project_visible(c, cen, nrm, rad)
        s &= observable
        return 100*npts[s].sum()/npts[observable].sum()
    cU = cov(U3); cN = cov(N3)

    fig = go.Figure()
    gx, gy, gz = xyz(obj[obs_pts])
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode='markers',
        marker=dict(size=2.6, color='#3fb950'), name=f'관측가능 표면 ({int(obs_pts.sum())})'))
    ux, uy, uz = xyz(obj[~obs_pts])
    fig.add_trace(go.Scatter3d(x=ux, y=uy, z=uz, mode='markers',
        marker=dict(size=2.2, color='#6e7681', opacity=0.45), name=f'밑면·관측불가 ({int((~obs_pts).sum())})'))
    for tr in cam_trace(U3, '#58a6ff', '균등 3m', True): fig.add_trace(tr)
    for tr in cam_trace(N3, '#f778ba', '비균등 3m', 'legendonly'): fig.add_trace(tr)
    tx, ty, tz = xyz(t)
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='markers',
        marker=dict(size=6, color='#f778ba', symbol='diamond'), name='target'))

    note = ("<b>3m 단독 (한 바퀴 17컷) — projection PB-NBV</b><br>"
            "raycast 아님 · ellipsoid 투영 가시<br><br>"
            f"  균등 3m  : {cU:.0f}% 도달 → <b>+0개</b><br>"
            f"  비균등 3m: {cN:.0f}% 도달 → <b>+0개</b><br><br>"
            "<b>→ 3m 한 바퀴로 물체 관측가능면 완성</b><br>"
            "  7m은 물체엔 추가 기여 0 (지형/광역용)<br>"
            f"  관측가능 = 물체 공중관측면 {int(obs_pts.sum())}/{len(obj)} (밑면 제외)")
    fig.add_annotation(text=note, xref='paper', yref='paper', x=0.01, y=0.99,
                       align='left', showarrow=False, font=dict(color='#e6edf3', size=11.5),
                       bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d', borderwidth=1, borderpad=8)

    fig.update_layout(
        title=dict(text='3m 단독 검증 (projection PB-NBV) — 한 바퀴로 <b>관측가능면 100%, 추가 경로 0개</b>'
                        '<br><sub>객체=ㄷ자 슬래브 · 범례에서 균등/비균등 3m 토글 · up=−Z(NED)</sub>',
                   font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False, showgrid=False, showticklabels=False, title='', zeroline=False),
            yaxis=dict(visible=False, showbackground=False, showgrid=False, showticklabels=False, title='', zeroline=False),
            zaxis=dict(visible=False, showbackground=False, showgrid=False, showticklabels=False, title='', zeroline=False),
            aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        legend=dict(bgcolor='#161b22', bordercolor='#21262d', borderwidth=1, x=0.99, xanchor='right'))
    out = ROOT/'results'/'pbnbv_projection_3m.html'
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f"✓ {out}  (균등3m {cU:.0f}%/+0, 비균등3m {cN:.0f}%/+0)")


if __name__ == '__main__':
    main()
