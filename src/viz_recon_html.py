"""
viz_recon_html.py — PB-NBV가 실제로 복원하는 'point cloud'를 인터랙티브 3D HTML로

각 시점(궤도/NBV 카메라)이 raycast로 실제 캡처하는 GT 점을 누적 = 복원 점군.
복원된 점 vs 못 잡은 점을 3D로 보여주고 마우스로 회전/줌.

출력: results/recon_orbit.html , results/recon_nbv.html
"""
import sys, types, json, math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P
import pbnbv_path as A

ROOT = HERE.parent
MAX_DIST = 10.5
INC_LIMIT = 60                      # 복원 인정 입사각 (품질 기준)
P.MAX_DIST = MAX_DIST
_NRM = np.load(ROOT/'real_test'/'airsim_gt_pts.npz')['normals'].astype(float)
_COS = math.cos(math.radians(INC_LIMIT))


def capture(cams):
    """각 점을 입사각<60°(양질)로 잡은 카메라 수. raycast 가시성 ∩ 입사각 품질."""
    A.RAYCAST_OCCLUSION = True
    pts = P._pts_raw
    seen_cnt = np.zeros(len(pts), int)
    for c in cams:
        live = np.ones(len(pts), bool)
        _, vis = A.information_gain(np.asarray(c), P.TARGET, pts, live, P.FOV_DEG, MAX_DIST)
        # 입사각 품질: 점→카메라 방향과 법선
        view = np.asarray(c) - pts
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
        cos_inc = (_NRM * view).sum(1)
        good = vis & (cos_inc >= _COS)
        seen_cnt += good.astype(int)
    return seen_cnt


def build_html(cams, ring, title, out, with_loop=True):
    pts = P._pts_raw
    seen = capture(cams)
    rec = seen > 0
    n_rec, n_tot = int(rec.sum()), len(pts)

    # NED z(아래+) → 보기 좋게 up = -z
    def xyz(a):
        a = np.atleast_2d(a)
        return a[:, 0], a[:, 1], -a[:, 2]

    gx, gy, gz = xyz(pts[rec])
    mx, my, mz = xyz(pts[~rec])
    cx, cy, cz = xyz(np.array(cams))

    fig = go.Figure()
    # 복원된 부분 (보임) — 단색 초록
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode='markers',
        marker=dict(size=3.0, color='#3fb950'),
        name=f'복원됨/보임 ({n_rec})'))
    # 미복원 부분 (안보임, 밑면 등) — 단색 빨강
    if (~rec).any():
        fig.add_trace(go.Scatter3d(
            x=mx, y=my, z=mz, mode='markers',
            marker=dict(size=3.0, color='#f85149'),
            name=f'미복원/안보임 ({n_tot-n_rec})'))
    # 카메라
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz, mode='markers',
        marker=dict(size=5, color='#58a6ff', symbol='diamond',
                    line=dict(color='white', width=1)),
        name=f'cameras ({len(cams)})'))
    # 카메라 시선 방향 화살표 (타겟 방향, 작게)
    Cn = np.array(cams, float)
    dirs = P.TARGET - Cn
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    L = 1.0                                   # 화살표 길이(m)
    End = Cn + dirs * L
    sx, sy, sz = xyz(Cn); ex, ey, ez = xyz(End)
    # 화살대(선)
    lx, ly, lz = [], [], []
    for i in range(len(Cn)):
        lx += [sx[i], ex[i], None]; ly += [sy[i], ey[i], None]; lz += [sz[i], ez[i], None]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                 line=dict(color='#ffa657', width=3), name='view dir', showlegend=True))
    # 화살촉(cone) — plot 좌표계 방향 = end-start
    fig.add_trace(go.Cone(
        x=ex, y=ey, z=ez, u=ex-sx, v=ey-sy, w=ez-sz,
        sizemode='absolute', sizeref=0.35, anchor='tip',
        showscale=False, colorscale=[[0, '#ffa657'], [1, '#ffa657']],
        name='view dir', hoverinfo='skip'))
    # 궤도/경로 선
    if with_loop and ring is not None and len(ring) > 1:
        rr = np.vstack([ring, ring[0]]) if with_loop else ring
        lx, ly, lz = xyz(rr)
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                     line=dict(color='#58a6ff', width=3), name='flight path'))
    # 타겟
    tx, ty, tz = xyz(P.TARGET)
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='markers',
                 marker=dict(size=6, color='#f778ba', symbol='diamond'), name='target'))

    pct = 100 * n_rec / n_tot
    fig.update_layout(
        title=dict(text=f'{title}<br><sub>reconstructed {n_rec}/{n_tot} = {pct:.1f}% of GT points '
                        f'(raycast capture) | up = -Z (NED)</sub>', font=dict(color='#e6edf3')),
        paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='up (m)',
                   xaxis=dict(backgroundcolor='#161b22', color='#e6edf3', gridcolor='#21262d'),
                   yaxis=dict(backgroundcolor='#161b22', color='#e6edf3', gridcolor='#21262d'),
                   zaxis=dict(backgroundcolor='#161b22', color='#e6edf3', gridcolor='#21262d'),
                   aspectmode='data'),
        legend=dict(bgcolor='#161b22', bordercolor='#21262d', borderwidth=1))
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f'✓ {out}  복원 {n_rec}/{n_tot}={pct:.1f}%')
    return pct


def main():
    t = P.TARGET
    top = P._pts_raw[:, 2].min()
    az = lambda p: math.degrees(math.atan2(p[1]-t[1], p[0]-t[0])) % 360

    # 1) 원형 궤도 36대 (방위 10° 간격 — ㄷ 안쪽 벽까지 정면 확보)
    orbit = A.gen_candidates_tilt45(t, [round(top-3, 2)], n_az=36, max_dist=MAX_DIST)
    orbit = np.array(sorted(orbit, key=az))
    build_html(orbit, orbit, 'PB-NBV reconstruction — Circular orbit (36 cams)',
               ROOT/'results'/'recon_orbit.html', with_loop=True)

    # 2) 순수 NBV 경로 (방금 결과)
    d = json.load(open(ROOT/'results'/'pbnbv_nocam_constrained_path.json'))
    nbv = [np.array(d['start_pos'])] + [np.array(w['pos']) for w in d['waypoints']]
    build_html(nbv, np.array(nbv), 'PB-NBV reconstruction — NBV path (no orbit, 7 WP)',
               ROOT/'results'/'recon_nbv.html', with_loop=False)


if __name__ == '__main__':
    main()
