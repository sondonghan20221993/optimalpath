#!/usr/bin/env python3
"""논문 PB-NBV: 각 viewpoint가 '처음 복원하는' 표면 영역을 색으로 표시 (HTML)"""
import json, math
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import pbnbv_paper as P

TARGET = P.TARGET
def alt(z): return TARGET[2] - np.asarray(z)

# 경로(유효 스텝)
path = json.load(open('results/pbnbv_paper/pbnbv_paper.json'))['path']
valid = [p for p in path if p['gained'] > 0]

# 각 스텝이 '처음' 복원하는 surface voxel 계산 (누적 차집합)
observed = np.zeros(P.N_SURF, dtype=bool)
first_step = np.full(P.N_SURF, -1, dtype=int)   # voxel별 최초 관측 스텝(0-based valid idx)
for si, p in enumerate(valid):
    idx = P.observed_by(np.array(p['pos']))
    newly = idx[~observed[idx]]
    first_step[newly] = si
    observed[idx] = True

# 원본 점군(조밀) → 각 점이 속한 surface voxel → first_step
v = np.load('real_test/real_test_pts_vis.npz')
PTS = v['points']
key_to_i = {k:i for i,k in enumerate(P.SURF_KEYS)}
pt_step = np.full(len(PTS), -1, dtype=int)
for j,pt in enumerate(PTS):
    vox = tuple(np.floor((pt-P.ORIGIN)/P.VOXEL).astype(int))
    i = key_to_i.get(vox, None)
    if i is not None: pt_step[j] = first_step[i]

palette = ['#e6194B','#f58231','#ffe119','#3cb44b','#4363d8','#911eb4','#42d4f4','#f032e6']

fig = go.Figure()

# 미관측 점(회색)
m0 = pt_step < 0
if m0.any():
    fig.add_trace(go.Scatter3d(
        x=PTS[m0,0], y=PTS[m0,1], z=alt(PTS[m0,2]), mode='markers',
        marker=dict(size=1.5, color='#444', opacity=0.4), name='미복원'))

# 스텝별: 그 viewpoint가 처음 복원한 점들 + viewpoint (같은 색, 같은 legendgroup)
for si, p in enumerate(valid):
    col = palette[si % len(palette)]
    m = pt_step == si
    grp = f'wp{p["step"]}'
    if m.any():
        fig.add_trace(go.Scatter3d(
            x=PTS[m,0], y=PTS[m,1], z=alt(PTS[m,2]), mode='markers',
            marker=dict(size=2.6, color=col, opacity=0.85),
            name=f'WP{p["step"]} 복원범위 (+{m.sum()}pts)', legendgroup=grp,
            hovertemplate=f'WP{p["step"]} 복원<extra></extra>'))
    # viewpoint
    pos = np.array(p['pos'])
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[alt(pos[2])], mode='markers+text',
        marker=dict(size=11, color=col, symbol='diamond', line=dict(color='white',width=1.5)),
        text=[f'{p["step"]}'], textposition='top center', textfont=dict(color='white',size=13),
        name=f'WP{p["step"]} 위치', legendgroup=grp, showlegend=False,
        hovertemplate=f'WP{p["step"]}  alt={p["alt"]:.1f}m az={p["azimuth"]:.0f}°<br>cov={p["coverage"]*100:.1f}%<extra></extra>'))
    # viewpoint→복원범위 중심 연결선 (옅게)
    if m.any():
        cen = PTS[m].mean(0)
        fig.add_trace(go.Scatter3d(
            x=[pos[0],cen[0]], y=[pos[1],cen[1]], z=[alt(pos[2]),alt(cen[2])],
            mode='lines', line=dict(color=col, width=2, dash='dot'),
            legendgroup=grp, showlegend=False, hoverinfo='skip'))

# 타겟
fig.add_trace(go.Scatter3d(
    x=[TARGET[0]], y=[TARGET[1]], z=[alt(TARGET[2])], mode='markers',
    marker=dict(size=8, color='yellow'), name='타겟'))

fig.update_layout(
    title=dict(text='논문 PB-NBV — 각 viewpoint가 처음 복원하는 표면 영역(색 매칭)',
               font=dict(color='white', size=14)),
    scene=dict(
        xaxis=dict(title=dict(text='X (m)',font=dict(color='white')), backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Y (m)',font=dict(color='white')), backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        zaxis=dict(title=dict(text='고도 (m)',font=dict(color='white')), backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        bgcolor='#0d0d1e', aspectmode='data', camera=dict(eye=dict(x=1.4,y=1.4,z=1.0))),
    paper_bgcolor='#0a0a14', legend=dict(font=dict(color='white'), bgcolor='#1a1a30'))

out = Path('results/pbnbv_paper/pbnbv_paper_coverage.html')
fig.write_html(str(out))
print("스텝별 복원 점수:")
for si,p in enumerate(valid):
    print(f"  WP{p['step']}: +{(pt_step==si).sum()} pts (cov {p['coverage']*100:.1f}%)")
print(f"✓ {out}")
