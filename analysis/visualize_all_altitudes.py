#!/usr/bin/env python3
"""고도 1~7m 각각의 PB-NBV 경로를 한 HTML에 (고도별 legend 토글)"""
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

TARGET = np.array([-33.67,-50.83,0.18])
def alt(z): return TARGET[2]-np.asarray(z)

# 점군
v=np.load('real_test/real_test_pts_vis.npz')
PTS,C=v['points'],v['colors']
rgb=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b in C]

palette={1:'#e6194B',2:'#f58231',3:'#ffe119',4:'#3cb44b',
         5:'#4363d8',6:'#911eb4',7:'#f032e6'}

fig=go.Figure()
# 점군 (항상 표시)
fig.add_trace(go.Scatter3d(
    x=PTS[:,0],y=PTS[:,1],z=alt(PTS[:,2]),mode='markers',
    marker=dict(size=1.4,color=rgb,opacity=0.35),name='점군'))

for a in [1,2,3,4,5,6,7]:
    fp=Path(f'results/pbnbv_paper/alt_{a}m.json')
    if not fp.exists(): continue
    d=json.load(open(fp))
    path=[p for p in d['path'] if p['gained']>0]
    if not path: continue
    pos=np.array([p['pos'] for p in path])
    col=palette[a]; grp=f'{a}m'
    final=path[-1]['coverage']*100
    # 경로 선+viewpoint
    fig.add_trace(go.Scatter3d(
        x=pos[:,0],y=pos[:,1],z=alt(pos[:,2]),
        mode='lines+markers+text',
        line=dict(color=col,width=4),
        marker=dict(size=9,color=col,symbol='diamond',line=dict(color='white',width=1)),
        text=[str(p['step']) for p in path],textposition='top center',
        textfont=dict(color='white',size=11),
        name=f'{a}m ({len(path)}스텝→{final:.0f}%)',legendgroup=grp,
        customdata=np.array([[p['alt'],p['azimuth'],p['gained'],p['coverage']*100] for p in path]),
        hovertemplate=(f'{a}m WP%{{text}}<br>az=%{{customdata[1]:.0f}}° '
                       '+%{customdata[2]}voxel cov=%{customdata[3]:.0f}%<extra></extra>'),
        visible=True if a in (1,4,7) else 'legendonly'))

# 타겟
fig.add_trace(go.Scatter3d(
    x=[TARGET[0]],y=[TARGET[1]],z=[alt(TARGET[2])],mode='markers',
    marker=dict(size=8,color='yellow'),name='타겟'))

fig.update_layout(
    title=dict(text='고도별 PB-NBV 경로 비교 (범례 클릭으로 고도 토글)',font=dict(color='white',size=14)),
    scene=dict(
        xaxis=dict(title=dict(text='X (m)',font=dict(color='white')),backgroundcolor='#0d0d1e',gridcolor='#333',tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Y (m)',font=dict(color='white')),backgroundcolor='#0d0d1e',gridcolor='#333',tickfont=dict(color='white')),
        zaxis=dict(title=dict(text='고도 (m)',font=dict(color='white')),backgroundcolor='#0d0d1e',gridcolor='#333',tickfont=dict(color='white')),
        bgcolor='#0d0d1e',aspectmode='data',camera=dict(eye=dict(x=1.4,y=1.4,z=1.1))),
    paper_bgcolor='#0a0a14',legend=dict(font=dict(color='white'),bgcolor='#1a1a30'))

out=Path('results/pbnbv_paper/all_altitudes.html')
fig.write_html(str(out))
print(f"✓ {out}")
print("  기본 표시: 1m,4m,7m / 나머지는 범례 클릭")