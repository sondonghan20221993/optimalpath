#!/usr/bin/env python3
"""논문 PB-NBV 경로 시각화 (HTML): 점군 + 선택 viewpoint + 경로 + 타겟"""
import json, math
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

TARGET = np.array([-33.67, -50.83, 0.18])
def alt(z): return TARGET[2] - np.asarray(z)   # NED z → 고도(양수)

d = json.load(open('results/pbnbv_paper/pbnbv_paper.json'))
path = d['path']
# 유효 스텝 = 새 voxel을 실제로 얻은 스텝만 (frontier 소진 전)
valid = [p for p in path if p['gained'] > 0]
pos = np.array([p['pos'] for p in valid])
cov = [p['coverage']*100 for p in valid]
order = [p['step'] for p in valid]

# 점군
v = np.load('real_test/real_test_pts_vis.npz')
P, C = v['points'], v['colors']
rgb = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b in C]

fig = go.Figure()

# 1) 점군
fig.add_trace(go.Scatter3d(
    x=P[:,0], y=P[:,1], z=alt(P[:,2]), mode='markers',
    marker=dict(size=1.6, color=rgb, opacity=0.45), name='점군(물체)',
    hovertemplate='x=%{x:.2f} y=%{y:.2f} alt=%{z:.2f}m<extra>점군</extra>'))

# 2) 경로 선 + viewpoint (coverage로 색)
fig.add_trace(go.Scatter3d(
    x=pos[:,0], y=pos[:,1], z=alt(pos[:,2]),
    mode='lines+markers+text',
    line=dict(color='cyan', width=5),
    marker=dict(size=9, color=cov, colorscale='Turbo', cmin=min(cov), cmax=100,
                showscale=True,
                colorbar=dict(title=dict(text='Coverage %', font=dict(color='white')),
                              tickfont=dict(color='white'), thickness=14),
                line=dict(color='white', width=1)),
    text=[f"{s}" for s in order], textposition='top center',
    textfont=dict(color='white', size=13),
    name='PB-NBV 경로(유효)',
    customdata=np.array([[p['alt'],p['azimuth'],p['gained'],p['coverage']*100,p['score']] for p in valid]),
    hovertemplate=('WP%{text}<br>alt=%{customdata[0]:.1f}m az=%{customdata[1]:.0f}°<br>'
                   '+%{customdata[2]}voxel  cov=%{customdata[3]:.1f}%<br>'
                   'F=%{customdata[4]:.0f}<extra>viewpoint</extra>')))

# 3) 기존 34프레임 카메라 (참고)
mp = Path('real_test/meta')
cams=[]
for mf in sorted(mp.glob('*.json')):
    pp = json.load(open(mf))['camera']['pose']['position']
    cams.append([pp['x'],pp['y'],pp['z']])
cams=np.array(cams)
fig.add_trace(go.Scatter3d(
    x=cams[:,0], y=cams[:,1], z=alt(cams[:,2]), mode='markers',
    marker=dict(size=3, color='gray', opacity=0.5, symbol='diamond'),
    name='기존 34프레임'))

# 4) 타겟
fig.add_trace(go.Scatter3d(
    x=[TARGET[0]], y=[TARGET[1]], z=[alt(TARGET[2])],
    mode='markers+text', marker=dict(size=10, color='yellow'),
    text=['Target'], textfont=dict(color='yellow', size=12), name='타겟'))

fig.update_layout(
    title=dict(text=f'논문 PB-NBV 경로 — {len(valid)}스텝에 coverage {cov[-1]:.1f}% (voxel+ellipsoid projection)',
               font=dict(color='white', size=14)),
    scene=dict(
        xaxis=dict(title=dict(text='X (m)',font=dict(color='white')), backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Y (m)',font=dict(color='white')), backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        zaxis=dict(title=dict(text='고도 (m)',font=dict(color='white')), backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        bgcolor='#0d0d1e', aspectmode='data',
        camera=dict(eye=dict(x=1.4,y=1.4,z=1.0))),
    paper_bgcolor='#0a0a14',
    legend=dict(font=dict(color='white'), bgcolor='#1a1a30'))

out = Path('results/pbnbv_paper/pbnbv_paper_path.html')
fig.write_html(str(out))
print(f"유효 스텝 {len(valid)}개, 최종 coverage {cov[-1]:.1f}%")
print(f"✓ {out}")
