#!/usr/bin/env python3
"""
두 초기 경로 비교 시각화:
1. 최적 (형상 앎): 고도 4m, 3시점
2. 초기경로 (형상 미확인): 고도 4m, 8시점
"""
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

TARGET = np.array([-33.67, -50.83, 0.18])

def make_orbit(alt, n_points, radius=7.0):
    """고도 alt, 시점 n개인 원형 궤적."""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    positions = []
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = TARGET[0] + radius * np.cos(az_rad)
        y = TARGET[1] + radius * np.sin(az_rad)
        z = TARGET[2] - alt
        positions.append([x, y, z])
    return np.array(positions), azimuths

def alt(z):
    """NED Z → 고도로 변환."""
    return TARGET[2] - np.asarray(z)

# 점군 로드
v = np.load('real_test/real_test_pts_vis.npz')
PTS, C = v['points'], v['colors']
rgb = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b in C]

# 두 경로 생성
opt_pos, opt_az = make_orbit(4, 3)      # 최적: 3시점
init_pos, init_az = make_orbit(4, 8)    # 초기경로: 8시점

fig = go.Figure()

# 점군 (항상 표시)
fig.add_trace(go.Scatter3d(
    x=PTS[:,0], y=PTS[:,1], z=alt(PTS[:,2]),
    mode='markers',
    marker=dict(size=1.5, color=rgb, opacity=0.35),
    name='점군'))

# 경로 1: 최적 (3시점, Cyan, 실선)
fig.add_trace(go.Scatter3d(
    x=opt_pos[:,0], y=opt_pos[:,1], z=alt(opt_pos[:,2]),
    mode='lines+markers+text',
    line=dict(color='cyan', width=4),
    marker=dict(size=11, color='cyan', symbol='diamond',
                line=dict(color='white', width=1.5)),
    text=[f'{i+1}' for i in range(len(opt_pos))],
    textposition='top center',
    textfont=dict(color='white', size=12),
    name='최적 (형상 앎): 고도4m, 3시점 → 100%',
    customdata=np.array([[az, 4.0, "최적"] for az in opt_az]),
    hovertemplate='최적 WP%{text}<br>az=%{customdata[0]:.0f}°<extra></extra>',
    visible=True))

# 경로 2: 초기경로 (8시점, Orange, 점선)
fig.add_trace(go.Scatter3d(
    x=init_pos[:,0], y=init_pos[:,1], z=alt(init_pos[:,2]),
    mode='lines+markers+text',
    line=dict(color='orange', width=3, dash='dash'),
    marker=dict(size=9, color='orange', symbol='circle',
                line=dict(color='white', width=1.5)),
    text=[f'{i+1}' for i in range(len(init_pos))],
    textposition='middle right',
    textfont=dict(color='white', size=10),
    name='초기경로 (형상 미확인): 고도4m, 8시점 → ~95%+NBV',
    customdata=np.array([[az, 4.0, "초기"] for az in init_az]),
    hovertemplate='초기 WP%{text}<br>az=%{customdata[0]:.0f}°<extra></extra>',
    visible=True))

# 타깃
fig.add_trace(go.Scatter3d(
    x=[TARGET[0]], y=[TARGET[1]], z=[alt(TARGET[2])],
    mode='markers',
    marker=dict(size=12, color='yellow', symbol='diamond'),
    name='타깃'))

fig.update_layout(
    title=dict(
        text='초기경로 비교: 최적(3시점) vs 일반(8시점)',
        font=dict(color='white', size=15)),
    scene=dict(
        xaxis=dict(title=dict(text='X (m)', font=dict(color='white')),
                   backgroundcolor='#0d0d1e', gridcolor='#333',
                   tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Y (m)', font=dict(color='white')),
                   backgroundcolor='#0d0d1e', gridcolor='#333',
                   tickfont=dict(color='white')),
        zaxis=dict(title=dict(text='고도 (m)', font=dict(color='white')),
                   backgroundcolor='#0d0d1e', gridcolor='#333',
                   tickfont=dict(color='white')),
        bgcolor='#0d0d1e',
        aspectmode='data',
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.0))),
    paper_bgcolor='#0a0a14',
    legend=dict(
        font=dict(color='white'),
        bgcolor='#1a1a30',
        x=0.02, y=0.98))

out = Path('results/initial_paths_comparison.html')
fig.write_html(str(out))
print(f"✓ {out}")
print(f"\n범례 클릭으로 경로 토글 가능")
print(f"\n최적 (Cyan 다이아몬드, 실선):")
print(f"  - 3시점, 방위 0°, 120°, 240° (120° 간격)")
print(f"  - 예상 커버: 100%")
print(f"\n초기경로 (Orange 원, 점선):")
print(f"  - 8시점, 방위 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° (45° 간격)")
print(f"  - 예상 커버: ~95% + 온라인 NBV 보강")
