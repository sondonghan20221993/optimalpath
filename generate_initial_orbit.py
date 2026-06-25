#!/usr/bin/env python3
"""
초기 경로: 고고도 비스듬 원형 1바퀴
타깃을 중심으로 원형 궤적, 카메라는 항상 타깃을 내려다봄
"""
import numpy as np
import json
from pathlib import Path

TARGET = np.array([-33.67, -50.83, 0.18])
ALTITUDE = 4.5  # 타깃 위 4.5m
RADIUS = 7.0    # 수평 반경
N_POINTS = 8    # 1바퀴 시점 개수

# 원형 궤적 생성 (azimuth 0~360도)
azimuths = np.linspace(0, 360, N_POINTS, endpoint=False)
positions = []
for az_deg in azimuths:
    az_rad = np.radians(az_deg)
    # 2D에서 원 위의 점
    x = TARGET[0] + RADIUS * np.cos(az_rad)
    y = TARGET[1] + RADIUS * np.sin(az_rad)
    z = TARGET[2] - ALTITUDE  # AirSim NED: 음수 = 위
    positions.append([x, y, z])

positions = np.array(positions)

# HTML 생성 (점군 + 경로)
import plotly.graph_objects as go

# 점군 로드
v = np.load('real_test/real_test_pts_vis.npz')
PTS, C = v['points'], v['colors']
rgb = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b in C]

def alt(z):
    return TARGET[2] - np.asarray(z)

fig = go.Figure()

# 점군
fig.add_trace(go.Scatter3d(
    x=PTS[:,0], y=PTS[:,1], z=alt(PTS[:,2]), mode='markers',
    marker=dict(size=1.5, color=rgb, opacity=0.35),
    name='점군'))

# 초기 경로 (궤적)
fig.add_trace(go.Scatter3d(
    x=positions[:,0], y=positions[:,1], z=alt(positions[:,2]),
    mode='lines+markers+text',
    line=dict(color='cyan', width=4),
    marker=dict(size=10, color='cyan', symbol='diamond', line=dict(color='white', width=1.5)),
    text=[str(i+1) for i in range(len(positions))],
    textposition='top center',
    textfont=dict(color='white', size=11),
    name='초기경로 (고도4.5m, 반경7m, 1바퀴)',
    customdata=np.array([[az, ALTITUDE] for az in azimuths]),
    hovertemplate='WP%{text}<br>az=%{customdata[0]:.0f}° alt=%{customdata[1]:.1f}m<extra></extra>'))

# 각 시점에서 타깃으로의 시선 방향 (화살표)
for pos in positions:
    vec = TARGET - pos
    vec_norm = vec / np.linalg.norm(vec)
    arrow_end = pos + vec_norm * 2.0  # 2m 길이 화살표
    fig.add_trace(go.Scatter3d(
        x=[pos[0], arrow_end[0]],
        y=[pos[1], arrow_end[1]],
        z=[alt(pos[2]), alt(arrow_end[2])],
        mode='lines',
        line=dict(color='orange', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip'))

# 타깃
fig.add_trace(go.Scatter3d(
    x=[TARGET[0]], y=[TARGET[1]], z=[alt(TARGET[2])],
    mode='markers',
    marker=dict(size=8, color='yellow'),
    name='타깃'))

fig.update_layout(
    title=dict(text='초기경로: 고고도 비스듬 원형 1바퀴',
               font=dict(color='white', size=14)),
    scene=dict(
        xaxis=dict(title=dict(text='X (m)', font=dict(color='white')),
                   backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        yaxis=dict(title=dict(text='Y (m)', font=dict(color='white')),
                   backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        zaxis=dict(title=dict(text='고도 (m)', font=dict(color='white')),
                   backgroundcolor='#0d0d1e', gridcolor='#333', tickfont=dict(color='white')),
        bgcolor='#0d0d1e', aspectmode='data',
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.0))),
    paper_bgcolor='#0a0a14',
    legend=dict(font=dict(color='white'), bgcolor='#1a1a30'))

out = Path('results/initial_orbit.html')
fig.write_html(str(out))
print(f"✓ {out}")
print(f"  타깃: {TARGET}")
print(f"  고도: {ALTITUDE}m, 반경: {RADIUS}m, 시점: {N_POINTS}개")
print(f"  부각: {np.degrees(np.arctan(ALTITUDE/RADIUS)):.1f}°")

# JSON 저장
data = {
    "schema_version": 1,
    "name": "초기경로 고고도원형1바퀴",
    "path_step_count": N_POINTS,
    "config": {
        "altitude": ALTITUDE,
        "radius": RADIUS,
        "depression_angle_deg": np.degrees(np.arctan(ALTITUDE/RADIUS))
    },
    "waypoints": [
        {
            "index": i+1,
            "position": list(positions[i]),
            "azimuth_deg": float(azimuths[i]),
            "relative_to_target": list(positions[i] - TARGET)
        }
        for i in range(len(positions))
    ]
}

json_out = Path('results/initial_orbit.json')
json.dump(data, open(json_out, 'w'), indent=2)
print(f"✓ {json_out}")
