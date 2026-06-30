#!/usr/bin/env python3
"""
Orbit vs Greedy 경로 시각화 (3D)

각 물체별로:
- surface voxel (파란점)
- target (빨간별)
- Orbit 경로 (청색 선)
- Greedy 경로 (빨강 선)
- 각 시점의 시야각 시각화 (원뿔 선택)
"""
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import math

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

with open('results/step2_v3/results.json') as f:
    all_results = json.load(f)

TARGET = np.array([-33.67, -50.83, 0.18])
RADIUS = 7.0
FOV_DEG = 89.9
MIN_DIST = 4.0
MAX_DIST = 13.0

CANDIDATES = []
for alt in np.arange(1.0, 9.5, 1.0):
    z = TARGET[2] - alt
    for rad in [4.5, 6.0, 7.5, 9.0]:
        n_az = max(6, int(2 * math.pi * rad / 1.5))
        for i in range(n_az):
            th = 2 * math.pi * i / n_az
            CANDIDATES.append([TARGET[0] + rad * math.cos(th),
                               TARGET[1] + rad * math.sin(th), z])
CANDIDATES = np.array(CANDIDATES)

# ─────────────────────────────────────────────────────────────────────────────
# 함수
# ─────────────────────────────────────────────────────────────────────────────

def make_orbit(n_points, alt=4.0):
    """Orbit-N 경로"""
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    positions = []
    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        x = TARGET[0] + RADIUS * np.cos(az_rad)
        y = TARGET[1] + RADIUS * np.sin(az_rad)
        z = TARGET[2] - alt
        positions.append([x, y, z])
    return np.array(positions)

def get_visible(cam_pos, voxel_centers, voxel_normals):
    """cam_pos에서 보이는 voxel 불리언 마스크"""
    cam_dir = TARGET - cam_pos
    cam_dir /= np.linalg.norm(cam_dir) + 1e-9

    to = voxel_centers - cam_pos
    dist = np.linalg.norm(to, axis=1)

    in_range = (dist >= MIN_DIST) & (dist <= MAX_DIST)
    in_fov = (to * cam_dir).sum(1) / (dist + 1e-9) >= math.cos(math.radians(FOV_DEG / 2))
    front = (voxel_normals * (cam_pos - voxel_centers)).sum(1) > 0

    return in_range & in_fov & front

def make_greedy(voxel_centers, voxel_normals, n_budget):
    """Greedy-N 경로"""
    vis_masks = np.array([
        get_visible(c, voxel_centers, voxel_normals)
        for c in CANDIDATES
    ])

    covered = np.zeros(len(voxel_centers), dtype=bool)
    selected = []

    for _ in range(n_budget):
        gains = np.sum(vis_masks & ~covered, axis=1)
        best = int(np.argmax(gains))
        selected.append(CANDIDATES[best])
        covered |= vis_masks[best]

    return np.array(selected)

# ─────────────────────────────────────────────────────────────────────────────
# 시각화 (대표 물체 2개: 평탄, 구형)
# ─────────────────────────────────────────────────────────────────────────────

test_objects = [
    ('flat-simple', 'Flat Simple (평탄 물체)'),
    ('sphere-large', 'Sphere Large (구형 물체)'),
]

for obj_name, obj_title in test_objects:
    # 데이터 로드
    obj_file = Path(f'data/test_objects/{obj_name}_0.150.npz')
    data = np.load(obj_file)
    voxel_centers = data['voxel_centers']
    voxel_normals = data['voxel_normals']

    # 경로 생성
    orbit_path = make_orbit(8)
    greedy_path = make_greedy(voxel_centers, voxel_normals, 8)

    # ─────────────────────────────────────────────────────────────────────────
    # 3D 그래프 생성
    # ─────────────────────────────────────────────────────────────────────────

    fig = go.Figure()

    # 1. Surface voxel (파란점, 물체)
    fig.add_trace(go.Scatter3d(
        x=voxel_centers[:, 0],
        y=voxel_centers[:, 1],
        z=voxel_centers[:, 2],
        mode='markers',
        marker=dict(size=3, color='lightblue', opacity=0.6),
        name='Surface Voxels',
        showlegend=True
    ))

    # 2. Target (빨간 다이아몬드)
    fig.add_trace(go.Scatter3d(
        x=[TARGET[0]],
        y=[TARGET[1]],
        z=[TARGET[2]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Target',
        showlegend=True
    ))

    # 3. Orbit 경로 (청색 선 + 점)
    fig.add_trace(go.Scatter3d(
        x=orbit_path[:, 0],
        y=orbit_path[:, 1],
        z=orbit_path[:, 2],
        mode='lines+markers',
        line=dict(color='cyan', width=4),
        marker=dict(size=8, color='cyan', symbol='circle'),
        name='Orbit Path',
        showlegend=True
    ))

    # Orbit 경로에 번호 매기기
    for i, pos in enumerate(orbit_path):
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode='text',
            text=[f'O{i+1}'],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip'
        ))

    # 4. Greedy 경로 (빨강 선 + 점)
    fig.add_trace(go.Scatter3d(
        x=greedy_path[:, 0],
        y=greedy_path[:, 1],
        z=greedy_path[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=4, dash='dash'),
        marker=dict(size=8, color='red', symbol='square'),
        name='Greedy Path',
        showlegend=True
    ))

    # Greedy 경로에 번호 매기기
    for i, pos in enumerate(greedy_path):
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode='text',
            text=[f'G{i+1}'],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip'
        ))

    # 5. 후보 시점 (회색점, 반투명)
    fig.add_trace(go.Scatter3d(
        x=CANDIDATES[:, 0],
        y=CANDIDATES[:, 1],
        z=CANDIDATES[:, 2],
        mode='markers',
        marker=dict(size=2, color='gray', opacity=0.2),
        name='Candidates (999)',
        showlegend=True
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 레이아웃
    # ─────────────────────────────────────────────────────────────────────────

    fig.update_layout(
        title=dict(
            text=f'Orbit vs Greedy 경로 비교<br>{obj_title} (N=8, 0.15m)',
            font=dict(size=14)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='data'
        ),
        hovermode='closest',
        width=1000,
        height=800,
        font=dict(size=10),
        legend=dict(x=0.7, y=0.95)
    )

    # 저장
    out_path = f'results/step2_v3/path_comparison_{obj_name}.html'
    fig.write_html(out_path)
    print(f"✓ {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 경로 통계 비교
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("경로 통계 비교 (N=8, 0.15m)")
print("=" * 80)
print(f"{'물체':<20} {'항목':<20} {'Orbit':<15} {'Greedy':<15}")
print("-" * 70)

START = np.array([-27.06, -50.51, -6.13])

for obj_name, obj_title in test_objects:
    obj_file = Path(f'data/test_objects/{obj_name}_0.150.npz')
    data = np.load(obj_file)
    voxel_centers = data['voxel_centers']
    voxel_normals = data['voxel_normals']

    orbit_path = make_orbit(8)
    greedy_path = make_greedy(voxel_centers, voxel_normals, 8)

    # 경로 길이
    def pathlen(positions):
        d = 0.0
        cur = START
        for p in positions:
            d += np.linalg.norm(np.array(p) - cur)
            cur = np.array(p)
        return d

    orbit_dist = pathlen(orbit_path)
    greedy_dist = pathlen(greedy_path)

    # 커버리지
    def eval_cov(positions, vc, vn):
        observed = np.zeros(len(vc), dtype=bool)
        for pos in positions:
            observed |= get_visible(pos, vc, vn)
        return np.mean(observed)

    orbit_cov = eval_cov(orbit_path, voxel_centers, voxel_normals)
    greedy_cov = eval_cov(greedy_path, voxel_centers, voxel_normals)

    # 분산 (각 시점 간 거리)
    def variance_between_points(positions):
        dists = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                dists.append(d)
        return np.std(dists) if dists else 0

    orbit_var = variance_between_points(orbit_path)
    greedy_var = variance_between_points(greedy_path)

    print(f"{obj_name:<20} {'경로 길이(m)':<20} {orbit_dist:>14.2f} {greedy_dist:>14.2f}")
    print(f"{'':<20} {'커버리지(%)':<20} {orbit_cov*100:>14.1f} {greedy_cov*100:>14.1f}")
    print(f"{'':<20} {'점 간 거리 분산':<20} {orbit_var:>14.2f} {greedy_var:>14.2f}")
    print()

print("=" * 80)
print("범례:")
print("  O1~O8 = Orbit 시점 번호 (균등 원형)")
print("  G1~G8 = Greedy 시점 번호 (oracle 최적화)")
print("  청색 선 = Orbit 경로")
print("  빨강 선(점선) = Greedy 경로")
print("  파란점 = Surface voxel (물체 표면)")
print("  빨간별 = Target (복원 대상)")
print("=" * 80)
