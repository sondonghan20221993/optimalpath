#!/usr/bin/env python3
"""
6개 테스트 물체 생성 및 PCA 검증

물체 1-2: flat-simple (real_test), flat-complex (+엣지 거칠기)
물체 3: box (1m×0.5m×0.3m)
물체 4-5: sphere-small (r=0.5m), sphere-large (r=0.7m)
물체 6: occluded-chair (의자)

각 물체를 2 해상도(0.15m, 0.075m)로 voxelize 및 PCA 검증
"""
import numpy as np
from sklearn.decomposition import PCA
import json
from pathlib import Path

TARGET = np.array([-33.67, -50.83, 0.18])
VOXEL_SIZES = [0.15, 0.075]

# ─────────────────────────────────────────────────────────────────────────────
# 물체 생성 함수
# ─────────────────────────────────────────────────────────────────────────────

def generate_flat_simple():
    """real_test 로드"""
    v = np.load('real_test/real_test_pts_normals.npz')
    return v['points'], v['normals']

def generate_flat_complex():
    """flat_simple + 모서리 거칠기"""
    points, normals = generate_flat_simple()

    # 모서리 voxel 판정 (간단히: xyz 범위의 상위 10% 근처)
    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = points[:,1].min(), points[:,1].max()

    edge_mask = (
        (np.abs(points[:,0] - x_min) < 0.1) |
        (np.abs(points[:,0] - x_max) < 0.1) |
        (np.abs(points[:,1] - y_min) < 0.1) |
        (np.abs(points[:,1] - y_max) < 0.1)
    )

    noisy = points.copy()
    noisy[edge_mask] += np.random.normal(0, 0.02, size=(edge_mask.sum(), 3))
    return noisy, normals

def generate_box(l=1.0, w=0.5, h=0.3, n_points=3000):
    """직육면체 1m×0.5m×0.3m"""
    points, normals = [], []
    center = TARGET

    # 6개 면
    faces = [
        # (점 범위, 법선)
        ('xy', np.array([1,0,0])),      # x+ 면
        ('xy', np.array([-1,0,0])),     # x- 면
        ('xz', np.array([0,1,0])),      # y+ 면
        ('xz', np.array([0,-1,0])),     # y- 면
        ('xy', np.array([0,0,1])),      # z+ 면 (상단)
        ('xy', np.array([0,0,-1])),     # z- 면 (하단)
    ]

    n_per_face = n_points // 6

    # x+ 면
    x = center[0] + l/2
    y_vals = np.random.uniform(center[1]-w/2, center[1]+w/2, n_per_face)
    z_vals = np.random.uniform(center[2]-h/2, center[2]+h/2, n_per_face)
    points.extend(np.stack([np.full(n_per_face, x), y_vals, z_vals], axis=1))
    normals.extend([np.array([1,0,0])] * n_per_face)

    # x- 면
    x = center[0] - l/2
    y_vals = np.random.uniform(center[1]-w/2, center[1]+w/2, n_per_face)
    z_vals = np.random.uniform(center[2]-h/2, center[2]+h/2, n_per_face)
    points.extend(np.stack([np.full(n_per_face, x), y_vals, z_vals], axis=1))
    normals.extend([np.array([-1,0,0])] * n_per_face)

    # y+ 면
    y = center[1] + w/2
    x_vals = np.random.uniform(center[0]-l/2, center[0]+l/2, n_per_face)
    z_vals = np.random.uniform(center[2]-h/2, center[2]+h/2, n_per_face)
    points.extend(np.stack([x_vals, np.full(n_per_face, y), z_vals], axis=1))
    normals.extend([np.array([0,1,0])] * n_per_face)

    # y- 면
    y = center[1] - w/2
    x_vals = np.random.uniform(center[0]-l/2, center[0]+l/2, n_per_face)
    z_vals = np.random.uniform(center[2]-h/2, center[2]+h/2, n_per_face)
    points.extend(np.stack([x_vals, np.full(n_per_face, y), z_vals], axis=1))
    normals.extend([np.array([0,-1,0])] * n_per_face)

    # z+ 면 (상단)
    z = center[2] + h/2
    x_vals = np.random.uniform(center[0]-l/2, center[0]+l/2, n_per_face)
    y_vals = np.random.uniform(center[1]-w/2, center[1]+w/2, n_per_face)
    points.extend(np.stack([x_vals, y_vals, np.full(n_per_face, z)], axis=1))
    normals.extend([np.array([0,0,1])] * n_per_face)

    # z- 면 (하단)
    z = center[2] - h/2
    x_vals = np.random.uniform(center[0]-l/2, center[0]+l/2, n_per_face)
    y_vals = np.random.uniform(center[1]-w/2, center[1]+w/2, n_per_face)
    points.extend(np.stack([x_vals, y_vals, np.full(n_per_face, z)], axis=1))
    normals.extend([np.array([0,0,-1])] * n_per_face)

    return np.array(points), np.array(normals)

def generate_sphere(radius=0.5, n_points=3000):
    """구 (fibonacci sphere sampling)"""
    indices = np.arange(n_points) + 0.5
    theta = np.arccos(1 - 2*indices/n_points)
    phi = np.pi * (1 + 5**0.5) * indices

    x = TARGET[0] + radius * np.cos(phi) * np.sin(theta)
    y = TARGET[1] + radius * np.sin(phi) * np.sin(theta)
    z = TARGET[2] + radius * np.cos(theta)

    points = np.stack([x, y, z], axis=1)
    normals = (points - TARGET) / (np.linalg.norm(points - TARGET, axis=1, keepdims=True) + 1e-9)

    return points, normals

def generate_chair(n_points=3000):
    """의자: 좌석(위 면만) + 등받이(앞쪽)"""
    points, normals = [], []

    # 좌석 (위 면만)
    seat_w, seat_d = 0.6, 0.5
    n_seat = n_points // 2

    x_seat = np.random.uniform(TARGET[0] - seat_w/2, TARGET[0] + seat_w/2, n_seat)
    y_seat = np.random.uniform(TARGET[1] - seat_d/2, TARGET[1] + seat_d/2, n_seat)
    z_seat = np.full(n_seat, TARGET[2] + 0.4)  # 좌석 높이

    points.extend(np.stack([x_seat, y_seat, z_seat], axis=1))
    normals.extend([np.array([0,0,1])] * n_seat)  # 위쪽 바라봄

    # 등받이 (앞쪽, y+ 방향)
    back_h = 0.8
    back_w = 0.6
    n_back = n_points // 2

    x_back = np.random.uniform(TARGET[0] - back_w/2, TARGET[0] + back_w/2, n_back)
    z_back = np.random.uniform(TARGET[2] + 0.4, TARGET[2] + 0.4 + back_h, n_back)
    y_back = np.full(n_back, TARGET[1] + seat_d/2)  # 좌석 앞쪽

    points.extend(np.stack([x_back, y_back, z_back], axis=1))
    normals.extend([np.array([0,1,0])] * n_back)  # 앞쪽 바라봄

    return np.array(points), np.array(normals)

# ─────────────────────────────────────────────────────────────────────────────
# Voxelization & PCA
# ─────────────────────────────────────────────────────────────────────────────

def voxelize_and_pca(points, point_normals, voxel_size):
    """Voxelize 및 PCA 평탄비 계산, voxel 레벨 법선 생성"""
    origin = points.min(0) - voxel_size
    voxel_indices = np.floor((points - origin) / voxel_size).astype(int)

    # Unique voxels (surface) with inverse index
    unique_voxels, inv = np.unique(voxel_indices, axis=0, return_inverse=True)
    voxel_centers = unique_voxels * voxel_size + origin + voxel_size/2

    # Voxel-level normals: average point normals within each voxel
    voxel_normals = np.zeros((len(unique_voxels), 3))
    for i in range(len(unique_voxels)):
        mask = inv == i
        avg = point_normals[mask].mean(0)
        norm = np.linalg.norm(avg)
        voxel_normals[i] = avg / (norm + 1e-9)

    # PCA
    pca = PCA()
    pca.fit(voxel_centers)
    var_ratio = pca.explained_variance_ratio_

    # Flatness: smallest eigenvalue / largest
    flatness = var_ratio[-1] / var_ratio[0]

    return {
        'n_surface_voxels': len(unique_voxels),
        'voxel_centers': voxel_centers,
        'voxel_normals': voxel_normals,
        'flatness': float(flatness),
        'var_ratio': var_ratio.tolist()
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

objects = {
    'flat-simple': generate_flat_simple,
    'flat-complex': generate_flat_complex,
    'box': generate_box,
    'sphere-small': lambda: generate_sphere(0.5),
    'sphere-large': lambda: generate_sphere(0.7),
    'occluded-chair': generate_chair,
}

results = {}
out_dir = Path('data/test_objects')
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("6개 테스트 물체 생성 및 검증")
print("=" * 80)

for obj_name, obj_func in objects.items():
    print(f"\n{obj_name}:")

    points, normals = obj_func()

    results[obj_name] = {}

    for voxel_size in VOXEL_SIZES:
        vox_info = voxelize_and_pca(points, normals, voxel_size)

        results[obj_name][voxel_size] = vox_info

        print(f"  {voxel_size}m: {vox_info['n_surface_voxels']} voxels, "
              f"flatness={vox_info['flatness']:.3f}")

        # Save npz
        np.savez(
            out_dir / f'{obj_name}_{voxel_size:.3f}.npz',
            points=points,
            normals=normals,
            voxel_centers=vox_info['voxel_centers'],
            voxel_normals=vox_info['voxel_normals'],
            flatness=vox_info['flatness']
        )

# Summary
print("\n" + "=" * 80)
print("PCA 평탄비 요약")
print("=" * 80)
print(f"{'물체':<20} {'0.15m flatness':<15} {'0.075m flatness':<15}")
print("-" * 80)
for obj_name in objects.keys():
    f1 = results[obj_name][0.15]['flatness']
    f2 = results[obj_name][0.075]['flatness']
    print(f"{obj_name:<20} {f1:<15.3f} {f2:<15.3f}")

# Save metadata
metadata = {
    'objects': {
        name: {
            '0.15m': {
                'n_voxels': results[name][0.15]['n_surface_voxels'],
                'flatness': results[name][0.15]['flatness']
            },
            '0.075m': {
                'n_voxels': results[name][0.075]['n_surface_voxels'],
                'flatness': results[name][0.075]['flatness']
            }
        }
        for name in objects.keys()
    }
}

with open(out_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ 물체 데이터 저장: {out_dir}")
print(f"✓ 메타데이터: {out_dir / 'metadata.json'}")
