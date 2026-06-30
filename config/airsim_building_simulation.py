#!/usr/bin/env python3
"""
AirSim 시뮬레이션: 건축물 모형 주변 3개 경로 비행
- Orbit-16: 균등 원형 배치 (45도 간격)
- Greedy-16: Coverage 기반 탐욕 선택
- Hybrid-16: Orbit-8 + Greedy-8 조합

Output: 각 경로별 카메라 이미지 + pose 정보
"""

import airsim
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

class BuildingSimulation:
    def __init__(self, building_center=(0, 0, 0), radius=5.0):
        """
        Args:
            building_center: 건축물 중심 (X, Y, Z in NED)
            radius: 비행 원형 반경
        """
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.building_center = np.array(building_center)
        self.radius = radius
        self.altitude = -2.0  # NED: 음수 = 위

        # 결과 저장 디렉토리
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("results/airsim_building") / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {self.output_dir}")

    def generate_orbit_path(self, n_points=16):
        """균등 원형 경로 생성"""
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        path = []

        for i, angle in enumerate(angles):
            x = self.building_center[0] + self.radius * np.cos(angle)
            y = self.building_center[1] + self.radius * np.sin(angle)
            z = self.building_center[2] + self.altitude
            path.append({'x': x, 'y': y, 'z': z, 'index': i, 'type': 'orbit'})

        return path

    def generate_greedy_path(self, n_points=16):
        """
        Greedy 경로: Coverage 기반 선택
        (단순화: 원 위의 점에서 탐욕적 선택)
        """
        candidates = []
        num_candidates = n_points * 3  # 후보 증가

        for i in range(num_candidates):
            angle = 2*np.pi * i / num_candidates
            x = self.building_center[0] + self.radius * np.cos(angle)
            y = self.building_center[1] + self.radius * np.sin(angle)
            z = self.building_center[2] + self.altitude
            candidates.append({
                'x': x, 'y': y, 'z': z,
                'angle': angle,
                'coverage': self._compute_coverage_score(angle)
            })

        # Greedy 선택: 높은 coverage score 순서
        sorted_cand = sorted(candidates, key=lambda c: c['coverage'], reverse=True)
        path = []
        selected_indices = set()

        for i in range(n_points):
            for j, cand in enumerate(sorted_cand):
                if j not in selected_indices:
                    cand['index'] = i
                    cand['type'] = 'greedy'
                    path.append(cand)
                    selected_indices.add(j)
                    break

        # 각도순 정렬
        path.sort(key=lambda p: p['angle'])
        return path

    def generate_hybrid_path(self):
        """Hybrid: Orbit-8 (Stage 1) + Greedy-8 (Stage 2)"""
        orbit_8 = self.generate_orbit_path(n_points=8)

        # Stage 2: Orbit 사이 공백 채우기
        greedy_8 = self.generate_greedy_path(n_points=8)

        # Orbit과 다른 위치에서 선택
        for wp in greedy_8:
            wp['stage'] = 2

        hybrid = orbit_8 + greedy_8
        hybrid.sort(key=lambda p: p.get('angle', 0))

        return hybrid

    def _compute_coverage_score(self, angle):
        """커버리지 점수: 중심 근처일수록 높음 (간단한 모델)"""
        # Greedy의 특징: 특정 각도 편중 (불균등)
        distance_from_zero = min(abs(angle), 2*np.pi - abs(angle))
        return 1.0 / (1.0 + distance_from_zero)

    def fly_path(self, path, path_name, speed=1.0):
        """
        경로를 따라 비행하며 이미지 수집

        Args:
            path: waypoint 리스트
            path_name: 경로 이름 ('orbit', 'greedy', 'hybrid')
            speed: 이동 속도 (m/s)
        """
        print(f"\n{'='*60}")
        print(f"Flying {path_name.upper()} path ({len(path)} waypoints)")
        print(f"{'='*60}")

        # Takeoff
        print("Taking off...")
        self.client.takeoffAsync().join()

        # 이미지/pose 저장 디렉토리
        image_dir = self.output_dir / path_name / "images"
        pose_dir = self.output_dir / path_name / "poses"
        image_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        poses = []

        for idx, wp in enumerate(path):
            target_pos = airsim.Vector3r(wp['x'], wp['y'], wp['z'])

            # 이동
            print(f"  [{idx+1}/{len(path)}] Moving to ({wp['x']:.2f}, {wp['y']:.2f}, {wp['z']:.2f})")
            self.client.moveToPositionAsync(
                target_pos.x_val,
                target_pos.y_val,
                target_pos.z_val,
                speed
            ).join()

            # 카메라 이미지 수집
            responses = self.client.simGetImages([
                airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
            ])

            if responses and len(responses) > 0:
                img = responses[0]
                img_path = image_dir / f"frame_{idx:04d}.png"
                airsim.write_png(str(img_path), img.image_data_uint8)

            # Pose 수집
            pose = self.client.simGetObjectPose('Building')
            camera_pose = self.client.simGetCameraPose(0)

            pose_info = {
                'index': idx,
                'position': {'x': wp['x'], 'y': wp['y'], 'z': wp['z']},
                'building_pose': {
                    'position': {
                        'x': pose.position.x_val,
                        'y': pose.position.y_val,
                        'z': pose.position.z_val
                    },
                    'orientation': {
                        'w': pose.orientation.w_val,
                        'x': pose.orientation.x_val,
                        'y': pose.orientation.y_val,
                        'z': pose.orientation.z_val
                    }
                },
                'camera_pose': {
                    'position': {
                        'x': camera_pose.position.x_val,
                        'y': camera_pose.position.y_val,
                        'z': camera_pose.position.z_val
                    },
                    'orientation': {
                        'w': camera_pose.orientation.w_val,
                        'x': camera_pose.orientation.x_val,
                        'y': camera_pose.orientation.y_val,
                        'z': camera_pose.orientation.z_val
                    }
                }
            }
            poses.append(pose_info)

        # Landing
        print("Landing...")
        self.client.landAsync().join()

        # Pose 저장
        poses_file = pose_dir / "poses.json"
        with open(poses_file, 'w') as f:
            json.dump(poses, f, indent=2)

        print(f"✓ Collected {len(poses)} poses")
        print(f"✓ Images saved to {image_dir}")
        print(f"✓ Poses saved to {poses_file}")

        return poses

    def run_simulation(self):
        """전체 시뮬레이션 실행"""
        print("\n" + "="*60)
        print("AIRSIM BUILDING SIMULATION")
        print("="*60)

        # 3개 경로 생성
        paths = {
            'orbit': self.generate_orbit_path(n_points=16),
            'greedy': self.generate_greedy_path(n_points=16),
            'hybrid': self.generate_hybrid_path()
        }

        # 경로 정보 저장
        for name, path in paths.items():
            path_file = self.output_dir / f"{name}_path.json"
            with open(path_file, 'w') as f:
                json.dump(path, f, indent=2)
            print(f"✓ Generated {name} path with {len(path)} waypoints")

        # 각 경로 비행
        all_poses = {}
        for path_name, path in paths.items():
            try:
                poses = self.fly_path(path, path_name, speed=1.0)
                all_poses[path_name] = poses
            except Exception as e:
                print(f"⚠ Error flying {path_name}: {e}")

        # 최종 보고서
        summary = {
            'timestamp': self.timestamp,
            'building_center': self.building_center.tolist(),
            'radius': self.radius,
            'altitude': self.altitude,
            'paths': {
                name: {
                    'waypoints': len(path),
                    'max_azimuth_gap': self._compute_max_gap([p.get('angle', 0) for p in path])
                }
                for name, path in paths.items()
            },
            'results_dir': str(self.output_dir)
        }

        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Simulation complete!")
        print(f"✓ All results saved to {self.output_dir}")
        print(f"✓ Summary: {summary_file}")

        return self.output_dir

    def _compute_max_gap(self, angles):
        """최대 인접 간격 계산"""
        if len(angles) < 2:
            return 360
        sorted_angles = sorted(angles)
        gaps = []
        for i in range(len(sorted_angles)):
            gap = (sorted_angles[(i+1) % len(sorted_angles)] - sorted_angles[i]) % 360
            gaps.append(gap)
        return max(gaps) if gaps else 360


if __name__ == "__main__":
    sim = BuildingSimulation(
        building_center=(0, 0, 0),
        radius=5.0
    )

    result_dir = sim.run_simulation()
    print(f"\n🎉 Results: {result_dir}")
