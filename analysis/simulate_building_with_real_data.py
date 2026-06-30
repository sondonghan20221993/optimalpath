#!/usr/bin/env python3
"""
실제 Phase 1-3 데이터로 건축물 복원 시뮬레이션 재실행
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm

class BuildingSimulationWithRealData:
    def __init__(self):
        self.phase1_file = Path("results/05_controlled_phases/phase1_controlled_comparison.json")

    def load_phase1_data(self):
        """Phase1 실제 데이터 로드"""
        with open(self.phase1_file) as f:
            data = json.load(f)
        return data

    def compute_azimuth_from_coords(self, coords):
        """좌표에서 방위각 계산"""
        building_center = np.array([-33.67, -50.83, 0])

        azimuths = []
        for wp in coords:
            wp_pos = np.array(wp[:2])  # x, y만
            rel_pos = wp_pos - building_center[:2]
            angle = np.arctan2(rel_pos[1], rel_pos[0])
            angle_deg = np.degrees(angle) % 360
            azimuths.append(angle_deg)

        return sorted(azimuths)

    def compute_gap_stats(self, angles):
        """인접 간격 통계"""
        if len(angles) < 2:
            return {'max': 360, 'mean': 360, 'std': 0, 'gaps': [360]}

        gaps = []
        for i in range(len(angles)):
            gap = (angles[(i+1) % len(angles)] - angles[i]) % 360
            gaps.append(gap)

        return {
            'max': max(gaps),
            'mean': np.mean(gaps),
            'std': np.std(gaps),
            'gaps': gaps
        }

    def run_analysis(self):
        """분석 실행"""
        data = self.load_phase1_data()

        print("\n" + "="*70)
        print("BUILDING RECONSTRUCTION WITH REAL PHASE1 DATA")
        print("="*70)

        paths = {
            'Orbit-8': data['orbit_path'],
            'Greedy-8': data['greedy_path'],
            'Hybrid-8': data['hybrid_path']
        }

        results = {}

        for name, coords in paths.items():
            print(f"\n{name}:")
            print("-" * 50)

            # 좌표 출력
            print(f"  Waypoints: {len(coords)}")
            unique_coords = list(set(tuple(c) for c in coords))
            print(f"  Unique positions: {len(unique_coords)}")

            if len(unique_coords) < len(coords):
                print(f"  ⚠️ {len(coords) - len(unique_coords)} 중복!")

            # 방위각 계산
            azimuths = self.compute_azimuth_from_coords(coords)
            gaps = self.compute_gap_stats(azimuths)

            print(f"  Azimuths: {[f'{a:.1f}°' for a in azimuths[:4]]}...")
            print(f"  Max gap: {gaps['max']:.1f}°")
            print(f"  Mean gap: {gaps['mean']:.1f}°")
            print(f"  Std gap: {gaps['std']:.1f}°")

            # 균등성 점수
            if gaps['mean'] > 0:
                uniformity = 1.0 - (gaps['std'] / gaps['mean'])
            else:
                uniformity = 0
            print(f"  Uniformity score: {uniformity:.2f}")

            results[name] = {
                'coords': coords,
                'azimuths': azimuths,
                'gaps': gaps,
                'uniformity': uniformity
            }

        # 비교 분석
        self._comparative_analysis(results)

        # 시각화
        self._visualize_paths(results)

        return results

    def _comparative_analysis(self, results):
        """비교 분석"""
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS")
        print("="*70)

        print("\n[Method Comparison]")
        for name, data in results.items():
            gap_info = data['gaps']
            uni = data['uniformity']
            print(f"  {name:10} → Max gap: {gap_info['max']:6.1f}° | Uniformity: {uni:.2f}")

        print("\n[Interpretation]")
        orbit_gaps = results['Orbit-8']['gaps']['max']
        greedy_gaps = results['Greedy-8']['gaps']['max']

        if greedy_gaps > orbit_gaps * 2:
            print(f"  ✓ Greedy is {greedy_gaps/orbit_gaps:.1f}x worse than Orbit")
            print(f"    → Greedy clustering confirmed!")

        # 최악의 경우 (Greedy의 복제)
        if len(set(tuple(c) for c in results['Greedy-8']['coords'])) < len(results['Greedy-8']['coords']):
            print(f"\n  ⚠️ ISSUE FOUND:")
            print(f"    Greedy has {len(results['Greedy-8']['coords'])} waypoints")
            print(f"    but only {len(set(tuple(c) for c in results['Greedy-8']['coords']))} unique positions")
            print(f"    → Algorithm bug: 100% coverage after Stage 1 leaves no room for Stage 2")

    def _visualize_paths(self, results):
        """시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))

        colors = {'Orbit-8': 'blue', 'Greedy-8': 'red', 'Hybrid-8': 'green'}

        for ax, (name, data) in zip(axes, results.items()):
            angles = np.radians(data['azimuths'])
            radii = np.ones_like(angles)

            ax.scatter(angles, radii, s=150, alpha=0.6, color=colors[name], label=name)
            ax.plot(np.concatenate([angles, [angles[0]]]),
                   np.concatenate([radii, [radii[0]]]),
                   'o-', linewidth=2, color=colors[name])

            gap_info = data['gaps']
            ax.set_ylim(0, 1.5)
            ax.set_title(f"{name}\n(Max gap: {gap_info['max']:.1f}° | Uniformity: {data['uniformity']:.2f})")
            ax.grid(True)

        plt.tight_layout()
        viz_file = Path("results/06_building_sim") / "phase1_real_data_azimuth.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {viz_file}")
        plt.close()

if __name__ == "__main__":
    sim = BuildingSimulationWithRealData()
    results = sim.run_analysis()
