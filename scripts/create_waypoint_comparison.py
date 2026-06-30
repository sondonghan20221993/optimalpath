#!/usr/bin/env python3
"""
Waypoint-only comparison: Greedy vs PB-NBV (Python) vs PB-NBV (C++)
Extracts waypoints and visualizes them as scattered points (no trajectory lines)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# ============ Load Python Results ============
with open('/mnt/c/Users/sdh97/Documents/GitHub/optimalpath/results/greedy_vs_pbnbv/result.json') as f:
    python_results = json.load(f)

greedy_path = python_results['greedy']['path']
pbnbv_py_path = python_results['pbnbv']['path']

# ============ Load C++ Results ============
cpp_path = [
    {'step': 1, 'pos': [15, 0, 5], 'azimuth': 0.0},
    {'step': 2, 'pos': [8.43, -5.42, 10], 'azimuth': 327.2},
    {'step': 3, 'pos': [5.12, -7.04, 11.7], 'azimuth': 306.0},
    {'step': 4, 'pos': [6.23, 2.53, 11.7], 'azimuth': 22.1},
    {'step': 5, 'pos': [6.41, 7.7, 8.51], 'azimuth': 50.2},
    {'step': 6, 'pos': [2.44, -9.78, 11.3], 'azimuth': 283.9},
    {'step': 7, 'pos': [0.955, 10.5, 8.51], 'azimuth': 84.7},
    {'step': 8, 'pos': [-1.08, -9.6, 12.1], 'azimuth': 263.5},
]

# ============ Extract Waypoint Data ============
def extract_waypoints(path, method_name):
    """Extract (x, y, z) coordinates from path"""
    waypoints = []
    for p in path:
        x, y, z = p['pos']
        step = p['step']
        azimuth = p['azimuth']
        waypoints.append({
            'method': method_name,
            'step': step,
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'azimuth': float(azimuth),
        })
    return waypoints

greedy_wp = extract_waypoints(greedy_path, 'Greedy')
pbnbv_py_wp = extract_waypoints(pbnbv_py_path, 'PB-NBV (Python)')
pbnbv_cpp_wp = extract_waypoints(cpp_path, 'PB-NBV (C++)')

all_waypoints = greedy_wp + pbnbv_py_wp + pbnbv_cpp_wp

# ============ Print Data ============
print("\n" + "="*80)
print("WAYPOINT COMPARISON (Points Only)")
print("="*80)
print(f"{'Method':<20} {'Step':<6} {'X':>8} {'Y':>8} {'Z':>8} {'Azimuth':>10}")
print("-"*80)
for wp in sorted(all_waypoints, key=lambda x: (x['method'], x['step'])):
    print(f"{wp['method']:<20} {wp['step']:<6} {wp['x']:>8.2f} {wp['y']:>8.2f} {wp['z']:>8.2f} {wp['azimuth']:>10.1f}°")

# ============ Helper Functions ============
def get_subset(waypoints, method):
    """Filter waypoints by method"""
    return [w for w in waypoints if w['method'] == method]

def sort_by_step(waypoints):
    """Sort waypoints by step"""
    return sorted(waypoints, key=lambda x: x['step'])

# ============ 3D Scatter Visualization ============
fig = plt.figure(figsize=(16, 12))
colors = {'Greedy': '#1f77b4', 'PB-NBV (Python)': '#ff7f0e', 'PB-NBV (C++)': '#2ca02c'}
methods = ['Greedy', 'PB-NBV (Python)', 'PB-NBV (C++)']

# 1. 3D Scatter (all methods)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
for method in methods:
    subset = get_subset(all_waypoints, method)
    xs = [w['x'] for w in subset]
    ys = [w['y'] for w in subset]
    zs = [w['z'] for w in subset]

    ax1.scatter(xs, ys, zs, s=150, label=method, color=colors[method],
               alpha=0.7, edgecolors='black', linewidth=1.5)

    # Add step numbers
    for w in subset:
        ax1.text(w['x'], w['y'], w['z'], str(int(w['step'])),
                fontsize=9, fontweight='bold', ha='center', va='center')

ax1.set_xlabel('X (m)', fontweight='bold')
ax1.set_ylabel('Y (m)', fontweight='bold')
ax1.set_zlabel('Z (m)', fontweight='bold')
ax1.set_title('3D Waypoints Comparison', fontweight='bold', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. XY Plane (top view)
ax2 = fig.add_subplot(2, 3, 2)
for method in methods:
    subset = get_subset(all_waypoints, method)
    xs = [w['x'] for w in subset]
    ys = [w['y'] for w in subset]

    ax2.scatter(xs, ys, s=150, label=method, color=colors[method],
               alpha=0.7, edgecolors='black', linewidth=1.5)

    for w in subset:
        ax2.annotate(str(int(w['step'])), (w['x'], w['y']),
                    fontsize=9, fontweight='bold', ha='center', va='center')

ax2.set_xlabel('X (m)', fontweight='bold')
ax2.set_ylabel('Y (m)', fontweight='bold')
ax2.set_title('XY Plane (Top View)', fontweight='bold', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# 3. XZ Plane (side view)
ax3 = fig.add_subplot(2, 3, 3)
for method in methods:
    subset = get_subset(all_waypoints, method)
    xs = [w['x'] for w in subset]
    zs = [w['z'] for w in subset]

    ax3.scatter(xs, zs, s=150, label=method, color=colors[method],
               alpha=0.7, edgecolors='black', linewidth=1.5)

    for w in subset:
        ax3.annotate(str(int(w['step'])), (w['x'], w['z']),
                    fontsize=9, fontweight='bold', ha='center', va='center')

ax3.set_xlabel('X (m)', fontweight='bold')
ax3.set_ylabel('Z (m)', fontweight='bold')
ax3.set_title('XZ Plane (Side View)', fontweight='bold', fontsize=12)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# 4. Z-height variation
ax4 = fig.add_subplot(2, 3, 4)
for method in methods:
    subset = sort_by_step(get_subset(all_waypoints, method))
    steps = [w['step'] for w in subset]
    zs = [w['z'] for w in subset]

    ax4.plot(steps, zs, marker='o', linewidth=2, markersize=8,
            label=method, color=colors[method], alpha=0.7)

ax4.set_xlabel('Step', fontweight='bold')
ax4.set_ylabel('Altitude Z (m)', fontweight='bold')
ax4.set_title('Altitude Variation', fontweight='bold', fontsize=12)
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, 9))

# 5. Azimuth variation
ax5 = fig.add_subplot(2, 3, 5)
for method in methods:
    subset = sort_by_step(get_subset(all_waypoints, method))
    steps = [w['step'] for w in subset]
    azimuths = [w['azimuth'] for w in subset]

    ax5.plot(steps, azimuths, marker='s', linewidth=2, markersize=8,
            label=method, color=colors[method], alpha=0.7)

ax5.set_xlabel('Step', fontweight='bold')
ax5.set_ylabel('Azimuth (degrees)', fontweight='bold')
ax5.set_title('Heading Direction Variation', fontweight='bold', fontsize=12)
ax5.legend(loc='best')
ax5.grid(True, alpha=0.3)
ax5.set_xticks(range(1, 9))

# 6. Distance from origin
ax6 = fig.add_subplot(2, 3, 6)
for method in methods:
    subset = sort_by_step(get_subset(all_waypoints, method))
    steps = [w['step'] for w in subset]
    dists = [np.sqrt(w['x']**2 + w['y']**2) for w in subset]

    ax6.plot(steps, dists, marker='D', linewidth=2, markersize=8,
            label=method, color=colors[method], alpha=0.7)

ax6.set_xlabel('Step', fontweight='bold')
ax6.set_ylabel('Horizontal Distance (m)', fontweight='bold')
ax6.set_title('Distance from Origin (XY Plane)', fontweight='bold', fontsize=12)
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)
ax6.set_xticks(range(1, 9))

plt.tight_layout()
plt.savefig('/mnt/c/Users/sdh97/Documents/GitHub/optimalpath/results/waypoints_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: waypoints_comparison.png")

# ============ Save Waypoint Data as CSV ============
csv_file = '/mnt/c/Users/sdh97/Documents/GitHub/optimalpath/results/waypoints_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['method', 'step', 'x', 'y', 'z', 'azimuth'])
    writer.writeheader()
    for wp in sorted(all_waypoints, key=lambda x: (x['method'], x['step'])):
        writer.writerow(wp)

print("✓ Data saved: waypoints_data.csv")

# ============ Summary Statistics ============
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for method in methods:
    subset = get_subset(all_waypoints, method)
    zs = [w['z'] for w in subset]
    azimuths = [w['azimuth'] for w in subset]
    horiz_dists = [np.sqrt(w['x']**2 + w['y']**2) for w in subset]

    print(f"\n{method}:")
    print(f"  Steps: {len(subset)}")
    print(f"  Avg Z: {np.mean(zs):.2f} m")
    print(f"  Z range: {min(zs):.2f} - {max(zs):.2f} m")
    print(f"  Avg Azimuth: {np.mean(azimuths):.1f}°")
    print(f"  Avg Horizontal Distance: {np.mean(horiz_dists):.2f} m")
    print(f"  Horiz Dist Range: {min(horiz_dists):.2f} - {max(horiz_dists):.2f} m")

print("\n" + "="*80)
