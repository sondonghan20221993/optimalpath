"""
clean_outliers.py — Statistical Outlier Removal (SOR)
real_test_pts_normals.npz 의 공중 노이즈 점 제거.
각 점의 16-NN 평균거리가 μ+1σ 초과면 이상치로 제거. points·normals 동시 적용.
"""
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "real_test" / "real_test_pts_normals.npz"
OUT  = ROOT / "real_test" / "real_test_pts_normals_clean.npz"

K, ALPHA = 16, 1.0

d = np.load(SRC)
pts, nrm = d["points"].astype(float), d["normals"].astype(float)
N = len(pts)

tree = cKDTree(pts)
dd, _ = tree.query(pts, k=K + 1)
mdist = dd[:, 1:].mean(1)
mu, sd = mdist.mean(), mdist.std()
keep = mdist < mu + ALPHA * sd

pts_c, nrm_c = pts[keep], nrm[keep]
np.savez(OUT, points=pts_c, normals=nrm_c)

def bb(p): return (p.max(0) - p.min(0))
print(f"SOR (k={K}, μ+{ALPHA}σ): {N} → {keep.sum()}점 (제거 {(~keep).sum()}개)")
print(f"  bbox 전: {bb(pts).round(2)}")
print(f"  bbox 후: {bb(pts_c).round(2)}")
print(f"  Z범위 전: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")
print(f"  Z범위 후: [{pts_c[:,2].min():.2f}, {pts_c[:,2].max():.2f}]  ← obj_z_top 정상화")
print(f"✓ 저장: {OUT}")
