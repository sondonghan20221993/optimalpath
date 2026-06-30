"""
pbnbv_projection_uniform.py — raycast 대신 **projection 기반(실제 PB-NBV)** 으로
uniform vs 비균등 비행의 복원율 + PB-NBV 추가시점 비교.

핵심(논문 PB-NBV):
  1) GT 점군을 K개 ELLIPSOID 단위로 클러스터링 (center, normal, 투영반경, 소속점)
  2) 카메라마다 ellipsoid를 이미지평면에 투영:
       - in-frustum(시야각) ∩ 정면(normal·view≥cos60) ∩ '투영 가림 없음'
       - 가림 = 더 가까운 ellipsoid의 투영 원이 덮으면 occluded (2D, 해상도 무관)
  3) 관측가능면 = 조밀 후보들이 투영으로 본 ellipsoid 합집합 (raycast bin 해상도 의존 제거)
  4) PB-NBV 이득 = 새로 보이는 미관측 ellipsoid의 투영 '면적'(=소속 점수), argmax
출력: results/pbnbv_projection_uniform.png, .json
"""
import sys, types, json, glob, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P
import pbnbv_path as A

ROOT = HERE.parent
UNI = ROOT / 'real_test_uniform'
NEW = ROOT / 'real_test_new'
MAX_DIST = 13.0
COS = math.cos(math.radians(60))
HALF_FOV = math.radians(89.9/2)
COS_FOV = math.cos(HALF_FOV)
OCC_FACTOR = 0.8        # 투영 가림 민감도
K_ELL = 90              # ellipsoid 개수

t = np.array(P.TARGET)
obj = P._pts_raw
obj_n = np.load(ROOT/'real_test'/'airsim_gt_pts.npz')['normals'].astype(float)
top = obj[:, 2].min()
WORLD_UP = np.array([0., 0., -1.])     # NED: up = -z
az = lambda c: math.degrees(math.atan2(c[1]-t[1], c[0]-t[0])) % 360


def load_poses(d):
    pos = []
    for f in sorted(glob.glob(str(d/'meta'/'*.json'))):
        p = json.load(open(f))['camera']['pose']['position']
        pos.append([p['x'], p['y'], p['z']])
    return np.array(pos)


def build_ellipsoids(K):
    km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(obj)
    lab = km.labels_
    cen, nrm, rad, pts_idx, npts = [], [], [], [], []
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) == 0: continue
        c = obj[idx].mean(0)
        n = obj_n[idx].mean(0); n /= np.linalg.norm(n)+1e-9
        r = np.linalg.norm(obj[idx]-c, axis=1).mean()    # 평균 퍼짐(투영반경 기준)
        cen.append(c); nrm.append(n); rad.append(max(r, 0.03))
        pts_idx.append(idx); npts.append(len(idx))
    return (np.array(cen), np.array(nrm), np.array(rad),
            pts_idx, np.array(npts))


def project_visible(c, cen, nrm, rad):
    """카메라 c(타겟 look-at)에서 투영으로 보이는 ellipsoid bool mask."""
    f = t - c; f /= np.linalg.norm(f)+1e-9
    right = np.cross(f, WORLD_UP); right /= np.linalg.norm(right)+1e-9
    up = np.cross(right, f)
    v = cen - c
    depth = v @ f
    dist = np.linalg.norm(v, axis=1)+1e-9
    cosang = depth / dist
    front = depth > 1e-3
    inview = cosang > COS_FOV
    vv = (c - cen) / (np.linalg.norm(c-cen, axis=1, keepdims=True)+1e-9)
    facing = (nrm*vv).sum(1) >= COS
    cand = front & inview & facing & (dist < MAX_DIST)
    # 이미지평면 좌표 + 투영반경
    u = (v @ right) / np.maximum(depth, 1e-3)
    w = (v @ up) / np.maximum(depth, 1e-3)
    prad = rad / np.maximum(depth, 1e-3)
    idx = np.where(cand)[0]
    order = idx[np.argsort(depth[idx])]      # 가까운 것부터
    occluded = np.zeros(len(cen), bool)
    for a_pos in range(len(order)):
        a = order[a_pos]
        for b in order[:a_pos]:              # a보다 가까운 것들
            if occluded[b]:
                continue
            if math.hypot(u[a]-u[b], w[a]-w[b]) < prad[b]*OCC_FACTOR:
                occluded[a] = True; break
    return cand & ~occluded


def main():
    cen, nrm, rad, pts_idx, npts = build_ellipsoids(K_ELL)
    Kc = len(cen)

    # 후보(조밀 링) 투영 가시 마스크 → 관측가능 ellipsoid
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=MAX_DIST)
                       for a in [3., 5., 7.]])
    dense_alt = np.concatenate([[a]*(len(dense)//3) for a in [3., 5., 7.]])[:len(dense)]
    dense_vis = [project_visible(c, cen, nrm, rad) for c in dense]
    observable = np.zeros(Kc, bool)
    for m in dense_vis: observable |= m
    obs_pts = int(npts[observable].sum())
    print(f"ellipsoid {Kc}개 | 관측가능 ellipsoid {int(observable.sum())}/{Kc} "
          f"= 점 {obs_pts}/{len(obj)} ({100*obs_pts/len(obj):.1f}%)")

    def flight_cov(cams):
        seen = np.zeros(Kc, bool)
        for c in cams: seen |= project_visible(c, cen, nrm, rad)
        seen &= observable
        return seen, 100*npts[seen].sum()/obs_pts

    def residual_pbnbv(seen0):
        seen = seen0.copy(); used = np.zeros(len(dense), bool)
        add_idx, curve = [], [100*npts[seen].sum()/obs_pts]
        for _ in range(len(dense)):
            remaining = observable & ~seen
            if remaining.sum() == 0: break
            best, bg = -1, 0
            for i in range(len(dense)):
                if used[i]: continue
                g = int(npts[dense_vis[i] & remaining].sum())   # 투영 면적(=점수) 이득
                if g > bg: bg, best = g, i
            if best < 0 or bg == 0: break
            used[best] = True; seen |= dense_vis[best] & observable
            add_idx.append(best); curve.append(100*npts[seen].sum()/obs_pts)
        return add_idx, curve, seen

    # UNIFORM
    u3, u7 = load_poses(UNI/'real_test_3m_uniform'), load_poses(UNI/'real_test_7m_uniform')
    seenU, covU = flight_cov(np.vstack([u3, u7]))
    c3 = flight_cov(u3)[1]; c7 = flight_cov(u7)[1]
    addU, curveU, _ = residual_pbnbv(seenU)
    # 비균등
    n3, n7 = load_poses(NEW/'real_test_3m'), load_poses(NEW/'real_test_7m')
    seenN, covN = flight_cov(np.vstack([n3, n7]))
    addN, curveN, _ = residual_pbnbv(seenN)

    print("\n== projection 기반 복원율 (관측가능면) ==")
    print(f"  UNIFORM : 3m {c3:.1f}% | 7m {c7:.1f}% | 결합 {covU:.1f}%  → PB-NBV 추가 {len(addU)}개 → {curveU[-1]:.1f}%")
    print(f"  비균등  :                                결합 {covN:.1f}%  → PB-NBV 추가 {len(addN)}개 → {curveN[-1]:.1f}%")

    out = {
        "name": "pbnbv_projection_uniform",
        "method": "PROJECTION-based PB-NBV (ellipsoid projection, 2D occlusion) — NO raycasting",
        "n_ellipsoids": Kc, "observable_points": obs_pts,
        "observable_pct_of_full": round(100*obs_pts/len(obj), 2),
        "uniform": {"cov_3m": round(c3,2), "cov_7m": round(c7,2), "cov_combined": round(covU,2),
                    "pbnbv_additional": len(addU), "final": round(curveU[-1],2)},
        "nonuniform": {"cov_combined": round(covN,2),
                       "pbnbv_additional": len(addN), "final": round(curveN[-1],2)},
        "uniform_additional_az": [round(az(dense[i]),1) for i in addU],
        "nonuniform_additional_az": [round(az(dense[i]),1) for i in addN],
    }
    js = ROOT/'results'/'pbnbv_projection_uniform.json'
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {js.name}")

    # ── 시각화 ──
    C = {'txt':'#e6edf3','ax':'#161b22','bg':'#0d1117','grid':'#21262d','obj':'#30363d',
         'seen':'#3fb950','res':'#f85149','add':'#ffa657','uni':'#58a6ff','non':'#f778ba','tgt':'#f778ba'}
    # ellipsoid→점 마스크 헬퍼
    def pmask(ellmask):
        m = np.zeros(len(obj), bool)
        for k in np.where(ellmask)[0]: m[pts_idx[k]] = True
        return m
    fig = plt.figure(figsize=(15.5, 6.4), facecolor=C['bg'])

    ax = fig.add_subplot(121, facecolor=C['ax'])
    pseen = pmask(seenU); pres = pmask(observable & ~seenU)
    ax.scatter(obj[pseen,0], obj[pseen,1], c=C['seen'], s=5, alpha=0.4, zorder=2, label=f'covered')
    if pres.any():
        ax.scatter(obj[pres,0], obj[pres,1], c=C['res'], s=22, zorder=4,
                   edgecolors='white', linewidths=0.3, label='missed (projection)')
    uA = np.vstack([u3, u7])
    ax.scatter(uA[:,0], uA[:,1], c=C['uni'], s=28, marker='D', alpha=0.7, zorder=3,
               edgecolors='white', linewidths=0.3, label=f'uniform poses ({len(uA)})')
    if addU:
        ac = dense[addU]
        ax.scatter(ac[:,0], ac[:,1], c=C['add'], s=170, marker='*', zorder=6,
                   edgecolors='white', linewidths=0.7, label=f'PB-NBV added ({len(addU)})')
        for r, i in enumerate(addU):
            ax.text(dense[i,0], dense[i,1], str(r+1), color='black', fontsize=7,
                    ha='center', va='center', zorder=7, fontweight='bold')
    ax.scatter(*t[:2], marker='*', s=240, c=C['tgt'], zorder=9)
    ax.set_title(f'(a) UNIFORM — projection-based residual + PB-NBV\n'
                 f'uniform {covU:.1f}% → +{len(addU)} → {curveU[-1]:.1f}%',
                 color=C['txt'], fontsize=11, pad=6)
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt']); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=8.5, loc='upper right')

    ax2 = fig.add_subplot(122, facecolor=C['ax'])
    ax2.plot(np.arange(len(curveN)), curveN, '-s', color=C['non'], lw=2, ms=5, alpha=0.9,
             label=f'NON-uniform: +{len(addN)} viewpoints')
    ax2.plot(np.arange(len(curveU)), curveU, '-*', color=C['uni'], lw=2.2, ms=11,
             label=f'UNIFORM: +{len(addU)} viewpoints')
    ax2.set_title('(b) PROJECTION-based PB-NBV additions\n(no raycast resolution circularity)',
                  color=C['txt'], fontsize=11, pad=6)
    ax2.set_xlabel('PB-NBV additional viewpoints after the flight', color=C['txt'])
    ax2.set_ylabel('observable-surface coverage (%)', color=C['txt'])
    ax2.tick_params(colors=C['txt'])
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax2.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9.5, loc='lower right')

    fig.suptitle('PROJECTION-based PB-NBV (real method, no raycasting) — '
                 f'UNIFORM +{len(addU)} vs NON-uniform +{len(addN)} additional viewpoints',
                 color=C['txt'], fontsize=12)
    png = ROOT/'results'/'pbnbv_projection_uniform.png'
    fig.savefig(png, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {png.name}")


if __name__ == '__main__':
    main()
