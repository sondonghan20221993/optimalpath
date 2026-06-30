"""
pbnbv_uniform_check.py — UNIFORM(균등) 비행 데이터로 복원율 + 실제 PB-NBV 추가시점 검증.

가설: '균등 궤도라면 방위 구멍이 없어 PB-NBV 추가 시점 ≈ 0'.
비균등(real_test_new)에서는 실측 96.2% 도달 후 PB-NBV가 22개 추가 시점을 제안했음.
같은 분석을 UNIFORM(real_test_uniform) 으로 돌려 비교한다.

평가: raycast 가시 ∩ 입사각<60°, 분모=관측가능 표면.
PB-NBV: information_gain_ellipsoid_rank (ellipsoid 0.5^r), argmax.
출력: results/pbnbv_uniform_check.png, results/pbnbv_uniform_check.json
"""
import sys, types, json, glob, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
P.MAX_DIST = MAX_DIST
A.RAYCAST_OCCLUSION = True

t = np.array(P.TARGET)
obj = P._pts_raw
obj_n = np.load(ROOT/'real_test'/'airsim_gt_pts.npz')['normals'].astype(float)
top = obj[:, 2].min()
az = lambda c: math.degrees(math.atan2(c[1]-t[1], c[0]-t[0])) % 360


def load_poses(d):
    pos = []
    for f in sorted(glob.glob(str(d/'meta'/'*.json'))):
        p = json.load(open(f))['camera']['pose']['position']
        pos.append([p['x'], p['y'], p['z']])
    return np.array(pos)


def obj_mask(c):
    live = np.ones(len(obj), bool)
    _, vis = A.information_gain(c, t, obj, live, P.FOV_DEG, MAX_DIST)
    v = c - obj; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
    return vis & ((obj_n*v).sum(1) >= COS)


def residual_pbnbv(rec0, dense, dense_recon, observable):
    """rec0 상태에서 실제 PB-NBV(ellipsoid 0.5^r) argmax 로 추가 시점 선택."""
    rec = rec0.copy(); used = np.zeros(len(dense), bool)
    add_idx, add_gain = [], []
    n_obs = int(observable.sum())
    curve = [100*rec.sum()/n_obs]
    for _ in range(len(dense)):
        remaining = observable & ~rec
        if remaining.sum() == 0: break
        best, bS = -1, 0.0
        for i in range(len(dense)):
            if used[i]: continue
            s, _ = A.information_gain_ellipsoid_rank(
                dense[i], t, obj, remaining, P.FOV_DEG, MAX_DIST, pts_normals=obj_n)
            if s > bS: bS, best = s, i
        if best < 0 or bS <= 0: break
        used[best] = True
        gain = int((dense_recon[best] & remaining).sum())
        if gain == 0: continue
        rec |= dense_recon[best] & observable
        add_idx.append(best); add_gain.append(gain)
        curve.append(100*rec.sum()/n_obs)
    return add_idx, add_gain, curve, rec


def main():
    # 관측가능 표면 + 후보 복원마스크 캐시
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=MAX_DIST)
                       for a in [3., 5., 7.]])
    dense_alt = np.concatenate([[a]*(len(dense)//3) for a in [3., 5., 7.]])[:len(dense)]
    dense_recon = [obj_mask(c) for c in dense]
    observable = np.zeros(len(obj), bool)
    for m in dense_recon: observable |= m
    n_obs = int(observable.sum())

    # UNIFORM 포즈
    u3 = load_poses(UNI/'real_test_3m_uniform')
    u7 = load_poses(UNI/'real_test_7m_uniform')
    uA = np.vstack([u3, u7])

    def cov_of(cams):
        rec = np.zeros(len(obj), bool)
        for c in cams: rec |= obj_mask(c) & observable
        return rec
    rec3 = cov_of(u3); rec7 = cov_of(u7); recU = cov_of(uA)
    c3 = 100*rec3.sum()/n_obs; c7 = 100*rec7.sum()/n_obs; cU = 100*recU.sum()/n_obs
    n_res_U = int((observable & ~recU).sum())

    print("== UNIFORM 복원율 (관측가능 표면 기준) ==")
    print(f"  3m 단독(17): {c3:.1f}%")
    print(f"  7m 단독(17): {c7:.1f}%")
    print(f"  3m+7m (34) : {cU:.1f}%  (잔여 {n_res_U} pts)")

    # PB-NBV 추가 시점: UNIFORM
    addU, gainU, curveU, recU_after = residual_pbnbv(recU, dense, dense_recon, observable)
    cU_after = 100*recU_after.sum()/n_obs

    # 비교용: 비균등(real_test_new)도 동일 절차로
    n3 = load_poses(NEW/'real_test_3m'); n7 = load_poses(NEW/'real_test_7m')
    recN = cov_of(np.vstack([n3, n7])); cN = 100*recN.sum()/n_obs
    n_res_N = int((observable & ~recN).sum())
    addN, gainN, curveN, recN_after = residual_pbnbv(recN, dense, dense_recon, observable)

    print("\n== PB-NBV 추가 시점 (실제 PB-NBV, ellipsoid 0.5^r) ==")
    print(f"  UNIFORM  : 실측 {cU:.1f}% → 추가 {len(addU)}개 → {cU_after:.1f}%  (잔여 {n_res_U})")
    print(f"  비균등   : 실측 {cN:.1f}% → 추가 {len(addN)}개 → {100*recN_after.sum()/n_obs:.1f}%  (잔여 {n_res_N})")
    if addU:
        print("  UNIFORM 추가 시점(고도·방위·이득):")
        for r, i in enumerate(addU):
            print(f"    +{r+1}: {dense_alt[i]:.0f}m  az={az(dense[i]):5.1f}°  gain={gainU[r]} pts")

    out = {
        "name": "pbnbv_uniform_check",
        "metric": "raycast∩incidence<60, denom=observable surface",
        "pbnbv": "information_gain_ellipsoid_rank (0.5^r), argmax",
        "observable_surface_points": n_obs,
        "uniform": {
            "cov_3m_pct": round(c3, 2), "cov_7m_pct": round(c7, 2),
            "cov_combined_pct": round(cU, 2), "residual_pts": n_res_U,
            "pbnbv_additional_viewpoints": len(addU),
            "coverage_after_additions_pct": round(cU_after, 2),
            "additional": [{"rank": r+1, "altitude_m": float(dense_alt[i]),
                            "azimuth_deg": round(az(dense[i]), 1), "gain_pts": int(gainU[r])}
                           for r, i in enumerate(addU)],
        },
        "nonuniform_for_comparison": {
            "cov_combined_pct": round(cN, 2), "residual_pts": n_res_N,
            "pbnbv_additional_viewpoints": len(addN),
        },
    }
    js = ROOT/'results'/'pbnbv_uniform_check.json'
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {js.name}")

    # ── 시각화 ──
    C = {'txt':'#e6edf3','ax':'#161b22','bg':'#0d1117','grid':'#21262d','obj':'#30363d',
         'seen':'#3fb950','res':'#f85149','add':'#ffa657','uni':'#58a6ff','non':'#f778ba','tgt':'#f778ba'}
    fig = plt.figure(figsize=(15.5, 6.4), facecolor=C['bg'])

    # (a) UNIFORM top-down: 잔여 + PB-NBV 추가
    ax = fig.add_subplot(121, facecolor=C['ax'])
    residual = observable & ~recU
    ax.scatter(obj[recU,0], obj[recU,1], c=C['seen'], s=5, alpha=0.4, zorder=2,
               label=f'covered ({int(recU.sum())})')
    if residual.any():
        ax.scatter(obj[residual,0], obj[residual,1], c=C['res'], s=24, zorder=4,
                   edgecolors='white', linewidths=0.3, label=f'missed ({n_res_U})')
    ax.scatter(uA[:,0], uA[:,1], c=C['uni'], s=30, marker='D', alpha=0.75, zorder=3,
               edgecolors='white', linewidths=0.3, label=f'uniform poses ({len(uA)})')
    if addU:
        addc = dense[addU]
        ax.scatter(addc[:,0], addc[:,1], c=C['add'], s=170, marker='*', zorder=6,
                   edgecolors='white', linewidths=0.7, label=f'PB-NBV added ({len(addU)})')
        for r, i in enumerate(addU):
            ax.text(dense[i,0], dense[i,1], str(r+1), color='black', fontsize=7,
                    ha='center', va='center', zorder=7, fontweight='bold')
    ax.scatter(*t[:2], marker='*', s=240, c=C['tgt'], zorder=9)
    ax.set_title(f'(a) UNIFORM flight residual + PB-NBV\n'
                 f'uniform {cU:.1f}% → +{len(addU)} viewpoints → {cU_after:.1f}%',
                 color=C['txt'], fontsize=11, pad=6)
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt']); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=8.5, loc='upper right')

    # (b) PB-NBV 추가 시점 수: uniform vs 비균등
    ax2 = fig.add_subplot(122, facecolor=C['ax'])
    xu = np.arange(len(curveU)); xn = np.arange(len(curveN))
    ax2.plot(xn, curveN, '-s', color=C['non'], lw=2, ms=5, alpha=0.9,
             label=f'NON-uniform: +{len(addN)} viewpoints')
    ax2.plot(xu, curveU, '-*', color=C['uni'], lw=2.2, ms=11,
             label=f'UNIFORM: +{len(addU)} viewpoints')
    ax2.set_title(f'(b) PB-NBV additional viewpoints needed\n'
                  f'uniform {len(addU)} vs non-uniform {len(addN)}  '
                  f'(fewer gaps → fewer additions)', color=C['txt'], fontsize=11, pad=6)
    ax2.set_xlabel('PB-NBV additional viewpoints after the flight', color=C['txt'])
    ax2.set_ylabel('observable-surface coverage (%)', color=C['txt'])
    ax2.tick_params(colors=C['txt'])
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax2.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9.5, loc='lower right')

    msg = (f'UNIFORM needs +{len(addU)} PB-NBV viewpoints vs NON-uniform +{len(addN)} — '
           'uniform orbit leaves fewer azimuth gaps')
    fig.suptitle(f'Uniform vs non-uniform flight: does PB-NBV still add viewpoints?  —  {msg}',
                 color=C['txt'], fontsize=12)
    png = ROOT/'results'/'pbnbv_uniform_check.png'
    fig.savefig(png, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {png.name}")


if __name__ == '__main__':
    main()
