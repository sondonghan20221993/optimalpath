"""
pbnbv_residual_actual.py — 실측 비행 '이후' **실제 PB-NBV**(ellipsoid 0.5^r 이득)로
                            추가 시점을 제안하는가?

앞선 pbnbv_residual_3m7m.py 는 단순 점-카운트(U=gain/dist)였음.
이번엔 논문 PB-NBV의 핵심 이득함수 `information_gain_ellipsoid_rank`:
  hard raycast → visible 미관측 점을 KMeans 클러스터(=ellipsoid 단위) →
  카메라 거리순 rank r → score = Σ_k count_k · 0.5^r_k.
선택 규칙도 논문대로 **argmax score**(거리가중 안 함).

평가면/복원지표는 동일: raycast 가시 ∩ 입사각<60°, 분모=관측가능 표면.
출력: results/pbnbv_residual_actual.png, results/pbnbv_residual_actual.json
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
    """복원 지표: raycast 가시 ∩ 입사각<60°."""
    live = np.ones(len(obj), bool)
    _, vis = A.information_gain(c, t, obj, live, P.FOV_DEG, MAX_DIST)
    v = c - obj; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
    return vis & ((obj_n*v).sum(1) >= COS)


def main():
    # 관측가능 표면 (조밀 후보 합집합) + 후보별 복원마스크 캐시
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=MAX_DIST)
                       for a in [3., 5., 7.]])
    dense_alt = np.concatenate([[a]*(len(dense)//3) for a in [3., 5., 7.]])[:len(dense)]
    dense_recon = [obj_mask(c) for c in dense]       # raycast∩입사각 (복원 회계용)
    observable = np.zeros(len(obj), bool)
    for m in dense_recon: observable |= m
    n_obs = int(observable.sum())

    # 실측 34포즈 복원 (관측가능면 한정)
    cams = np.vstack([load_poses(NEW/'real_test_3m'), load_poses(NEW/'real_test_7m')])
    rec = np.zeros(len(obj), bool)
    for c in cams: rec |= obj_mask(c) & observable
    cov_real = 100*rec.sum()/n_obs
    n_res = int((observable & ~rec).sum())

    # ── 실제 PB-NBV(ellipsoid 0.5^r) 그리디, argmax score ──
    used = np.zeros(len(dense), bool)
    add_idx, add_score, add_gain, cov_curve = [], [], [], [cov_real]
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
        gain = int((dense_recon[best] & remaining).sum())   # 실제 복원 증가분
        used[best] = True
        if gain == 0:           # raycast로는 보이나 입사각<60 충족 못함 → 회계 0, 계속
            continue
        rec |= dense_recon[best] & observable
        add_idx.append(best); add_score.append(float(bS)); add_gain.append(gain)
        cov_curve.append(100*rec.sum()/n_obs)
    cov_final = 100*rec.sum()/n_obs
    n_add = len(add_idx)
    n_closed = n_res - int((observable & ~rec).sum())

    print(f"관측가능 표면        : {n_obs} pts")
    print(f"실측 34컷 도달        : {cov_real:.1f}%  (잔여 미복원 {n_res} pts)")
    print(f"실제 PB-NBV 추가 시점 : {n_add}개  (ellipsoid 0.5^r, argmax)")
    print(f"추가 후 도달          : {cov_final:.1f}%  (+{cov_final-cov_real:.1f}%p, {n_closed}/{n_res} 닫음)")
    if n_add:
        print("추가 시점(고도·방위·PB-NBV score·실복원이득):")
        for r, i in enumerate(add_idx):
            print(f"  +{r+1}: {dense_alt[i]:.0f}m  az={az(dense[i]):5.1f}°  "
                  f"score={add_score[r]:6.2f}  gain={add_gain[r]} pts")

    out = {
        "name": "pbnbv_residual_actual",
        "method": "ACTUAL PB-NBV ellipsoid 0.5^r gain (information_gain_ellipsoid_rank), argmax selection",
        "vs_previous": "pbnbv_residual_3m7m used plain point-count U=gain/dist",
        "observable_surface_points": n_obs,
        "real_coverage_pct": round(float(cov_real), 2),
        "residual_unseen_points": n_res,
        "pbnbv_additional_viewpoints": n_add,
        "coverage_after_additions_pct": round(float(cov_final), 2),
        "gain_pp": round(float(cov_final-cov_real), 2),
        "residual_points_closed": int(n_closed),
        "additional_viewpoints": [
            {"rank": r+1, "altitude_m": float(dense_alt[i]), "azimuth_deg": round(az(dense[i]), 1),
             "pbnbv_score": round(add_score[r], 2), "recon_gain_points": int(add_gain[r]),
             "position": [round(float(x), 4) for x in dense[i]]}
            for r, i in enumerate(add_idx)],
    }
    js = ROOT/'results'/'pbnbv_residual_actual.json'
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {js.name}")

    # ── 시각화 ──
    C = {'txt':'#e6edf3','ax':'#161b22','bg':'#0d1117','grid':'#21262d','obj':'#30363d',
         'seen':'#3fb950','res':'#f85149','add':'#ffa657','real':'#58a6ff','tgt':'#f778ba'}
    residual0 = observable & ~np.zeros(len(obj), bool)  # placeholder
    # 잔여(실측 직후) 다시 계산용: rec 변형됐으니 재계산
    rec_real = np.zeros(len(obj), bool)
    for c in cams: rec_real |= obj_mask(c) & observable
    residual = observable & ~rec_real

    fig = plt.figure(figsize=(15.5, 6.4), facecolor=C['bg'])
    ax = fig.add_subplot(121, facecolor=C['ax'])
    ax.scatter(obj[rec_real,0], obj[rec_real,1], c=C['seen'], s=5, alpha=0.4, zorder=2,
               label=f'real-covered ({int(rec_real.sum())})')
    ax.scatter(obj[residual,0], obj[residual,1], c=C['res'], s=22, zorder=4,
               edgecolors='white', linewidths=0.3, label=f'missed by real ({n_res})')
    ax.scatter(cams[:,0], cams[:,1], c=C['real'], s=30, marker='D', alpha=0.7, zorder=3,
               edgecolors='white', linewidths=0.3, label=f'real poses ({len(cams)})')
    if n_add:
        addc = dense[add_idx]
        ax.scatter(addc[:,0], addc[:,1], c=C['add'], s=170, marker='*', zorder=6,
                   edgecolors='white', linewidths=0.7, label=f'PB-NBV added ({n_add})')
        for r, i in enumerate(add_idx):
            ax.text(dense[i,0], dense[i,1], str(r+1), color='black', fontsize=7,
                    ha='center', va='center', zorder=7, fontweight='bold')
    ax.scatter(*t[:2], marker='*', s=240, c=C['tgt'], zorder=9)
    ax.set_title(f'(a) Real residual + ACTUAL PB-NBV additions\n'
                 f'real {cov_real:.1f}% → +{n_add} viewpoints → {cov_final:.1f}%',
                 color=C['txt'], fontsize=11, pad=6)
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt']); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=8.5, loc='upper right')

    ax2 = fig.add_subplot(122, facecolor=C['ax'])
    xs = np.arange(len(cov_curve))
    ax2.axhline(cov_real, color=C['real'], lw=1, ls='--', alpha=0.8)
    ax2.text(len(xs)*0.45, cov_real-0.6, f'real-flight ceiling {cov_real:.1f}%', color=C['real'], fontsize=8.5)
    ax2.plot(xs, cov_curve, '-*', color=C['add'], lw=2, ms=11, label='ACTUAL PB-NBV (ellipsoid 0.5^r)')
    ax2.scatter([0],[cov_real], s=80, c=C['real'], zorder=6, label='after real 34 shots')
    ax2.set_title(f'(b) Extra coverage ACTUAL PB-NBV recovers\n'
                  f'{n_add} added shots close {n_closed}/{n_res} missed pts (+{cov_final-cov_real:.1f}%p)',
                  color=C['txt'], fontsize=11, pad=6)
    ax2.set_xlabel('PB-NBV additional viewpoints', color=C['txt'])
    ax2.set_ylabel('observable-surface coverage (%)', color=C['txt'])
    ax2.tick_params(colors=C['txt'])
    ax2.set_ylim(cov_real-1.2, min(100.6, cov_final+0.8)); ax2.set_xticks(xs)
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax2.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9, loc='lower right')

    verdict = (f'ACTUAL PB-NBV proposes {n_add} additional viewpoints → +{cov_final-cov_real:.1f}%p'
               if n_add else 'ACTUAL PB-NBV proposes ZERO additional viewpoints')
    fig.suptitle(f'Real PB-NBV (ellipsoid 0.5^r gain) after the 3m+7m flight  —  {verdict}',
                 color=C['txt'], fontsize=12.5)
    png = ROOT/'results'/'pbnbv_residual_actual.png'
    fig.savefig(png, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {png.name}")


if __name__ == '__main__':
    main()
