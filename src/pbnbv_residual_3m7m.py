"""
pbnbv_residual_3m7m.py — 실측 비행 '이후' PB-NBV가 추가 시점을 제안하는가?

질문: 실측 3m+7m(34컷)이 관측가능 표면의 96.2%를 채웠다. 남은 ~3.8%를
PB-NBV가 '새로운 후보 시점'으로 더 잡으라고 하는가, 아니면 추가 경로가 없는가?

절차:
  1) rec_real = 실측 34포즈의 관측가능면 복원 합집합
  2) 후보 = 조밀 링(3m·7m, 방위 72분할 = 풍부한 시점 풀)
  3) rec_real 상태에서 PB-NBV 그리디를 '이어서' 실행 → 추가 이득>0인 시점만 선택
  4) 추가 시점 수 / 채운 잔여 coverage / 위치(방위·고도) 보고
출력: results/pbnbv_residual_3m7m.png, results/pbnbv_residual_3m7m.json
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
    live = np.ones(len(obj), bool)
    _, vis = A.information_gain(c, t, obj, live, P.FOV_DEG, MAX_DIST)
    v = c - obj; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
    return vis & ((obj_n*v).sum(1) >= COS)


def main():
    # 관측가능 표면 (조밀 후보 합집합)
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=MAX_DIST)
                       for a in [3., 5., 7.]])
    dense_alt = np.concatenate([[a]*(len(dense)//3) for a in [3., 5., 7.]])[:len(dense)]
    dense_masks = [obj_mask(c) for c in dense]
    observable = np.zeros(len(obj), bool)
    for m in dense_masks: observable |= m
    n_obs = int(observable.sum())

    # 실측 34포즈 복원 (관측가능면 한정)
    cams = np.vstack([load_poses(NEW/'real_test_3m'), load_poses(NEW/'real_test_7m')])
    rec_real = np.zeros(len(obj), bool)
    for c in cams: rec_real |= obj_mask(c) & observable
    cov_real = 100*rec_real.sum()/n_obs
    residual = observable & ~rec_real           # 실측이 놓친 관측가능 점
    n_res = int(residual.sum())

    # 후보 마스크를 관측가능면으로 제한
    cand_masks = [m & observable for m in dense_masks]

    # PB-NBV 그리디를 rec_real 상태에서 '이어서' 실행 (추가 이득>0 시점만)
    rec = rec_real.copy()
    cur = cams[-1].copy()                        # 실측 마지막 포즈에서 출발
    add_idx, add_gain, cov_curve = [], [], [cov_real]
    for _ in range(len(dense)):
        best, bU, bg = -1, -1.0, 0
        for i in range(len(dense)):
            if i in add_idx: continue
            g = int((cand_masks[i] & ~rec).sum())
            if g == 0: continue
            d = np.linalg.norm(dense[i]-cur)+1e-6; U = g/d
            if U > bU: bU, best, bg = U, i, g
        if best < 0: break
        rec |= cand_masks[best]; cur = dense[best].copy()
        add_idx.append(best); add_gain.append(bg)
        cov_curve.append(100*rec.sum()/n_obs)
    cov_final = 100*rec.sum()/n_obs
    n_add = len(add_idx)
    n_closed = int((rec & residual).sum())       # 추가로 닫은 잔여 점

    print(f"관측가능 표면        : {n_obs} pts")
    print(f"실측 34컷 도달        : {cov_real:.1f}%  (잔여 미복원 {n_res} pts)")
    print(f"PB-NBV 추가 제안 시점 : {n_add}개")
    print(f"추가 후 도달          : {cov_final:.1f}%  (+{cov_final-cov_real:.1f}%p, 잔여 {n_res-n_closed} pts 닫음/{n_res})")
    if n_add:
        print("추가 시점(고도·방위·이득):")
        for r, i in enumerate(add_idx):
            print(f"  +{r+1}: {dense_alt[i]:.0f}m  az={az(dense[i]):5.1f}°  gain={add_gain[r]} pts")

    out = {
        "name": "pbnbv_residual_3m7m",
        "question": "After the real 3m+7m flight, does PB-NBV propose ADDITIONAL viewpoints?",
        "observable_surface_points": n_obs,
        "real_coverage_pct": round(float(cov_real), 2),
        "residual_unseen_points": n_res,
        "pbnbv_additional_viewpoints": n_add,
        "coverage_after_additions_pct": round(float(cov_final), 2),
        "gain_pp": round(float(cov_final-cov_real), 2),
        "residual_points_closed": n_closed,
        "additional_viewpoints": [
            {"rank": r+1, "altitude_m": float(dense_alt[i]), "azimuth_deg": round(az(dense[i]), 1),
             "gain_points": int(add_gain[r]),
             "position": [round(float(x), 4) for x in dense[i]]}
            for r, i in enumerate(add_idx)],
    }
    js = ROOT/'results'/'pbnbv_residual_3m7m.json'
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {js.name}")

    # ── 시각화 ──
    C = {'txt':'#e6edf3','ax':'#161b22','bg':'#0d1117','grid':'#21262d','obj':'#30363d',
         'seen':'#3fb950','res':'#f85149','add':'#ffa657','real':'#58a6ff','tgt':'#f778ba'}
    fig = plt.figure(figsize=(15.5, 6.4), facecolor=C['bg'])

    # (a) top-down: 실측 포즈 + 잔여 미복원 점 + PB-NBV 추가 시점
    ax = fig.add_subplot(121, facecolor=C['ax'])
    ax.scatter(obj[rec_real,0][:], obj[rec_real & observable,1]*0+obj[rec_real,1] if False else obj[rec_real,1],
               c=C['seen'], s=5, alpha=0.4, zorder=2, label=f'real-covered ({int(rec_real.sum())})')
    ax.scatter(obj[residual,0], obj[residual,1], c=C['res'], s=22, zorder=4,
               edgecolors='white', linewidths=0.3, label=f'missed by real ({n_res})')
    ax.scatter(cams[:,0], cams[:,1], c=C['real'], s=30, marker='D', alpha=0.7, zorder=3,
               edgecolors='white', linewidths=0.3, label=f'real poses ({len(cams)})')
    if n_add:
        addc = dense[add_idx]
        ax.scatter(addc[:,0], addc[:,1], c=C['add'], s=160, marker='*', zorder=6,
                   edgecolors='white', linewidths=0.7, label=f'PB-NBV added ({n_add})')
        for r, i in enumerate(add_idx):
            ax.text(dense[i,0], dense[i,1], str(r+1), color='black', fontsize=7,
                    ha='center', va='center', zorder=7, fontweight='bold')
    ax.scatter(*t[:2], marker='*', s=240, c=C['tgt'], zorder=9)
    ax.set_title(f'(a) Residual after real flight + PB-NBV additions\n'
                 f'real {cov_real:.1f}% → +{n_add} viewpoints → {cov_final:.1f}%',
                 color=C['txt'], fontsize=11, pad=6)
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt']); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=8.5, loc='upper right')

    # (b) coverage 곡선: 실측 천장 → 추가 시점이 메우는 잔여
    ax2 = fig.add_subplot(122, facecolor=C['ax'])
    xs = np.arange(len(cov_curve))
    ax2.axhline(cov_real, color=C['real'], lw=1, ls='--', alpha=0.8)
    ax2.text(len(xs)*0.5, cov_real-0.6, f'real-flight ceiling {cov_real:.1f}%', color=C['real'], fontsize=8.5)
    ax2.plot(xs, cov_curve, '-*', color=C['add'], lw=2, ms=10, label='PB-NBV additional viewpoints')
    ax2.scatter([0],[cov_real], s=80, c=C['real'], zorder=6, label='after real 34 shots')
    ax2.set_title(f'(b) Extra coverage PB-NBV recovers\n'
                  f'{n_add} added shots close {n_closed}/{n_res} missed pts (+{cov_final-cov_real:.1f}%p)',
                  color=C['txt'], fontsize=11, pad=6)
    ax2.set_xlabel('PB-NBV additional viewpoints', color=C['txt'])
    ax2.set_ylabel('observable-surface coverage (%)', color=C['txt'])
    ax2.tick_params(colors=C['txt'])
    ax2.set_ylim(cov_real-1.2, min(100.5, cov_final+0.8))
    ax2.set_xticks(xs)
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax2.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9, loc='lower right')

    verdict = (f'PB-NBV proposes {n_add} ADDITIONAL viewpoints → +{cov_final-cov_real:.1f}%p'
               if n_add else 'PB-NBV proposes ZERO additional viewpoints (real flight already maximal)')
    fig.suptitle(f'Does PB-NBV add viewpoints after the real 3m+7m flight?  —  {verdict}',
                 color=C['txt'], fontsize=12.5)
    png = ROOT/'results'/'pbnbv_residual_3m7m.png'
    fig.savefig(png, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {png.name}")


if __name__ == '__main__':
    main()
