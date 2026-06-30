"""
validate_pbnbv_3m7m.py — 실측 3m/7m 데이터로 PB-NBV '검증'.

핵심 아이디어: PB-NBV에게 **실제로 촬영된 34개 실측 포즈**(3m 17 + 7m 17)를
후보 풀로 그대로 주고, 정보이득 효용(U = new_gain / dist)으로 순서를 매긴다.
그 누적 coverage 곡선을 실측 비행이 실제로 찍은 순서(as-flown)와 비교한다.
→ 같은 데이터·같은 지표에서 PB-NBV '시점 선택'의 효율만 분리해서 검증.

평가면: 물체의 관측가능 표면(공중에서 입사각<60°로 볼 수 있는 면, 밑면 제외).
복원 = raycast 가시 ∩ 입사각<60°.   (7m standoff~12.4m → MAX_DIST=13)

출력:
  results/validate_pbnbv_3m7m.png   (좌: coverage 곡선, 우: PB-NBV 선택 순서 top-down)
  results/validate_pbnbv_3m7m.json  (순서·임계값 도달 컷 수·요약)
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
    """카메라 c가 물체에서 복원하는 점(raycast 가시 ∩ 입사각<60°)."""
    live = np.ones(len(obj), bool)
    _, vis = A.information_gain(c, t, obj, live, P.FOV_DEG, MAX_DIST)
    v = c - obj; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
    return vis & ((obj_n*v).sum(1) >= COS)


def main():
    # ── 관측가능 표면 (조밀 후보 합집합, run_pbnbv_path_3m7m와 동일 기준) ──
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=MAX_DIST)
                       for a in [3., 5., 7.]])
    observable = np.zeros(len(obj), bool)
    for c in dense:
        observable |= obj_mask(c)
    n_obs = int(observable.sum())

    # ── 실측 34 포즈 + 각 포즈의 관측가능면 복원 마스크 ──
    pos3 = load_poses(NEW/'real_test_3m')      # 17
    pos7 = load_poses(NEW/'real_test_7m')       # 17
    cams = np.vstack([pos3, pos7])              # as-flown 순서: 3m 먼저 → 7m
    tier = np.array([3.0]*len(pos3) + [7.0]*len(pos7))
    masks = [obj_mask(c) & observable for c in cams]

    # ── (1) 실측 as-flown 누적 coverage (3m 0..16 → 7m 0..16) ──
    def cum_curve(order):
        rec = np.zeros(len(obj), bool); ys = [0.0]
        for i in order:
            rec |= masks[i]; ys.append(100*rec.sum()/n_obs)
        return np.array(ys)
    asflown = list(range(len(cams)))
    y_flown = cum_curve(asflown)

    # ── (2) PB-NBV 그리디 (U = new_gain / dist), 실측 포즈만 후보 ──
    rec = np.zeros(len(obj), bool); used = np.zeros(len(cams), bool)
    cur = np.array([t[0]+6, t[1], top-3]); order_nbv = []; util_log = []
    for _ in range(len(cams)):
        best, bU, bg = -1, -1.0, 0
        for i in range(len(cams)):
            if used[i]: continue
            gain = int((masks[i] & ~rec).sum())
            if gain == 0: continue
            d = np.linalg.norm(cams[i]-cur)+1e-6; U = gain/d
            if U > bU: bU, best, bg = U, i, gain
        if best < 0: break
        rec |= masks[best]; used[best] = True; cur = cams[best].copy()
        order_nbv.append(best); util_log.append((bg, float(np.linalg.norm(cams[best]-t))))
    # gain=0 이라 안 뽑힌 잉여 포즈는 뒤에 이어붙임(곡선 평탄 구간)
    tail = [i for i in range(len(cams)) if not used[i]]
    order_full = order_nbv + tail
    y_nbv = cum_curve(order_full)

    # ── 임계값 도달 컷 수 ──
    def shots_to(curve, thr):
        for k, y in enumerate(curve):
            if y >= thr: return k
        return None
    rows = []
    for thr in [80, 90, 95, 99, 99.9]:
        rows.append((thr, shots_to(y_nbv, thr), shots_to(y_flown, thr)))
    final_nbv = y_nbv[-1]; final_flown = y_flown[-1]
    n_eff = len(order_nbv)   # 실제로 이득을 준(선택된) 포즈 수

    print(f"관측가능 표면: {n_obs} pts (전체 {len(obj)} 중 {100*n_obs/len(obj):.1f}%)")
    print(f"실측 34컷 최종 관측가능 coverage: {final_flown:.1f}%")
    print(f"PB-NBV 선택(실측 포즈 풀)  최종: {final_nbv:.1f}%  | 유효 컷 {n_eff}/34")
    print("임계값 도달 컷 수 (PB-NBV vs as-flown):")
    for thr, a, b in rows:
        print(f"  {thr:>5}% : PB-NBV {str(a):>4}컷 | 실측 {str(b):>4}컷"
              + (f"  → {b-a:+d}컷 절감" if (a is not None and b is not None) else ""))

    # ── JSON ──
    out = {
        "name": "validate_pbnbv_3m7m",
        "idea": "PB-NBV utility (U=new_gain/dist) orders the 34 ACTUAL real poses; "
                "compare cumulative observable-surface coverage vs as-flown capture order",
        "metric": "raycast visibility ∩ incidence<60deg, denominator = observable object surface",
        "observable_surface_points": n_obs,
        "total_object_points": int(len(obj)),
        "n_real_poses": int(len(cams)),
        "final_coverage_pct": {"pbnbv_order": round(float(final_nbv), 2),
                                "as_flown": round(float(final_flown), 2)},
        "effective_shots_pbnbv": int(n_eff),
        "shots_to_threshold": [
            {"threshold_pct": thr, "pbnbv": a, "as_flown": b,
             "shots_saved": (None if (a is None or b is None) else b - a)}
            for thr, a, b in rows],
        "pbnbv_order": [
            {"rank": k+1, "real_index": int(i), "tier_m": float(tier[i]),
             "azimuth_deg": round(az(cams[i]), 1),
             "standoff_m": round(float(np.linalg.norm(cams[i]-t)), 2)}
            for k, i in enumerate(order_nbv)],
    }
    js = ROOT/'results'/'validate_pbnbv_3m7m.json'
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {js.name}")

    # ── 시각화 ──
    C = {'txt':'#e6edf3','ax':'#161b22','bg':'#0d1117','grid':'#21262d','obj':'#6e7681',
         'nbv':'#3fb950','flown':'#f778ba','c7':'#58a6ff','c3':'#d2a8ff','tgt':'#f778ba'}
    fig = plt.figure(figsize=(15.5, 6.4), facecolor=C['bg'])

    # (a) coverage 곡선
    ax = fig.add_subplot(121, facecolor=C['ax'])
    xs = np.arange(len(y_nbv))
    ax.plot(xs, y_nbv, '-o', color=C['nbv'], lw=2, ms=4, label='PB-NBV order (U=gain/dist)')
    ax.plot(xs, y_flown, '-s', color=C['flown'], lw=2, ms=3.5, alpha=0.9, label='Real as-flown order')
    for thr in [95, 99]:
        ax.axhline(thr, color=C['grid'], lw=0.8, ls='--')
        a = shots_to(y_nbv, thr); b = shots_to(y_flown, thr)
        if a is not None: ax.scatter([a],[y_nbv[a]], s=120, facecolors='none', edgecolors=C['nbv'], lw=1.8, zorder=6)
        if b is not None: ax.scatter([b],[y_flown[b]], s=120, facecolors='none', edgecolors=C['flown'], lw=1.8, zorder=6)
        ax.text(len(xs)*0.98, thr+0.4, f'{thr}%', color=C['txt'], fontsize=8, ha='right')
    ax.set_title(f'(a) Coverage vs #shots — same 34 real poses, reordered\n'
                 f'PB-NBV reaches 95% in {shots_to(y_nbv,95)} shots vs {shots_to(y_flown,95)} as-flown',
                 color=C['txt'], fontsize=11, pad=6)
    ax.set_xlabel('number of shots used', color=C['txt'])
    ax.set_ylabel('observable-surface coverage (%)', color=C['txt'])
    ax.tick_params(colors=C['txt']); ax.set_ylim(0, 102)
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9, loc='lower right')

    # (b) PB-NBV 선택 순서 top-down
    ax2 = fig.add_subplot(122, facecolor=C['ax'])
    ax2.scatter(obj[:,0], obj[:,1], c=C['obj'], s=6, alpha=0.5, zorder=2)
    ax2.scatter(*t[:2], marker='*', s=300, c=C['tgt'], zorder=9)
    seq = cams[order_nbv]
    ax2.plot(seq[:,0], seq[:,1], '-', color=C['nbv'], lw=1.3, alpha=0.7, zorder=3)
    for tval, col, lab in [(7, C['c7'], '7m'), (3, C['c3'], '3m')]:
        mk = np.array([tier[i] == tval for i in order_nbv])
        if mk.any():
            ax2.scatter(seq[mk,0], seq[mk,1], c=col, s=130, marker='o', zorder=5,
                        edgecolors='white', linewidths=0.6, label=f'{lab} pose')
    for k, i in enumerate(order_nbv):
        ax2.text(cams[i,0], cams[i,1], str(k+1), color='white', fontsize=6.5,
                 ha='center', va='center', zorder=7, fontweight='bold')
    ax2.set_title(f'(b) PB-NBV selection order over the real poses\n'
                  f'{n_eff} effective shots → {final_nbv:.0f}% observable', color=C['txt'], fontsize=11, pad=6)
    ax2.set_xlabel('X (m)', color=C['txt']); ax2.set_ylabel('Y (m)', color=C['txt'])
    ax2.tick_params(colors=C['txt']); ax2.set_aspect('equal')
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax2.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9, loc='upper right')

    fig.suptitle('PB-NBV validation on real 3m+7m data — info-gain ordering of the 34 captured poses '
                 f'front-loads coverage ({shots_to(y_nbv,95)} vs {shots_to(y_flown,95)} shots to 95%)',
                 color=C['txt'], fontsize=12.5)
    png = ROOT/'results'/'validate_pbnbv_3m7m.png'
    fig.savefig(png, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {png.name}")


if __name__ == '__main__':
    main()
