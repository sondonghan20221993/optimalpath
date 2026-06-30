"""
run_pbnbv_path_3m7m.py — 새 데이터(3m/7m) 기준 PB-NBV 경로 생성.
  1) 후보: 3m·7m 링(방위 36분할), tilt=45
  2) PB-NBV 그리디(U=new_gain/dist)로 관측가능 표면 100% 까지 시점 선택
  3) 비행 가능한 순서로 정렬: 7m 링(광역) → 3m 링(세부), 각 링 방위각 순
  4) JSON(AirSim NED) + 경로 시각화 출력
출력: results/pbnbv_path_3m7m.json, results/pbnbv_path_3m7m.png
"""
import sys, types, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_paper as P, pbnbv_path as A
A.RAYCAST_OCCLUSION = True
ROOT = HERE.parent
MAX_DIST = 13.0; COS = math.cos(math.radians(60)); P.MAX_DIST = MAX_DIST

t = np.array(P.TARGET); obj = P._pts_raw
obj_n = np.load(ROOT/'real_test'/'airsim_gt_pts.npz')['normals'].astype(float)
top = obj[:, 2].min()
az = lambda c: math.degrees(math.atan2(c[1]-t[1], c[0]-t[0])) % 360


def masks_of(cams):
    out = []
    for c in cams:
        live = np.ones(len(obj), bool)
        _, vis = A.information_gain(c, t, obj, live, P.FOV_DEG, MAX_DIST)
        v = c - obj; v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-9
        out.append(vis & ((obj_n*v).sum(1) >= COS))
    return out


def main():
    # 관측가능 표면
    dense = np.vstack([A.gen_candidates_tilt45(t, [round(top-a, 2)], n_az=72, max_dist=MAX_DIST) for a in [3., 5., 7.]])
    observable = np.zeros(len(obj), bool)
    for m in masks_of(dense): observable |= m
    n_obs = int(observable.sum())

    # 후보 (3m·7m 각 36) — 고도 라벨 보존
    c3 = A.gen_candidates_tilt45(t, [round(top-3, 2)], n_az=36, max_dist=MAX_DIST)
    c7 = A.gen_candidates_tilt45(t, [round(top-7, 2)], n_az=36, max_dist=MAX_DIST)
    cand = np.vstack([c3, c7]); alt_lab = np.array([3.0]*len(c3) + [7.0]*len(c7))
    M = masks_of(cand)

    # PB-NBV 그리디 (관측가능 100%까지)
    rec = np.zeros(len(obj), bool); used = np.zeros(len(cand), bool); sel_idx = []
    cur = np.array([t[0]+6, t[1], top-3])
    for _ in range(len(cand)):
        best, bU, bg = -1, -1, 0
        for i in range(len(cand)):
            if used[i]: continue
            gain = int((M[i] & ~rec & observable).sum())
            if gain == 0: continue
            d = np.linalg.norm(cand[i]-cur)+1e-6; U = gain/d
            if U > bU: bU, best, bg = U, i, gain
        if best < 0 or bg == 0: break
        rec |= M[best]; used[best] = True; cur = cand[best].copy(); sel_idx.append(best)
    sel_idx = np.array(sel_idx)
    sel = cand[sel_idx]; sel_alt = alt_lab[sel_idx]
    cov = 100*(rec & observable).sum()/n_obs

    # ── 비행 순서 정렬: 7m 링(광역) 먼저 → 3m 링(세부), 각 링 방위각 순 ──
    order = sorted(range(len(sel)), key=lambda i: (-sel_alt[i], az(sel[i])))  # 7m(고도大) 먼저
    path = sel[order]; path_alt = sel_alt[order]
    # 비행거리
    plen = sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1))

    # ── JSON (AirSim NED) ──
    out = {
        "schema_version": 1,
        "name": "pbnbv_path_3m7m",
        "algorithm": "PB-NBV (info-gain greedy) on real-GT object, 3m+7m candidates",
        "coordinate_system": "AirSim NED (z down +, up=-z)",
        "metric": "raycast visibility ∩ incidence<60deg",
        "target": t.tolist(),
        "n_candidates": int(len(cand)),
        "n_waypoints": int(len(path)),
        "observable_surface_points": n_obs,
        "observable_coverage_pct": round(cov, 2),
        "flight_order": "7m ring (survey) -> 3m ring (inspect), azimuth-sorted",
        "path_length_m": round(float(plen), 2),
        "waypoints": [
            {
                "index": i+1,
                "position": [round(float(x), 4) for x in path[i]],
                "altitude_m": round(float(-path[i][2]), 2),
                "tier_m": float(path_alt[i]),
                "azimuth_deg": round(az(path[i]), 1),
                "standoff_m": round(float(np.linalg.norm(path[i]-t)), 2),
                "tilt_deg": 45.0,
            }
            for i in range(len(path))
        ],
    }
    js = ROOT/'results'/'pbnbv_path_3m7m.json'
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {js.name}  {len(path)}WP, 관측가능 {cov:.1f}%, 비행거리 {plen:.1f}m")
    n7 = int((path_alt == 7).sum()); n3 = int((path_alt == 3).sum())
    print(f"  구성: 7m {n7}컷 + 3m {n3}컷")

    # ── 시각화: (a) top-down  (b) 측면 고도 프로파일  (c) 물체 close-up ──
    C = {'txt':'#e6edf3','ax':'#161b22','bg':'#0d1117','grid':'#21262d','obj':'#6e7681',
         'c7':'#58a6ff','c3':'#d2a8ff','tgt':'#f778ba','line':'#3fb950','og':'#3fb950'}
    fig = plt.figure(figsize=(19, 6.5), facecolor=C['bg'])

    # (a) top-down
    ax = fig.add_subplot(131, facecolor=C['ax'])
    ax.scatter(obj[:,0], obj[:,1], c=C['obj'], s=6, alpha=0.5, zorder=2)
    ax.scatter(*t[:2], marker='*', s=320, c=C['tgt'], zorder=9)
    ax.plot(path[:,0], path[:,1], '-', color=C['line'], lw=1.6, alpha=0.8, zorder=3)
    for tier, col in [(7, C['c7']), (3, C['c3'])]:
        mk = path_alt == tier
        ax.scatter(path[mk,0], path[mk,1], c=col, s=120, marker='o', zorder=5,
                   edgecolors='white', linewidths=0.6, label=f'{tier}m ring ({int(mk.sum())})')
    for i, c in enumerate(path):
        ax.text(c[0], c[1], str(i+1), color='white', fontsize=6.5, ha='center', va='center', zorder=7, fontweight='bold')
    ax.set_title(f'(a) Top-down (XY) — {len(path)} WP, {cov:.0f}% observable', color=C['txt'], fontsize=11, pad=6)
    ax.set_xlabel('X (m)', color=C['txt']); ax.set_ylabel('Y (m)', color=C['txt'])
    ax.tick_params(colors=C['txt']); ax.set_aspect('equal')
    for sp in ax.spines.values(): sp.set_color(C['grid'])
    ax.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9, loc='upper right')

    # (b) 측면 고도 프로파일 (X vs up) — 왜곡 없이 3m/7m 고도와 납작한 물체가 보임
    ax2 = fig.add_subplot(132, facecolor=C['ax'])
    ax2.scatter(obj[:,0], -obj[:,2], c=C['obj'], s=8, alpha=0.6, zorder=2, label='object (flat slab, 0.36m)')
    for tier, col in [(7, C['c7']), (3, C['c3'])]:
        mk = path_alt == tier
        ax2.scatter(path[mk,0], -path[mk,2], c=col, s=90, zorder=5, edgecolors='white',
                    linewidths=0.5, label=f'{tier}m waypoints')
    ax2.axhline(0, color=C['grid'], lw=1)
    ax2.set_title('(b) Side elevation (X vs altitude) — true vertical scale', color=C['txt'], fontsize=11, pad=6)
    ax2.set_xlabel('X (m)', color=C['txt']); ax2.set_ylabel('altitude up (m)', color=C['txt'])
    ax2.tick_params(colors=C['txt']); ax2.set_aspect('equal')
    for sp in ax2.spines.values(): sp.set_color(C['grid'])
    ax2.grid(True, color=C['grid'], alpha=0.35, lw=0.5)
    ax2.legend(facecolor=C['ax'], edgecolor=C['grid'], labelcolor=C['txt'], fontsize=9, loc='center right')

    # (c) 물체 close-up oblique (실제 비율)
    ax3 = fig.add_subplot(133, projection='3d', facecolor=C['ax'])
    OX, OY, OZ = obj[:,0], obj[:,1], -obj[:,2]
    ax3.scatter(OX, OY, OZ, c=C['og'], s=5, alpha=0.6)
    ax3.set_box_aspect((np.ptp(OX), np.ptp(OY), np.ptp(OZ)))   # 실제 비율 → 납작
    ax3.view_init(elev=24, azim=-60)
    ax3.set_title('(c) Object close-up (true proportion) — flat U-slab', color=C['txt'], fontsize=11)
    for a in (ax3.xaxis, ax3.yaxis, ax3.zaxis): a.pane.set_facecolor('#0d1117')
    ax3.tick_params(colors='#8b949e', labelsize=6)

    fig.suptitle('PB-NBV flight path (real-GT object, 3m+7m) — '
                 f'{len(path)} WP / {plen:.0f}m reach {cov:.0f}% observable | object = flat U-slab lying on ground',
                 color=C['txt'], fontsize=12.5)
    png = ROOT/'results'/'pbnbv_path_3m7m.png'
    fig.savefig(png, dpi=150, facecolor=C['bg'], bbox_inches='tight')
    print(f"✓ {png.name}")


if __name__ == '__main__':
    main()
