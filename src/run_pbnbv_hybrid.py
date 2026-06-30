"""
run_pbnbv_hybrid.py — Case 2: 하이브리드
논문 PB-NBV F점수로 시점 SET을 선택(pbnbv_paper.run_nbv)한 뒤,
선택 순서는 버리고 Greedy NN으로 방문 순서를 재정렬한다.

Case 1(순차 NBV)과의 차이: 방문 순서 → 이동거리.
커버리지는 동일 SET이므로 같다.
"""
import sys, types, json, math
from pathlib import Path
import numpy as np

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pbnbv_path as A
import pbnbv_paper as P

OUT = HERE.parent / "results" / "pbnbv_hybrid" / "pbnbv_hybrid.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

START = np.array([-27.06, -50.51, -6.13])   # Case1과 동일 시작점


def path_length(pts_seq, start):
    cur, total = np.array(start), 0.0
    for p in pts_seq:
        p = np.array(p)
        total += np.linalg.norm(p - cur)
        cur = p
    return total


def main():
    print("="*60)
    print("Case 2 (하이브리드): 논문 F점수 선택 → Greedy NN 재정렬")
    print("="*60)

    # 1) 논문 PB-NBV로 시점 SET 선택 (선택 순서 = NBV 자연 순서)
    path, curve, cands = P.run_nbv()
    nbv_order = [p["pos"] for p in path]
    print(f"\n[선택] 논문 F점수로 {len(nbv_order)}개 시점 선택")

    # 2) 선택 순서 버리고 Greedy NN 재정렬
    greedy_order = A.greedy_path(np.array(nbv_order), START)
    print(f"[정렬] Greedy NN으로 방문 순서 재정렬")

    # 3) 이동거리 비교
    len_nbv    = path_length(nbv_order, START)
    len_greedy = path_length(greedy_order, START)
    print(f"\n[이동거리 비교]")
    print(f"  Case 1 (NBV 순차 순서) : {len_nbv:.2f} m")
    print(f"  Case 2 (Greedy 재정렬) : {len_greedy:.2f} m")
    print(f"  개선                   : {len_nbv - len_greedy:+.2f} m "
          f"({100*(len_nbv-len_greedy)/len_nbv:+.1f}%)")

    # 4) 커버리지 (동일 SET → 동일, 실제 포인트+raycast로 확인)
    pt_curve, pt_left = P.eval_on_points([np.array(p) for p in greedy_order])
    pt_total = P.len_points if hasattr(P, "len_points") else len(P._pts_raw)
    pt_cov   = (pt_total - pt_left) / pt_total
    print(f"\n[커버리지] 실제 포인트 기준: {pt_total-pt_left}/{pt_total} ({pt_cov*100:.1f}%)")

    # 5) waypoint 저장 (Greedy 순서)
    target = P.TARGET
    def az(p):
        return math.degrees(math.atan2(p[1]-target[1], p[0]-target[0])) % 360
    waypoints = []
    for i, wp in enumerate(greedy_order):
        wp = np.array(wp)
        t = A.tilt_deg_to_target(wp, target)
        waypoints.append({"idx": i+1, "pos": wp.tolist(),
                          "tilt_deg": round(t, 1), "azimuth_deg": round(az(wp), 1)})
        print(f"  WP{i+1:02d} ({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) tilt={t:.1f}° az={az(wp):.0f}°")

    out = {
        "algorithm": "Case2 Hybrid (paper-F selection + Greedy NN ordering)",
        "n_waypoints": len(greedy_order),
        "path_length_nbv_order": round(len_nbv, 3),
        "path_length_greedy_order": round(len_greedy, 3),
        "path_improvement_m": round(len_nbv - len_greedy, 3),
        "coverage_points": round(pt_cov, 4),
        "points_covered": pt_total - pt_left,
        "points_total": pt_total,
        "waypoints": waypoints,
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
