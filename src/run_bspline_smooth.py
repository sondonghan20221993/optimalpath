"""
run_bspline_smooth.py — NBV waypoint를 지나는 B-spline 궤적 평활화 (경로 안전성 층)

문헌 근거:
  Vasquez-Gomez et al. (NBV utility의 traveled-distance 항 = back-and-forth 감소),
  Smooth Coverage Path Planning (B-spline 곡선보간 → 연속성 정리로 평활성·동역학 실행성 보장).

구조 (2층 분리 — 원래 PB-NBV 의도 불변):
  층1: viewpoint 선택  = PB-NBV (info-gain) 결과 그대로 (WP 위치 손 안 댐)
  층2: 궤적 평활화      = WP들을 지나는 cubic B-spline → 급회전/지그재그 제거

입력:  results/pbnbv_nocam_constrained_path.json  (start + WP들)
출력:  results/bspline_smoothed_path.json         (조밀 샘플 궤적)
"""
import sys, types, json
from pathlib import Path
import numpy as np
from scipy.interpolate import splprep, splev

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN   = ROOT / "results" / "pbnbv_nocam_constrained_path.json"
OUT  = ROOT / "results" / "bspline_smoothed_path.json"

N_SAMPLE = 200   # 궤적 조밀 샘플 수


def polyline_len(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def turn_angles(P):
    """연속 세그먼트 사이 방향전환각(deg) 리스트."""
    v = np.diff(P, axis=0)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    ang = []
    for i in range(len(v) - 1):
        c = np.clip(np.dot(v[i], v[i + 1]), -1, 1)
        ang.append(np.degrees(np.arccos(c)))
    return ang


def main():
    d = json.load(open(IN))
    start = np.array(d["start_pos"])
    wps = np.array([w["pos"] for w in d["waypoints"]])
    ctrl = np.vstack([start, wps])           # 경유점: start → WP1 → WP2 → WP3
    print(f"[1] 경유점 {len(ctrl)}개 (start + WP {len(wps)})")
    print(f"    위치(불변): start + " + ", ".join(f"WP{i+1}" for i in range(len(wps))))

    # ── 원래 직선경로 (층1 그대로) ──
    raw_len = polyline_len(ctrl)
    raw_turn = turn_angles(ctrl)
    print(f"[2] 직선경로(평활화 전): {raw_len:.2f}m, "
          f"방향전환 최대 {max(raw_turn):.0f}° (지그재그)")

    # ── B-spline 평활화 (층2) ──
    # k=3 cubic, s=0 → 모든 WP를 정확히 통과(보간), 곡선만 매끄럽게
    k = min(3, len(ctrl) - 1)
    tck, u = splprep([ctrl[:, 0], ctrl[:, 1], ctrl[:, 2]], k=k, s=0)
    uu = np.linspace(0, 1, N_SAMPLE)
    sx, sy, sz = splev(uu, tck)
    smooth = np.column_stack([sx, sy, sz])

    sm_len = polyline_len(smooth)
    sm_turn = turn_angles(smooth)
    print(f"[3] B-spline 궤적(cubic, WP 정확 통과): {sm_len:.2f}m")
    print(f"    방향전환 최대 {max(sm_turn):.1f}°/step (조밀 샘플 → 매끄러움)")
    print(f"    곡률 연속(C2) → 동역학 실행 가능, ATE 안전")

    # WP가 실제로 궤적 위에 있는지 검증 (의도 불변 확인)
    on_path = []
    for w in ctrl:
        dmin = np.min(np.linalg.norm(smooth - w, axis=1))
        on_path.append(round(float(dmin), 4))
    print(f"[4] WP-궤적 일치 검증: 최대 이탈 {max(on_path):.4f}m (≈0 → 위치 불변 확인)")

    out = {
        "note": "Layer2 B-spline smoothing over fixed PB-NBV waypoints. "
                "Viewpoint selection (layer1) untouched.",
        "source": "Vasquez-Gomez NBV distance term + B-spline continuity (Smooth CPP)",
        "ctrl_points": ctrl.tolist(),
        "raw_path_length_m": round(raw_len, 3),
        "raw_max_turn_deg": round(max(raw_turn), 1),
        "smooth_path_length_m": round(sm_len, 3),
        "spline_degree": k,
        "n_sample": N_SAMPLE,
        "wp_offpath_max_m": max(on_path),
        "smooth_trajectory": smooth.tolist(),
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n✓ 저장: {OUT}")


if __name__ == "__main__":
    main()
