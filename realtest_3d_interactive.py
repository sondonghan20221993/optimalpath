"""
realtest_3d_interactive.py

PB-NBV 선택 지점 + real_test 기존 경로를 인터랙티브 3D로 시각화.
브라우저에서 드래그·줌·회전 가능한 HTML 파일로 저장한다.

사용법:
  python realtest_3d_interactive.py
  python realtest_3d_interactive.py --json results/realtest_pbnbv_uniform/realtest_pbnbv_realband.json
"""

import argparse
import json
import glob
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

TARGET = np.array([-33.67, -50.83, 0.18])


def load_real_test_cams(meta_dir="real_test/meta"):
    cams = []
    for f in sorted(glob.glob(f"{meta_dir}/*.json")):
        d = json.load(open(f))
        p = d["camera"]["pose"]["position"]
        cams.append([p["x"], p["y"], p["z"]])
    return np.array(cams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/realtest_pbnbv_uniform/realtest_pbnbv_realband.json")
    ap.add_argument("--output", default="results/realtest_pbnbv_uniform/realtest_3d_interactive.html")
    ap.add_argument("--max-frames", type=int, default=None, help="기존 경로 최대 프레임 수 (예: 17=1바퀴)")
    ap.add_argument("--pts-npz", default=None, help="실제 점군 .npz 경로 (points+normals) — 물체 표시")
    ap.add_argument("--greedy-path", action="store_true", help="Greedy NN으로 경로 연결 후 표시")
    args = ap.parse_args()

    # PB-NBV 선택 지점 로드
    data = json.load(open(args.json))
    pts = np.array([[w["position"][0], w["position"][1], w["position"][2]]
                    for w in data["selected_points"]])
    sc  = np.array([w["pbnbv_score"] for w in data["selected_points"]])

    # Greedy Nearest Neighbor 경로 정렬
    if args.greedy_path and len(pts) > 1:
        unvisited = list(range(len(pts)))
        order = [unvisited.pop(0)]  # 첫 점 출발
        while unvisited:
            cur = pts[order[-1]]
            dists = [np.linalg.norm(pts[i] - cur) for i in unvisited]
            nearest = unvisited[int(np.argmin(dists))]
            order.append(nearest)
            unvisited.remove(nearest)
        pts = pts[order]
        sc  = sc[order]

    # real_test 기존 경로
    cams = load_real_test_cams()
    if args.max_frames is not None:
        cams = cams[:args.max_frames]

    fig = go.Figure()

    # NED Z → 고도 (양수=위) 변환
    def alt(z): return -np.asarray(z)

    # 1) real_test 기존 경로 선 (경로가 있을 때만)
    if len(cams) > 0:
        cams_loop = np.vstack([cams, cams[0]])
        fig.add_trace(go.Scatter3d(
            x=cams_loop[:, 0], y=cams_loop[:, 1], z=alt(cams_loop[:, 2]),
            mode="lines+markers",
            line=dict(color="#55aaff", width=3),
            marker=dict(size=5, color="#55aaff", symbol="circle",
                        line=dict(color="white", width=0.5)),
            name=f"real_test path ({len(cams)} frames)",
            hovertemplate="real_test<br>x=%{x:.2f} y=%{y:.2f} alt=%{z:.2f}m<extra></extra>",
        ))

    # 2) PB-NBV 선택 지점 (순위별 컬러, 1위=노랑)
    ranks = np.argsort(np.argsort(-sc)) + 1
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=alt(pts[:, 2]),
        mode="markers+text",
        text=[f"#{r}" for r in ranks],
        textposition="top center",
        textfont=dict(color="white", size=10),
        marker=dict(
            size=10,
            color=ranks,
            colorscale="Plasma_r",
            reversescale=False,
            showscale=True,
            cmin=1, cmax=len(ranks),
            colorbar=dict(title="Rank (1=best)", thickness=15,
                          tickfont=dict(color="white"),
                          titlefont=dict(color="white")),
            line=dict(color="white", width=0.8),
            symbol="circle",
        ),
        name=f"PB-NBV selected ({len(pts)})",
        hovertemplate="rank=%{text}<br>x=%{x:.2f} y=%{y:.2f} alt=%{z:.2f}m<br>score=%{customdata:.0f}<extra></extra>",
        customdata=sc,
    ))

    # 3) Greedy 경로 선
    if args.greedy_path:
        path_loop = np.vstack([pts, pts[0]])
        total_dist = sum(np.linalg.norm(path_loop[i+1] - path_loop[i])
                         for i in range(len(path_loop)-1))
        fig.add_trace(go.Scatter3d(
            x=path_loop[:, 0], y=path_loop[:, 1], z=alt(path_loop[:, 2]),
            mode="lines",
            line=dict(color="#ff6b35", width=3, dash="dash"),
            name=f"Greedy path ({total_dist:.1f}m)",
            hoverinfo="skip",
        ))

    # 4) 실제 물체 점군
    if args.pts_npz:
        obj_data = np.load(args.pts_npz)
        obj_pts = obj_data["points"]
        if "colors" in obj_data:
            c = obj_data["colors"]
            obj_colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                          for r, g, b in c]
        else:
            obj_colors = "#00e676"
        fig.add_trace(go.Scatter3d(
            x=obj_pts[:, 0], y=obj_pts[:, 1], z=alt(obj_pts[:, 2]),
            mode="markers",
            marker=dict(size=3, color=obj_colors, opacity=0.9,
                        line=dict(width=0)),
            name=f"object ({len(obj_pts)} pts)",
            hovertemplate="obj<br>x=%{x:.2f} y=%{y:.2f} alt=%{z:.2f}m<extra></extra>",
        ))

    # 5) 타겟
    fig.add_trace(go.Scatter3d(
        x=[TARGET[0]], y=[TARGET[1]], z=alt([TARGET[2]]),
        mode="markers+text",
        marker=dict(size=12, color="#ffcc02", symbol="diamond",
                    line=dict(color="white", width=1)),
        text=["TARGET"], textposition="top center",
        textfont=dict(color="#ffcc02", size=13),
        name="Target",
        hovertemplate="TARGET<br>x=%{x:.2f} y=%{y:.2f} alt=%{z:.2f}m<extra></extra>",
    ))

    # 6) 고도별 수평 링 (참고선) — 고도 양수로 표시
    for alt_label, alt_val, col in [("4.3m", 4.3, "rgba(85,170,255,0.15)"),
                                     ("6.1m", 6.1, "rgba(255,204,2,0.15)")]:
        theta = np.linspace(0, 2*math.pi, 60)
        rx = TARGET[0] + 8.5*np.cos(theta)
        ry = TARGET[1] + 8.5*np.sin(theta)
        rz = np.full_like(theta, alt_val)
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode="lines",
            line=dict(color=col, width=1.5, dash="dot"),
            name=f"alt ref {alt_label}",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(
            text="PB-NBV Viewpoint Selection — Interactive 3D<br>"
                 "<sup>drag to rotate · scroll to zoom · double-click to reset</sup>",
            font=dict(color="white", size=16),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="#0d0d1e",
                       gridcolor="#333", zerolinecolor="#555",
                       tickfont=dict(color="white"), titlefont=dict(color="white")),
            yaxis=dict(title="Y (m)", backgroundcolor="#0d0d1e",
                       gridcolor="#333", zerolinecolor="#555",
                       tickfont=dict(color="white"), titlefont=dict(color="white")),
            zaxis=dict(title="Altitude (m)", backgroundcolor="#0d0d1e",
                       gridcolor="#333", zerolinecolor="#555",
                       tickfont=dict(color="white"), titlefont=dict(color="white")),
            bgcolor="#0d0d1e",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            aspectmode="data",
        ),
        paper_bgcolor="#0a0a14",
        plot_bgcolor="#0a0a14",
        legend=dict(font=dict(color="white"), bgcolor="#1a1a30",
                    bordercolor="#444", x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=80, b=0),
        height=750,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    win_path = str(out).replace('/mnt/c/', 'C:\\').replace('/', '\\')
    print(f"저장 완료: {out}")
    print(f"Windows에서 열기: {win_path}")


if __name__ == "__main__":
    main()
