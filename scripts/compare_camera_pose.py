import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_estimated_pose_txt(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if len(values) != 8:
                raise ValueError(f"{path} line {idx + 1}: expected 8 values, got {len(values)}")
            t, x, y, z, qx, qy, qz, qw = values
            rows.append(
                {
                    "time": t,
                    "position": np.array([x, y, z], dtype=np.float64),
                    "quat_xyzw": np.array([qx, qy, qz, qw], dtype=np.float64),
                }
            )
    if not rows:
        raise ValueError(f"No pose rows found in {path}")
    return rows


def load_ground_truth_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level list")

    rows = []
    for idx, item in enumerate(data):
        if "T_wc" not in item:
            raise ValueError(f"{path} item {idx} missing T_wc")
        mat = np.array(item["T_wc"], dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"{path} item {idx} T_wc shape {mat.shape}, expected (4,4)")
        rows.append(
            {
                "time": float(item.get("timestamp", idx)),
                "position": mat[:3, 3].copy(),
                "rotation": mat[:3, :3].copy(),
                "image": item.get("image", f"frame_{idx:06d}"),
            }
        )
    if not rows:
        raise ValueError(f"No pose rows found in {path}")
    return rows


def quat_xyzw_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def umeyama_sim3(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must be Nx3 arrays with matching shape")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1

    rot = u @ s @ vt
    src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = np.trace(np.diag(d) @ s) / src_var
    trans = dst_mean - scale * (rot @ src_mean)
    return float(scale), rot, trans


def apply_sim3(points, scale, rot, trans):
    pts = np.asarray(points, dtype=np.float64)
    return (scale * (rot @ pts.T)).T + trans


def rotation_angle_deg(r_err):
    trace = np.trace(r_err)
    val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def build_comparison_figure(gt_positions, est_positions_raw, est_positions_aligned, position_errors, orientation_errors):
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene", "colspan": 2}, None], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Trajectory Comparison (3D)",
            "",
            "Position Error per Frame",
            "Orientation Error per Frame",
        ),
    )

    fig.add_trace(
        go.Scatter3d(
            x=gt_positions[:, 0],
            y=gt_positions[:, 1],
            z=gt_positions[:, 2],
            mode="lines+markers",
            name="Ground Truth",
            line=dict(color="#22c55e", width=5),
            marker=dict(size=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=est_positions_raw[:, 0],
            y=est_positions_raw[:, 1],
            z=est_positions_raw[:, 2],
            mode="lines+markers",
            name="Estimated (Raw)",
            line=dict(color="#f59e0b", width=4, dash="dot"),
            marker=dict(size=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=est_positions_aligned[:, 0],
            y=est_positions_aligned[:, 1],
            z=est_positions_aligned[:, 2],
            mode="lines+markers",
            name="Estimated (Aligned)",
            line=dict(color="#3b82f6", width=5),
            marker=dict(size=3),
        ),
        row=1,
        col=1,
    )

    frames = np.arange(len(position_errors))
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=position_errors,
            mode="lines+markers",
            name="Position Error (m)",
            line=dict(color="#ef4444"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=orientation_errors,
            mode="lines+markers",
            name="Orientation Error (deg)",
            line=dict(color="#8b5cf6"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Camera Pose Comparison",
        template="plotly_dark",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
    )
    fig.update_xaxes(title_text="Frame Index", row=2, col=1)
    fig.update_xaxes(title_text="Frame Index", row=2, col=2)
    fig.update_yaxes(title_text="Position Error", row=2, col=1)
    fig.update_yaxes(title_text="Orientation Error (deg)", row=2, col=2)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare MASt3R-SLAM pose output with AirSim ground truth.")
    parser.add_argument(
        "--base-dir",
        default=r"C:\Users\sdh97\Desktop\airsim_degree_notilt_more_distance",
        help="Directory containing rgb.txt and rgb_poses.json",
    )
    parser.add_argument("--est", default="rgb.txt", help="Estimated pose txt filename")
    parser.add_argument("--gt", default="rgb_poses.json", help="Ground-truth pose json filename")
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML path for the comparison figure",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    est_path = base_dir / args.est
    gt_path = base_dir / args.gt
    if not est_path.is_file():
        raise FileNotFoundError(f"Missing estimated pose file: {est_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth pose file: {gt_path}")

    est_rows = load_estimated_pose_txt(est_path)
    gt_rows = load_ground_truth_json(gt_path)

    count = min(len(est_rows), len(gt_rows))
    est_rows = est_rows[:count]
    gt_rows = gt_rows[:count]

    est_positions = np.stack([row["position"] for row in est_rows], axis=0)
    gt_positions = np.stack([row["position"] for row in gt_rows], axis=0)
    est_rotations = [quat_xyzw_to_matrix(row["quat_xyzw"]) for row in est_rows]
    gt_rotations = [row["rotation"] for row in gt_rows]

    scale, rot_align, trans_align = umeyama_sim3(est_positions, gt_positions)
    est_positions_aligned = apply_sim3(est_positions, scale, rot_align, trans_align)
    position_errors = np.linalg.norm(est_positions_aligned - gt_positions, axis=1)

    orientation_errors = []
    for r_est, r_gt in zip(est_rotations, gt_rotations):
        r_est_aligned = rot_align @ r_est
        r_err = r_gt @ r_est_aligned.T
        orientation_errors.append(rotation_angle_deg(r_err))
    orientation_errors = np.asarray(orientation_errors, dtype=np.float64)

    ate_rmse = float(np.sqrt(np.mean(position_errors ** 2)))
    ate_mean = float(np.mean(position_errors))
    orient_mean = float(np.mean(orientation_errors))

    fig = build_comparison_figure(
        gt_positions=gt_positions,
        est_positions_raw=est_positions,
        est_positions_aligned=est_positions_aligned,
        position_errors=position_errors,
        orientation_errors=orientation_errors,
    )

    summary = {
        "base_dir": str(base_dir),
        "frame_count_compared": count,
        "sim3_scale": scale,
        "sim3_rotation": rot_align.tolist(),
        "sim3_translation": trans_align.tolist(),
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "ate_max": float(np.max(position_errors)),
        "orientation_error_mean_deg": orient_mean,
        "orientation_error_max_deg": float(np.max(orientation_errors)),
    }

    if args.output_html:
        output_html = Path(args.output_html)
    else:
        output_html = base_dir / "camera_pose_comparison.html"
    fig.write_html(str(output_html), include_plotlyjs="cdn")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved comparison figure: {output_html}")


if __name__ == "__main__":
    main()
