import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_pose_matrix(item, index):
    if "T_wc" not in item:
        raise ValueError(f"Pose #{index} is missing required key 'T_wc'.")

    matrix = np.array(item["T_wc"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"Pose #{index} has invalid T_wc shape {matrix.shape}; expected (4, 4)."
        )

    return matrix


def main():
    parser = argparse.ArgumentParser(description="View point cloud with camera trajectory")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--ply", type=str, default="rgb.ply")
    parser.add_argument("--poses", type=str, default="rgb_poses.json")
    args = parser.parse_args()

    if args.base_dir is not None:
        base_dir = Path(args.base_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        working_dir = Path.cwd()
        if (working_dir / args.ply).is_file() and (working_dir / args.poses).is_file():
            base_dir = working_dir
        else:
            base_dir = script_dir

    ply_path = base_dir / args.ply
    pose_path = base_dir / args.poses

    if not ply_path.is_file():
        raise FileNotFoundError(f"Missing file: {ply_path}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing file: {pose_path}")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {ply_path}")

    with pose_path.open("r", encoding="utf-8") as f:
        poses = json.load(f)

    if not isinstance(poses, list):
        raise ValueError("rgb_poses.json must contain a top-level list of poses.")

    geoms = [pcd]
    centers = []

    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 500, 500, 320, 240)

    for index, item in enumerate(poses):
        if not isinstance(item, dict):
            raise ValueError(f"Pose #{index} is not a JSON object.")

        t_wc = load_pose_matrix(item, index)
        centers.append(t_wc[:3, 3])

        # Open3D camera visualization requires camera-to-world inverse.
        t_cw = np.linalg.inv(t_wc)
        cam = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic=intrinsic,
            extrinsic=t_cw,
            scale=0.2,
        )
        geoms.append(cam)

    centers = np.array(centers, dtype=float)

    if len(centers) >= 2:
        lines = [[i, i + 1] for i in range(len(centers) - 1)]
        traj = o3d.geometry.LineSet()
        traj.points = o3d.utility.Vector3dVector(centers)
        traj.lines = o3d.utility.Vector2iVector(lines)
        geoms.append(traj)

    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
