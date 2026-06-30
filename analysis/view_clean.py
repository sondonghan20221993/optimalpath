import json
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


# ===== Adjustable parameters =====
# File names
POINT_CLOUD_FILE = "airsim_degree_notilt_more_distance/rgb.ply"       # Point cloud file (ply, pcd, txt, npy, etc.)
POSE_FILE = "airsim_degree_notilt_more_distance/rgb_poses.json"       # Pose file (json, txt, npy, etc.)

# Visualization parameters
VOXEL_SIZE = 0.0000001        # 값이 클수록 포인트 수가 줄어들며, 더 깔끔하고 빠르게 처리됨
POINT_SIZE = 2     # 렌더링되는 포인트 크기
FRUSTUM_SCALE = 0.08     # 카메라 프러스텀 크기
FRUSTUM_STEP = 1         # N번째마다 카메라 프러스텀을 그림
TRAJ_STEP = 1            # N번째마다 궤적 점을 샘플링
BACKGROUND_BLACK = True # True = 검은 배경, False = 흰 배경
# ================================


def resolve_base_dir(pcd_name: str = POINT_CLOUD_FILE, pose_name: str = POSE_FILE) -> Path:
    script_dir = Path(__file__).resolve().parent
    working_dir = Path.cwd()

    required = (pcd_name, pose_name)
    if all((working_dir / name).is_file() for name in required):
        return working_dir
    if all((script_dir / name).is_file() for name in required):
        return script_dir

    raise FileNotFoundError(
        f"Could not find {pcd_name} and {pose_name} in the current folder "
        f"({working_dir}) or the script folder ({script_dir})."
    )


def load_point_cloud(pcd_path: Path) -> o3d.geometry.PointCloud:
    """Load point cloud from various formats (ply, pcd, txt, etc.)"""
    suffix = pcd_path.suffix.lower()
    
    if suffix in ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts']:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
    elif suffix in ['.txt', '.npy']:
        if suffix == '.npy':
            points = np.load(str(pcd_path))
        else:  # .txt
            points = np.loadtxt(str(pcd_path))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:  # RGB color included
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")
    
    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {pcd_path}")
    
    return pcd


def load_poses(pose_path: Path) -> list:
    """Load poses from various formats (json, txt, etc.)"""
    suffix = pose_path.suffix.lower()
    
    if suffix == '.json':
        with pose_path.open("r", encoding="utf-8") as f:
            poses = json.load(f)
    elif suffix == '.txt':
        poses = []
        with pose_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Parse line as space-separated values (16 elements for 4x4 matrix)
                values = list(map(float, line.split()))
                if len(values) == 16:
                    pose_dict = {"T_wc": np.array(values).reshape(4, 4).tolist()}
                    poses.append(pose_dict)
                else:
                    raise ValueError(f"Expected 16 values per line, got {len(values)}")
    elif suffix == '.npy':
        data = np.load(str(pose_path))
        if len(data.shape) == 3 and data.shape[1:] == (4, 4):
            poses = [{"T_wc": matrix.tolist()} for matrix in data]
        else:
            raise ValueError(f"NPY file shape {data.shape} not compatible. Expected (N, 4, 4)")
    else:
        raise ValueError(f"Unsupported pose format: {suffix}")
    
    if not isinstance(poses, list):
        raise ValueError("Poses must be a list.")
    
    return poses


def load_pose_matrix(item: dict, index: int) -> np.ndarray:
    if "T_wc" not in item:
        raise ValueError(f"Pose #{index} is missing required key 'T_wc'.")

    matrix = np.array(item["T_wc"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"Pose #{index} has invalid T_wc shape {matrix.shape}; expected (4, 4)."
        )

    return matrix


def main(pcd_file: str = POINT_CLOUD_FILE, pose_file: str = POSE_FILE, voxel_size: float = None):
    base_dir = resolve_base_dir(pcd_file, pose_file)
    ply_path = base_dir / pcd_file
    pose_path = base_dir / pose_file

    # Load point cloud
    pcd = load_point_cloud(ply_path)

    # Apply voxel downsampling if specified
    vs = voxel_size if voxel_size is not None else VOXEL_SIZE
    if vs > 0:
        pcd = pcd.voxel_down_sample(voxel_size=vs)

    # Load poses
    poses = load_poses(pose_path)

    geoms = [pcd]
    centers = []
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 500, 500, 320, 240)

    for index, item in enumerate(poses):
        if not isinstance(item, dict):
            raise ValueError(f"Pose #{index} is not a JSON object.")

        t_wc = load_pose_matrix(item, index)
        centers.append(t_wc[:3, 3])

        if FRUSTUM_STEP <= 0:
            raise ValueError("FRUSTUM_STEP must be >= 1.")
        if index % FRUSTUM_STEP != 0:
            continue

        t_cw = np.linalg.inv(t_wc)
        cam = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic=intrinsic,
            extrinsic=t_cw,
            scale=FRUSTUM_SCALE,
        )
        cam.paint_uniform_color([0.0, 0.0, 1.0])
        geoms.append(cam)

    centers = np.array(centers, dtype=float)

    if TRAJ_STEP <= 0:
        raise ValueError("TRAJ_STEP must be >= 1.")

    if len(centers) >= 2:
        sampled = centers[::TRAJ_STEP]
        lines = [[i, i + 1] for i in range(len(sampled) - 1)]

        traj = o3d.geometry.LineSet()
        traj.points = o3d.utility.Vector3dVector(sampled)
        traj.lines = o3d.utility.Vector2iVector(lines)
        traj.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(traj)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"PLY Viewer - {ply_path.parent.name}",
        width=1600,
        height=1000,
    )

    for geom in geoms:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    if BACKGROUND_BLACK:
        opt.background_color = np.array([0.0, 0.0, 0.0])
    else:
        opt.background_color = np.array([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize point cloud with camera trajectory"
    )
    parser.add_argument(
        "--pcd",
        default=POINT_CLOUD_FILE,
        help=f"Point cloud file (ply, pcd, txt, npy, etc.). Default: {POINT_CLOUD_FILE}"
    )
    parser.add_argument(
        "--poses",
        default=POSE_FILE,
        help=f"Pose file (json, txt, npy, etc.). Default: {POSE_FILE}"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size for downsampling. If not set, uses VOXEL_SIZE constant"
    )
    
    args = parser.parse_args()
    main(pcd_file=args.pcd, pose_file=args.poses, voxel_size=args.voxel_size)
