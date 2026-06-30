import argparse
import json
import math
from pathlib import Path


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def build_pose_entry(meta_path: Path, frame_idx: int):
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    camera_pose = data["camera"]["pose"]
    pos = camera_pose["position"]
    ori = camera_pose["orientation"]

    tx = float(pos["x"])
    ty = float(pos["y"])
    tz = float(pos["z"])
    qx = float(ori["x"])
    qy = float(ori["y"])
    qz = float(ori["z"])
    qw = float(ori["w"])

    if not all(math.isfinite(v) for v in [tx, ty, tz, qx, qy, qz, qw]):
        raise ValueError(f"Non-finite pose value in {meta_path}")

    rotation = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    t_wc = [
        [rotation[0][0], rotation[0][1], rotation[0][2], tx],
        [rotation[1][0], rotation[1][1], rotation[1][2], ty],
        [rotation[2][0], rotation[2][1], rotation[2][2], tz],
        [0.0, 0.0, 0.0, 1.0],
    ]

    frame_id = str(data.get("frame_id", frame_idx)).zfill(6)
    image_name = f"{frame_id}.png"
    timestamp = str(data.get("wall_time_ns", frame_idx))

    return {
        "frame_id": frame_idx,
        "timestamp": timestamp,
        "image": image_name,
        "source": str(meta_path),
        "T_wc": t_wc,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert AirSim per-frame meta JSON files into rgb_poses.json")
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)
    output = Path(args.output)

    meta_files = sorted(meta_dir.glob("*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No meta json files found in {meta_dir}")

    poses = [build_pose_entry(path, idx) for idx, path in enumerate(meta_files)]
    output.write_text(json.dumps(poses, indent=2), encoding="utf-8")
    print(f"wrote {len(poses)} poses to {output}")


if __name__ == "__main__":
    main()
