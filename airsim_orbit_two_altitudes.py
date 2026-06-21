import argparse
import importlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

AIRSIM_BACKEND = "cosysairsim"  # auto | airsim | cosysairsim
AIRSIM_HOST = "127.0.0.1"
AIRSIM_PORT = 41451
CONNECT_RETRIES = 5
RETRY_DELAY_SEC = 2.0
CAMERA_NAME = "0"
VEHICLE_NAME = ""
CAMERA_PITCH_DEG = -45.0
SAVE_DEPTH_NPY = False
SAVE_DEPTH_PNG = False
MIN_STABILIZE_WAIT_SEC = 0.3
MAX_STABILIZE_WAIT_SEC = 2.0
STABILIZE_CHECK_INTERVAL_SEC = 0.1
STABILIZE_LINEAR_SPEED_MPS = 0.15
STABILIZE_ANGULAR_SPEED_RADPS = 0.08


def load_airsim_backend():
    if AIRSIM_BACKEND == "airsim":
        return importlib.import_module("airsim")
    if AIRSIM_BACKEND == "cosysairsim":
        return importlib.import_module("cosysairsim")
    try:
        return importlib.import_module("airsim")
    except Exception:
        return importlib.import_module("cosysairsim")


airsim = load_airsim_backend()


def connect_client():
    client = airsim.MultirotorClient(AIRSIM_HOST, AIRSIM_PORT)
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            client.confirmConnection()
            return client
        except Exception as e:
            print(f"AirSim 연결 실패 ({attempt}/{CONNECT_RETRIES}): {e}")
            if attempt < CONNECT_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    raise RuntimeError(
        f"AirSim 서버에 연결하지 못했습니다. Unreal Editor/CitySample 실행 여부와 AirSim 포트({AIRSIM_PORT})를 확인하세요."
    )


def euler_to_quaternion(pitch, roll, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return airsim.Quaternionr(x, y, z, w)


def configure_camera_pitch(client, pitch_deg):
    camera_pose = airsim.Pose(
        airsim.Vector3r(0.0, 0.0, 0.0),
        euler_to_quaternion(math.radians(pitch_deg), 0.0, 0.0),
    )
    client.simSetCameraPose(CAMERA_NAME, camera_pose, vehicle_name=VEHICLE_NAME)
    print(f"camera pitch set: {pitch_deg:.1f} deg")


def vector3r_norm(v):
    return math.sqrt(v.x_val * v.x_val + v.y_val * v.y_val + v.z_val * v.z_val)


def wait_until_stable(
    client,
    min_wait_sec,
    max_wait_sec,
    check_interval_sec,
    linear_speed_threshold,
    angular_speed_threshold,
):
    start_time = time.time()
    time.sleep(max(0.0, min_wait_sec))

    while True:
        state = client.getMultirotorState()
        linear_speed = vector3r_norm(state.kinematics_estimated.linear_velocity)
        angular_speed = vector3r_norm(state.kinematics_estimated.angular_velocity)
        elapsed = time.time() - start_time

        if linear_speed <= linear_speed_threshold and angular_speed <= angular_speed_threshold:
            return elapsed
        if elapsed >= max_wait_sec:
            return elapsed

        time.sleep(max(0.0, check_interval_sec))


def yaw_to_target_deg(px, py, tx, ty):
    return math.degrees(math.atan2(ty - py, tx - px))


def make_orbit_points(target_x, target_y, target_z, altitude_m, radius_m, points_per_lap):
    z = target_z - altitude_m  # AirSim NED: higher altitude => smaller (more negative) z
    points = []
    for i in range(points_per_lap):
        theta = 2.0 * math.pi * i / points_per_lap
        x = target_x + radius_m * math.cos(theta)
        y = target_y + radius_m * math.sin(theta)
        points.append((x, y, z))
    return points


def ensure_dirs(root: Path):
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    if SAVE_DEPTH_NPY:
        (root / "depth_npy").mkdir(parents=True, exist_ok=True)
    if SAVE_DEPTH_PNG:
        (root / "depth_png").mkdir(parents=True, exist_ok=True)


def vector3r_to_dict(v):
    return {"x": float(v.x_val), "y": float(v.y_val), "z": float(v.z_val)}


def quaternionr_to_dict(q):
    return {
        "w": float(q.w_val),
        "x": float(q.x_val),
        "y": float(q.y_val),
        "z": float(q.z_val),
    }


def pose_to_dict(pose):
    return {
        "position": vector3r_to_dict(pose.position),
        "orientation": quaternionr_to_dict(pose.orientation),
    }


def camera_info_to_dict(camera_info):
    return {
        "pose": pose_to_dict(camera_info.pose),
        "fov": float(camera_info.fov),
    }


def imu_to_dict(imu):
    return {
        "time_stamp": int(imu.time_stamp),
        "orientation": quaternionr_to_dict(imu.orientation),
        "angular_velocity": vector3r_to_dict(imu.angular_velocity),
        "linear_acceleration": vector3r_to_dict(imu.linear_acceleration),
    }


def gps_to_dict(gps):
    gnss = gps.gnss
    data = {
        "time_stamp": int(getattr(gps, "time_stamp", 0)),
        "geo_point": {
            "latitude": float(getattr(gnss.geo_point, "latitude", 0.0)),
            "longitude": float(getattr(gnss.geo_point, "longitude", 0.0)),
            "altitude": float(getattr(gnss.geo_point, "altitude", 0.0)),
        },
        "velocity": {
            "x": float(getattr(gnss.velocity, "x_val", 0.0)),
            "y": float(getattr(gnss.velocity, "y_val", 0.0)),
            "z": float(getattr(gnss.velocity, "z_val", 0.0)),
        },
        "eph": float(getattr(gnss, "eph", 0.0)),
        "epv": float(getattr(gnss, "epv", 0.0)),
        "fix_type": int(getattr(gnss, "fix_type", 0)),
        "time_utc": int(getattr(gnss, "time_utc", 0)),
    }
    if hasattr(gps, "is_valid"):
        data["is_valid"] = bool(gps.is_valid)
    return data


def kinematics_to_dict(kin):
    return {
        "position": vector3r_to_dict(kin.position),
        "orientation": quaternionr_to_dict(kin.orientation),
        "linear_velocity": vector3r_to_dict(kin.linear_velocity),
        "angular_velocity": vector3r_to_dict(kin.angular_velocity),
        "linear_acceleration": vector3r_to_dict(kin.linear_acceleration),
        "angular_acceleration": vector3r_to_dict(kin.angular_acceleration),
    }


def save_depth_files(depth_2d, npy_path: Path, png_path: Path):
    depth = np.array(depth_2d, dtype=np.float32)
    if SAVE_DEPTH_NPY:
        np.save(str(npy_path), depth)
    if SAVE_DEPTH_PNG:
        valid = np.isfinite(depth)
        if np.any(valid):
            d = depth.copy()
            d[~valid] = 0.0
            d_min = float(np.min(d[valid]))
            d_max = float(np.max(d[valid]))
            if d_max > d_min:
                depth_norm = (d - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(d, dtype=np.float32)
            depth_u8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
        else:
            depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
        ok = cv2.imwrite(str(png_path), depth_u8)
        if not ok:
            raise RuntimeError(f"Depth PNG 저장 실패: {png_path}")


def save_capture(client, out_dir: Path, frame_idx: int):
    frame_id = f"{frame_idx:06d}"
    t_wall_ns = time.time_ns()

    requests = [
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPerspective, True, False),
    ]
    responses = client.simGetImages(requests)
    if len(responses) != 2:
        raise RuntimeError(f"이미지 응답 개수 이상: {len(responses)}")

    rgb_resp = responses[0]
    depth_resp = responses[1]

    if rgb_resp.width == 0 or rgb_resp.height == 0:
        raise RuntimeError("RGB 이미지가 비어 있음")
    if depth_resp.width == 0 or depth_resp.height == 0:
        raise RuntimeError("Depth 이미지가 비어 있음")

    pose = client.simGetVehiclePose()
    camera_info = client.simGetCameraInfo(CAMERA_NAME)
    imu = client.getImuData()
    gps = client.getGpsData()
    mstate = client.getMultirotorState()

    rgb_path = out_dir / "rgb" / f"{frame_id}.png"
    meta_path = out_dir / "meta" / f"{frame_id}.json"
    depth_npy_path = out_dir / "depth_npy" / f"{frame_id}.npy"
    depth_png_path = out_dir / "depth_png" / f"{frame_id}.png"

    rgb_image = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8)
    expected = rgb_resp.width * rgb_resp.height * 3
    if rgb_image.size != expected:
        raise RuntimeError(
            f"RGB 크기 불일치: got={rgb_image.size}, expected={expected} ({rgb_resp.width}x{rgb_resp.height}x3)"
        )
    rgb_image = rgb_image.reshape(rgb_resp.height, rgb_resp.width, 3)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(rgb_path), rgb_image):
        raise RuntimeError(f"RGB 저장 실패: {rgb_path}")

    depth_2d = airsim.list_to_2d_float_array(
        depth_resp.image_data_float,
        depth_resp.width,
        depth_resp.height,
    )
    if SAVE_DEPTH_NPY or SAVE_DEPTH_PNG:
        save_depth_files(depth_2d, depth_npy_path, depth_png_path)

    meta = {
        "frame_id": frame_id,
        "wall_time_ns": int(t_wall_ns),
        "camera_name": CAMERA_NAME,
        "camera": camera_info_to_dict(camera_info),
        "image": {
            "rgb_path": str(rgb_path.relative_to(out_dir)),
            "depth_npy_path": str(depth_npy_path.relative_to(out_dir)) if SAVE_DEPTH_NPY else None,
            "depth_png_path": str(depth_png_path.relative_to(out_dir)) if SAVE_DEPTH_PNG else None,
            "rgb_width": int(rgb_resp.width),
            "rgb_height": int(rgb_resp.height),
            "depth_width": int(depth_resp.width),
            "depth_height": int(depth_resp.height),
        },
        "vehicle_pose": pose_to_dict(pose),
        "multirotor_kinematics": kinematics_to_dict(mstate.kinematics_estimated),
        "gps": gps_to_dict(gps),
        "imu": imu_to_dict(imu),
        "airsim_timestamps": {
            "rgb_time_stamp": int(rgb_resp.time_stamp),
            "depth_time_stamp": int(depth_resp.time_stamp),
            "imu_time_stamp": int(imu.time_stamp),
            "gps_time_stamp": int(gps.time_stamp),
            "state_timestamp": int(mstate.timestamp),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def fly_orbit(client, out_dir, frame_idx, target_x, target_y, target_z, altitude_m, radius_m, speed_mps, points_per_lap, settle_sec):
    points = make_orbit_points(target_x, target_y, target_z, altitude_m, radius_m, points_per_lap)
    print(f"\n[Orbit] altitude={altitude_m:.2f}m radius={radius_m:.2f}m points={points_per_lap}")

    for idx, (x, y, z) in enumerate(points, start=1):
        yaw_deg = yaw_to_target_deg(x, y, target_x, target_y)
        print(f"  waypoint {idx:02d}: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw_deg:.2f}")
        client.moveToPositionAsync(
            x,
            y,
            z,
            speed_mps,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg),
        ).join()
        stabilize_wait = wait_until_stable(
            client,
            max(settle_sec, MIN_STABILIZE_WAIT_SEC),
            MAX_STABILIZE_WAIT_SEC,
            STABILIZE_CHECK_INTERVAL_SEC,
            STABILIZE_LINEAR_SPEED_MPS,
            STABILIZE_ANGULAR_SPEED_RADPS,
        )
        print(f"    stabilized for capture after {stabilize_wait:.2f}s")
        save_capture(client, out_dir, frame_idx)
        frame_idx += 1

    x, y, z = points[0]
    yaw_deg = yaw_to_target_deg(x, y, target_x, target_y)
    client.moveToPositionAsync(
        x,
        y,
        z,
        speed_mps,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg),
    ).join()
    stabilize_wait = wait_until_stable(
        client,
        max(settle_sec, MIN_STABILIZE_WAIT_SEC),
        MAX_STABILIZE_WAIT_SEC,
        STABILIZE_CHECK_INTERVAL_SEC,
        STABILIZE_LINEAR_SPEED_MPS,
        STABILIZE_ANGULAR_SPEED_RADPS,
    )
    print(f"    stabilized for capture after {stabilize_wait:.2f}s")
    save_capture(client, out_dir, frame_idx)
    frame_idx += 1
    return frame_idx


def main():
    parser = argparse.ArgumentParser(description="Orbit around a target at 8m and 5m altitude in AirSim with dataset capture.")
    parser.add_argument("--target-x", type=float, required=True, help="Target X coordinate")
    parser.add_argument("--target-y", type=float, required=True, help="Target Y coordinate")
    parser.add_argument("--target-z", type=float, required=True, help="Target Z coordinate")
    parser.add_argument("--coords-in-cm", action="store_true", help="Interpret target/start coordinates as Unreal centimeters instead of AirSim meters")
    parser.add_argument("--start-x", type=float, default=None, help="Start/reference X coordinate (required with --coords-in-cm)")
    parser.add_argument("--start-y", type=float, default=None, help="Start/reference Y coordinate (required with --coords-in-cm)")
    parser.add_argument("--start-z", type=float, default=None, help="Start/reference Z coordinate (required with --coords-in-cm)")
    parser.add_argument("--radius", type=float, default=30.0, help="Orbit radius in meters")
    parser.add_argument("--speed", type=float, default=3.0, help="Move speed in m/s")
    parser.add_argument("--points-per-lap", type=int, default=12, help="Waypoints used to approximate one circle")
    parser.add_argument("--settle-sec", type=float, default=0.05, help="Pause after each waypoint")
    parser.add_argument("--takeoff-altitude", type=float, default=8.0, help="Initial takeoff altitude above current position")
    parser.add_argument("--high-altitude", type=float, default=8.0, help="Higher orbit altitude above target")
    parser.add_argument("--low-altitude", type=float, default=5.0, help="Lower orbit altitude above target")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save rgb/meta captures")
    args = parser.parse_args()

    target_x = args.target_x
    target_y = args.target_y
    target_z = args.target_z

    if args.coords_in_cm:
        if args.start_x is None or args.start_y is None or args.start_z is None:
            raise ValueError("--coords-in-cm 사용 시 --start-x --start-y --start-z 를 함께 넣어야 합니다.")
        target_x = (args.target_x - args.start_x) / 100.0
        target_y = (args.target_y - args.start_y) / 100.0
        target_z = -(args.target_z - args.start_z) / 100.0
        print(
            "converted Unreal(cm) -> AirSim local NED(m): "
            f"target=({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )

    out_dir = Path(args.output_dir)
    ensure_dirs(out_dir)
    manifest = {
        "target_local_ned_m": {"x": target_x, "y": target_y, "z": target_z},
        "radius_m": args.radius,
        "speed_mps": args.speed,
        "points_per_lap": args.points_per_lap,
        "high_altitude_m": args.high_altitude,
        "low_altitude_m": args.low_altitude,
        "coords_in_cm": args.coords_in_cm,
        "source_target_input": {"x": args.target_x, "y": args.target_y, "z": args.target_z},
        "source_start_input": {"x": args.start_x, "y": args.start_y, "z": args.start_z},
        "airsim_backend": AIRSIM_BACKEND,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    client = connect_client()
    client.enableApiControl(True)
    client.armDisarm(True)
    configure_camera_pitch(client, CAMERA_PITCH_DEG)
    frame_idx = 0

    try:
        print("takeoff")
        client.takeoffAsync().join()

        state = client.getMultirotorState()
        start_z = state.kinematics_estimated.position.z_val
        safe_z = start_z - args.takeoff_altitude
        print(f"rise to safe altitude: {args.takeoff_altitude:.2f}m")
        client.moveToZAsync(safe_z, args.speed).join()
        client.hoverAsync().join()
        wait_until_stable(
            client,
            max(args.settle_sec, 0.5),
            MAX_STABILIZE_WAIT_SEC,
            STABILIZE_CHECK_INTERVAL_SEC,
            STABILIZE_LINEAR_SPEED_MPS,
            STABILIZE_ANGULAR_SPEED_RADPS,
        )

        frame_idx = fly_orbit(
            client,
            out_dir,
            frame_idx,
            target_x,
            target_y,
            target_z,
            args.high_altitude,
            args.radius,
            args.speed,
            args.points_per_lap,
            args.settle_sec,
        )
        frame_idx = fly_orbit(
            client,
            out_dir,
            frame_idx,
            target_x,
            target_y,
            target_z,
            args.low_altitude,
            args.radius,
            args.speed,
            args.points_per_lap,
            args.settle_sec,
        )

        print("hover")
        client.hoverAsync().join()
    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        print("done")


if __name__ == "__main__":
    main()
