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
        except Exception as exc:
            print(f"AirSim 연결 실패 ({attempt}/{CONNECT_RETRIES}): {exc}")
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


def yaw_to_target_deg(px, py, tx, ty):
    return math.degrees(math.atan2(ty - py, tx - px))


def make_orbit_points(target_x, target_y, target_z, altitude_m, radius_m, points_per_lap):
    z = target_z - altitude_m
    points = []
    for i in range(points_per_lap):
        theta = 2.0 * math.pi * i / points_per_lap
        x = target_x + radius_m * math.cos(theta)
        y = target_y + radius_m * math.sin(theta)
        points.append((x, y, z))
    return points


def select_evenly_spaced_points(points, capture_count, include_closing_point):
    selected = list(points)
    if include_closing_point and points:
        selected = selected + [points[0]]

    if capture_count > len(selected):
        raise ValueError(
            f"요청한 capture_count={capture_count} 가 사용 가능한 경로 포인트 수 {len(selected)} 보다 큽니다."
        )

    if capture_count == len(selected):
        return selected

    if capture_count == 1:
        return [selected[0]]

    indices = []
    last_idx = len(selected) - 1
    for i in range(capture_count):
        idx = round(i * last_idx / (capture_count - 1))
        if idx not in indices:
            indices.append(idx)

    if len(indices) != capture_count:
        raise RuntimeError(
            f"균등 샘플링 인덱스 생성 실패: capture_count={capture_count}, indices={indices}"
        )
    return [selected[idx] for idx in indices]


def build_circular_capture_points(target_x, target_y, target_z, altitude_m, radius_m, points_per_lap, capture_count):
    base_points = make_orbit_points(target_x, target_y, target_z, altitude_m, radius_m, points_per_lap)
    return select_evenly_spaced_points(base_points, capture_count, include_closing_point=True)


def load_recommended_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "scaled_target_airsim_est" in data:
        target = data["scaled_target_airsim_est"]
    else:
        target = data["target_region_center"]

    points = []
    relative_points = []
    for item in data["waypoints"]:
        if "scaled_position_airsim_est" in item:
            points.append(item["scaled_position_airsim_est"])
        elif "position" in item:
            points.append(item["position"])
        else:
            raise ValueError("waypoint entry must contain 'scaled_position_airsim_est' or 'position'")

        if "scaled_relative_to_target_m" in item:
            relative_points.append(item["scaled_relative_to_target_m"])
        elif "relative_to_target" in item:
            relative_points.append(item["relative_to_target"])
        else:
            relative_points.append(None)
    return target, points, relative_points, data


def is_valid_pose(pose):
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    if position is None or orientation is None:
        return False
    values = [
        position.x_val,
        position.y_val,
        position.z_val,
        orientation.x_val,
        orientation.y_val,
        orientation.z_val,
        orientation.w_val,
    ]
    return all(math.isfinite(v) for v in values)


def resolve_runtime_target(client, args, recommended_meta):
    actor_name = args.target_actor_name or recommended_meta.get("target_actor_name")
    if actor_name:
        try:
            pose = client.simGetObjectPose(actor_name)
            print(f"raw actor pose for '{actor_name}': {pose}")
        except Exception as exc:
            print(f"target actor pose lookup failed for '{actor_name}': {exc}")
        else:
            if is_valid_pose(pose):
                resolved = [
                    float(pose.position.x_val),
                    float(pose.position.y_val),
                    float(pose.position.z_val),
                ]
                print(
                    f"resolved target from actor '{actor_name}': "
                    f"x={resolved[0]:.3f}, y={resolved[1]:.3f}, z={resolved[2]:.3f}"
                )
                return resolved
            print(f"invalid pose from actor '{actor_name}', fallback to CLI target")
    return [
        float(args.target_x),
        float(args.target_y),
        float(args.target_z),
    ]


def reanchor_points_to_target(relative_points, fallback_points, raw_target, target):
    anchored = []
    for rel, fallback in zip(relative_points, fallback_points):
        if rel is not None:
            dx = float(rel[0])
            dy = float(rel[1])
            dz = float(rel[2])
        else:
            dx = float(fallback[0]) - float(raw_target[0])
            dy = float(fallback[1]) - float(raw_target[1])
            dz = float(fallback[2]) - float(raw_target[2])

        anchored.append(
            [
                float(target[0]) + dx,
                float(target[1]) + dy,
                float(target[2]) + dz,
            ]
        )
    return anchored


def build_relative_points(raw_target, raw_points, relative_points):
    rels = []
    for rel, point in zip(relative_points, raw_points):
        if rel is not None:
            rels.append([float(rel[0]), float(rel[1]), float(rel[2])])
        else:
            rels.append([
                float(point[0]) - float(raw_target[0]),
                float(point[1]) - float(raw_target[1]),
                float(point[2]) - float(raw_target[2]),
            ])
    return rels


def reanchor_points_with_known_radius(raw_target, raw_points, relative_points, target, desired_radius_m):
    rels = build_relative_points(raw_target, raw_points, relative_points)
    xy_radii = [math.hypot(rel[0], rel[1]) for rel in rels if math.hypot(rel[0], rel[1]) > 1e-8]
    if not xy_radii:
        return reanchor_points_to_target(relative_points, raw_points, raw_target, target)

    current_mean_radius = sum(xy_radii) / len(xy_radii)
    scale_xy = float(desired_radius_m) / float(current_mean_radius)

    anchored = []
    for rel in rels:
        anchored.append([
            float(target[0]) + rel[0] * scale_xy,
            float(target[1]) + rel[1] * scale_xy,
            float(target[2]) + rel[2],
        ])
    return anchored


def center_align_points(points, target_x, target_y):
    if not points:
        return points
    mean_x = sum(float(p[0]) for p in points) / len(points)
    mean_y = sum(float(p[1]) for p in points) / len(points)
    dx = float(target_x) - mean_x
    dy = float(target_y) - mean_y
    return [[float(p[0]) + dx, float(p[1]) + dy, float(p[2])] for p in points]


def project_points_to_circle(points, target_x, target_y, radius_m):
    if not points:
        return points
    projected = []
    for point in points:
        dx = float(point[0]) - float(target_x)
        dy = float(point[1]) - float(target_y)
        norm = math.hypot(dx, dy)
        if norm <= 1e-8:
            px = float(target_x) + float(radius_m)
            py = float(target_y)
        else:
            angle = math.atan2(dy, dx)
            px = float(target_x) + float(radius_m) * math.cos(angle)
            py = float(target_y) + float(radius_m) * math.sin(angle)
        projected.append([px, py, float(point[2])])
    return projected


def project_points_to_top_arc(points, target_x, target_y, radius_m):
    if not points:
        return []

    count = len(points)
    if count == 1:
        angles = [math.pi / 2.0]
    else:
        angles = [math.pi - (math.pi * idx / (count - 1)) for idx in range(count)]

    projected = []
    for point, angle in zip(points, angles):
        px = float(target_x) + float(radius_m) * math.cos(angle)
        py = float(target_y) + float(radius_m) * math.sin(angle)
        projected.append([px, py, float(point[2])])
    return projected


def offset_points_xy(points, offset_x, offset_y):
    adjusted = []
    for point in points:
        adjusted.append([
            float(point[0]) + float(offset_x),
            float(point[1]) + float(offset_y),
            float(point[2]),
        ])
    return adjusted


def force_fixed_altitude(points, target_z, altitude_m):
    fixed_z = float(target_z) - float(altitude_m)
    adjusted = []
    for x, y, _ in points:
        adjusted.append([float(x), float(y), fixed_z])
    return adjusted


def fly_capture_points(client, out_dir, frame_idx, target_x, target_y, points, speed_mps, settle_sec, label):
    print(f"\n[{label}] captures={len(points)}")
    for idx, (x, y, z) in enumerate(points, start=1):
        yaw_deg = yaw_to_target_deg(x, y, target_x, target_y)
        print(f"  capture {idx:02d}: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw_deg:.2f}")
        client.moveToPositionAsync(
            x,
            y,
            z,
            speed_mps,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
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
        client.rotateToYawAsync(yaw_deg).join()
        save_capture(client, out_dir, frame_idx)
        frame_idx += 1
    return frame_idx


def densify_polyline(points, max_step_m):
    if len(points) <= 1:
        return points
    if max_step_m <= 0:
        raise ValueError(f"max_step_m 는 0보다 커야 합니다: {max_step_m}")

    dense = [list(points[0])]
    for start, end in zip(points, points[1:]):
        dist = math.dist(start, end)
        num_segments = max(1, int(math.ceil(dist / max_step_m)))
        for idx in range(1, num_segments + 1):
            alpha = idx / num_segments
            dense.append([
                start[0] + alpha * (end[0] - start[0]),
                start[1] + alpha * (end[1] - start[1]),
                start[2] + alpha * (end[2] - start[2]),
            ])
    return dense


def densify_turn_aware(points, max_step_m, turn_threshold_deg, turn_step_m):
    if len(points) <= 2:
        return densify_polyline(points, max_step_m)
    if turn_step_m <= 0:
        raise ValueError(f"turn_step_m 는 0보다 커야 합니다: {turn_step_m}")

    dense = [list(points[0])]
    for idx in range(1, len(points)):
        start = np.asarray(points[idx - 1], dtype=np.float64)
        end = np.asarray(points[idx], dtype=np.float64)
        step_limit = max_step_m

        if 0 < idx < len(points) - 1:
            prev_vec = start - np.asarray(points[idx - 2], dtype=np.float64)
            next_vec = np.asarray(points[idx + 1], dtype=np.float64) - end
            prev_norm = np.linalg.norm(prev_vec)
            next_norm = np.linalg.norm(next_vec)
            if prev_norm > 1e-8 and next_norm > 1e-8:
                cos_angle = float(np.clip(np.dot(prev_vec, next_vec) / (prev_norm * next_norm), -1.0, 1.0))
                turn_angle_deg = math.degrees(math.acos(cos_angle))
                if turn_angle_deg >= turn_threshold_deg:
                    step_limit = min(step_limit, turn_step_m)

        dist = float(np.linalg.norm(end - start))
        num_segments = max(1, int(math.ceil(dist / step_limit)))
        for seg_idx in range(1, num_segments + 1):
            alpha = seg_idx / num_segments
            dense.append([
                float(start[0] + alpha * (end[0] - start[0])),
                float(start[1] + alpha * (end[1] - start[1])),
                float(start[2] + alpha * (end[2] - start[2])),
            ])
    return dense


def select_best_entry_index(anchor_point, target_x, target_y, points):
    if not points:
        return 0

    anchor_xy = np.asarray(anchor_point[:2], dtype=np.float64)
    anchor_yaw = yaw_to_target_deg(anchor_point[0], anchor_point[1], target_x, target_y)
    best_idx = 0
    best_score = None

    for idx, point in enumerate(points):
        point_xy = np.asarray(point[:2], dtype=np.float64)
        dist_xy = float(np.linalg.norm(point_xy - anchor_xy))
        point_yaw = yaw_to_target_deg(point[0], point[1], target_x, target_y)
        yaw_diff = abs(point_yaw - anchor_yaw)
        while yaw_diff > 180.0:
            yaw_diff -= 360.0
        yaw_diff = abs(yaw_diff)

        score = dist_xy + 0.1 * yaw_diff
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def rotate_points(points, start_idx):
    if not points:
        return points
    start_idx = int(start_idx) % len(points)
    return list(points[start_idx:]) + list(points[:start_idx])


def fly_continuous_path_capture(
    client,
    out_dir,
    frame_idx,
    target_x,
    target_y,
    points,
    speed_mps,
    settle_sec,
    label,
):
    if len(points) == 0:
        return frame_idx

    print(f"\n[{label}] entering first waypoint without turn")

    first_x, first_y, first_z = points[0]
    client.moveToPositionAsync(
        first_x,
        first_y,
        first_z,
        speed_mps,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
    ).join()

    stabilize_wait = wait_until_stable(
        client,
        max(settle_sec, MIN_STABILIZE_WAIT_SEC),
        MAX_STABILIZE_WAIT_SEC,
        STABILIZE_CHECK_INTERVAL_SEC,
        STABILIZE_LINEAR_SPEED_MPS,
        STABILIZE_ANGULAR_SPEED_RADPS,
    )
    print(f"    stabilized at first waypoint after {stabilize_wait:.2f}s")

    first_yaw = yaw_to_target_deg(first_x, first_y, target_x, target_y)
    client.rotateToYawAsync(first_yaw).join()
    save_capture(client, out_dir, frame_idx)
    frame_idx += 1

    if len(points) == 1:
        return frame_idx

    sampled_points = [list(p) for p in points[1:]]

    print(f"[{label}] captures={len(sampled_points) + 1} (recommended path)")

    for idx, (x, y, z) in enumerate(sampled_points, start=2):
        yaw_deg = yaw_to_target_deg(x, y, target_x, target_y)
        client.moveToPositionAsync(
            x,
            y,
            z,
            speed_mps,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
        ).join()
        stabilize_wait = wait_until_stable(
            client,
            max(settle_sec, MIN_STABILIZE_WAIT_SEC),
            MAX_STABILIZE_WAIT_SEC,
            STABILIZE_CHECK_INTERVAL_SEC,
            STABILIZE_LINEAR_SPEED_MPS,
            STABILIZE_ANGULAR_SPEED_RADPS,
        )
        print(f"    stabilized at waypoint {idx:02d} after {stabilize_wait:.2f}s")
        client.rotateToYawAsync(yaw_deg).join()
        save_capture(client, out_dir, frame_idx)
        frame_idx += 1

    return frame_idx


def build_recommended_points(args, recommended_target, raw_points, relative_points, straight_target):
    if args.recommended_xy_mode == "scaled_absolute":
        points = [[float(p[0]), float(p[1]), float(p[2])] for p in raw_points]
    else:
        if args.recommended_xy_mode == "shape_preserving_center_aligned":
            points = reanchor_points_to_target(
                relative_points,
                raw_points,
                recommended_target,
                straight_target,
            )
            points = center_align_points(points, straight_target[0], straight_target[1])
        elif args.recommended_xy_mode == "circular_reprojected":
            points = reanchor_points_with_known_radius(
                recommended_target,
                raw_points,
                relative_points,
                straight_target,
                args.orbit_radius,
            )
            points = center_align_points(points, straight_target[0], straight_target[1])
            points = project_points_to_circle(
                points,
                straight_target[0],
                straight_target[1],
                args.orbit_radius,
            )
        elif args.recommended_xy_mode == "top_arc_reprojected":
            points = reanchor_points_with_known_radius(
                recommended_target,
                raw_points,
                relative_points,
                straight_target,
                args.orbit_radius,
            )
            points = center_align_points(points, straight_target[0], straight_target[1])
            points = project_points_to_top_arc(
                points,
                straight_target[0],
                straight_target[1],
                args.orbit_radius,
            )
        elif args.recommended_xy_mode == "center_aligned_radius_calibrated":
            points = reanchor_points_with_known_radius(
                recommended_target,
                raw_points,
                relative_points,
                straight_target,
                args.orbit_radius,
            )
            points = center_align_points(points, straight_target[0], straight_target[1])
        elif args.recommended_xy_mode == "radius_calibrated":
            points = reanchor_points_with_known_radius(
                recommended_target,
                raw_points,
                relative_points,
                straight_target,
                args.orbit_radius,
            )
        else:
            points = reanchor_points_to_target(
                relative_points,
                raw_points,
                recommended_target,
                straight_target,
            )

        points = offset_points_xy(
            points,
            args.recommended_center_offset_x,
            args.recommended_center_offset_y,
        )
        points = force_fixed_altitude(points, straight_target[2], args.recommended_altitude)

    if args.limit_count is not None:
        points = points[: args.limit_count]

    return points


def main():
    parser = argparse.ArgumentParser(
        description="Car-centered AirSim capture runner. Default workflow is a simple circular orbit."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="orbit_only",
        choices=["orbit_only", "straight_then_recommended", "orbit_then_recommended"],
    )
    parser.add_argument("--recommended-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-z", type=float, required=True)
    parser.add_argument("--target-actor-name", type=str, default=None)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--settle-sec", type=float, default=0.3)
    parser.add_argument("--takeoff-altitude", type=float, default=8.0)
    parser.add_argument("--orbit-altitude", type=float, default=8.0)
    parser.add_argument("--orbit-radius", type=float, default=7.0)
    parser.add_argument("--orbit-capture-count", type=int, default=17)
    parser.add_argument("--orbit-points-per-lap", type=int, default=24)
    parser.add_argument("--limit-count", type=int, default=17)
    parser.add_argument("--recommended-altitude", type=float, default=8.0)
    parser.add_argument("--recommended-max-step", type=float, default=9999.0)
    parser.add_argument("--recommended-turn-threshold-deg", type=float, default=20.0)
    parser.add_argument("--recommended-turn-step", type=float, default=9999.0)
    parser.add_argument("--auto-select-entry", action="store_true")
    parser.add_argument("--recommended-center-offset-x", type=float, default=0.0)
    parser.add_argument("--recommended-center-offset-y", type=float, default=0.0)
    parser.add_argument(
        "--recommended-xy-mode",
        type=str,
        default="shape_preserving_center_aligned",
        choices=[
            "scaled_absolute",
            "relative_to_target",
            "radius_calibrated",
            "center_aligned_radius_calibrated",
            "circular_reprojected",
            "top_arc_reprojected",
            "shape_preserving_center_aligned",
        ],
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dirs(out_dir)

    fallback_target = [
        float(args.target_x),
        float(args.target_y),
        float(args.target_z),
    ]
    straight_target = list(fallback_target)

    recommended_json = Path(args.recommended_json) if args.recommended_json else None
    recommended_target = None
    raw_points = []
    relative_points = []
    meta = {}
    points = []

    if args.mode in {"straight_then_recommended", "orbit_then_recommended"}:
        if recommended_json is None:
            raise ValueError("--recommended-json is required in recommended mode")
        recommended_target, raw_points, relative_points, meta = load_recommended_json(recommended_json)

    manifest = {
        "mode": args.mode,
        "recommended_json": str(recommended_json) if recommended_json else None,
        "target_region_center": recommended_target,
        "straight_prefix_target": straight_target,
        "target_actor_name": args.target_actor_name,
        "recommended_altitude_m": args.recommended_altitude,
        "recommended_max_step_m": args.recommended_max_step,
        "recommended_turn_threshold_deg": args.recommended_turn_threshold_deg,
        "recommended_turn_step_m": args.recommended_turn_step,
        "auto_select_entry": args.auto_select_entry,
        "recommended_xy_mode": args.recommended_xy_mode,
        "recommended_center_offset_x_m": args.recommended_center_offset_x,
        "recommended_center_offset_y_m": args.recommended_center_offset_y,
        "waypoints": points,
        "reanchored_from_relative_offsets": any(rel is not None for rel in relative_points),
        "coordinate_mode": "scaled_airsim_est" if "scaled_target_airsim_est" in meta else "raw_json",
        "source_meta": meta,
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

        straight_target = resolve_runtime_target(client, args, meta)
        print(f"using runtime target: {straight_target}")
        if args.target_actor_name and straight_target == fallback_target:
            print("actor pose lookup did not produce a usable target; using CLI target coordinates")

        if args.mode in {"straight_then_recommended", "orbit_then_recommended"}:
            points = build_recommended_points(
                args,
                recommended_target,
                raw_points,
                relative_points,
                straight_target,
            )
            manifest["straight_prefix_target"] = straight_target
            manifest["waypoints"] = points
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.mode == "orbit_only":
            orbit_points = build_circular_capture_points(
                straight_target[0],
                straight_target[1],
                straight_target[2],
                args.orbit_altitude,
                args.orbit_radius,
                args.orbit_points_per_lap,
                args.orbit_capture_count,
            )
            frame_idx = fly_capture_points(
                client,
                out_dir,
                frame_idx,
                straight_target[0],
                straight_target[1],
                orbit_points,
                args.speed,
                args.settle_sec,
                label=(
                    "OrbitOnly"
                    f" altitude={args.orbit_altitude:.2f}m"
                    f" radius={args.orbit_radius:.2f}m"
                    f" captures={len(orbit_points)}"
                ),
            )
            prefix_end_points = orbit_points
        elif args.mode == "straight_then_recommended":
            prefix_end_points = []
        else:
            orbit_points = build_circular_capture_points(
                straight_target[0],
                straight_target[1],
                straight_target[2],
                args.orbit_altitude,
                args.orbit_radius,
                args.orbit_points_per_lap,
                args.orbit_capture_count,
            )
            frame_idx = fly_capture_points(
                client,
                out_dir,
                frame_idx,
                straight_target[0],
                straight_target[1],
                orbit_points,
                args.speed,
                args.settle_sec,
                label=(
                    "OrbitPrefix"
                    f" altitude={args.orbit_altitude:.2f}m"
                    f" radius={args.orbit_radius:.2f}m"
                    f" captures={len(orbit_points)}"
                ),
            )
            prefix_end_points = orbit_points

        if points and args.auto_select_entry and prefix_end_points:
            entry_idx = select_best_entry_index(
                prefix_end_points[-1],
                straight_target[0],
                straight_target[1],
                points,
            )
            points = rotate_points(points, entry_idx)
            manifest["selected_entry_index"] = entry_idx
        else:
            manifest["selected_entry_index"] = 0

        if points and args.mode != "orbit_only":
            dense_recommended_points = densify_turn_aware(
                points,
                args.recommended_max_step,
                args.recommended_turn_threshold_deg,
                args.recommended_turn_step,
            )
            frame_idx = fly_continuous_path_capture(
                client,
                out_dir,
                frame_idx,
                straight_target[0],
                straight_target[1],
                dense_recommended_points,
                args.speed,
                args.settle_sec,
                label=(
                    f"RecommendedJsonDirect mode={manifest['coordinate_mode']}"
                    f" xy_mode={manifest['recommended_xy_mode']}"
                    f" reanchored={manifest['reanchored_from_relative_offsets']}"
                    f" captures={len(dense_recommended_points)}"
                ),
            )

        print("hover")
        client.hoverAsync().join()
    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        print("done")


if __name__ == "__main__":
    main()
