import json
import time
from pathlib import Path

import airsim
import numpy as np


DATA_DIR = Path(r"C:\Users\sdh97\Desktop\o_circle_3")
PATH_FILE = DATA_DIR / "recommended_candidate_path.json"

# AirSim NED transform.
# If the recorded path does not line up with your map, only adjust these values.
PATH_SCALE = 1.0
PATH_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)
INVERT_Z = True

FLIGHT_SPEED = 2.5
LOOKAHEAD = 1.0
WAYPOINT_TIMEOUT = 120.0
STARTUP_ALTITUDE = -3.0
VEHICLE_NAME = ""


def load_path_points(path_file: Path) -> np.ndarray:
    with path_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    points = [data["start_position"]]
    points.extend(waypoint["position"] for waypoint in data["waypoints"])
    return np.asarray(points, dtype=float)


def build_dense_path(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    dense_parts = []
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
        dense_parts.append(start + (end - start) * t)

    dense_parts.append(points[-1][None, :])
    return np.vstack(dense_parts)


def to_airsim_points(points: np.ndarray) -> list[airsim.Vector3r]:
    transformed = points * PATH_SCALE + PATH_OFFSET
    if INVERT_Z:
        transformed[:, 2] *= -1.0

    return [airsim.Vector3r(float(x), float(y), float(z)) for x, y, z in transformed]


def connect_client() -> airsim.MultirotorClient:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)
    return client


def plot_path(client: airsim.MultirotorClient, path: list[airsim.Vector3r]) -> None:
    client.simFlushPersistentMarkers()
    client.simPlotLineStrip(
        path,
        color_rgba=[0.0, 0.6, 1.0, 1.0],
        thickness=12.0,
        is_persistent=True,
    )
    client.simPlotPoints(
        path,
        color_rgba=[1.0, 0.8, 0.0, 1.0],
        size=25.0,
        is_persistent=True,
    )


def ensure_start_altitude(client: airsim.MultirotorClient) -> None:
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    current_z = state.kinematics_estimated.position.z_val

    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
    if current_z > STARTUP_ALTITUDE:
        client.moveToZAsync(
            z=STARTUP_ALTITUDE,
            velocity=1.5,
            vehicle_name=VEHICLE_NAME,
        ).join()


def fly_path(client: airsim.MultirotorClient, path: list[airsim.Vector3r]) -> None:
    if not path:
        raise ValueError("Path is empty.")

    client.moveToPositionAsync(
        path[0].x_val,
        path[0].y_val,
        path[0].z_val,
        velocity=FLIGHT_SPEED,
        timeout_sec=WAYPOINT_TIMEOUT,
        vehicle_name=VEHICLE_NAME,
    ).join()

    client.moveOnPathAsync(
        path,
        velocity=FLIGHT_SPEED,
        timeout_sec=WAYPOINT_TIMEOUT,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
        lookahead=LOOKAHEAD,
        adaptive_lookahead=1,
        vehicle_name=VEHICLE_NAME,
    ).join()


def main() -> None:
    raw_points = load_path_points(PATH_FILE)
    dense_points = build_dense_path(raw_points)
    airsim_path = to_airsim_points(dense_points)

    print(f"Loaded {len(raw_points)} path points from: {PATH_FILE}")
    print(f"Generated {len(airsim_path)} AirSim samples")
    print(
        "Transform:",
        {
            "scale": PATH_SCALE,
            "offset": PATH_OFFSET.tolist(),
            "invert_z": INVERT_Z,
        },
    )

    client = connect_client()

    try:
        plot_path(client, airsim_path)
        ensure_start_altitude(client)
        time.sleep(0.5)
        fly_path(client, airsim_path)
        client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
    finally:
        client.armDisarm(False, vehicle_name=VEHICLE_NAME)
        client.enableApiControl(False, vehicle_name=VEHICLE_NAME)


if __name__ == "__main__":
    main()
