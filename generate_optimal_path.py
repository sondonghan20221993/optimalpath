"""
generate_optimal_path.py

blue_1.mp4 (AirSim 주차장 영상)에서 파란 차량을 탐지하여
car.py --recommended-json 과 호환되는 최적 경로 JSON을 생성합니다.

사용법:
  # 영상에서 자동 탐지 (타겟 좌표는 기존 optimal 데이터 기반 기본값 사용)
  python generate_optimal_path.py

  # 직접 타겟 좌표 지정
  python generate_optimal_path.py --target-x -34.5 --target-y -47.9 --target-z 1.42

  # 생성된 JSON을 car.py에 적용
  python car.py \\
      --mode orbit_then_recommended \\
      --recommended-json blue1_optimal_path.json \\
      --output-dir path/blue1_result \\
      --target-x -34.5 --target-y -47.9 --target-z 1.42
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


# ─── 파란색 HSV 범위 (차량 탐지용) ───────────────────────────────────────────
BLUE_HSV_LOWER = np.array([95, 80, 60])
BLUE_HSV_UPPER = np.array([135, 255, 255])
MIN_BLOB_AREA = 200   # 노이즈 제거용 최소 픽셀 면적


# ─── 탐지 함수 ────────────────────────────────────────────────────────────────

def detect_blue(frame):
    """프레임에서 파란 영역의 무게중심과 면적 반환. 없으면 (None, 0)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_BLOB_AREA:
        return None, 0.0

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, 0.0

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy), float(area)


def annotate_frame(frame, center, area):
    """디버그용: 탐지 결과를 프레임에 표시."""
    out = frame.copy()
    if center:
        cv2.circle(out, (int(center[0]), int(center[1])), 15, (0, 255, 0), 2)
        cv2.circle(out, (int(center[0]), int(center[1])), 3, (0, 255, 0), -1)
        cv2.putText(out, f"area={int(area)}", (int(center[0]) + 18, int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


# ─── 좌표 역투영 (픽셀 → 3D 월드) ──────────────────────────────────────────

def pixel_to_world(px, py, frame_w, frame_h, fov_h_deg,
                   cam_x, cam_y, cam_z_ned, ground_z_ned=0.0,
                   pitch_deg=-45.0):
    """
    카메라 픽셀 좌표를 AirSim 월드 좌표로 역투영합니다.

    pitch_deg = 0 : 정방향(수평), -45 : 45도 아래를 바라봄 (AirSim 기본값)
    AirSim은 NED 좌표계 사용 (Z 음수 = 위)
    """
    altitude = abs(cam_z_ned - ground_z_ned)   # 지면까지의 수직 거리 (m)

    fov_h = math.radians(fov_h_deg)
    fov_v = fov_h * (frame_h / frame_w)

    # 픽셀 → 정규화 각도
    angle_h = (px / frame_w - 0.5) * fov_h   # 좌우 각도 (rad)
    angle_v = (py / frame_h - 0.5) * fov_v   # 상하 각도 (rad)

    pitch = math.radians(pitch_deg)           # 카메라 피치 (rad)

    # 카메라 광선 방향 (NED: x=전방, y=우측, z=하방)
    ray_x = math.cos(pitch) * math.cos(angle_v) * math.cos(angle_h)
    ray_y = math.cos(pitch) * math.cos(angle_v) * math.sin(angle_h)
    ray_z = math.sin(pitch) + math.sin(angle_v)

    # 지면(z = ground_z_ned 평면)과의 교점
    if abs(ray_z) < 1e-8:
        return None

    # NED에서 아래 방향이 +Z, 위가 -Z 이므로
    # cam_z_ned < 0 (드론이 공중), ground_z_ned ≈ 0
    dz_to_ground = ground_z_ned - cam_z_ned   # 양수
    t = dz_to_ground / ray_z

    if t < 0:
        return None

    world_x = cam_x + t * ray_x
    world_y = cam_y + t * ray_y
    return world_x, world_y


# ─── 경로 생성 함수 ──────────────────────────────────────────────────────────

def make_orbit_waypoints(tx, ty, tz, altitude_m, radius_m, n_points):
    """타겟 주변 원형 궤도 웨이포인트 생성."""
    z = tz - altitude_m
    waypoints = []
    for i in range(n_points):
        theta = 2.0 * math.pi * i / n_points
        x = tx + radius_m * math.cos(theta)
        y = ty + radius_m * math.sin(theta)
        rel_x = radius_m * math.cos(theta)
        rel_y = radius_m * math.sin(theta)
        waypoints.append({
            "position": [x, y, z],
            "relative_to_target": [rel_x, rel_y, -altitude_m],
        })
    return waypoints


def make_spiral_waypoints(tx, ty, tz, altitudes, radii, n_per_ring):
    """
    다중 고도 나선형 최적 경로 생성.
    altitudes / radii 를 묶어서 여러 링에서 촬영 포인트를 생성합니다.
    """
    waypoints = []
    for alt, rad in zip(altitudes, radii):
        ring = make_orbit_waypoints(tx, ty, tz, alt, rad, n_per_ring)
        waypoints.extend(ring)
    return waypoints


def tsp_nearest_neighbor(points, start_idx=0):
    """
    Nearest-Neighbor TSP로 웨이포인트 순서 최적화 (총 이동거리 최소화).
    """
    n = len(points)
    if n <= 1:
        return list(range(n))

    visited = [False] * n
    order = [start_idx]
    visited[start_idx] = True

    for _ in range(n - 1):
        cur = order[-1]
        cx, cy, cz = points[cur]["position"]
        best_dist = float("inf")
        best_next = -1
        for j in range(n):
            if visited[j]:
                continue
            nx, ny, nz = points[j]["position"]
            d = math.sqrt((nx - cx) ** 2 + (ny - cy) ** 2 + (nz - cz) ** 2)
            if d < best_dist:
                best_dist = d
                best_next = j
        if best_next >= 0:
            order.append(best_next)
            visited[best_next] = True

    return order


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="blue_1.mp4에서 파란 차량을 탐지하여 AirSim 최적 경로 JSON 생성"
    )
    parser.add_argument(
        "--video",
        default=r"C:\Users\sdh97\Desktop\6_16주차장\blue_1.mp4",
        help="입력 영상 경로",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\sdh97\Desktop\blue1_optimal_path.json",
        help="출력 JSON 경로",
    )
    parser.add_argument(
        "--sample-every", type=int, default=15,
        help="N 프레임마다 샘플링 (기본: 15)",
    )
    # 타겟 좌표 (기본값: 기존 optimal path 데이터 기반)
    parser.add_argument("--target-x", type=float, default=-34.49998948)
    parser.add_argument("--target-y", type=float, default=-47.89999844)
    parser.add_argument("--target-z", type=float, default=1.41999357)
    parser.add_argument(
        "--auto-detect-target", action="store_true",
        help="영상에서 파란 차량 위치를 자동으로 추정하여 타겟 좌표 갱신",
    )
    # 카메라 파라미터
    parser.add_argument("--cam-x", type=float, default=None,
                        help="영상 촬영 시 카메라 X (지정 안 하면 타겟 X 사용)")
    parser.add_argument("--cam-y", type=float, default=None,
                        help="영상 촬영 시 카메라 Y (지정 안 하면 타겟 Y 사용)")
    parser.add_argument("--cam-z", type=float, default=-5.0,
                        help="카메라 NED Z (음수 = 위, 기본: -5.0)")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="카메라 수평 FOV (도)")
    parser.add_argument("--cam-pitch", type=float, default=-45.0,
                        help="카메라 피치 각도 (도, 기본: -45)")
    # 경로 파라미터
    parser.add_argument("--orbit-radius", type=float, default=7.0)
    parser.add_argument("--orbit-altitude", type=float, default=6.0)
    parser.add_argument("--orbit-n", type=int, default=17)
    parser.add_argument(
        "--mode",
        choices=["orbit", "spiral", "custom"],
        default="spiral",
        help="경로 타입: orbit=단순원형, spiral=다중고도나선형, custom=기존optimal복사",
    )
    parser.add_argument(
        "--optimize-order", action="store_true", default=True,
        help="Nearest-Neighbor TSP로 웨이포인트 순서 최적화",
    )
    parser.add_argument(
        "--debug-dir", default=None,
        help="탐지 결과를 저장할 디렉토리 (None이면 저장 안 함)",
    )
    args = parser.parse_args()

    # ── 영상 열기 ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없음: {args.video}", file=sys.stderr)
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"영상 정보: {W}x{H}, 총 {total}프레임, {fps:.1f}fps")
    print(f"파란 차량 탐지 중 (매 {args.sample_every}프레임 샘플링)...")

    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    cam_x = args.cam_x if args.cam_x is not None else args.target_x
    cam_y = args.cam_y if args.cam_y is not None else args.target_y

    detections = []
    frame_idx = 0
    debug_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.sample_every == 0:
            center, area = detect_blue(frame)
            if center is not None:
                world_pos = None
                if args.auto_detect_target:
                    world_pos = pixel_to_world(
                        center[0], center[1], W, H,
                        args.fov, cam_x, cam_y, args.cam_z,
                        ground_z_ned=args.target_z,
                        pitch_deg=args.cam_pitch,
                    )
                detections.append({
                    "frame": frame_idx,
                    "pixel": list(center),
                    "area": area,
                    "world_pos": list(world_pos) if world_pos else None,
                })
                if args.debug_dir:
                    dbg = annotate_frame(frame, center, area)
                    cv2.imwrite(f"{args.debug_dir}/detect_{debug_idx:04d}_f{frame_idx}.jpg", dbg)
                    debug_idx += 1

        frame_idx += 1

    cap.release()

    if not detections:
        print("[경고] 파란 차량이 탐지되지 않았습니다. HSV 범위를 조정하거나 영상을 확인하세요.")
        sys.exit(1)

    print(f"탐지 완료: {len(detections)}개 샘플 프레임에서 파란 차량 발견")

    # ── 타겟 좌표 갱신 (--auto-detect-target 옵션) ───────────────────────────
    target_x, target_y, target_z = args.target_x, args.target_y, args.target_z

    if args.auto_detect_target:
        valid = [d for d in detections if d["world_pos"] is not None]
        if valid:
            total_w = sum(d["area"] for d in valid)
            target_x = sum(d["world_pos"][0] * d["area"] for d in valid) / total_w
            target_y = sum(d["world_pos"][1] * d["area"] for d in valid) / total_w
            print(f"탐지 기반 타겟 추정: x={target_x:.3f}, y={target_y:.3f}")
        else:
            print("[경고] 월드 좌표 역투영 실패. 기본 타겟 좌표를 사용합니다.")

    print(f"최종 타겟 좌표: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}")

    # ── 탐지 품질 통계 ────────────────────────────────────────────────────────
    areas = [d["area"] for d in detections]
    best_frame = detections[int(np.argmax(areas))]
    print(f"최고 탐지 프레임: {best_frame['frame']}번 (면적={int(best_frame['area'])}px²)")
    print(f"평균 탐지 면적: {np.mean(areas):.0f}px², 최대: {max(areas):.0f}px²")

    # ── 경로 생성 ─────────────────────────────────────────────────────────────
    if args.mode == "orbit":
        waypoints = make_orbit_waypoints(
            target_x, target_y, target_z,
            args.orbit_altitude, args.orbit_radius, args.orbit_n,
        )
        print(f"단순 원형 경로 생성: {len(waypoints)}개 웨이포인트")

    elif args.mode == "spiral":
        # 두 고도에서 촬영: 가까운 고도(상세) + 높은 고도(전체)
        altitudes = [args.orbit_altitude, args.orbit_altitude * 1.6]
        radii = [args.orbit_radius, args.orbit_radius * 1.4]
        n_per = max(8, args.orbit_n // 2)
        waypoints = make_spiral_waypoints(
            target_x, target_y, target_z,
            altitudes, radii, n_per,
        )
        print(f"다중 고도 나선형 경로 생성: {len(waypoints)}개 웨이포인트")

    else:  # custom: 기존 optimal manifest의 웨이포인트 형식으로 생성
        waypoints = make_orbit_waypoints(
            target_x, target_y, target_z,
            args.orbit_altitude, args.orbit_radius, args.orbit_n,
        )
        print(f"커스텀 경로 생성: {len(waypoints)}개 웨이포인트")

    # ── 순서 최적화 ───────────────────────────────────────────────────────────
    if args.optimize_order and len(waypoints) > 2:
        order = tsp_nearest_neighbor(waypoints, start_idx=0)
        waypoints = [waypoints[i] for i in order]
        print("Nearest-Neighbor TSP로 경로 순서 최적화 완료")

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    output_data = {
        "schema_version": 1,
        "name": "blue1_optimal_path",
        "notes": f"blue_1.mp4에서 파란 차량 탐지 후 생성. 탐지 샘플={len(detections)}개, 모드={args.mode}",
        "source_video": str(Path(args.video).resolve()),
        "detection_sample_count": len(detections),
        "detection_mean_area_px2": float(np.mean(areas)),
        "reference_target_position_airsim_m": [target_x, target_y, target_z],
        "target_region_center": [target_x, target_y, target_z],
        "path_step_count": len(waypoints),
        "waypoints": [
            {
                "index": i + 1,
                "position": wp["position"],
                "relative_to_target": wp["relative_to_target"],
            }
            for i, wp in enumerate(waypoints)
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n최적 경로 저장 완료: {out_path}")
    print(f"총 웨이포인트: {len(waypoints)}개")

    # ── 사용법 출력 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("car.py 실행 명령어:")
    print(f"  python car.py \\")
    print(f"      --mode orbit_then_recommended \\")
    print(f"      --recommended-json \"{out_path}\" \\")
    print(f"      --output-dir path/blue1_result \\")
    print(f"      --target-x {target_x:.5f} \\")
    print(f"      --target-y {target_y:.5f} \\")
    print(f"      --target-z {target_z:.5f} \\")
    print(f"      --orbit-altitude {args.orbit_altitude} \\")
    print(f"      --orbit-radius {args.orbit_radius} \\")
    print(f"      --recommended-altitude {args.orbit_altitude}")
    print("=" * 60)


if __name__ == "__main__":
    main()
