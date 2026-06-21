# optimalpath

`cfs-telemetry-app`에 입력할 경로를 생성하는 경로 계획 프로젝트.

> **실험 데이터**: `real_test/` 폴더의 AirSim 실비행 데이터셋을 기준으로 진행한다.
> RGB 이미지 34장 + 프레임별 카메라·GPS·IMU 메타데이터 포함 (`Downloads/real_test.zip` 압축 해제본).

## 개요

사용자가 지정한 waypoint 또는 영상/포인트클라우드 분석 결과를 바탕으로 최적 경로를 계산하고,
`cfs-telemetry-app`의 `uplink_app`으로 전달할 route update를 생성한다.

## 관련 프로젝트

| 프로젝트 | 역할 |
|---|---|
| `cfs-telemetry-app` | 이 프로젝트에서 생성한 경로를 수신하여 FC에 업로드 |
| `cansat_2` | 전체 시스템 통합 프로젝트 |

---

## 스크립트

### 1. `pbnbv_path.py` — PB-NBV(2) + Greedy Path Planning (메인 알고리즘)

MASt3R SfM 결과(포인트클라우드 + 카메라 포즈)를 입력받아 미관측 영역을 탐색하고 최적 시점을 선정한다.

**파이프라인**

```
poses.npy / focals.npy / pointcloud.ply
        │
        ▼
[1] MASt3R 데이터 로드   → 카메라 위치·방향, FOV, 타겟 추정
        │
        ▼
[2] Coverage 추정        → 각 포인트의 기존 카메라 관측 횟수 계산
        │
        ▼
[3] 후보 시점 생성       → 타겟 주변 다중 고도·반경 구면 격자 (기본 150개)
        │
        ▼
[4] PB-NBV(2) 스코어링  → 1-step IG + 0.5 × lookahead 최대 IG
        │
        ▼
[5] Greedy Path Planning → Nearest-Neighbor로 방문 순서 결정
        │
        ▼
pbnbv_path.json + pbnbv_result.png
```

**주요 파라미터**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `N_PCD_SAMPLE` | 40,000 | 포인트클라우드 서브샘플 수 |
| `MAX_DIST` | 8.0 m | 최대 가시 거리 |
| `UNDEROBS_THRESH` | 3 | 미관측 판정 기준 (관측 횟수) |
| `N_CANDIDATES` | 150 | 후보 시점 수 |
| `N_SELECT` | 10 | 최종 선택 웨이포인트 수 |
| `LOOKAHEAD` | 2 | PB-NBV lookahead 단계 |
| `ORBIT_ALTITUDES` | [0.5, 1.0, 1.5] m | 후보 고도 오프셋 |
| `ORBIT_RADII` | [2.0, 3.5, 5.0] m | 후보 수평 반경 |

**입력 경로 (스크립트 상단에서 수정)**

```python
BASE_DIR    = r"C:\...\blue_1_fhd_sfm(pp팍스 mast3r결과)"
PLY_PATH    = BASE_DIR + r"\pointcloud.ply"
POSES_PATH  = BASE_DIR + r"\poses.npy"
FOCALS_PATH = BASE_DIR + r"\focals.npy"
OUT_JSON    = r"...\pbnbv_path.json"
OUT_IMG     = r"...\pbnbv_result.png"
```

**실행**

```bash
python pbnbv_path.py
```

---

### 2. `generate_optimal_path.py` — 영상 기반 경로 생성

AirSim 녹화 영상에서 파란 차량을 HSV 탐지하여 타겟 좌표를 추정하고,
나선형(다중 고도) 또는 원형 궤도 경로를 생성한다.

**파이프라인**

```
blue_1.mp4
    │
    ▼
HSV 탐지 (파란색 blob)    → 프레임별 무게중심·면적 계산
    │
    ▼
픽셀 → 3D 역투영 (선택)   → --auto-detect-target 옵션 사용 시
    │
    ▼
경로 생성
  orbit  : 단순 원형 웨이포인트
  spiral : 2-ring 다중 고도 나선형 (기본)
  custom : orbit과 동일
    │
    ▼
TSP Nearest-Neighbor 순서 최적화
    │
    ▼
blue1_optimal_path.json
```

**실행 예시**

```bash
# 기본 (영상 자동 탐지, 기본 타겟 좌표 사용)
python generate_optimal_path.py

# 타겟 좌표 직접 지정
python generate_optimal_path.py --target-x -34.5 --target-y -47.9 --target-z 1.42

# 영상에서 타겟 자동 추정
python generate_optimal_path.py --auto-detect-target

# car.py 연계 실행
python car.py \
    --mode orbit_then_recommended \
    --recommended-json blue1_optimal_path.json \
    --output-dir path/blue1_result \
    --target-x -34.5 --target-y -47.9 --target-z 1.42
```

**주요 옵션**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--video` | `blue_1.mp4` | 입력 영상 |
| `--mode` | `spiral` | `orbit` / `spiral` / `custom` |
| `--orbit-radius` | 7.0 m | 궤도 반경 |
| `--orbit-altitude` | 6.0 m | 비행 고도 |
| `--orbit-n` | 17 | 웨이포인트 수 |
| `--auto-detect-target` | off | 영상에서 타겟 좌표 자동 추정 |
| `--sample-every` | 15 | N프레임마다 탐지 샘플링 |

---

### 3. `visualize_optimal_path.py` — 결과 시각화

`generate_optimal_path.py`의 출력 JSON과 기존 경로 manifest를 비교 시각화한다.

**출력 패널 구성 (`blue1_result_visual.png`)**

| 패널 | 내용 |
|---|---|
| 최고 탐지 프레임 | 파란 차량 탐지 면적이 가장 큰 프레임 |
| HSV 마스크 | 파란색 필터링 결과 |
| 탐지 면적 타임라인 | 프레임별 탐지 면적 추이 |
| 2D 평면도 | 기존 경로 vs 신규 경로 Top-Down 비교 |
| 3D 경로 | 기존 경로 vs 신규 경로 3D 비교 |

**실행**

```bash
python visualize_optimal_path.py
```

---

## 출력 파일

| 파일 | 생성 스크립트 | 설명 |
|---|---|---|
| `blue1_optimal_path.json` | `generate_optimal_path.py` | 영상 탐지 기반 경로 (16 WP, spiral) |
| `blue1_orbit_path.json` | `generate_optimal_path.py` | 단순 원형 궤도 경로 |
| `pbnbv_path.json` | `pbnbv_path.py` | PB-NBV(2) 경로 (10 WP) |
| `blue1_result_visual.png` | `visualize_optimal_path.py` | 경로 비교 시각화 |
| `pbnbv_result.png` | `pbnbv_path.py` | PB-NBV 분석 결과 시각화 |

## JSON 스키마 (공통)

```json
{
  "schema_version": 1,
  "name": "...",
  "path_step_count": 10,
  "waypoints": [
    {
      "index": 1,
      "position": [x, y, z],
      "relative_to_target": [dx, dy, dz]
    }
  ]
}
```

- `position`: AirSim NED 좌표 (m). Z 음수 = 위
- `relative_to_target`: 타겟 기준 상대 좌표

## 사용 알고리즘

| 알고리즘 | 사용처 | 역할 |
|---|---|---|
| PB-NBV(2) | `pbnbv_path.py` | 2-step lookahead로 정보 이득 최대 시점 선정 |
| Greedy (Nearest-Neighbor) | `pbnbv_path.py` | 선정된 시점의 방문 순서 최적화 |
| HSV Blob Detection | `generate_optimal_path.py` | 영상에서 파란 차량 위치 탐지 |
| TSP Nearest-Neighbor | `generate_optimal_path.py` | 웨이포인트 순서 최적화 |
| 픽셀→3D 역투영 | `generate_optimal_path.py` | 탐지 픽셀 좌표를 AirSim 월드 좌표로 변환 |

## 의존성

```
opencv-python
numpy
open3d          # pbnbv_path.py
matplotlib
```
