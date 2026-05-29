# optimalpath

`cfs-telemetry-app`에 입력할 경로를 생성하는 경로 계획 프로젝트다.

## 역할

사용자가 지정한 waypoint를 기반으로 최적 경로를 계산하고,
`cfs-telemetry-app`의 `uplink_app`으로 전달할 route update를 생성한다.

## 관련 프로젝트

| 프로젝트 | 역할 |
| --- | --- |
| `cfs-telemetry-app` | 이 프로젝트에서 생성한 경로를 수신하여 FC에 업로드 |
| `cansat_2` | 전체 시스템 통합 프로젝트 |

## 사용 알고리즘

- **PB-NBV(2)**: 최적 지점 선정
- **Greedy path planning**: 경로 생성
