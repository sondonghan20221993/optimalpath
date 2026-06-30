# PB-NBV 비교 실험 TODO (빠른 실행판)

## 목표
- Greedy vs PB-NBV를 같은 조건에서 공정 비교
- ㄷ자 물체에서 가림(occlusion)까지 반영한 결과 확보

## 현재 결론
- Greedy 거리 가중치는 이미 적용/검증됨 (alpha=1.0에서 커버리지 유지 + 이동거리 감소)
- PB-NBV 코어 로직은 유지해야 함 (타원체 투영, frontier-occupied, 0.5^rank)
- 지금 가장 큰 오차 원인은 거리 가중치보다 가림 미반영

## 우선순위 체크리스트

### P0 (필수) Python sim에 가림 추가
- [ ] `observed_by()`에 ray-cast occlusion 추가
- [ ] 기존 결과(가림 X)와 신규 결과(가림 O) 비교
- [ ] ㄷ자 오목부에서 과대관측 감소 확인

완료 기준:
- [ ] 동일 시작점/동일 예산에서 Greedy, PB-NBV 모두 실행 가능
- [ ] 커버리지/이동거리/스텝 수 표 1개 생성

### P1 (권장) 거리 가중치 공정 적용
- [ ] Python PB-NBV에 거리 항 추가 (실험 분기만, 기본식 보존)
- [ ] 식 1: `F' = F / dist^alpha`
- [ ] 식 2: `F' = F - lambda * dist`
- [ ] alpha: 0, 0.5, 1.0, 2.0 스윕

완료 기준:
- [ ] alpha=0에서 원래 PB-NBV와 동일 결과 재현
- [ ] 비교표(coverage, path_length, efficiency) 작성

### P2 (검증용) C++ PB-NBV 정상화
- [ ] `drone_runner`에 frustum culling 연결
- [ ] frontier_voxel이 0 고정인지 재확인
- [ ] Python 결과와 경향성 교차검증 1회

완료 기준:
- [ ] 점수가 `0-occupied` 퇴화에서 벗어남
- [ ] 최소 1개 시나리오에서 Python/C++ 경향 일치

## 오늘 바로 할 일 (순서 고정)
1. P0 구현
2. P0 전/후 리포트 출력
3. P1 alpha=0,1.0만 먼저 실행
4. 시간 남으면 P2 착수

## 발표용 최소 산출물
- [ ] 커버리지 곡선 1장
- [ ] 이동거리 막대그래프 1장
- [ ] waypoint 3D 시각화 1장
- [ ] 핵심 문장 3줄

핵심 문장 템플릿:
- 동일 가시성 모델에서 Greedy/PB-NBV를 공정 비교했다.
- occlusion 반영 후 ㄷ자 오목부 과대관측이 줄어 결과 신뢰도가 상승했다.
- 거리 가중치는 성능-효율 절충 파라미터이며 alpha=0에서 원식으로 복원된다.

---

## 2026-06-25 실행 기록 (3m_1 물체)

실행 입력:
- 데이터 폴더: C:/Users/sdh97/Desktop/학교/캔위성/2차 발표자료/drone_real_sfm/3m_1
- 산출물: `results/pbnbv_3m1_path.json`, `results/pbnbv_3m1_result.png`

관찰 결과:
- 경로 생성 자체는 성공 (waypoint 10개 생성)
- 하지만 PB-NBV score가 전 step에서 0
- `underobserved_ratio = 0.0` 확인

확정 원인:
1. 미관측 집합이 공집합
	- `underobs_mask = coverage < UNDEROBS_THRESH`에서 전체 False
2. FOV가 과도하게 넓게 계산됨
	- focal 평균 약 359.88, IMG_W=1920 기준 수평 FOV 약 138.9도
3. 가시성 모델이 단순 원뿔+거리 기준
	- occlusion/표면 가림 미반영으로 과관측 발생

검증 수치:
- visibility count 분포: min=20, max=55, mean=53.2 (카메라 55개)
- `<3회 관측` 포인트 수: 0개

영향:
- PB-NBV 정보이득이 0으로 고정
- 후보 점수 동점 -> 인덱스 순 정렬 의존
- 결과적으로 정보이득 기반 경로가 아니라 형식적 경로가 생성됨

## 즉시 수정 계획 (다음 실행 전)

P0. 가시성 현실화 (필수)
- [ ] `point_in_frustum`에 front-face 조건 추가
- [ ] 가능하면 간단 occlusion(ray-cast 또는 depth bin) 추가

P1. FOV 보정 (필수)
- [ ] 이미지 해상도/초점거리 일치 여부 확인 (실제 센서 기준)
- [ ] 필요 시 FOV 상한 도입 (예: 80~100도 범위 실험)

P2. 미관측 기준 동적화 (권장)
- [ ] 고정 임계값(<3) 대신 분포 기반 하위 p%로 underobserved 정의
- [ ] 예: coverage 하위 20%를 미관측으로 설정

완료 판정:
- [ ] `underobserved_ratio > 0` 확인
- [ ] PB-NBV score가 step별로 0에서 분리됨
- [ ] 동점 정렬 의존이 아닌 점수 기반 선택 확인

---

## 2026-06-25 수정/재실행 기록 (v2, v3)

### 적용한 코드 수정
1. 가시성/미관측 개선
- `point_in_frustum`에 front-face 필터 추가
- focal 기반 FOV 상한 적용 (`MAX_FOV_DEG = 95.0`)
- 고정 임계(<3) 대신 하위 비율 기반 underobserved 선택 (`UNDEROBS_FRACTION = 0.20`)

2. 카메라 45도 제약 추가
- 후보 생성 시 타겟 기준 틸트각 계산
- `MAX_TILT_DEG = 45.0` 초과 후보 제외
- 결과 JSON에 `max_tilt_deg`, waypoint별 `tilt_deg` 저장

### 재실행 결과 (3m_1)

v2 결과:
- 산출물: `results/pbnbv_3m1_path_v2.json`, `results/pbnbv_3m1_result_v2.png`
- `underobserved_ratio = 0.2`
- 후보 스코어 범위: 3918~7734
- 전 step 점수 0 문제 해소

v3 (tilt 45 적용) 결과:
- 산출물: `results/pbnbv_3m1_path_v3_tilt45.json`, `results/pbnbv_3m1_result_v3_tilt45.png`
- `max_tilt_deg = 45.0` 기록 확인
- waypoint `tilt_deg` 범위: 약 7.3° ~ 17.3° (전부 45° 이하)

### 체크리스트 상태 업데이트
- [x] `underobserved_ratio > 0` 확인
- [x] PB-NBV score가 step별로 0에서 분리됨
- [x] 동점 정렬 의존이 아닌 점수 기반 선택 확인

### 다음 실험 메모
- 45도 제약은 현재 데이터에서 여유가 큼(최대 약 17.3°)
- 다음은 30°/45°/60° 제약 비교로 경로/점수 민감도 확인

---

## 2026-06-26 real_test 데이터 실행 기록

### 알고리즘 구조 확정
**PB-NBV(2) → IG 임계값 → Greedy NN 정렬** (두 단계 명확히 분리)
- PB-NBV: live_mask 갱신으로 실제 정보이득 계산 (시점 선택)
- Greedy NN: 선택된 시점의 방문 순서 결정

### 실행 결과 (real_test_pts_normals.npz, 2438점)
- 전체 미관측: 487개 (하위 20%)
- **결과: 3개 waypoint** (IG > 5 기준)
  - WP01: IG=342 (az=51°, tilt=15.6°, z=-1.36m)
  - WP02: IG=95 (az=196°, tilt=26.4°, z=-2.60m)
  - WP03: IG=12 (az=223°, tilt=14.7°, z=-1.36m)
  - 커버: 487개 중 449개(92%)

### 문제: 남은 38개 포인트
위치: 거의 지면 (Z ≈ -0.02m, 타겟 Z=-0.13m)
현재 코드에서 front-face 필터가 이 포인트들을 모두 탈락시킴
- 법선이 지면 방향(+Z)
- 카메라가 위에 있으니 법선 · view_vec < 0 → 뒷면으로 판정

### 가시성 모델의 두 극단
| 설정 | 결과 | 문제 |
|------|------|------|
| `pts_normals=nrm` (front-face O) | 3개 waypoint, 38개 영구 미관측 | 바닥 법선 때문에 과소 추정 |
| `pts_normals=None` (법선 X) | 1개 waypoint, IG=487 (전부 커버) | 가림 없이 과대 추정 |

### P0 완료: Ray-cast occlusion 적용 ✓
- [x] 방향 bin 단위 occlusion 체크 (36개 azimuth, 5개 elevation bin)
- [x] 각 bin에서 가장 가까운 포인트만 visible로 처리
- [x] pbnbv_path.py `RAYCAST_OCCLUSION` 플래그로 런타임 제어 가능

### 결과 비교 (real_test 데이터)
| 가시성 모델 | Waypoint | 최대IG | 커버 | 특징 |
|-----------|----------|--------|------|------|
| 법선 O (front-face) | 3개 | 342 | 449/487 | 바닥면 탈락(과소) |
| 법선 X (무제약) | 1개 | 487 | 487/487 | 가림 무시(과대) |
| **Ray-cast** | **4개** | **225** | **415/487** | **균형잡힘** |

Ray-cast가 두 극단의 중간 결과를 제공: 4개 waypoint로 85% 커버 (38개만 미관측)
- WP01: IG=225 (2차 궤도, 낮은 고도)
- WP02: IG=129
- WP03: IG=61
- WP04: IG=25

---

## 2026-06-26 알고리즘 완성 기록 (최종 구현)

### 구현 완료 항목

#### 1. tilt 보정 (`project_to_tilt_boundary`)
- 기존: tilt > 45° 후보 **제외**
- 변경: tilt > 45° 후보를 수평 이동 → tilt=45° 경계로 **보정** 후 포함
- 위치: `pbnbv_path.py` `project_to_tilt_boundary()` 함수
- 논문 기술 "제약 범위를 벗어난 후보는 허용 범위에 가깝도록 보정"과 일치

#### 2. Salvage Pass (`salvage_coverage`)
- 위치: `pbnbv_path.py` `salvage_coverage()` 함수
- 동작: IG_THRESH 컷오프 이후 남은 미관측 포인트 중 어떤 후보라도 볼 수 있으면 IG 무시하고 추가
- 종료 조건: 어떤 후보도 남은 포인트를 볼 수 없음 → 물리적 한계로 판단하고 종료
- 의미: "못 본 부분을 보는 waypoint는 무조건 살린다"

#### 3. No-Camera 시나리오 (`run_pbnbv_nocam.py`)
- 기존 카메라 pose 없이 포인트클라우드만으로 전체 커버리지 경로 계획
- underobs_mask = 전체 True (모든 포인트를 미관측으로 간주)
- 오브젝트 centroid 기준 다중 고도·반경 후보 생성

### No-Camera 실행 결과 (real_test_pts_normals.npz, 2438점)

| 단계 | WP 수 | 커버 | 비고 |
|------|-------|------|------|
| PB-NBV(2) 선택 (IG > 5) | 7개 | 2433/2438 (99.8%) | |
| Salvage 추가 | +3개 | 2438/2438 (100.0%) | IG=3,1,1 |
| **최종** | **10개** | **100%** | |

WP별 PB-NBV 스코어:
- WP01: 2006 (첫 시점이 전체의 72% 커버)
- WP02: 534
- WP03: 114
- WP04: 57
- WP05: 26
- WP06: 13
- WP07: 9 → IG_THRESH=5 컷오프
- SV01~03: IG=3,1,1 (salvage)

### IG 유의미성 기준 (2438점 기준)

| raw-IG 범위 | 전체 대비 | 판단 |
|------------|----------|------|
| ≥ 10 | ≥ 0.4% | 명확히 유의미 |
| 5 ~ 10 | 0.2~0.4% | 경계 (현재 컷오프=5 합리적) |
| < 5 | < 0.2% | Salvage 영역 — 유일성 기준으로만 판단 |

### 최종 알고리즘 파이프라인

```
① 포인트클라우드 로드
② underobs_mask 설정 (재관측: 하위 20% / no-camera: 전체)
③ 후보 생성 + tilt 보정 (project_to_tilt_boundary)
④ PB-NBV(2) 선택 루프 (IG_THRESH 컷오프)
⑤ Salvage Pass (남은 커버 가능 포인트 무조건 추가)
⑥ Greedy NN 정렬
```

### 파일 목록

| 파일 | 설명 |
|------|------|
| `src/pbnbv_path.py` | 핵심 알고리즘 (project_to_tilt_boundary, salvage_coverage 포함) |
| `src/run_pbnbv_realtest.py` | 재관측 시나리오 (기존 카메라 34개 활용) |
| `src/run_pbnbv_nocam.py` | No-camera 시나리오 (전체 커버리지) |
| `src/viz_nocam.py` | No-camera 결과 시각화 |
| `results/pbnbv_nocam_path.json` | No-camera 경로 결과 |
| `results/pbnbv_nocam_viz.png` | No-camera 시각화 이미지 |
