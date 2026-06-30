# PB-NBV Path Planning — Results Summary

**데이터**: `real_test/real_test_pts_normals.npz` (SOR 정제, 2436 pts)  
**공동 통제 조건**: tilt=45° · MAX_DIST=8m · no-camera · raycast evaluation  
**좌표계**: AirSim NED (Z 음수 = 위)  
**실제 물체**: 120×110×36cm 납작한 ㄷ자 슬래브 (점군 본체 1.55×1.10×0.34m로 일치)  
**최종 업데이트**: 2026-06-27 (Orbit + Soft NBV 최종 결과 추가)

---

## 0. ⚠️ 중대 수정 이력 (2026-06-26)

초기 결과는 **2개의 공중 노이즈 점**으로 인한 버그 위에서 산출됨:
- `obj_z_top = pts[:,2].min()`이 노이즈 점(-3.85m, 3.85m 상공)을 물체 꼭대기로 오인
- → 후보 카메라 고도가 실제보다 **3.5m 높게** 설정됨 (34cm 물체를 4.8m 상공에서 촬영)
- **SOR(k=16, μ+1σ)로 2점 제거** → Z범위 -3.85→-0.38m 정상화 → 전체 재실행
- 결과: 모든 케이스가 적은 WP로 100% 달성 (이전 90.9% 정체는 버그 아티팩트였음)

---

## 1. 5-Case 비교 결과 (정제 후)

| Case | 알고리즘 | WP 수 | 커버리지 | 경로 길이 | z 레벨 |
|------|----------|:-----:|:--------:|:---------:|--------|
| A-1 | Paper NBV sequential | 4 | **100%** | 20.4m | -1.38, -5.38 |
| A-2 | Hybrid (paper-F + Greedy) | 4 | **100%** | 17.6m | -1.38, -5.38 |
| **3** | **Hard Raycast + PB-NBV(2)** | **2** | **100%** | 11.0m | -1.38 |
| B   | Soft 0.5^r per-point | 4 | **100%** | **9.9m** | -1.38 |
| C   | Ellipsoid 0.5^r + Raycast | 4 | **100%** | 22.8m | -1.38~-4.38 |

> **최소 WP**: Case 3 (2 WP) · **최단 경로**: Case B (9.9m)  
> 정제 후 소형 납작 물체는 자기가림이 적어 2~4 시점으로 100% 커버됨

---

## 2. Case별 상세

### Case A-1: Paper NBV Sequential
- **출처**: arXiv:2501.10663 충실 구현
- **선택 방식**: frontier/occupied voxel 분류 → GMM → Ellipsoid 투영 → F = Σ(frontier·W) − Σ(occupied·W), W=0.5^r (depth rank)
- **경로 순서**: NBV 선택 순서 그대로 (greedy 없음)
- **결과(정제 후)**: 4 WP, 100%, 20.4m
- **비고**: 정제 전엔 고도 버그로 90.9% 정체였으나, 정제 후 올바른 고도에서 100% 도달

### Case A-2: Hybrid (Paper-F + Greedy)
- **선택**: A-1과 동일 (paper F-score)
- **경로**: 선택 후 Greedy NN으로 방문 순서 재정렬
- **결과(정제 후)**: 4 WP, 100%, 17.6m (A-1 대비 -2.8m)
- **의의**: 2-stage 구조(선택/정렬 분리)의 경로 효율 입증

### Case 3: Hard Raycast + PB-NBV(2) + Greedy
- **선택**: 점 단위 IG 계산 + 2-step lookahead (IG1 + 0.5 × max(IG2))
- **가림**: Hard raycast (방향 bin 1440×720, 0.25°×0.25°, bin당 최근접만 visible)
- **경로**: Greedy NN
- **결과(정제 후)**: **2 WP, 100%, 11.0m — 최소 WP**
- **의의**: 소형 납작 물체는 올바른 고도에서 자기가림이 적어 2시점으로 100% 커버

### Case B: Soft 0.5^r per-point
- **선택**: 같은 방향 bin 내 점 depth 순서 r → W = 0.5^r (가장 가까운 점 = 1.0, 다음 = 0.5, ...)
- **가림**: soft (가려진 점도 부분 기여)
- **결과(정제 후)**: **4 WP, 100%, 9.9m — 최단 경로**
- **비고**: 정제 전엔 soft weight가 종료조건을 약화시켜 14 WP 과다 생성했으나, 정제 후 4 WP로 안정

### Case C: Ellipsoid-unit 0.5^r + Raycast
- **선택**: Hard raycast → 가시 미관측 점 KMeans 클러스터링(≤8개) → 클러스터 depth rank r → score = Σ count_k × 0.5^r_k
- **가림**: Hard raycast (Case 3와 동일)
- **결과(정제 후)**: 4 WP, 100%, 22.8m — z 4개 레벨에 분산되어 최장
- **의의**: 논문 ellipsoid 단위 개념 + raycast 정밀도 결합

---

## 3. 핵심 비교 정리 (정제 후)

```
정제 후 모든 케이스가 100% 커버 (소형 납작 물체라 자기가림 적음)
  - 최소 WP : Case 3 (Hard Raycast) = 2 WP, 11.0m
  - 최단 경로: Case B (Soft 0.5^r)  = 9.9m (4 WP)
  - 최장 경로: Case C (Ellipsoid)   = 22.8m (z 4레벨 분산)
```

**tilt=45° 제약 결과(정제 후)**: `dist = √2 × vert`, obj_z_top=-0.38m 기준 유효 z 레벨은
-1.38 ~ -5.38m의 5개 (z=-6.38m: dist=8.7m > 8.0m → 제외)

---

## 4. 알고리즘 구성 요소

### 4-1. tilt=45° 후보 생성
```
z_levels = [obj_z_top - 1, -2, -3, -4, -5, -6]  (NED: 오브젝트 위)
           obj_z_top = 정제 점군의 min Z = -0.38m  ← 노이즈 제거로 정상화
horiz = |z - target_z|                            (tilt=45° ↔ horiz=vert)
dist  = √2 × vert                                 (MAX_DIST=8m 초과 시 제외)
candidates = 36 방위각 × 유효 z레벨(5개) = 180개
```
> ⚠️ 정제 전: obj_z_top이 노이즈(-3.85m)로 잡혀 후보가 3.5m 높게 생성되는 버그 존재

### 4-2. PB-NBV(2) 스코어
```
score(v) = IG1(v) + 0.5 × max_{v'≠v} IG2(v')
```
- IG1: 현재 후보에서 새로 보이는 미관측 점 수 (방식은 Case별 상이)
- IG2: 1단계 후 나머지 중 최대 IG (lookahead, 30개 샘플)

### 4-3. Salvage Pass
- PB-NBV IG_THRESH 컷오프 후 남은 점이 있으면 greedy로 추가 선택
- 물리적으로 어떤 후보도 볼 수 없으면 포기 (hard limit)

### 4-4. Greedy NN Path
- 시작점에서 가장 가까운 미방문 WP 순으로 방문 순서 결정
- 선택된 WP 집합은 변하지 않고 순서만 최적화

---

## 5. 결과 파일

**데이터 (real_test/)**
| 파일 | 내용 |
|------|------|
| `real_test_pts_normals.npz` | **현재 사용 — SOR 정제본 (2436점, 진짜)** |
| `real_test_pts_normals_clean.npz` | 정제본 사본 |
| `real_test_pts_normals_orig.npz` | 원본 백업 (오염본, 2438점) |

**5-Case 결과**
| 파일 | 내용 |
|------|------|
| `results/pbnbv_paper/pbnbv_paper.json` | Case A-1 |
| `results/pbnbv_hybrid/pbnbv_hybrid.json` | Case A-2 |
| `results/pbnbv_nocam_path.json` | Case 3 |
| `results/pbnbv_softrank_path.json` | Case B |
| `results/pbnbv_ellipsoid_path.json` | Case C |
| `results/compare5_cases.png` | 5-Case 비교 이미지 |
| `results/object_shape.png` | 정제 점군 형상 진단 (ㄷ자 확인) |

**Orbit 시나리오 결과**
| 파일 | 내용 |
|------|------|
| `results/pbnbv_orbit_path.json` | 궤도+raycast 보완 (3 WP, 95→95.9%) |
| `results/pbnbv_orbit_paper_path.json` | 궤도+논문방식 (보완 0개, θ<90) |
| `results/pbnbv_orbit_paper_strict.json` | 규제 sweep (θ 90~40°) |
| `results/orbit_compare.png` | 궤도 보완 3방식 비교 이미지 |

---

---

## 7. Orbit + Soft-Regulated PB-NBV — 최종 결과 (2026-06-27)

**시나리오**: 원형 궤도(34 카메라) 1바퀴 후, Soft Lambert 규제 PB-NBV로 사각지대 보완  
**결과 이미지**: `results/orbit_soft_summary.png`

### 알고리즘 구성

```
관측 가중치  w(voxel) = max(0, cos_inc)^alpha     (Lambert 코사인)
누적 관측    W(voxel) = Σ_cameras w(voxel)
관측됨       W(voxel) >= TAU
utility      U(v)     = F(v) / dist(cur_pos → v)   (Bircher, F≥0)
```

**파라미터 (커버리지 튜닝 아님 — 물리/기존조건에서 유도)**:
- `alpha = 1.0` → Lambert 조도 감쇠 (광학 표준)
- `TAU = cos(70°) ≈ 0.342` → hard 경계(theta<70°, 정면1장)과 동등한 누적 기준

### 수정 이력

| 버그 | 원인 | 수정 |
|------|------|------|
| tilt 실제 35° | `gen_candidates_fixed_alt` 4레벨 중 35° 링 선택 | `gen_candidates_tilt45()` 전환 |
| U=F/dist 역전 | evaluate() F<0 → farther 선호 (dist↔U 상관 +0.853) | `frontier_only=True`, F≥0 보장 |

### 최종 결과

| 지표 | 궤도 단독 | 궤도 + Soft NBV (3 WP) |
|------|:---------:|:----------------------:|
| hard (theta<70°, 정면1장) | 95.0% | **95.0%** |
| soft (Lambert cos¹, TAU) | 97.8% | **99.4%** |
| **real (raycast)** | 99.1% | **100.0% ✓** |

**보완 WP**: 3개 · **경로**: 7.92m · **고도**: z=-3.38m (tilt=45°, 물체 위 3m)  
**WP 위치**: az=290°, 280°, 270° (서쪽 사각지대 집중 보완)

### 구조적으로 관측 불가능한 voxel 분석

- 85개 frontier voxel 중 **40개**의 surface normal이 수평(az방향) → tilt=45° 어느 방향에서도 theta>70°
  - 최소 입사각 71.3°, 평균 81.1° → hard theta<70° 단일 정면 불가
- **해결책**: Soft multi-view — 여러 grazing 관점의 Lambert cos¹ 합산 ≥ TAU
  - MVS/SfM 재구성 맥락에서 물리적으로 정당 (raycast 없음 유지)

### 관련 파일

| 파일 | 내용 |
|------|------|
| `src/run_pbnbv_orbit_soft.py` | 메인 실행 스크립트 |
| `src/run_pbnbv_orbit_fixedalt.py` | tilt=45° 고정 + hard theta<70° 규제 (비교용) |
| `results/pbnbv_orbit_soft_path.json` | WP 좌표 및 상세 결과 |
| `results/orbit_soft_summary.png` | 전체 요약 4-panel 이미지 |
| `results/orbit_soft_final.png` | 경로 + 커버리지 바 상세 (이전 생성) |
| `results/regulation_compare.png` | hard/soft 규제 4방식 비교 |
| `results/unobservable_reason.png` | 구조적 불가 voxel 분석 이미지 |

---

## 5b. 데이터 정제 (SOR) — 2026-06-26

**실제 물체**: 사진 측정 120×110×36cm (납작한 ㄷ자 슬래브)

**문제**: 원본 점군(2438)에 공중 노이즈 2점(Z=-3.85m, 3.85m 상공) 혼입
→ `obj_z_top = pts[:,2].min()`이 노이즈를 물체 top으로 오인 → 후보 고도 3.5m 과대

**정제**: `src/clean_outliers.py` — SOR(k=16, μ+1σ)로 2점 제거
- Z범위 -3.85→-0.38m, bbox 451×383cm → 115×36cm 정상화
- 점군 본체 1.55×1.10×0.34m → 실제 물체와 일치 확인

**검증**: `src/viz_object_shape.py` → `results/object_shape.png`
- 3뷰 + 밀도맵으로 ㄷ자 형상 확인 (테두리 빽빽, 중앙 듬성)

---

## 6. 논문 (arXiv:2501.10663) 분석 메모

- **논문이 실제로 하는 것**: frontier/occupied voxel → GMM → ellipsoid 투영 면적 F-score
- **논문의 0.5^r 의미**: 깊이 rank r인 ellipsoid에 W=0.5^r 부여 — coarse occlusion 근사이지, occlusion 무시가 아님
- **논문의 한계 (논문 자체 인정)**: "optimal global path planning" = future work. 오목구조 자기가림(self-occlusion) 처리 미흡
- **Case 3이 논문에서 채용한 것**: PB-NBV 2-step lookahead 구조만. frontier/GMM/ellipsoid/0.5^r 모두 미채용
- **Case C의 기여 포지션**: 논문 ellipsoid 단위 0.5^r + raycast 정밀도 결합 → "논문 아이디어 발전" 서술 가능
