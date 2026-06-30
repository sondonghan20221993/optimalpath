# src/ 파일 기능 정리

**최종 업데이트**: 2026-06-27

---

## 핵심 라이브러리 (import 대상)

### `pbnbv_path.py` — 공통 알고리즘 라이브러리
모든 run_*.py, viz_*.py가 `import pbnbv_path as A`로 사용하는 핵심 모듈.

| 함수 | 역할 |
|------|------|
| `point_in_frustum()` | FOV + 거리 + 법선(backface) + raycast occlusion 가시성 판정 |
| `information_gain()` | 후보 시점에서 미관측 점 중 새로 보이는 수 반환 (hard count) |
| `information_gain_ellipsoid_rank()` | Hard raycast → KMeans 클러스터 → 0.5^r depth rank 가중 IG (Case C) |
| `information_gain_softrank()` | 방향 bin 내 depth rank → 0.5^r 가중 IG, hard raycast 없음 (Case B) |
| `pbnbv_score()` | PB-NBV(2) 스코어: IG1 + 0.5 × max(IG2) — hard count 기반 |
| `pbnbv_score_ellipsoid()` | PB-NBV(2) — ellipsoid-unit 0.5^r 기반 (Case C) |
| `pbnbv_score_softrank()` | PB-NBV(2) — soft per-point 0.5^r 기반 (Case B) |
| `gen_candidates_tilt45()` | tilt=45° 고정 후보 생성, max_dist 초과 고도 자동 제외 |
| `generate_candidates()` | MASt3R 좌표계용 구면 후보 생성 (project_to_tilt_boundary 적용) |
| `project_to_tilt_boundary()` | tilt 초과 후보를 수평 이동해 tilt=MAX_TILT 경계로 보정 |
| `salvage_coverage()` | IG_THRESH 컷 후 남은 점을 greedy로 추가 커버 |
| `greedy_path()` | Nearest-Neighbor Greedy 방문 순서 결정 |
| `tilt_deg_to_target()` | 카메라 위치 → target 기준 tilt 각도 계산 |
| `compute_coverage()` | 기존 카메라 전체 대비 점별 관측 횟수 계산 |

**주요 전역 파라미터**
```python
MAX_DIST       = 8.0   # 최대 가시거리 (m)
MAX_TILT_DEG   = 45.0  # 최대 tilt 각도
LOOKAHEAD      = 2     # PB-NBV lookahead 단계
RAYCAST_OCCLUSION = False  # 런타임 설정 가능
```

---

### `pbnbv_paper.py` — 논문 충실 구현 (Case A-1)
arXiv:2501.10663 PB-NBV를 real_test 데이터에 직접 구현.

| 함수 | 역할 |
|------|------|
| `make_candidates()` | gen_candidates_tilt45() 호출, tilt=45° 후보 생성 |
| `make_K()` | 카메라 intrinsic 행렬 (FOV 기반) |
| `look_at_R()` | 카메라 → target 방향 회전 행렬 |
| `observed_by()` | 후보 시점이 관측하는 surface voxel 인덱스 반환 |
| `compute_frontier()` | 미관측 voxel 중 관측된 voxel에 인접한 것 = frontier |
| `fit_ellipsoids()` | GMM 클러스터링 → (center, covariance, mean_normal) 리스트 |
| `project_area_depth()` | Ellipsoid → 이미지 평면 투영 면적 + 깊이 (closed-form Jacobian) |
| `evaluate()` | F = Σ(frontier·W) − Σ(occupied·W), W=0.5^r (depth rank) |
| `run_nbv()` | NBV 반복 루프 → path 반환 |
| `eval_on_points()` | 실제 점 2438개 + raycast 기준 커버리지 평가 (공동 통제) |

**설정 상수**
```python
VOXEL    = 0.05   # 복셀 크기 (점 간격 ~0.035m 기준)
MAX_DIST = 8.0    # 공동 통제
TARGET   = pts.mean(axis=0)  # 포인트클라우드 centroid
```

---

## 실험 실행 스크립트 (run_*.py)

### `run_pbnbv_nocam.py` → **Case 3: Hard Raycast + PB-NBV(2) + Greedy**
- no-camera 시나리오 (전체 점 미관측)
- `pbnbv_score()` + `salvage_coverage()` + `greedy_path()` 순서
- 출력: `results/pbnbv_nocam_path.json`
- 결과: 7 WP, 100%, 29.9m

### `run_pbnbv_hybrid.py` → **Case A-2: Hybrid (paper-F + Greedy)**
- `pbnbv_paper.run_nbv()` 로 시점 집합 선택 (paper F-score)
- 선택 순서를 버리고 `greedy_path()` 로 방문 순서 재정렬
- 출력: `results/pbnbv_hybrid/pbnbv_hybrid.json`
- 결과: 3 WP, 90.9%, 22.7m

### `run_pbnbv_softrank.py` → **Case B: Soft 0.5^r per-point**
- `pbnbv_score_softrank()` 사용 (방향 bin 내 depth rank 가중)
- live_mask 갱신은 hard_vis(최근접) 기준
- 출력: `results/pbnbv_softrank_path.json`
- 결과: 14 WP, 100%, 34.7m

### `run_pbnbv_ellipsoid.py` → **Case C: Ellipsoid-unit 0.5^r + Raycast**
- `pbnbv_score_ellipsoid()` 사용 (hard raycast → KMeans → 0.5^r)
- 출력: `results/pbnbv_ellipsoid_path.json`
- 결과: 8 WP, 100%, **20.3m (100% 케이스 최단)**

### `run_pbnbv_realtest.py` — real_test + 기존 카메라 포즈
- real_test meta/*.json 기존 카메라 pose 로드
- 기존 카메라 커버리지 계산 후 미관측 영역에 PB-NBV 적용
- 출력: `results/pbnbv_realtest_path.json`

### `run_pbnbv_orbit_soft.py` — **최종: Orbit + Soft Lambert PB-NBV**
- **메인 결과**: 원형 궤도 후 Soft Lambert cos¹ 규제로 사각지대 보완
- Soft 누적 관측: `W(voxel) = Σ cos^alpha(theta)`, 관측 기준 `W >= TAU=cos(70°)`
- U = F/dist (frontier_only=True, F≥0 보장)
- **파라미터**: alpha=1.0 (Lambert 표준), TAU=cos(70°) (hard 경계 동등)
- 출력: `results/pbnbv_orbit_soft_path.json`
- **결과: 3 WP, 7.92m, real raycast 100.0%**

### `run_pbnbv_orbit_fixedalt.py` — Orbit 보완 + tilt=45° 고정 + hard theta<70°
- tilt=45° 고정 고도(z=-3.38m) + 입사각 theta<70° hard 규제
- `gen_candidates_tilt45()` + `frontier_only=True` 적용
- 출력: `results/pbnbv_orbit_fixedalt_path.json`
- 결과: 1 WP, 8.24m, real 99.6% (F<0 버그 수정 후)
- **비교 역할**: soft 규제와 hard 규제의 직접 비교

### `run_pbnbv_orbit.py` — 원형 궤도 보완 (raycast)
- 기존 34개 원형 궤도 카메라 → raycast 커버리지 → 사각지대만 PB-NBV(2) 보완
- 출력: `results/pbnbv_orbit_path.json` (궤도 95% → 보완 95.9%)

### `run_pbnbv_orbit_paper.py` — 원형 궤도 보완 (논문 0.5^r, raycast 없음)
- 궤도 voxel 커버 → frontier → ellipsoid F-score 보완
- 출력: `results/pbnbv_orbit_paper_path.json` (θ<90: voxel 100% 거짓판정 → 보완 0개)

### `run_pbnbv_orbit_paper_strict.py` — 논문방식 + 입사각 규제 sweep
- observed 판정에 입사각(θ_max) 제한 추가 (raycast 없이 가림 부분 근사)
- θ=90/70/60/50/40° sweep → θ≤60에서 실제 100% 회복
- 출력: `results/pbnbv_orbit_paper_strict.json`
- **비교 역할**: 단순 threshold sweep과 soft 규제의 차이 시각화

---

## 데이터 정제 스크립트

### `clean_outliers.py` — SOR 이상치 제거
- 16-NN 평균거리 > μ+1σ 인 점 제거 (공중 노이즈 2점)
- points·normals 동시 적용
- 출력: `real_test/real_test_pts_normals_clean.npz`
- ⚠️ 실행 후 원본 백업(`_orig.npz`)하고 메인 npz 교체함

---

## 시각화 스크립트 (viz_*.py)

### `viz_compare3.py` — Case A-1 / A-2 / 3 비교
- 3-case top-down 경로 + coverage 곡선 + 요약 bar + 텍스트 분석
- 출력: `results/compare3_cases.png`

### `viz_compare4.py` — Case A-1 / A-2 / 3 / B 비교 (4-case)
- 4-case top-down 경로 + coverage 곡선 + 요약 bar + 분석
- 출력: `results/compare4_cases.png`

### `viz_compare5.py` — 전체 5-Case 비교 (최신 기준)
- 5-case top-down + coverage 곡선 + 요약 bar + 핵심 분석 텍스트
- 출력: `results/compare5_cases.png`

### `viz_orbit_soft_summary.py` — **Orbit + Soft NBV 최종 요약 (4-panel)**
- (a) 개발 이력 타임라인 (버그 수정 포함)
- (b) 최종 경로 top-down (orbit + 3 WP)
- (c) 커버리지 3지표 비교 bar (hard/soft/real)
- (d) 파라미터 정당화 & 방어 포인트
- 출력: `results/orbit_soft_summary.png`

### `viz_orbit_compare.py` — 원형 궤도 보완 3방식 비교
- 논문(0.5^r) vs 규제 vs raycast / 궤도 사각지대 + 규제 sweep + claimed vs real
- 출력: `results/orbit_compare.png`

### `viz_object_shape.py` — 정제 점군 형상 진단
- 3뷰(top/front/side) + XY 밀도맵으로 ㄷ자 형상 확인
- 출력: `results/object_shape.png`

### `viz_nocam.py` — Case 3 단독 시각화
- no-camera 시나리오 결과 단독 상세 시각화
- 출력: `results/pbnbv_nocam_viz.png`

### `viz_realtest.py` — real_test 단독 시각화
- 기존 카메라 + PB-NBV 신규 경로 비교
- 출력: `results/pbnbv_realtest_viz.png`

---

## 구버전 / 실험용 스크립트

### `coverage_greedy.py`
- voxel 기반 순수 coverage greedy (F-score 없음)
- 고도를 고정하고 방위각 순서로 선택하는 baseline
- `pbnbv_paper.py` import 필요

### `pbnbv_path_azimuth_variants.py`
- 방위각 다양성 항 λ × azimuth_penalty 추가 실험
- λ = 0, 0.25, 0.5, 1.0, 2.0 및 Hierarchical(커버 95% 이후 azimuth 우선) 비교
- MASt3R 좌표계 전용, real_test 미지원

---

## 실행 순서 (처음부터 재현 시)

```bash
# 논문 방식 (Case A-1, A-2)
python src/pbnbv_paper.py
python src/run_pbnbv_hybrid.py

# No-camera 시나리오
python src/run_pbnbv_nocam.py       # Case 3
python src/run_pbnbv_softrank.py    # Case B
python src/run_pbnbv_ellipsoid.py   # Case C

# 시각화
python src/viz_compare5.py          # 전체 5-Case 비교

# ── Orbit 시나리오 (최신 / 메인) ──────────────────────────────────────────
# 최종 결과: Orbit + Soft Lambert PB-NBV
python src/run_pbnbv_orbit_soft.py      # 3 WP, 7.92m, real 100.0%

# 비교용
python src/run_pbnbv_orbit_fixedalt.py  # hard theta<70° (1 WP, 99.6%)
python src/run_pbnbv_orbit_paper.py     # 논문 원본 방식
python src/run_pbnbv_orbit_paper_strict.py  # theta sweep

# 시각화
python src/viz_orbit_soft_summary.py    # 최종 4-panel 요약
```

> 모든 스크립트는 `real_test/real_test_pts_normals.npz` 를 기준 데이터로 사용.  
> 실행 위치: `optimalpath/` 루트 디렉토리
