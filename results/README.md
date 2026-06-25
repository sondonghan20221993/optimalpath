# results/ — 실험 결과 디렉토리

연구 단계별로 정리. 번호순이 실험 진행 순서.

---

## 📁 디렉토리 구조

| 폴더 | 내용 |
|---|---|
| `docs/` | 설계 문서 / 분석 요약 (MD) |
| `01_initial_paths/` | 초기 Orbit 경로 생성 및 시각화 |
| `02_pbnbv_baseline/` | PB-NBV 기반 baseline 경로 실험 |
| `03_step2_comparison/` | Step2: Orbit vs Greedy 강건성 비교 |
| `04_hybrid_design/` | Hybrid 경로 설계 + 방위각 분석 |
| `05_controlled_phases/` | Phase 1-3: 통제 변수 비교 실험 |
| `06_building_sim/` | 건축물 모형 복원 시뮬레이션 |

---

## 📄 docs/ — 핵심 문서

| 파일 | 내용 |
|---|---|
| `HYBRID_PATH_DESIGN.md` | Hybrid 2단계 경로 설계 문서 |
| `FINAL_ATE_ANALYSIS.md` | 방위각 균등성 → SfM 복원 최종 분석 |
| `EXPERIMENT_SUMMARY.txt` | 전체 실험 요약 (한눈에 보기) |

---

## 01 — 초기 경로 탐색

Orbit 경로 초기 생성 및 시각화.

| 파일 | 내용 |
|---|---|
| `initial_orbit.json` | Orbit 기본 경로 좌표 |
| `initial_orbit.html` | 3D 시각화 |
| `initial_paths_comparison.html` | 초기 경로 비교 |

---

## 02 — PB-NBV Baseline

논문 PB-NBV 알고리즘 기반 기준선 실험.

| 폴더 | 내용 |
|---|---|
| `pbnbv_paper/` | 논문 충실 PB-NBV (고도 1~7m) |
| `realtest_optimal_path/` | PB-NBV + Greedy 최종 경로 (12 WP) |
| `realtest_pbnbv_path/` | 단순 NBV 경로 (10 WP) |
| `realtest_pbnbv_ring/` | 후보 ring 모드 |
| `realtest_pbnbv_uniform/` | 후보 uniform 모드 |
| `camera_range/` | 카메라 거리 범위 실험 |

---

## 03 — Step2: Orbit vs Greedy 비교

180회 비교 실험 (6 물체 × 2 해상도 × 3 경로 × 5N).

| 파일/폴더 | 내용 |
|---|---|
| `step2_comparison/` | 1차 비교 결과 |
| `step2_v2/` | 2차 비교 결과 |
| `step2_v3/` | 3차 비교 결과 (최종, 분석 포함) |
| `eval_compare.png` | 단순 NBV vs 논문 PB-NBV |
| `compare_two_methods.png` | 온라인 NBV vs 배치+Greedy |
| `speed_scaling.png` | ray-casting vs ellipsoid 속도 |

**핵심 결과**: Orbit std 14.4% vs Greedy std 16.1% → Orbit이 더 강건

---

## 04 — Hybrid 경로 설계

Orbit-8 (Stage 1) + Greedy-8 (Stage 2) 2단계 전략.

| 파일/폴더 | 내용 |
|---|---|
| `hybrid_orbit_then_greedy.json` | Hybrid 경로 좌표 (16 WP) |
| `hybrid_azimuth_visualization.png` | 극좌표 방위각 분포 (Stage 1/2/Combined) |
| `hybrid_gap_comparison.png` | 간격 비교 막대그래프 |
| `path_comparison_3d.html` | Orbit/Greedy/Hybrid 3D 비교 |
| `real_test_greedy_analysis/` | Greedy 방위각 분석 |
| `mast3r_pathcomparison/` | MASt3R 기반 경로 비교 |

---

## 05 — 통제 변수 비교 (Phase 1-3)

고도·N 통제 후 방위각 균등성만 비교.

| 파일 | 내용 |
|---|---|
| `phase1_controlled_comparison.json` | N=8, 고도=-2m 고정 비교 |
| `phase2_n_effect.json` | 고도=-2m 고정, N=[4,6,8,12,16] |
| `phase3_altitude_effect.json` | N=8 고정, 고도=[1~5m] |

**결론**:
- Orbit: 45° 균등 (N에 비례)
- Greedy: 160° 고정 불균등 (N 무관)
- 방위각은 고도와 독립적

---

## 06 — 건축물 모형 시뮬레이션

3개 경로(Orbit/Greedy/Hybrid)의 건축물 복원 품질 비교.

| 파일/폴더 | 내용 |
|---|---|
| `building_reconstruction/` | 시각화 + 경로 JSON + 요약 |
| `unknown_object_simulation.json` | 미지 물체 시뮬레이션 |

**결과**:
- Orbit-16, Hybrid-16: Max gap 22.5° (균등성 1.00) ✅
- Greedy-16: Max gap 101.2° (균등성 0.03) ❌

---

## 🎯 최종 권장 경로

**Hybrid-16** (Stage 1: Orbit-8 + Stage 2: Greedy-8)  
→ 균등성 + 정보성 모두 달성
