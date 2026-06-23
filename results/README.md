# results/ — 실행 결과 모음

실행한 스크립트·모드별로 JSON/PNG 출력을 폴더로 정리한다.

### 단순 NBV 계열 (기존)

| 폴더 | 생성 스크립트 / 명령 | 내용 |
|---|---|---|
| `realtest_optimal_path/` | `python realtest_optimal_path.py` | PB-NBV(2) + Greedy 최종 경로 (12 WP). `realtest_optimal.json` + 6패널 시각화 `realtest_result.png` |
| `realtest_pbnbv_path/` | `python realtest_pbnbv_path.py` | 단순 NBV 경로 (10 WP) + PNG + HTML |
| `realtest_pbnbv_ring/` | `python realtest_pbnbv_only.py --candidate-mode ring` | 지점 생성만 (경로 연결 없음). 후보를 **동심원** 위에 생성 → 선택 결과가 원형 링 |
| `realtest_pbnbv_uniform/` | `python realtest_pbnbv_only.py --candidate-mode uniform ...` | 지점 생성만. 후보를 **공간 균일 무작위**로 생성 → 선택 결과가 공(ball) 모양 |

### 논문 충실 PB-NBV (`pbnbv_paper.py`)

| 파일 | 내용 |
|---|---|
| `pbnbv_paper/pbnbv_paper.json` | 논문 PB-NBV 경로 (voxel+ellipsoid 투영, 온라인 NBV) |
| `pbnbv_paper/alt_1m.json ~ alt_7m.json` | 고도 1~7m **단일고도별** PB-NBV 경로 |
| `pbnbv_paper/pbnbv_paper_path.html` | 경로 3D (coverage 색) |
| `pbnbv_paper/pbnbv_paper_coverage.html` | viewpoint별 **복원 범위** 색 매칭 |
| `pbnbv_paper/all_altitudes.html` | 고도 1~7m 경로 토글 비교 |
| `pbnbv_paper/all_altitudes_two_methods.html` | 고도별 × (온라인 A vs 배치 B) 비교 |
| `pbnbv_paper/altitude_comparison.png` | 고도별 커버상한·스텝 수 |
| `eval_compare.png` | 단순 NBV vs 논문 PB-NBV |
| `compare_two_methods.png` | 온라인 NBV vs 배치선택+greedy |
| `speed_scaling.png` | ray-casting O(N) vs ellipsoid O(1) 속도 |

## 핵심 관찰

**기존 단순 NBV 계열:**
- ring 모드는 원형, uniform 모드는 공 모양 — "원형" 결과는 알고리즘 결론이 아니라 **후보 생성 방식(cos/sin 원)** 이 반영된 것.
- 상위 점수가 만점에 포화되어 우열이 안 갈림 (occlusion 미적용 한계).

**논문 PB-NBV (`pbnbv_paper.py`):**
- occlusion(법선/가시성) + frontier + occupied 페널티 적용 → 점수가 제대로 갈림.
- **이 물체(real_test)는 평평해서 1바퀴(또는 고고도 2장)로 충분.** "원형 2바퀴"는 데이터상 불필요.
- 고고도(4~7m)가 저고도보다 효율적(100% 커버, 2장). 단 현재 평가함수 F는 투영면적 편향이라 저고도를 선호하는 한계가 있음.
- 속도: 점/voxel 수가 클수록 ellipsoid 투영이 압승(2.4M점 1,600배). real_test는 교차점 규모.
