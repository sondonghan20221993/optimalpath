# results/ — 실행 결과 모음

실행한 스크립트·모드별로 JSON/PNG 출력을 폴더로 정리한다.

| 폴더 | 생성 스크립트 / 명령 | 내용 |
|---|---|---|
| `realtest_optimal_path/` | `python realtest_optimal_path.py` | PB-NBV(2) + Greedy 최종 경로 (12 WP). `realtest_optimal.json` + 6패널 시각화 `realtest_result.png` |
| `realtest_pbnbv_ring/` | `python realtest_pbnbv_only.py --candidate-mode ring` | 지점 생성만 (경로 연결 없음). 후보를 **동심원** 위에 생성 → 선택 결과가 원형 링. `_200.png`은 200개 선택 버전 |
| `realtest_pbnbv_uniform/` | `python realtest_pbnbv_only.py --candidate-mode uniform --n-candidates 1500 --n-select 200` | 지점 생성만. 후보를 **공간 균일 무작위**로 생성 → 선택 결과가 공(ball) 모양 |

## 핵심 관찰

- ring 모드는 원형, uniform 모드는 공 모양이 나온다.
- 두 경우 모두 상위 시점 점수가 **1800점(만점)에 포화**되어 우열이 거의 갈리지 않는다.
- 즉 현재 파이프라인에서 "원형" 결과는 알고리즘의 객관적 결론이 아니라 **후보 생성 방식(cos/sin 원)** 이 반영된 것이다.
- 근본 원인: **occlusion(자기 가림) 미적용** → 어느 시점에서든 박스 점 1800개가 전부 보여 점수가 갈리지 않음.
