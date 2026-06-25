# 최종 실험 설계 (확정본)

## 핵심 질문

> **미지의 물체를 복원할 때, 형상을 모르고 정한 원형 경로(Orbit)가 형상을 보고 적응하는 Greedy 경로보다 강건한가?**

---

## 주력 비교 기준

**측정 지표** (우선순위):
1. **std(blind 절대 커버리지)**: 물체 6개에서 분산이 작은가 (강건성)
   - `std(Orbit-N)` vs `std(Greedy-N)` 
   - 작을수록 형상 무지에 강건함
   
2. **mean(blind 절대 커버리지)**: 절대 효율성
   - `mean(Orbit-N)` vs `mean(Greedy-N)`
   
3. **min(blind 절대 커버리지)**: worst-case 보장
   - `min(Orbit-N)` vs `min(Greedy-N)`
   - Orbit의 셀링포인트

**보조 지표**:
- gap (informed - blind): STEP 2 분석에서만 사용
- 형상-복잡도와 (Greedy-Orbit 격차)의 상관성

---

## 실험 구조 (2 STEP)

### STEP 2: 비교 실험 (유일한 실행 단계)

**입력**:
- 6개 물체 (flat-simple, flat-complex, box, sphere-small, sphere-large, occluded-chair)
- 각 물체 2 해상도 (0.15m, 0.075m)
- 3가지 경로 (Orbit, Greedy, Random)
- 시점 예산 N = 4, 6, 8, 12, 16

**실행**: 각 (물체, 해상도, N) 조합마다

```
경로 A: Orbit-N (blind)
  - N등분 균등 고도 4m, 반경 7m 원형
  - 형상 미관측 상태, 사전 정의
  
경로 B: Greedy-N (informed 상한)
  - Oracle (전체 surface voxel 알고 있음)
  - set-cover greedy로 N개 시점 선택
  - 실세계에 없는 상한값
  
경로 C: Random-N (baseline)
  - 후보에서 N개 무작위 선택

출력: 각 경로의 coverage (%)
```

**측정** (사전 등록):

```python
for (object, voxel_size, N):
  Orbit_cov = run_orbit(N)      # blind
  Greedy_cov = run_greedy(N)    # informed
  Random_cov = run_random(N)    # baseline
  
  record: (Orbit_cov, Greedy_cov, Random_cov)
```

**결과 테이블** (실행 후):

```
| 물체 | voxel | N | Orbit | Greedy | Random | Orbit-Greedy |
|------|-------|---|-------|--------|--------|--------------|
| flat | 0.15m | 4 | 85%   | 92%    | 45%    | -7%          |
| ...  | ...   |...|  ...  | ...    | ...    | ...          |
```

---

### STEP 1: 분석 (사후 해석, 독립 실행 아님)

**입력**: STEP 2의 Orbit 커버리지 곡선

**분석 1: 포화점 분포**

```
각 물체에서 Orbit-N의 커버리지 곡선을 보고,
포화점(증가분 < 1%p)을 읽음

예시:
  flat-simple:  N=4에서 포화 (88% → 88.5% → 88.7%)
  box:          N=8에서 포화
  sphere-large: N=12에서 포화
  occluded-chair: N=16에서 포화? ← 검증 필수
```

**검증**: 의자가 N=16에서 정말 포화했는가?
```
의자 Orbit-16 커버리지 증가분이 < 1%p인가?

Yes:  → N=16이 worst-case, 종료
No:   → N=20, 24로 상한 확장, 다시 계산
      → 진짜 포화점 찾기 (이 경우 설계에 반영)
```

**분석 2: 형상 복잡도와 필요 등분의 관계**

```
PCA flatness vs 포화점 N을 산점도로 그리기

예상: 
  PCA 0.25 → N=4
  PCA 0.45 → N=8~12
  PCA 0.90 → N=12~16
  
패턴: 형상 복잡도 ↑ → 필요 N ↑
```

---

## 사전등록 실패 기준 (동결, 수정 불가)

| # | 항목 | 판정 기준 | 결과 | 조치 |
|---|------|---------|------|------|
| 1 | **강건성** (주력) | `std(Orbit-N) ≥ std(Greedy-N)` 모든 N에서 | 기각 | 가설 폐기, "Greedy가 더 강건" 결론 |
| 2 | **근접성** | `mean(Orbit-N) < mean(Greedy-N) × 0.9` 모든 N | 기각 | "Orbit이 비효율적" 결론 |
| 3 | **형상 상관** | (Greedy-Orbit 격차)와 형상 복잡도의 피어슨 r < 0.3 | 기각 | "형상 복잡도가 경로 성능을 결정 안 함" 결론 |
| 4 | **해상도 의존** | 0.15m→0.075m에서 Orbit의 우위 통계유의(p<0.05) 사라짐 | 약화 | "강건성은 해상도에 제한적" 명시 |
| 5 | **포화점 신뢰** | 의자가 N=16에서 증가분 ≥1%p (도달 못 함) | 경고 | "16은 절단점, worst-case 미입증" 명시 |
| 6 | **분모 명시** | 모든 커버리지가 N_achievable(바닥 제외) 기준이 아님 | 재계산 | 분모 재정의 후 재분석 |

**규칙**: 
- 어떤 기준도 사후에 수정 불가
- 기각 → 가설 폐기 또는 범위 한정
- 약화/경고 → 결론에 전제 조건 추가
- 재계산 → 원점에서 다시 (설계 오류)

---

## 최종 권고 (결론 템플릿, 아직 비움)

```
[STEP 2 결과 보기 전]

가능 시나리오:

### 시나리오 A: Orbit 강건함 (기대)
std(Orbit) << std(Greedy) 모든 N에서
→ "형상을 모르는 상황에서 원형이 강건하다"

### 시나리오 B: 형상 복잡할수록 중요
std 차이가 N이 작을 때만 (2~6)
→ "예산 제약 상황에서만 Orbit 유리"

### 시나리오 C: Greedy 우월
std(Greedy) ≤ std(Orbit) 모든 N
→ "형상 보고 적응하는 게 필수"

[STEP 2 완료 후]
→ 포화점 분포 본 후
→ 6가지 실패 기준 모두 통과/실패 기록
→ 그에 따라 위 3개 중 하나 또는 복합 결론 작성
```

---

## 구현 체크리스트

### 준비 단계
- [ ] 6개 물체 생성 및 PCA 검증 (0.25~0.92)
- [ ] 각 물체 2 해상도 voxelize (0.15m, 0.075m)
- [ ] N_total, N_achievable 계산
- [ ] Greedy oracle 구현 (set-cover)
- [ ] Random baseline 코드

### 실행 단계 (STEP 2만)
- [ ] 6물체 × 2해상도 × 3경로 × 5N = 180 실험
- [ ] 병렬 실행 (각 실험 ~5분, 총 3~4시간)
- [ ] 결과 저장: CSV (object, voxel_size, N, path, coverage, steps, distance)

### 분석 단계 (STEP 1)
- [ ] Orbit 커버리지 곡선 그리기 (6물체 × 2해상도 = 12곡선)
- [ ] 포화점 자동 탐지 (증가분 < 1%p)
- [ ] **의자 N=16 검증**: 포화 도달했나? → 없으면 상한 확장
- [ ] PCA vs 포화점 N 산점도
- [ ] std, mean, min 계산 (각 N에서)

### 결과 정리
- [ ] 테이블 (물체, N별 Orbit/Greedy/Random)
- [ ] 그래프 4개:
  - std 비교 (N별, 물체별)
  - mean 비교
  - min 비교 (worst-case)
  - PCA vs 포화점
- [ ] 6가지 실패 기준 결과 (통과/실패 ✓✗)
- [ ] 최종 결론 (위 템플릿 중 선택)

---

## 타임라인

```
Day 1-2: 물체 생성 + 검증
Day 3-4: STEP 2 실행 (180 실험)
Day 5:   STEP 1 분석 + 의자 포화점 검증
Day 6-7: 결과 정리 + 최종 결론

총 1주일 (병렬 가능)
```

---

## 정직함의 증거

**이 설계가 정직하다는 증거**:

1. ✅ 결론 칸 비움 (결과 보기 전)
2. ✅ 실패 기준 6가지 사전 등록 (사후 수정 불가)
3. ✅ "N=16 worst-case"를 검증 단계로 묶음 (미검증 주장 차단)
4. ✅ 형상 상관성을 명시적 기준으로 (숨은 가정 제거)
5. ✅ 해상도 의존성을 별도 기준으로 (한계 명시)
6. ✅ Orbit이 졌을 때도 "형상 복잡할 때만 유리"라 결론 수정 가능

→ **어느 결과가 나와도 과학적으로 유효한 발견**

---

## Go

구현 시작. 데이터가 말하게 하자.
