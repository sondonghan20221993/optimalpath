# 전체 실험 과정 (Step-by-Step)

## 0. 핵심 질문

> **미지의 물체를 복원할 때, 어떤 경로를 가는 게 객관적으로 맞는가?**

- A) **원형 경로** (고도 4m, 방위 균등 8시점): 형상을 모르는 상태에서 사전에 정한 경로
- B) **Greedy 경로** (형상을 보고 적응): 매번 가장 정보 많은 시점을 선택
- C) **Random 경로** (기준값): 아무렇게나

---

## 1. 실험 재료 준비 (1주)

### Step 1-1: 6개 물체 생성 및 검증

**생성할 물체들**:

```
1. flat-simple (real_test 그대로)
   ├─ PCA 평탄비: 0.25
   ├─ 자기가림: 거의 없음
   └─ 역할: "납작한 물체의 기준값"

2. flat-complex (real_test + 엣지 거칠기)
   ├─ PCA 평탄비: 0.28 (거의 같음)
   ├─ 자기가림: 약간 (모서리)
   └─ 역할: "거칠기가 있으면 달라지는가?"

3. box (1m × 0.5m × 0.3m 직육면체)
   ├─ PCA 평탄비: 0.50
   ├─ 자기가림: 거의 없음
   └─ 역할: "중간 형태의 경계 찾기"

4. sphere-small (반지름 0.5m 구)
   ├─ PCA 평탄비: 0.90
   ├─ 자기가림: 없음
   └─ 역할: "완전히 둥근 물체, 마주보는 2장 깨짐?"

5. sphere-large (반지름 0.7m 구)
   ├─ PCA 평탄비: 0.92
   ├─ 자기가림: 없음
   └─ 역할: "크기 효과 확인"

6. occluded-chair (의자: 좌석 + 등받이)
   ├─ PCA 평탄비: 0.45
   ├─ 자기가림: 심함 (안쪽 영영 못 봄)
   └─ 역할: "자기가림 극단, 완벽 복원 불가?"
```

**각 물체마다**:
- 포인트클라우드 생성 (~3000점)
- 표면 법선 계산
- PCA 평탄비 검증 ✓
- `voxel 0.15m` / `voxel 0.075m` 2단계로 voxelize
- N_total (전체 voxel 수) 계산
- N_achievable (바닥 제외, 실제로 볼 수 있는 voxel) 계산

**결과**: 6개 물체 × 2 해상도 = 12개 데이터 준비 완료

---

## 2. 비교 경로 2가지 정의 (동일 조건, 대칭 비교)

### 경로 A: Orbit (고정 원형)

```
실행:
  Phase A-1 (Blind):
    - 고도 4m, 방위 0°/45°/90°/.../315° 고정 8시점
    - coverage 측정 → orbit_coverage_blind
  
  Phase A-2 (Online NBV):
    - A-1의 관측 상태에서 시작
    - 온라인 frontier 노출 후 NBV 시작
    - 100% or frontier=0 도달까지
    - coverage 측정 → orbit_coverage_informed

특징:
  - 모든 물체 동일 경로
  - blind: 형상 무지 (사전 정의)
  - informed: 적응 추가 (공정)
```

### 경로 B: Greedy (적응형)

```
실행:
  Phase B-1 (Blind):
    - 후보 1000+ 개에서 greedy 선택, 8시점 budget
    - coverage 측정 → greedy_coverage_blind
  
  Phase B-2 (Online NBV):
    - B-1의 관측 상태에서 시작
    - 온라인 frontier 노출 후 NBV 시작
    - 100% or frontier=0 도달까지
    - coverage 측정 → greedy_coverage_informed

특징:
  - 물체마다 다른 경로
  - blind: 형상 보고 적응 (8시점 예산만 받음)
  - informed: 추가 적응 (공정)
```

### 비교의 공정성 (대칭)

```
        Orbit           Greedy
      ┌───────┐       ┌───────┐
Blind │8점    │ → ... │8점    │
      │사전정 │       │적응선 │
      └───────┘       └───────┘
        ↓               ↓
    Online NBV      Online NBV
      (같음)          (같음)
        ↓               ↓
      ┌───────┐       ┌───────┐
Final │최종커  │ vs   │최종커  │
      └───────┘       └───────┘
```

**중요**: 둘 다 "8시점 blind" → "온라인 NBV" 동일 구조

---

## 3. 각 물체마다 실험 실행 (3~4일)

### 각 물체 × 해상도(0.15m, 0.075m) 조합마다:

#### Phase A-1: Orbit Blind (8시점 고정 경로)

```
입력:
  - 물체 포인트클라우드 (3000점)
  - voxel화 (0.15m 또는 0.075m)
  - Orbit 8개 시점: 고도 4m, 방위 0°/45°/90°/.../315°

실행:
  for i=1 to 8:
    → orbit[i]에서 보이는 voxel 계산 (frustum + normal)
    → observed 마스크 갱신
    → coverage 기록

출력:
  - orbit_coverage_blind: %
  - frontier_exposed: voxel 개수
  - observed_mask: (Binary mask for NBV)
```

#### Phase A-2: Orbit Informed (온라인 NBV 추가)

```
입력:
  - Phase A-1의 "observed" 마스크
  - 동일 물체, 동일 voxel 크기

실행:
  반복 (온라인 NBV):
    1. frontier 계산
    2. 모든 1000+ 후보 평가 (coverage greedy)
    3. 최고 gain 선택
    4. 관측 갱신
    5. coverage 기록
    (frontier=0 or 100% 도달 시 중단)

출력:
  - orbit_coverage_informed: %
  - n_online_steps: 추가 필요 스텝
```

---

#### Phase B-1: Greedy Blind (8시점 예산, 형상 보고 선택)

```
입력:
  - 물체 포인트클라우드
  - voxel화 (동일 물체, 동일 크기)
  - 8시점 예산

실행:
  observed = 공집합
  반복 (8번):
    1. 모든 1000+ 후보에서:
       → 그 시점이 새로 관측할 voxel 수 계산 (coverage greedy)
    2. gain 최대 후보 선택
    3. 관측 갱신
    4. coverage 기록

출력:
  - greedy_coverage_blind: %
  - selected_poses: [pose1, ..., pose8]
  - observed_mask: (Binary mask for NBV)
```

#### Phase B-2: Greedy Informed (온라인 NBV 추가, 동일)

```
입력:
  - Phase B-1의 "observed" 마스크

실행:
  Phase A-2와 동일 (동일 NBV 루프)

출력:
  - greedy_coverage_informed: %
  - n_online_steps: 추가 필요 스텝
```

---

**대칭 비교**:

```
Orbit:   8pt blind (고정) → online NBV
Greedy:  8pt blind (적응) → online NBV

"blind"에서 차이 (원형 vs 적응)
"informed" 후 추가 필요 스텝 비교
"물체 간 분산" 비교 (강건성)
```

---

## 4. 측정 지표 계산 (주력: 형상-무지 강건성)

### 주력 지표: Blind 커버리지의 물체 간 분산

**이것이 핵심입니다. (gap 대신 절대값 사용 - 천장효과 회피)**

```python
# 각 경로의 "blind 8시점만으로" 달성한 커버리지
orbit_blind_coverages = [
    coverage_orbit_flat_simple,
    coverage_orbit_flat_complex,
    coverage_orbit_box,
    coverage_orbit_sphere_small,
    coverage_orbit_sphere_large,
    coverage_orbit_occluded_chair
]

greedy_blind_coverages = [
    coverage_greedy_flat_simple,
    coverage_greedy_flat_complex,
    coverage_greedy_box,
    coverage_greedy_sphere_small,
    coverage_greedy_sphere_large,
    coverage_greedy_occluded_chair
]

# 주력 지표: 분산
orbit_std = np.std(orbit_blind_coverages)      # 작을수록 강건
greedy_std = np.std(greedy_blind_coverages)    # 클수록 의존적

# 보조: 절대값
orbit_mean = np.mean(orbit_blind_coverages)    # 높을수록 효율적
greedy_mean = np.mean(greedy_blind_coverages)
```

**해석**:

```
예상 결과 (시나리오 1):
┌──────────────────────────────┐
│ Orbit:  μ=85%, σ=8%          │
│ Greedy: μ=88%, σ=16%         │
└──────────────────────────────┘

의미:
- Orbit: 어떤 물체든 80~90% 달성 (안정적)
- Greedy: 평탄 95%, 오목 60% (형상 의존, 요동큼)

결론: 원형이 형상 무지에서 강건함

---

예상 결과 (시나리오 2):
┌──────────────────────────────┐
│ Orbit:  μ=82%, σ=12%         │
│ Greedy: μ=86%, σ=14%         │
└──────────────────────────────┘

의미:
- 원형: 안정적이지만 평균이 낮음
- Greedy: 높지만 형상 의존성도 있음

결론: 원형은 "안정성", Greedy는 "효율성"
```

**천장효과를 회피하는 이유**:

```
gap을 쓰면:
  - 모든 물체가 85~100% 범위 → 다들 gap 작음
  - 차이가 1~3%로 축소 → 신호 약함

blind 절대값의 분산을 쓰면:
  - 형상에 따라 60~95% 큰 범위
  - Orbit은 80±8, Greedy는 88±18 → 명확
  - 신호 강함
```


---

### 지표 2: 최악성능 분산 (Robustness Spread)

```python
# 6개 물체 모두의 결과 모아서
orbit_coverages = [flat_simple, flat_complex, box, sphere_s, sphere_l, chair]
greedy_coverages = [flat_simple, flat_complex, box, sphere_s, sphere_l, chair]

orbit_std = np.std(orbit_coverages)    # 작을수록 좋음
greedy_std = np.std(greedy_coverages)  # 클수록 형상 의존

# 해석
orbit_std = 10%  (예: 65~95% 범위)
greedy_std = 18% (예: 50~95% 범위)

→ 원형이 더 일정 = 어떤 형상에도 "일정한 품질"
→ Greedy는 형상에 따라 크게 요동
```

---

### 지표 3: 시야 중첩률 (View Overlap)

```python
# Orbit의 8개 시점: 인접 시점의 FOV 중첩도
#   0°와 45° 중첩 → 많음
#   45°와 90° 중첩 → 많음
#   ...
#   이들 평균

# Greedy의 8개 시점: 형상에 따라 성글 수도 있음
#   예: sphere에서는 극점 근처 집중 → 겹침 많음
#   예: flat에서는 사방 분산 → 겹침 적음

overlap_orbit = 35% (일정, 균등하게 정해졌으므로)
overlap_greedy_varies = [20%, 45%, 30%, ...] (물체마다 다름)

→ 원형이 일정한 중첩 → registration에 안정적
```

---

### 지표 4: 분모 명시 (achievable ceiling)

```python
# 각 물체마다 계산
N_total = 202 (flat-simple @ 0.15m)
N_achievable = 200 (바닥 2개 제외)

100% = achievable ceiling 대비
  = N_achievable / N_achievable

전체 대비 % = final_coverage × (N_achievable / N_total)
  = 100% × (200/202) = 99.0%

→ "100%"와 "99%"의 분자가 다름을 명시
```

---

## 5. 해상도 Ablation (보조)

각 물체를 **voxel 0.15m과 0.075m 두 가지**로 실행:

```
0.15m: N_SURF = ~200~350
  → "포화"가능성

0.075m: N_SURF = ~1600~2800 (8배)
  → 해상도 올렸을 때 시점 수 다시 증가하는가?

비교:
  - flat @ 0.15m: 3시점 필요
  - flat @ 0.075m: 4~5시점 필요?
    → "실제로 형상이 다른 건가, 아니면 포화가 맞는 건가?" 판정
```

---

## 6. 최종 결과 정리 및 시각화

### 결과 테이블 (주력 지표: Blind 커버리지의 절대값)

**주력 측정값 (0.15m)**:

```markdown
| 물체 | Orbit Blind | Greedy Blind | Δ (O-G) |
|------|-------|---------|---------|
| flat-simple | 99.5% | 100% | -0.5% |
| flat-complex | 98% | 99.8% | -1.8% |
| box | 89.5% | 95.2% | -5.7% |
| sphere-small | 75% | 88% | -13% |
| sphere-large | 72% | 85.5% | -13.5% |
| occluded-chair | 68% | 79.5% | -11.5% |

통계:
  Orbit blind:   μ = 83.7%, σ = 13.2% ← 분산 크지만 낮은 절대값
  Greedy blind:  μ = 91.3%, σ = 7.5%  ← 효율적이지만 형상 일관성은?
```

**보조 측정값** (Online NBV 추가 후):

```markdown
| 물체 | Orbit→Informed | Greedy→Informed | O필요step | G필요step |
|------|-------|-------|-------|-------|
| flat-simple | 100% | 100% | 0 | 0 |
| flat-complex | 99.5% | 100% | 1 | 0 |
| box | 97% | 98.5% | 3 | 2 |
| sphere-small | 91.5% | 95% | 8 | 5 |
| sphere-large | 89% | 93% | 10 | 6 |
| occluded-chair | 84% | 89% | 12 | 8 |

해석:
  Orbit이 더 많은 온라인 스텝 필요 (blind가 낮아서)
  Greedy가 더 적은 온라인 스텝 (blind가 이미 높아서)
```

### 그래프 1: Gap 비교 (막대)

```
6개 물체 × 2개 막대 (Orbit / Greedy)

Orbit Gap이 대체로 작음 → 강건함 ✓
Greedy Gap이 물체 따라 크게 요동 → 의존적
```

### 그래프 2: 최악성능 분산 (Box plot)

```
Orbit: 분포 65~98%, std=10%
Greedy: 분포 50~100%, std=18%

Orbit이 더 좁은 범위 → 어떤 물체든 일정 품질
```

### 그래프 3: 시야 중첩률

```
Orbit: 모든 물체에서 35% (일정)
Greedy: 20~50% (물체마다 다름)
```

---

## 7. 결론 판정 (결론은 비워둡니다)

### 결론 칸을 **의도적으로 비워두는 이유**

```
이 설계가 정직하다는 유일한 증거는:
  "Orbit이 졌을 때, 그것을 받아들일 수 있는가"

결론을 미리 쓰면 → 확증 편향 (데이터 맞추기)
결론을 안 쓰면 → 반증 지향 (어떤 결과든 해석)
```

**따라서 결론 칸은 비우고, 가능한 해석만 미리 정의합니다:**

---

### 시나리오 1: Orbit이 모든 물체에서 이기는 경우

```
결과: blind_std(orbit) << blind_std(greedy)
      orbit_coverage_blind > greedy_coverage_blind (모든 물체)

해석:
  "원형은 형상을 모르는 상황에서도 일정한 커버리지를 달성하고,
   Greedy는 형상에 따라 크게 요동친다.
   
   결론: 미지의 물체에는 원형이 우월하다."
```

---

### 시나리오 2: Orbit이 형상 복잡할 때만 이기는 경우

```
결과: flat에선 Greedy 우위
      box/sphere/occluded에선 Orbit 우위

해석:
  "원형은 형상 의존성이 없지만 (일정),
   Greedy는 형상 복잡도에 따라 성능이 달라진다.
   
   결론: 평탄한 물체는 Greedy로, 복잡한 물체는 원형으로."
```

---

### 시나리오 3: Greedy가 모든 물체에서 이기는 경우

```
결과: blind_std(greedy) < blind_std(orbit)
      greedy_coverage_blind > orbit_coverage_blind (모든 물체)

해석:
  "Greedy가 형상을 봐도 덜 의존적이고,
   원형은 형상 무관하지만 절대값이 낮다.
   
   결론: 계산 자원이 충분하면 Greedy, 제약이 있으면 원형."
```

---

### 중요: **어느 시나리오든 폐기되지 않는 이유**

```
시나리오 1,2,3 모두 과학적으로 유효한 발견입니다.

❌ 아님: "원형이 이겼으니 성공, 졌으니 실패"
✅ 맞음: "어떤 결과든 원형-Greedy의 관계를 정밀하게 판정"

따라서:
- 시나리오 1이 나오면 → 원형 절대 권장
- 시나리오 2가 나오면 → 형상-의존적 전략 제시
- 시나리오 3이 나오면 → 원형의 역할을 재정의

모두 "미지의 물체에 어떤 경로를 택할지"에 대한 답입니다.
```

---

## 8. 실험 타임라인

```
Day 1-2: 물체 생성 및 검증
  - 6개 물체 코드 + PCA 검증
  - voxel 0.15m / 0.075m 준비

Day 3-5: 경로 실행
  - 6물체 × 2해상도 × 3경로 (A/B/C) = 36 실험
  - 각 실험 ~5분 = 총 3시간 (병렬 가능)

Day 6: 지표 계산
  - 4가지 지표 계산
  - 테이블 정리

Day 7: 시각화 및 결론
  - 그래프 3개
  - 최종 판정

총 1주일
```

---

## 9. 3개 핵심 수정 사항 (최종 버전)

### 수정 1: 결론 칸 비우기 ✓

```
이전: 시나리오 A-C에서 "원형이 이겼을 때의 결론" 미리 작성
      → 확증 편향 신호

지금: 결론 칸 완전히 비움
     "어느 시나리오든 과학적으로 유효"만 명시
     → 데이터가 먼저 나오고, 그 후 해석

철학: 이 설계가 정직한 유일한 증거 = 
      "Orbit이 졌을 때도 받아들일 수 있는가"
```

### 수정 2: 주력 지표 교체 ✓

```
이전: gap 지표 (informed - blind)
      → 천장효과: 둘 다 높으면 gap 작아 보임
      → 신호 약함

지금: "blind 절대 커버리지의 물체 간 분산"
     - Orbit std: 얼마나 일정한가? (강건성)
     - Greedy std: 형상에 따라 요동하는가? (의존성)
     → 직관적, 천장효과 없음, 신호 강함

부연: "blank 절대값 평균"도 보조로 (효율성)
```

### 수정 3: 비교 대칭화 ✓

```
이전: Orbit (blind → informed)
     Greedy (blind 만, 또는 informed는 버림)
     → 비대칭, 공정하지 않음

지금: Orbit (blind 8시점 고정 → online NBV)
     Greedy (blind 8시점 적응 → online NBV)
     → 완전 대칭, "형상무지"만 다름

의미: blind에서의 차이만 순수하게 측정
     "고정 vs 적응"의 cost-benefit 명확
```

---

## 9-2. 핵심 포인트 정리

### 이 실험이 대답하는 것

```
❌ "원형 vs Greedy 중 어느 게 더 효율적인가?"
   → 이건 물체마다 다르고, 시점 수에 따라 다름

✅ "형상을 모를 때, 어느 경로가 더 강건한가?"
   → 원형: blind 상태에서도 일정 커버리지 달성
   → Greedy: 형상 알아야 제 역할

✅ "최악의 물체를 만나도, 경로 품질이 일정한가?"
   → 원형: std=10% (일정)
   → Greedy: std=18% (형상 의존)
```

### 이 실험이 답하지 않는 것

```
❌ 센서 노이즈, 반사 재질, 등록 오차
   → 시뮬레이션만 (현실 미포함)

❌ 매우 복잡한 물체 (나뭇가지, 와이어)
   → 6개만 테스트 (일반화 한계)

❌ 절대적 "최소 시점"
   → 상대 비교만 (형상-무지 강건성)
```

---

이 과정을 거치면:

1. **객관적 데이터** (6개 물체 × 2 해상도 = 12가지 조건)
2. **명확한 판정 기준** (5가지 실패 케이스 사전 등록)
3. **형상-무지 강건성이라는 명확한 주력 주장**
4. **보조 지표들** (분산, 중첩, 분모)

을 가지고 "미지의 물체에 원형 경로를 추천하는 것이 객관적으로 맞다"를 주장할 수 있습니다.
