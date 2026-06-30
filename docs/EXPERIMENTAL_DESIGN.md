# 실험 설계: 다양성 검증 (선택지 2)

## 0. 목표 & 주력 주장

**주력 주장**: 원형 경로는 **형상 무지(blind) 상황에서 greedy보다 강건하다**
- Greedy = 형상을 알아야만 적응 최적화 가능 (사후 최적)
- 원형 = 형상을 모르는 상태에서 모든 방위 균등 스캔 (사전 robust)

**판정 축**:
1. **형상-무지 전이 손실** (메인): blind(초기경로만) vs informed(+온라인NBV) 사이의 격차
   - 원형이 작은 격차? → 형상무지 강건함
   - Greedy가 큰 격차? → 형상 의존성 높음

2. **최악성능 분산** (따름정리): 다양한 형상에서 원형의 안정성
   - 원형: 물체 상관없이 일정한 커버리지?
   - Greedy: 형상에 따라 요동?

3. **시야 중첩률** (기술 benefit): 정합(registration) 시 겹침의 이점

---

**기존 비판 해결**:
- **비판 2,4,7**: 해상도 ablation + 분모 명시 (보조로 유지)
- **새로운 초점**: "최소 시점 수"는 버리고, "형상무지 강건성" 측정

---

## 1. 비교군 설정 (3가지 경로, 동일 시점 예산)

### 1.0 핵심: 같은 8시점으로 비교

```
A) 원형 경로 (Orbit)
   - 고도: 4m 고정
   - 방위: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° (균등)
   - 특성: 형상 독립적, blind choice, 모든 방위 균등 노출

B) Greedy 경로 (Coverage Greedy)
   - 같은 8시점 budget
   - 하지만 고도 변화 허용 (1~7m 범위)
   - 형상을 "안다고 가정" → 적응 선택
   - 특성: 형상 의존적, informed choice

C) 랜덤 경로 (Random baseline)
   - 후보에서 random 선택 8개
   - 특성: 최악 baseline
```

**핵심**: "같은 시점 수(8개)"에서, 누가 더 잘 견디는가?
- 원형이 어떤 형상에서나 일정? → 강건함 입증
- Greedy가 형상에 따라 요동? → 형상 의존성 입증

---

## 1.1. 물체 선정 (6~8개, 축 정화)

축을 정화하고 교란을 제거한 **6개 물체** (n≥6):

| # | 이름 | PCA flatness | Occlusion | 의미 | 데이터 |
|---|---|----|----|----|---|
| 1 | flat-simple | 0.25 | 거의없음 | real_test 그대로 | 기존 |
| 2 | flat-complex | 0.28 | 약간 (모서리) | real_test + 엣지 | synthetic |
| 3 | box | 0.50 | 거의없음 | 1m×0.5m×0.3m 직육면체 | synthetic |
| 4 | sphere-small | 0.90 | 없음 | r=0.5m 구 | synthetic |
| 5 | sphere-large | 0.92 | 없음 | r=0.7m 구 (더 크게) | synthetic |
| 6 | occluded-chair | 0.45 | 심함 (의자 등받이) | 깊은 오목 | synthetic |

**각 축의 분포**:
- PCA: 0.25, 0.28, 0.50, 0.90, 0.92, 0.45 → 균등하지는 않지만 "전체 범위" 포함
- Occlusion: 3개 (거의없음) + 1개 (약간) + 1개 (심함) → 분포 커버
- 크기 scale: 모두 ~0.5~1m (일정함) → 교란 제거

---

### 각 물체 생성 상세

#### 1. flat-simple, flat-complex
```python
# real_test 그대로 로드, 또는 모서리 노이즈 추가
def add_complexity(points, normals, noise_level=0.02):
    """엣지(모서리) 거칠기 추가 → flat-complex"""
    # 경계 voxel에 ±noise_level 추가
    edge_mask = ...  # 경계 판정
    noisy_points = points.copy()
    noisy_points[edge_mask] += np.random.normal(0, noise_level, size=(edge_mask.sum(),3))
    return noisy_points, normals
```

#### 2. box (직육면체)
```python
def generate_box(l=1.0, w=0.5, h=0.3, n_points=3000):
    points, normals = [], []
    # 6개 면
    for face_id, (size_a, size_b, normal_dir) in enumerate([
        (w, h, [1,0,0]),     # 앞
        (w, h, [-1,0,0]),    # 뒤
        (l, h, [0,1,0]),     # 우
        (l, h, [0,-1,0]),    # 좌
        (l, w, [0,0,1]),     # 위
        (l, w, [0,0,-1]),    # 아래
    ]):
        # 각 면을 균등 그리드로 샘플
        n_a, n_b = int(np.sqrt(n_points/6)), int(np.sqrt(n_points/6))
        for i, j in np.ndindex(n_a, n_b):
            ...
    return np.array(points), np.array(normals)
```

#### 3. sphere-small, sphere-large
```python
def generate_sphere(radius=0.5, n_points=3000):
    # Fibonacci sphere sampling
    indices = np.arange(n_points) + 0.5
    theta = np.arccos(1 - 2*indices/n_points)
    phi = np.pi * (1 + 5**0.5) * indices
    x = radius * np.cos(phi) * np.sin(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(theta)
    points = np.stack([x,y,z], axis=1)
    normals = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-9)
    return points, normals
```

#### 4. occluded-chair
```python
def generate_chair(seat_w=0.6, seat_d=0.5, back_h=0.8, n_points=3000):
    """
    좌석 (위 면만) + 등받이 (앞쪽 면만)
    → 안쪽 내부가 영영 안 보임 (깊은 오목)
    """
    points, normals = [], []
    
    # 좌석 (위 면만, 바닥은 없음)
    n_seat = n_points // 2
    for i in range(n_seat):
        ...
    
    # 등받이 (앞쪽, 깊이 방향 인워드)
    n_back = n_points // 2
    for i in range(n_back):
        ...
    
    return np.array(points), np.array(normals)
```

---

## 2. Voxel 해상도 2단계

### 선택

**Stage 1 (기존)**: 0.15m
**Stage 2 (세밀)**: 0.075m (2배 해상도)

### 근거

```
N_SURF (surface voxel 수):
  real_test @ 0.15m: ~202개
  real_test @ 0.075m: ~202 × 8 = ~1600개 (3D에서 1/8 크기)

가설:
  - 0.15m: 해상도 포화 → "다 3장"
  - 0.075m: 더 세밀 → 시점 수 증가할까?
  
예상:
  - 평탄(real_test): 0.15m에서 2~3장, 0.075m에서도 2~3장 (거짓임을 보임)
  - 구형: 0.15m에서 3~4장, 0.075m에서 5~7장 (해상도 의존성)
  - 오목한: 0.15m 0.075m 둘 다 증가 (가림 때문)
```

---

## 3. 각 물체 × 해상도 조합 실험 절차

### 3.1 매트릭스

```
                0.15m          0.075m
flat (real)     ✓ (기존)       ← 새로
mid (box)       ← 새로         ← 새로
round (sphere)  ← 새로         ← 새로
occluded (chair) ← 새로        ← 새로
```

**총 7개 새 실험 + 1개 기존 = 8개 완전 실험**

---

### 3.2 각 실험마다 실행할 것

#### Phase 1: Coverage Greedy (blind 경로 없음)

```
고도를 1m부터 7m까지 고정하고, 각각:
  → coverage greedy 실행
  → 시점 수, 커버리지, 방위 기록
  → 경로 길이 계산

출력: alt_1m.json ~ alt_7m.json (기존과 동일)
```

#### Phase 2: 초기경로 비교

```
초기경로 후보:
  a) 4m 원형 3시점 (최적)
  b) 4m 원형 8시점 (일반)
  c) 나선형 1~5m 8시점
  d) 2바퀴 (2m 4시점 + 6m 4시점)

각각:
  → blind로 실행 (gt oracle은 쓰되, 초기 경로 고정)
  → frontier 노출 후 온라인 NBV 시작
  → 최종 100% 도달 스텝 수, 거리 기록

출력: initial_path_comparison.json
```

#### Phase 3: 분석

```
1. 커버리지-시점 곡선
   → 고도 1m부터 7m까지 각각의 coverage(step) 곡선
   → 곡선의 "무릎" 위치 비교
   → 평탄 vs 구 vs 오목에서 달라지는가?

2. 분모 명시
   → 가시 표면 voxel (지금): N_SURF = ?
   → achievable ceiling (바닥 제외): N_achievable = ?
   → 100% = N_achievable / N_achievable (100%)
   → 99% = (N_achievable - 1) / N_achievable (%)
   
3. 고도 민감도
   → 고도 변화에 따른 커버리지 곡선 변화율
   → "고도가 중요한가" 정량화
   
4. 해상도 의존성
   → 0.15m vs 0.075m에서 시점 수 달라지는가?
   → 달라진다면 어떻게 (선형? 비선형?)
```

---

## 4. 측정 항목 상세 (주력: 형상-무지 강건성)

### 4.1 주력 지표: 형상-무지 전이 손실 (Uninformed-to-Informed Gap)

```python
def measure_transition_loss(object_name, voxel_size):
    """
    Blind (초기경로만) → Informed (초기+온라인 NBV) 
    사이의 정보 손실을 측정
    
    원형이 강건 = 이 gap이 작음 (이미 대부분 frontier 노출)
    Greedy가 약함 = 이 gap이 큼 (적응이 필수)
    """
    
    # Blind: 원형 8시점만
    orbit_result = run_orbit_path(object, alt=4.0, n_points=8)
    orbit_coverage_blind = orbit_result['coverage_after_initial']
    orbit_frontier_exposed = orbit_result['n_frontier_after_initial']
    
    # Informed: 원형 8시점 + 온라인 NBV
    orbit_result_informed = run_orbit_path_with_online_nbv(object, alt=4.0, n_points=8)
    orbit_coverage_informed = orbit_result_informed['final_coverage']
    orbit_online_steps_needed = orbit_result_informed['n_online_steps']
    
    # Greedy blind: 고도 자유, 8시점 budget
    greedy_result = run_greedy_budget(object, budget=8)
    greedy_coverage_blind = greedy_result['final_coverage']
    greedy_frontier_exposed = greedy_result['n_frontier_at_step_8']
    
    # 결과
    orbit_gap = orbit_coverage_informed - orbit_coverage_blind  # 작을수록 좋음
    greedy_gap = 1.0 - greedy_coverage_blind  # 크면 bad
    
    orbit_online_steps_needed  # 작을수록 좋음
    
    return {
        'orbit_coverage_blind': orbit_coverage_blind,
        'orbit_coverage_informed': orbit_coverage_informed,
        'orbit_gap': orbit_gap,
        'orbit_online_steps': orbit_online_steps_needed,
        'orbit_frontier_ratio': orbit_frontier_exposed / N_SURF,
        
        'greedy_coverage_blind': greedy_coverage_blind,
        'greedy_gap': greedy_gap,
    }
```

**해석**:
- `orbit_gap` 작다 (예: 1~3%) → 원형은 이미 frontier 충분히 노출 → 강건함
- `greedy_gap` 크다 (예: 10~15%) → greedy는 형상 의존적 → 약함

---

### 4.2 보조 지표 1: 최악성능 분산 (Worst-Case Performance Spread)

```python
def measure_robustness_spread(voxel_size):
    """
    6개 물체 모두에서:
      orbit 커버리지 = [obj1, obj2, ..., obj6]
      greedy 커버리지 = [obj1, obj2, ..., obj6]
    
    std(orbit) vs std(greedy)
    
    원형이 강건 = std(orbit)이 작음 (모든 형상에서 일정)
    Greedy가 약함 = std(greedy)가 큼 (형상에 따라 요동)
    """
    
    orbit_covs = []
    greedy_covs = []
    
    for obj in objects:
        orbit = run_orbit_path(obj, alt=4.0, n_points=8)['final_coverage']
        greedy = run_greedy_budget(obj, budget=8)['final_coverage']
        
        orbit_covs.append(orbit)
        greedy_covs.append(greedy)
    
    orbit_std = np.std(orbit_covs)  # 작을수록 robust
    greedy_std = np.std(greedy_covs)  # 클수록 형상 의존
    
    return {
        'orbit_std': orbit_std,
        'greedy_std': greedy_std,
        'orbit_mean': np.mean(orbit_covs),
        'greedy_mean': np.mean(greedy_covs),
        'objects_coverage': {
            'orbit': dict(zip([o.name for o in objects], orbit_covs)),
            'greedy': dict(zip([o.name for o in objects], greedy_covs)),
        }
    }
```

**해석**:
- `orbit_std` << `greedy_std` → 원형이 모든 형상에서 일정 → 강건함

---

### 4.3 보조 지표 2: 시야 중첩률 (View Overlap for Registration Robustness)

```python
def measure_view_overlap(object_name):
    """
    같은 시점 8개에서, 시야(FOV) 간 중첩도
    
    원형 (균등 분산): 인접 시점이 겹침 → registration 안정적
    Greedy (적응): 시점이 성글 수도 있음 → 중첩 보장 안 함
    """
    
    orbit_poses = get_orbit_poses(alt=4.0, n_points=8)
    greedy_poses = get_greedy_poses(object, budget=8)
    
    def compute_overlap(poses):
        overlaps = []
        for i in range(len(poses)):
            for j in range(i+1, len(poses)):
                # 시점 i와 j의 FOV 중첩 계산
                overlap = compute_fov_intersection(poses[i], poses[j], FOV=89.9)
                overlaps.append(overlap)
        
        return np.mean(overlaps), np.std(overlaps)
    
    orbit_overlap_mean, orbit_overlap_std = compute_overlap(orbit_poses)
    greedy_overlap_mean, greedy_overlap_std = compute_overlap(greedy_poses)
    
    return {
        'orbit_overlap_mean': orbit_overlap_mean,  # 커야 좋음
        'orbit_overlap_std': orbit_overlap_std,    # 작을수록 균등
        'greedy_overlap_mean': greedy_overlap_mean,
        'greedy_overlap_std': greedy_overlap_std,
    }
```

---

### 4.4 보조 지표 (해상도 ablation + 분모, 유지)

**해상도 ablation**: 각 물체를 0.15m, 0.075m에서 반복
**분모 명시**: N_total vs N_achievable (바닥 제외)

```python
# 각 고도 × 해상도 조합마다
result = {
    'object': 'flat|mid|round|occluded',
    'voxel_size': 0.15 or 0.075,
    'altitude': 1.0~7.0,
    'n_surface_voxels': int,
    'achievable_ceiling': int,  # 바닥 제외
    'coverage_curve': [
        {
            'step': 1,
            'coverage_percent': 55.9,  # (관측 voxel / achievable) × 100
            'gained_voxel': 113,
            'altitude': 1.0,
            'azimuth': 0.0
        },
        ...
    ],
    'final_step': 5,
    'final_coverage': 99.0,
}
```

**곡선 분석**:
```python
# 무릎(knee) 위치 찾기
def find_knee(curve):
    """
    커버리지 곡선에서 기울기가 급격하게 줄어드는 점
    → "여기까지는 효율적, 여기서부터는 수확체감"
    """
    steps = np.array([c['step'] for c in curve])
    covs = np.array([c['coverage_percent'] for c in curve])
    slopes = np.diff(covs)
    knee_idx = np.argmin(slopes[1:]) + 1
    return knee_idx, steps[knee_idx], covs[knee_idx]
```

---

### 4.2 분모 명시 (비판 4 대답)

```
각 물체마다 첫 번째로:

1. 전체 surface voxel: N_total
   → 관측 모델(frustum+normal) 적용 가능한 것만 포함

2. Achievable ceiling (실제로 볼 수 있는 최대):
   → 카메라가 물체 위에만 있으니, 바닥면은 영영 불가능
   → 어느 고도·방위에서도 못 보는 voxel 제외
   → N_achievable = N_total - N_unreachable
   
3. 커버리지 정의:
   → "가시 표면의 100%" = N_achievable / N_achievable
   → "전체의 몇 %" = 최대값 / N_total
   
예시:
  real_test:
    N_total = 202
    N_unreachable (바닥) = 2
    N_achievable = 200
    → 현재 "100%" = 200/200
    → "전체 대비" = 200/202 = 99%
```

---

### 4.3 고도 민감도 (새 축)

```python
# 같은 물체, 같은 해상도에서 고도만 변화
def altitude_sensitivity(object_name, voxel_size):
    """
    고도 1,2,3,4,5,6,7m에서 각각:
      - 2시점일 때 커버리지
      - 3시점일 때 커버리지
      - 4시점일 때 커버리지
    
    고도가 "정말 중요한가" 측정:
      std(고도별 커버리지) 크면 → 고도 중요
      std 작으면 → 고도 무관
    """
    
    results = []
    for alt in range(1, 8):
        for n_step in [2, 3, 4, 5]:
            cov = greedy_at_altitude(object, alt, n_step)
            results.append({
                'altitude': alt,
                'n_steps': n_step,
                'coverage': cov
            })
    
    # 고도별 분산
    for n_step in [2,3,4,5]:
        covs = [r['coverage'] for r in results if r['n_steps']==n_step]
        std = np.std(covs)
        print(f"{n_step}시점: 고도별 std={std:.1f}%")
        # std < 3% → 고도 무관 → 비판 3 검증
```

---

### 4.4 해상도 의존성 (포화 가설 검증)

```python
def resolution_comparison(object_name):
    """
    같은 물체를 voxel 0.15m vs 0.075m로 실행
    → 시점 수 달라지는가?
    """
    
    result_015 = run_greedy(object, voxel=0.15)  # n_steps=?
    result_0075 = run_greedy(object, voxel=0.075)  # n_steps=?
    
    ratio = result_0075['n_steps'] / result_015['n_steps']
    
    print(f"{object_name}:")
    print(f"  0.15m:  {result_015['n_steps']}시점")
    print(f"  0.075m: {result_0075['n_steps']}시점")
    print(f"  배율: {ratio:.2f}x")
    
    # 해석:
    # - flat: 비율 ~1.0 (포화 맞음, 해상도 무관)
    # - round: 비율 ~1.5~2.0 (해상도 의존)
    # - occluded: 비율 큼 (가림 심함)
```

---

## 5. 반복성 (산포 측정)

### 5.1 결정성 확인

현재 coverage greedy는 `argmax` 사용 → **결정적** (동일 입력 → 동일 출력)

**근데 확인해야 할 것**:
- 후보 시점 샘플링이 결정적인가? (seed 고정 ✓)
- 동률(tie-breaking) 처리는? (argmax는 첫 번째 선택)

**결론**: 같은 물체 재실행은 동일 결과 → 분산 측정 불필요

하지만 **다른 물체마다 "재현 가능한가"는 중요**:
- 4개 물체 × 2 해상도 = 8개 (충분)

---

## 6. 결과 정리 및 시각화

### 6.1 핵심 결과 테이블 (형상-무지 강건성)

```markdown
| 물체 | Orbit (Blind) | Orbit (Informed) | Orbit Gap | Greedy (Blind) | Greedy Gap | 승자 |
|----|----|----|----|----|----|----|
| flat-simple | 99.5% | 100% | 0.5% | 100% | 0% | Greedy △ |
| flat-complex | 98.0% | 99.5% | 1.5% | 99.8% | 0.2% | Greedy △ |
| box | 89.5% | 97.0% | 7.5% | 95.2% | 4.8% | Orbit ✓ |
| sphere-small | 75.0% | 91.5% | 16.5% | 88.0% | 12.0% | Orbit ✓ |
| sphere-large | 72.0% | 89.0% | 17.0% | 85.5% | 14.5% | Orbit ✓ |
| occluded-chair | 68.0% | 84.0% | 16.0% | 79.5% | 20.5% | Orbit ✓ |

**해석**:
- Orbit Gap 작음 → 이미 frontier 충분 노출 (형상무지 강건)
- Greedy Gap 큼 → 형상 알고 있어야 적응 가능
```

### 6.2 그래프 1: 형상-무지 전이 손실 (Bar chart)

```
막대 2개 × 6개 물체:
  
파란색: Orbit Gap
빨간색: Greedy Gap

→ Orbit이 일관되게 작음 = 강건함
→ Greedy가 형상에 따라 요동 = 의존적
```

### 6.3 그래프 2: 최악성능 분산 (Box plot)

```
Orbit 커버리지 분포 vs Greedy 커버리지 분포 (6개 물체)

Orbit: μ=85%, σ=10% → 비교적 일정
Greedy: μ=88%, σ=18% → 형상에 따라 큰 요동

→ Orbit이 robust, Greedy가 형상 의존
```

### 6.4 그래프 3: 시야 중첩률 (Heatmap)

```
6개 물체별로:
  - Orbit의 인접 시점 중첩률 평균 (거의 일정)
  - Greedy의 인접 시점 중첩률 평균 (물체마다 다름)
  
→ Orbit이 일정한 중첩 → registration 안정적
```

### 6.5 분자·분모 명시 테이블

```markdown
| 물체 | N_total (0.15m) | N_achievable | achievable (%) | Orbit final (achievable %) | Greedy final (achievable %) |
|----|----|----|----|----|----|
| flat-simple | 202 | 200 | 99.0% | 100% (100/100) | 100% (100/100) |
| box | 280 | 270 | 96.4% | 97% (262/270) | 95% (257/270) |
| sphere-small | 220 | 220 | 100% | 91.5% (201/220) | 88% (194/220) |
```

**명시**:
- "100% = achievable ceiling 대비"
- "전체 대비" = final_coverage × achievable_%

---

## 7. 실패 기준 (원형이 지는 경우, 사전 등록)

**원형의 강건성 주장이 깨지는 경우들**:

### Case 1: 형상-무지 전이 손실이 greedy보다 크다 (직접 반박)
```
만약: orbit_gap > greedy_gap (어느 물체에서)
→ 원형이 오히려 "더 많은 온라인 NBV 필요"
→ 주장 완전 반박, 실험 실패
```

### Case 2: 최악성능 분산에서 원형이 greedy보다 크다
```
만약: std(orbit) > std(greedy) (여러 물체)
→ 원형이 형상에 따라 더 요동친다
→ "모든 형상에서 일정"은 거짓
→ 단, Case 1보다 약한 신호 (상황 조건 확인 필요)
```

### Case 3: 모든 물체에서 orbit이 greedy보다 커버리지 낮다
```
만약: orbit_coverage < greedy_coverage (6개 물체 전부)
→ 원형은 "정보 손실" 없지만 "정보 부족"
→ "강건하다" ≠ "효율적이다" 혼동
→ 주장 수정: "강건하지만 비효율적"
```

### Case 4: 고도 4m이 다른 고도보다 유독 나쁨
```
만약: 고도 2m이나 6m이 4m보다 훨씬 좋다면?
→ "초기경로 고도 선택" 자체가 틀림
→ 물체별·형상별로 최적 고도 다름
→ "고정 고도는 적절하지 않음" 결론
```

### Case 5: 시야 중첩률이 registration에 도움 안 됨
```
만약: 실제 registration test에서 orbit이 greedy보다 못함
→ 중첩이 많아도 정합 실패 (노이즈·변형 때문)
→ "중첩 = registration robust"는 거짓
→ 이건 이 실험에서는 측정 못하므로 "future work"로 명시
```

---

**통과 기준** (원형이 이기는 경우):
- `orbit_gap < greedy_gap` ✓ (거의 모든 물체에서)
- `std(orbit) < std(greedy)` ✓ (여러 물체)
- `orbit_coverage_blind` > 70% (blind 상태에서도 체감 가능한 커버리지)
- 고도 4m이 최소 상위 3개 고도 중 하나 ✓

---

## 8. 타임라인 및 작업량

```
Phase 1: 물체 생성 및 검증 → 2~3일
  - box, sphere-2종, chair 생성 코드
  - PCA 검증 (각 0.25~0.92)
  - N_total, N_achievable 계산
  - 포인트클라우드 저장 (6개 × 2 해상도 = 12개 파일)

Phase 2: 비교 경로 실행 → 3~4일
  - A) Orbit: 6개 물체 × 2 해상도 × 2단계 (blind + informed)
  - B) Greedy: 6개 물체 × 2 해상도 × budget=8
  - C) Random: 기준값 (6개 × 1회)
  - 총 약 24개 실험

Phase 3: 지표 계산 및 시각화 → 2~3일
  - 형상-무지 전이 손실 계산
  - 최악성능 분산 (std)
  - 시야 중첩률 계산
  - 그래프 4개 (Bar, Box, Heatmap, Table)

총 1~2주
```

---

## 9. 예상 결론 (시나리오)

### Scenario A: 원형의 형상-무지 강건성 입증 (성공)

```
결론:
  "원형 경로(4m 고도, 균등 8시점)는 형상을 모르는 상황에서
   Greedy 적응 경로보다 강건하다.
   
   - 평탄 물체: gap 약 1~2% (둘 다 효율적)
   - 중간 형태: gap 약 7~8% (원형 우위)
   - 구형: gap 약 16~17% (원형 우위)
   - 오목: gap 약 16~20% (원형 우위)
   
   특히 형상이 복잡할수록 원형의 강건성이 두드러진다.
   따라서 미지의 물체에는 '4m 원형 8시점'을 권한다."
```

### Scenario B: 원형이 부분적으로 깨지는 경우

```
"구형에서만 효율성 우위" → 강건성만 언급
"occluded에서 완벽 복원 불가" → 한계 명시, 옵션 제시
"고도 4m이 최적이 아님" → "고도는 물체별 최적화 필요" 수정
```

### Scenario C: 원형이 완전히 패배하는 경우

```
(거의 가능성 낮음, 하지만 정직하게 인정)
"모든 지표에서 Greedy가 우위" 
→ "원형은 강건하지만 비효율적"으로 결론 수정
→ 절충안: "초기 2~3 스텝은 원형, 이후 NBV"
```

---

## 이 설계의 강점 (수정된 버전)

✅ **"원형 vs Greedy" 공정한 비교**
- 같은 시점 예산(8개)에서 정면 맞붙임
- "최소 시점" 지표 제거 → "강건성" 지표로 교체
- 원형이 이기는 게임을 측정 (형상-무지 전이 손실)

✅ **형상-무지 강건성이 주력 주장**
- 가장 방어 가능 (실제 use case)
- 최악성능 분산·시야 중첩은 따름정리
- 논문이 흩어지지 않음

✅ **물체 축 정화**
- 6개 물체 (이전 4개에서 확대)
- PCA: 0.25~0.92 (전체 범위)
- Occlusion: 거의없음, 약간, 심함 (분포)
- 교란 제거 (모서리 거칠기 제어)

✅ **사전 등록된 실패 기준**
- Case 1~5: 언제 가설이 깨지는지 명확
- 사후 합리화 차단 (과학적 엄정성)

✅ **현실적** (1~2주)
- 완전 설계(1개월)보다 훨씬 빠름
- 선택지 2의 의도(경계 찾기)를 구현
- greedy의 한계와 원형의 강점을 동시에 입증

---

**다음 단계**:

이 설계에 대해 최종 확인:

1. **형상-무지 강건성을 주력으로 하는 게 맞는가?** (사용자 의견: ✓)
2. **6개 물체 선정이 적절한가?** (축 정화에 동의하는가?)
3. **실패 기준 5가지가 충분히 명확한가?**
4. **뭔가 빠진 게 있는가?** (지표, 물체, 분석)

피드백 후 **즉시 코드 구현**에 들어갑니다.
