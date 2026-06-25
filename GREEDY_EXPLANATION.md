# Greedy 경로 선택 방식 상세 설명

## 1️⃣ 개념 (일반적)

**Greedy** = "욕심쟁이" 알고리즘
- 매 단계에서 **지금 당장 최고 이득**을 주는 선택을 함
- 전체 최적해를 보장하지 않지만, 빠르고 합리적인 해를 찾음

**예시**:
```
지금 시점 1에서 본 것: 50개 voxel
지금 시점 2에서 본 것 (새로 추가): 40개 voxel
지금 시점 3에서 본 것 (새로 추가): 30개 voxel

Greedy는 지금 당장 "가장 많이 보는" 시점 1을 선택
→ 그 다음 단계에서 또 "가장 많이 새로 보는" 시점을 선택
```

---

## 2️⃣ 우리 실험에서 Greedy의 작동

### 입력
```python
voxel_centers = 202개 surface voxel 위치
voxel_normals = 각 voxel의 법선 방향
CANDIDATES = 999개 후보 시점 (999개 위치)
n_budget = N (시점 개수, 4/6/8/12/16)
```

### 알고리즘 (의사코드)

```python
Step 1: 모든 999개 후보 시점에서 각각 "뭘 볼 수 있는지" 미리 계산
  for each candidate in 999개:
    vis_mask[candidate] = boolean array (202개 voxel)
    # True = 이 시점에서 이 voxel을 볼 수 있음
    # False = 볼 수 없음

Step 2: N번 반복 (N=8이면 8번)
  covered = 아직 관측 안 된 voxel 집합 (처음엔 202개 모두)
  
  for i in range(N):
    # 각 후보의 "새로 얻을 이득" 계산
    for each candidate in 999개:
      gain[candidate] = (vis_mask[candidate] AND NOT covered).sum()
      # = 이 시점이 "아직 안 본 voxel" 중 몇 개를 새로 볼 수 있나?
    
    # 최고 이득의 시점 선택
    best = argmax(gain)  # 가장 이득이 큰 시점
    selected.append(CANDIDATES[best])
    
    # 관측 집합 갱신
    covered = covered OR vis_mask[best]  # 이제 이 voxel들도 "봤음"

return selected (N개 시점)
```

---

## 3️⃣ 구체적 예시 (flat-simple, N=8)

### 상황
- surface voxel: 202개
- 후보 시점: 999개

### 동작

**1번째 시점 선택**:
```
후보 1: 새로 볼 수 있는 voxel = 70개
후보 2: 새로 볼 수 있는 voxel = 120개 ← 최고 (선택!)
후보 3: 새로 볼 수 있는 voxel = 95개
...
후보 999: 새로 볼 수 있는 voxel = 45개

→ 후보 2 선택. 이제 covered = 120개
```

**2번째 시점 선택** (다시 계산, 이번엔 이미 120개는 "봤음"):
```
후보 1: 새로 추가로 볼 수 있는 voxel = 65개 - 20개(중복) = 45개
후보 2: 새로 추가로 볼 수 있는 voxel = 115개 - 90개(중복) = 25개
후보 3: 새로 추가로 볼 수 있는 voxel = 88개 - 50개(중복) = 38개
후보 5: 새로 추가로 볼 수 있는 voxel = 105개 - 80개(중복) = 25개

→ 후보 1 선택 (45개 새로 추가). 이제 covered = 165개
```

**3~8번째도 같은 방식** (매번 새로 추가로 볼 수 있는 voxel 가장 많은 곳 선택)

---

## 4️⃣ 이게 왜 "Oracle"이고 "상한값"인가?

### Oracle (신탁, 전지전능한 존재)
```
Greedy는 모든 202개 voxel의 위치를 미리 알고 있음:
  "이 voxel이 어디에 있고, 어떤 시점에서 봐야 하는지"

현실에서는:
  드론이 비행하면서 "아직 못 본 부분"을 찾아야 함
  → Online NBV (online Next-Best-View)
  → 하나하나 실제로 본 후 다음을 결정

Greedy는:
  "너는 GT(지상진실)를 알고 있으니, 완벽하게 플래닝해봐"
  → 따라서 현실보다 훨씬 좋은 성능을 낼 수 있음
```

### 상한값 (Upper Bound)
```
어떤 경로 선택 방식이든, 절대 Greedy를 이길 수 없다
(GT를 모르므로)

예시 (N=8, flat-simple):
  Orbit:   100.0%  (사전 정의)
  Greedy: 100.0%   (oracle로 최적 선택)
  Random:  100.0%  (운 좋게 다 봄)

이 경우 Greedy = 최고 달성 가능한 점수 (상한)
```

---

## 5️⃣ 왜 Sphere에서 Greedy가 잘하는가?

### Sphere-small (반지름 0.5m)

**Orbit 전략** (균등 원형):
```
8개 시점이 고르게 원형으로 배치
→ 어느 각도에서 본다는 보장 없음
→ 일부 영역은 역광(back-facing)이 될 수 있음
→ 커버리지 ~88%
```

**Greedy 전략** (oracle 최적화):
```
"202개 surface voxel을 보려면, 어디서 봐야 할까?"
→ 각 voxel의 법선 방향을 알고 있음
→ 법선이 카메라를 바라보는 위치들을 선택
→ 겹침을 최소화하면서 모든 voxel 보기
→ 커버리지 ~99.5%
```

**차이점**:
```
Sphere 표면은 모든 방향으로 법선이 바깥을 향함
→ 정보가 풍부함 (어디서든 뭔가를 볼 수 있음)
→ Greedy가 이 정보를 활용해 최적화 가능
→ +10.6%p 이득

Box나 Chair 같은 평탄 물체:
→ 어디서나 보이는 표면이 정해져 있음
→ Greedy도 Orbit도 "봐야 할 부분"은 같음
→ 추가 최적화의 여지가 없음
→ 0% 차이
```

---

## 6️⃣ 수식으로 보기

### Greedy의 set-cover 목표

```
목표: N개 시점으로 최대한 많은 voxel을 보기

수식:
  C = coverage(N) = |union of all observed voxels| / |total voxels|

Greedy는 각 단계에서:
  p_i = argmax_p ( |V_p - C_{i-1}| )
  # p_i = i번째 선택할 시점
  # V_p = 시점 p에서 볼 수 있는 voxel 집합
  # C_{i-1} = (i-1)단계까지 본 voxel의 합집합
  # |V_p - C_{i-1}| = 새로 추가할 수 있는 voxel 수

  C_i = C_{i-1} union V_{p_i}
```

### Orbit과의 비교

```
Orbit: p_i = TARGET + RADIUS * (cos(2πi/N), sin(2πi/N), -ALT)
       → 위치는 고정, 수학식으로 정의
       → 물체의 형상을 모름

Greedy: p_i = argmax ( C_i가 커지도록 하는 위치 )
        → 위치를 동적으로 선택
        → 물체의 모든 voxel을 알고 있음 (oracle)
```

---

## 7️⃣ 왜 우리 실험에서 Greedy는 "현실적이지 않은가"?

### 1. Oracle 정보 필요
```python
# Greedy는 이걸 알고 있음:
for i, point in enumerate(point_cloud):
    x, y, z = point
    nx, ny, nz = normal[i]
    # ↑ 모든 점의 위치와 법선을 미리 알고 있음!
```

### 2. 현실에서는
```
드론이 처음 본 부분 → "여기서 뭘 못 봤나?"
→ 다음 시점 선택 (online NBV)
→ 또 본다
→ 또 결정
→ ...

온라인이므로, 아직 못 본 영역이 뭔지 모름
```

### 3. 따라서 Greedy는
```
"이론적 상한값" (theoretical upper bound)
"최고로 잘했을 때" (best case scenario)
"GT를 완벽히 아는 경우" (perfect information)
```

---

## 📊 세 경로의 관계

```
상한값:    Greedy (oracle, 전지전능)
         ↓ 현실 gap
현실:     Online NBV (드론이 실제 비행)
         ↓ 강건성 문제 (형상 미지)
盲目적:   Orbit (사전 정의, 정보 없음)
         ↓ 
기저선:   Random (무작위)
```

---

## 🎯 우리 실험에서의 의미

### Greedy가 이긴 이유 (sphere의 +10.6%)

```
1. Sphere는 정보 풍부: 모든 점이 관측 가능
2. Greedy는 oracle: 어디가 최적인지 알고 있음
3. 결과: 완벽한 최적화

→ 이건 "Greedy가 좋다"가 아니라
  "정보가 있으면 최적화할 여지가 많다"를 보여줌
```

### Greedy가 못 이긴 이유 (box/chair의 0%)

```
1. Box/Chair는 정보 부족: 법선 방향이 정해져 있음
2. Greedy도 "봐야 할 부분"을 다 봐야 함
3. 추가 최적화 가능성 = 거의 0

→ "형상이 단순하면 경로 선택도 덜 중요"
```

---

## 💡 결론

**Greedy = "모든 정보를 아는 최고 선택권"**

- **구형 물체에서 강함**: 많은 정보 활용 가능
- **평탄/가려진 물체에서 약함**: 활용할 정보 부족
- **현실 적용 불가**: Oracle 정보 불가능

실험에서 Greedy를 사용한 이유:
```
"Orbit이 정말 강건한지 보려면,
가장 강한 상대(Greedy)와 비교해야 한다"

만약 Greedy도 못 이기면,
정말로 Orbit이 강건한 것이다.

하지만 실제로는:
- 4개 물체: 동률 (비교 불가)
- 2개 물체: Greedy 우위 (Orbit 약함)

→ Orbit의 강건성 입증 실패
```
