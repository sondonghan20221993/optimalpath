# drone_runner_cpp

`pb_nbv`(외부: dspangpang/pb_nbv) 및 `pb_nbv_minimal`(로컬 고아 repo)에 흩어져 있던
**본인 작성 C++/ROS 작업물 + 드론 실험 데이터**를 백업·버전관리를 위해 이곳으로 흡수한 것.

> 외부 원본(pb_core/see_common/scvp_core 등 dspangpang 코드)은 포함하지 않는다.
> 그쪽은 https://github.com/dspangpang/pb_nbv 에서 재clone 가능.

## 구성

| 경로 | 출처 | 내용 |
|---|---|---|
| `pb_nbv/drone_runner/` | pb_nbv repo (untracked 본인 추가분) | ROS drone_runner 노드 (8K판) |
| `pb_nbv/Dockerfile` | pb_nbv repo | ROS noetic + 드론 의존성 컨테이너 |
| `pb_nbv/ground_first.pcd` | pb_nbv repo | 포인트클라우드 |
| `pb_nbv_minimal/drone_runner/` | pb_nbv_minimal | ROS drone_runner 노드 (16K판, pb_nbv판과 **상이**) |
| `pb_nbv_minimal/*.pcd` | pb_nbv_minimal | 드론 실비행 포인트클라우드 (`drone_real_3m`, `ground_first`) |
| `pb_nbv_minimal/real_test/`, `results/` | pb_nbv_minimal | 실험 입력·결과 |

## 주의
- `pb_nbv/drone_runner`와 `pb_nbv_minimal/drone_runner`는 **CMakeLists.txt / node.cpp가 서로 다름**.
  추후 어느 쪽을 정식 버전으로 둘지 정리 필요.
