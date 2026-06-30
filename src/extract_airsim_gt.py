"""
extract_airsim_gt.py — AirSim에서 카메라 독립 GT 점군 추출 (정직한 커버리지 실험용)

배경:
  지금까지 쓴 점군은 MASt3R-SfM 출력 = "카메라가 본 것"이라 커버리지 평가가 자기참조였음.
  AirSim 시뮬레이터에는 물체의 실제 메시(GT)가 있으므로, 거기서 직접 뽑으면
  카메라와 무관한 ground truth가 된다.

핵심 API:
  client.simGetMeshPositionVertexBuffers()
    → 씬 모든 물체의 메시 정점을 'AirSim NED 월드좌표(미터)'로 반환.
    → meta/*.json 카메라 pose와 같은 좌표계 → 추가 변환 없이 정렬됨.

실행 (★ AirSim 시뮬레이터가 켜진, 이 물체가 있는 환경에서):
  pip install airsim          # 또는 cosysairsim (cosys-airsim 포크면)
  python src/extract_airsim_gt.py

출력:
  real_test/airsim_gt_pts.npz   (points, [tri_indices])
  → 이후 이걸 GT로 궤도/NBV 커버리지를 정직하게 재평가.
"""
import sys, json
from pathlib import Path
import numpy as np

try:
    import airsim
except ImportError:
    try:
        import cosysairsim as airsim
    except ImportError:
        sys.exit("airsim 패키지 필요:  pip install airsim   (또는 cosysairsim)")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "real_test" / "airsim_gt_pts.npz"

# 물체가 있는 월드 영역 (SfM 점군 범위 + 여유). NED 미터.
#  SfM 원본 범위: X[-34.5,-32.8] Y[-54.5,-50.0] Z[-3.85,-0.02]
BBOX_MIN = np.array([-36.0, -56.0, -5.0])
BBOX_MAX = np.array([-31.0, -48.0,  1.5])

# 타겟 물체 이름 필터 (정규식). 모르면 None → 전체에서 BBOX로만 자름.
NAME_REGEX = None        # 예: "Slab.*" 또는 "StaticMeshActor.*"


def main():
    client = airsim.VehicleClient()
    client.confirmConnection()

    # 1) 씬 물체 목록 — 타겟 이름 파악용
    print("[1] 씬 물체 목록 (BBOX 근처일 가능성 높은 것 위주로 확인):")
    try:
        names = client.simListSceneObjects()
        print(f"    총 {len(names)}개. 처음 40개:")
        for n in names[:40]:
            print("     ", n)
    except Exception as e:
        print("    simListSceneObjects 실패:", e)

    # 2) 메시 정점 버퍼 (NED 미터)
    print("[2] simGetMeshPositionVertexBuffers() 호출...")
    bufs = client.simGetMeshPositionVertexBuffers()
    print(f"    메시 {len(bufs)}개 수신")

    import re
    pat = re.compile(NAME_REGEX) if NAME_REGEX else None

    all_pts, all_names = [], []
    for b in bufs:
        name = getattr(b, "name", "")
        if pat and not pat.search(name):
            continue
        v = np.array(b.vertices, dtype=np.float64).reshape(-1, 3)
        if len(v) == 0:
            continue
        # BBOX 필터
        m = np.all((v >= BBOX_MIN) & (v <= BBOX_MAX), axis=1)
        if m.sum() == 0:
            continue
        all_pts.append(v[m])
        all_names.append((name, int(m.sum()), int(len(v))))

    if not all_pts:
        print("    [경고] BBOX 안에 메시 정점 없음. BBOX_MIN/MAX 또는 NAME_REGEX 조정 필요.")
        print("    팁: 위 물체목록에서 타겟 이름 찾아 NAME_REGEX 지정, 또는 BBOX 넓혀서 재실행.")
        return

    pts = np.vstack(all_pts)
    print(f"[3] BBOX 내 GT 정점 {len(pts):,}개  (기여 메시 {len(all_names)}개):")
    for nm, k, tot in sorted(all_names, key=lambda x: -x[1])[:15]:
        print(f"     {nm}: {k}/{tot}")
    print(f"    범위: X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
          f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

    np.savez(OUT, points=pts.astype(np.float32))
    print(f"\n✓ 저장: {OUT}")
    print("  다음: 이 GT로 궤도/NBV 커버리지를 재평가 (eval_against_gt.py).")
    print("  먼저 SfM 점군과 겹쳐보고 정렬·스케일이 맞는지 육안 확인 권장.")


if __name__ == "__main__":
    main()
