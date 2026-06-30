#!/usr/bin/env python3
"""
eval_compare.py — 두 NBV 방법을 동일 조건에서 비교 평가
  A) 단순 NBV    : frustum 안 미관측 점 수 × 거리가중치 (기존 우리 구현)
  B) 논문 PB-NBV : voxel→ellipsoid 투영, F = Σfrontier − Σoccupied

동일: 후보 시점 집합, 관측 모델(observed_by), voxel coverage 기준.
평가: 스텝별 coverage curve, 완전커버 도달 스텝, 고도/방위 분포.
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pbnbv_paper as P   # 논문 구현 재사용

cands = P.make_candidates()
N = P.N_SURF
print(f"surface voxels={N}, candidates={len(cands)}")

START = np.array([-27.06,-50.51,-6.13])

def run(method):
    observed=np.zeros(N,dtype=bool); used=np.zeros(len(cands),dtype=bool)
    cur=START; path=[]; curve=[]
    for step in range(15):
        if method=='simple':
            # frustum 안 미관측 점 수 × 거리가중치
            scores=np.full(len(cands),-1e18)
            for i,c in enumerate(cands):
                if used[i]: continue
                idx=P.observed_by(c)
                new=(~observed[idx]).sum()
                d=np.linalg.norm(c-P.TARGET)
                w=(1/(d**2+1e-9))/(1/(P.MIN_DIST**2+1e-9))
                scores[i]=new*w
        else:  # paper
            fro_idx=P.compute_frontier(observed)
            occ_pts=P.SURF_CEN[observed]; occ_nrm=P.SURF_NRM[observed]
            if len(fro_idx)>0:
                fro_pts=P.SURF_CEN[fro_idx]; fro_nrm=P.SURF_NRM[fro_idx]
            else: fro_pts=np.empty((0,3)); fro_nrm=np.empty((0,3))
            occ_ell=P.fit_ellipsoids(occ_pts,occ_nrm)
            fro_ell=P.fit_ellipsoids(fro_pts,fro_nrm) if len(fro_pts)>0 else []
            if len(occ_pts)==0 and len(fro_pts)==0:
                fro_ell=P.fit_ellipsoids(P.SURF_CEN,P.SURF_NRM)
            scores=np.array([(-1e18 if used[i] else P.evaluate(c,occ_ell,fro_ell))
                             for i,c in enumerate(cands)])
        best=int(np.argmax(scores)); used[best]=True; ch=cands[best]
        gained=(~observed[P.observed_by(ch)]).sum()
        observed[P.observed_by(ch)]=True
        cov=observed.sum()/N; curve.append(cov)
        alt=P.TARGET[2]-ch[2]; az=math.degrees(math.atan2(ch[1]-P.TARGET[1],ch[0]-P.TARGET[0]))
        path.append({'alt':round(alt,1),'az':round(az,1),'gained':int(gained),'cov':round(cov,3)})
        cur=ch
        if cov>0.999 or (method!='simple' and scores[best]<=0 and gained==0): break
    return path,curve

pa,ca=run('simple')
pb,cb=run('paper')

def to95(curve):
    for i,c in enumerate(curve):
        if c>=0.95: return i+1
    return None

print("\n=== 비교 ===")
print(f"{'':14}| {'단순 NBV':>12} | {'논문 PB-NBV':>14}")
print(f"{'95% 도달스텝':14}| {str(to95(ca)):>12} | {str(to95(cb)):>14}")
print(f"{'최종 coverage':14}| {ca[-1]*100:11.1f}% | {cb[-1]*100:13.1f}%")
print(f"{'사용 스텝수':14}| {len(ca):>12} | {len(cb):>14}")

def altdist(p):
    a=[round(x['alt']) for x in p]
    return ' '.join(f"{v}m×{a.count(v)}" for v in sorted(set(a)))
print(f"{'고도분포(단순)':14}| {altdist(pa)}")
print(f"{'고도분포(논문)':14}| {altdist(pb)}")

# 그래프
fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].plot(range(1,len(ca)+1),[c*100 for c in ca],'o-',label='Simple NBV (frustum×dist)',lw=2)
ax[0].plot(range(1,len(cb)+1),[c*100 for c in cb],'s-',label='Paper PB-NBV (ellipsoid proj)',lw=2)
ax[0].axhline(95,ls='--',c='gray',alpha=0.5)
ax[0].set_xlabel('NBV step'); ax[0].set_ylabel('Surface coverage (%)')
ax[0].set_title('Coverage vs Steps'); ax[0].legend(); ax[0].grid(alpha=0.3)

# 고도-방위 산점도
for name,p,m,c in [('Simple',pa,'o','C0'),('Paper',pb,'s','C1')]:
    az=[x['az'] for x in p]; al=[x['alt'] for x in p]
    ax[1].scatter(az,al,marker=m,s=80,label=name,c=c,alpha=0.7)
    for i,(a,l) in enumerate(zip(az,al)):
        ax[1].annotate(str(i+1),(a,l),fontsize=7)
ax[1].set_xlabel('Azimuth (deg)'); ax[1].set_ylabel('Altitude (m)')
ax[1].set_title('Selected viewpoints'); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/eval_compare.png',dpi=130)
print("\n✓ results/eval_compare.png")
json.dump({'simple':{'curve':ca,'path':pa},'paper':{'curve':cb,'path':pb}},
          open('results/eval_compare.json','w'),indent=2)
