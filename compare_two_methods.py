#!/usr/bin/env python3
"""
방식A(온라인 greedy NBV) vs 방식B(배치선택+greedy순서) 비교
공통: 동일 후보, 동일 평가함수(논문 ellipsoid 투영)
"""
import json, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pbnbv_paper as P

T=P.TARGET
cands=P.make_candidates()
N_SEL=6
START=np.array([-27.06,-50.51,-6.13])

def fit_state(observed):
    fro=P.compute_frontier(observed)
    occ_e=P.fit_ellipsoids(P.SURF_CEN[observed],P.SURF_NRM[observed])
    if len(fro)>0:
        fro_e=P.fit_ellipsoids(P.SURF_CEN[fro],P.SURF_NRM[fro])
    else: fro_e=[]
    if observed.sum()==0 and len(fro)==0:
        fro_e=P.fit_ellipsoids(P.SURF_CEN,P.SURF_NRM)
    return occ_e,fro_e

def greedy_order(pts,start):
    rem=list(range(len(pts))); order=[]; cur=start
    while rem:
        i=min(rem,key=lambda k:np.linalg.norm(pts[k]-cur))
        order.append(i); cur=pts[i]; rem.remove(i)
    return order

# ── 방식A: 온라인 greedy NBV (매 스텝 갱신) ─────────────────────────────────
def run_A():
    observed=np.zeros(P.N_SURF,bool); used=np.zeros(len(cands),bool)
    chosen=[]; curve=[]; cur=START
    for _ in range(N_SEL):
        occ_e,fro_e=fit_state(observed)
        sc=np.array([(-1e18 if used[i] else P.evaluate(c,occ_e,fro_e)) for i,c in enumerate(cands)])
        b=int(np.argmax(sc)); used[b]=True; ch=cands[b]
        observed[P.observed_by(ch)]=True
        chosen.append(ch); curve.append(observed.sum()/P.N_SURF); cur=ch
    return np.array(chosen),curve

# ── 방식B: 초기 1회 스코어링 → 상위 N → greedy 순서 ────────────────────────
def run_B():
    observed0=np.zeros(P.N_SURF,bool)
    occ_e,fro_e=fit_state(observed0)        # 초기상태(전부 미관측)
    sc=np.array([P.evaluate(c,occ_e,fro_e) for c in cands])
    top=np.argsort(sc)[::-1][:N_SEL]        # 상위 N개 (비적응)
    sel=cands[top]
    order=greedy_order(sel,START)           # 순서만 greedy
    sel=sel[order]
    # 그 순서대로 관측하며 coverage
    observed=np.zeros(P.N_SURF,bool); curve=[]
    for ch in sel:
        observed[P.observed_by(ch)]=True
        curve.append(observed.sum()/P.N_SURF)
    return sel,curve

def info(pts):
    out=[]
    cur=START
    for p in pts:
        a=T[2]-p[2]; az=math.degrees(math.atan2(p[1]-T[1],p[0]-T[0]))
        out.append((round(a,1),round(az,0)))
    return out

def pathlen(pts):
    d=0; cur=START
    for p in pts: d+=np.linalg.norm(p-cur); cur=p
    return d

A,cA=run_A()
B,cB=run_B()

print("="*64)
print(f"{'':20}| {'방식A 온라인NBV':>18} | {'방식B 배치+순서':>18}")
print("-"*64)
print(f"{'최종 coverage':20}| {cA[-1]*100:17.1f}% | {cB[-1]*100:17.1f}%")
def r(c,th):
    for i,x in enumerate(c):
        if x*100>=th: return i+1
    return None
print(f"{'95% 도달스텝':20}| {str(r(cA,95)):>18} | {str(r(cB,95)):>18}")
print(f"{'경로 이동거리(m)':20}| {pathlen(A):17.1f}  | {pathlen(B):17.1f} ")
print(f"\n방식A viewpoint(고도,방위): {info(A)}")
print(f"방식B viewpoint(고도,방위): {info(B)}")

# 그래프
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(range(1,len(cA)+1),[c*100 for c in cA],'o-',lw=2,label='A: online greedy NBV')
ax[0].plot(range(1,len(cB)+1),[c*100 for c in cB],'s-',lw=2,label='B: batch select + greedy order')
ax[0].axhline(95,ls='--',c='gray',alpha=0.5)
ax[0].set_xlabel('step'); ax[0].set_ylabel('coverage %'); ax[0].set_title('Coverage vs step')
ax[0].legend(); ax[0].grid(alpha=0.3)
for name,pts,m,c in [('A',A,'o','C0'),('B',B,'s','C1')]:
    az=[math.degrees(math.atan2(p[1]-T[1],p[0]-T[0])) for p in pts]
    al=[T[2]-p[2] for p in pts]
    ax[1].scatter(az,al,marker=m,s=90,c=c,label=name,alpha=0.7)
ax[1].set_xlabel('azimuth deg'); ax[1].set_ylabel('altitude m')
ax[1].set_title('selected viewpoints'); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('results/compare_two_methods.png',dpi=130)
print("\n✓ results/compare_two_methods.png")
