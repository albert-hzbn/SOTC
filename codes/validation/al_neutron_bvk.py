#!/usr/bin/env python3
"""X- and L-point frequencies of Al from the Born--von Karman force constants
fitted to inelastic neutron scattering (Kresch et al., PRB 77, 024301 (2008);
Kresch, Caltech PhD thesis (2009), Table 9.2), for comparison with the
converged harmonic reference. Prints frequencies in THz at 10, 150, 300 K."""
import numpy as np, itertools
# Kresch thesis Table 9.2 (N/m); columns 10,150,300,525,775 K
T={'10':0,'150':1,'300':2}
raw="""K1xx 10.112 9.708 9.708
K1xy 11.148 10.697 10.378
K1zz -1.356 -1.201 -2.059
K2xx 2.454 2.408 2.224
K2yy -0.532 -0.508 -0.367
K3xx -0.634 -0.636 -0.635
K3xy -0.185 -0.301 -0.294
K3yy -0.298 -0.183 -0.181
K3yz -0.149 -0.147 -0.148
K4xx 0.273 0.273 0.273
K4xy -0.051 -0.051 -0.051
K4zz 0.324 0.324 0.324
K5xx 0.469 0.469 0.469
K5xy 0.090 0.090 0.090
K5yy 0.229 0.229 0.229
K5zz 0.199 0.199 0.199
K6xx 0.144 0.144 0.144
K6xy -0.110 -0.110 -0.110
K7xx -0.061 -0.061 -0.061
K7xy 0.032 0.032 0.032
K7xz 0.016 0.016 0.016
K7yy -0.088 -0.088 -0.088
K7yz 0.011 0.011 0.011
K7zz -0.105 -0.105 -0.105
K8xx -0.536 -0.536 -0.536
K8yy -0.117 -0.117 -0.117"""
V={l.split()[0]:list(map(float,l.split()[1:])) for l in raw.splitlines()}
def tens(s,i):
    g=lambda k:V.get(f'K{s}{k}',[0,0,0])[i]
    xx,xy,xz,yy,yz,zz=[g(k) for k in('xx','xy','xz','yy','yz','zz')]
    if s in(1,4): yy=xx; xz=yz=0
    if s in(2,8): zz=yy; xy=xz=yz=0
    if s==3: xz=xy; zz=yy
    if s==5: xz=yz=0
    if s==6: yy=zz=xx; xz=yz=xy
    return np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
bonds={1:(1,1,0),2:(2,0,0),3:(2,1,1),4:(2,2,0),5:(3,1,0),6:(2,2,2),7:(3,2,1),8:(4,0,0)}
ops=[]
for p in itertools.permutations(range(3)):
    for sg in itertools.product((1,-1),repeat=3):
        M=np.zeros((3,3));
        for r,c in enumerate(p): M[r,c]=sg[r]
        ops.append(M)
a=4.05e-10  # m, room temperature
m=26.9815*1.66054e-27
def freqs(q,i):  # q in units 2pi/a
    Dm=np.zeros((3,3))
    for s,b in bonds.items():
        K=tens(s,i); seen={}
        for O in ops:
            R=tuple(O@np.array(b,float)); 
            if R in seen: continue
            Kr=O@K@O.T; seen[R]=Kr
            Rv=np.array(R)*a/2
            Dm+=Kr*(1-np.cos(2*np.pi/a*np.dot(q,Rv)))
    w=np.sqrt(np.clip(np.linalg.eigvalsh(Dm/m),0,None))/(2*np.pi)/1e12
    return np.sort(w)
for t,i in T.items():
    print(t,'X',freqs(np.array([1,0,0]),i).round(2),'L',freqs(np.array([.5,.5,.5]),i).round(2))
