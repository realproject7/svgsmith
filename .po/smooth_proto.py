import numpy as np
from svgsmith.postprocess import parse_path, _subpath_points

def _perim(pts): return float(np.sum(np.linalg.norm(np.diff(np.r_[pts,pts[:1]],axis=0),axis=1)))
def _resample_closed(pts,n):
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.r_[pts,pts[:1]],axis=0),axis=1))]
    if d[-1]==0 or n<4: return pts
    u=np.linspace(0,d[-1],n,endpoint=False)
    return np.column_stack([np.interp(u,d,np.r_[pts[:,0],pts[0,0]]),np.interp(u,d,np.r_[pts[:,1],pts[0,1]])])
def _corners(pts,angle_deg):
    n=len(pts); cor=[]; thr=np.cos(np.radians(180-angle_deg))
    for i in range(n):
        a=pts[i]-pts[(i-1)%n]; b=pts[(i+1)%n]-pts[i]
        na=np.linalg.norm(a); nb=np.linalg.norm(b)
        if na<1e-6 or nb<1e-6: continue
        if np.dot(a,b)/(na*nb) < thr: cor.append(i)
    return cor
def _maxdev(arc):
    a=arc[0]; b=arc[-1]; ab=b-a; L=np.linalg.norm(ab)
    if L<1e-6: return 0.0
    n=np.array([-ab[1],ab[0]])/L
    return float(np.max(np.abs((arc-a)@n)))
def _gauss_arc(pts,sigma):
    if sigma<=0 or len(pts)<5: return pts
    r=max(1,int(sigma*3)); k=np.exp(-0.5*(np.arange(-r,r+1)/sigma)**2); k/=k.sum()
    xp=np.r_[pts[0:1].repeat(r,0),pts,pts[-1:].repeat(r,0)]
    out=pts.copy()
    out[:,0]=np.convolve(xp[:,0],k,'same')[r:-r]; out[:,1]=np.convolve(xp[:,1],k,'same')[r:-r]
    out[0]=pts[0]; out[-1]=pts[-1]; return out
def _cr_open(pts):
    n=len(pts); segs=[]
    for i in range(n-1):
        p0=pts[max(0,i-1)];p1=pts[i];p2=pts[i+1];p3=pts[min(n-1,i+2)]
        segs.append((tuple(p1+(p2-p0)/6.0),tuple(p2-(p3-p1)/6.0),tuple(p2)))
    return segs

def smooth_d(d, min_perim=120, sigma=1.2, pts_per_100=10, corner_deg=40, straight_tol=2.0, samples=6):
    out=[]
    for sub in parse_path(d):
        if not sub.closed or len(sub.segments)<3:
            out.append(("RAW",sub)); continue
        pts=np.asarray(_subpath_points(sub,samples),float)[:-1]
        per=_perim(pts)
        if per<min_perim: out.append(("RAW",sub)); continue
        n=max(12,int(per/100*pts_per_100)); rs=_resample_closed(pts,n)
        cor=sorted(set(_corners(rs,corner_deg)))
        if len(cor)<2:  # treat whole loop as one arc
            cor=[0, len(rs)//2]
        start=tuple(rs[cor[0]]); segs=[]
        for j in range(len(cor)):
            a=cor[j]; b=cor[(j+1)%len(cor)]
            idx=list(range(a,b+1)) if b>a else list(range(a,len(rs)))+list(range(0,b+1))
            arc=np.array([rs[k] for k in idx])
            if len(arc)<3 or _maxdev(arc)<=straight_tol:
                # straight edge: keep as a single line to the corner (crisp)
                segs.append(("L",tuple(arc[-1])))
            else:
                for c1,c2,e in _cr_open(_gauss_arc(arc,sigma)): segs.append(("C",c1,c2,e))
        out.append(("MIX",(start,segs)))
    return out

def emit(items,prec=2):
    def f(p): return f"{p[0]:.{prec}f} {p[1]:.{prec}f}"
    ch=[]
    for kind,data in items:
        if kind=="RAW":
            sub=data; ch.append(f"M{f(sub.start)}")
            for seg in sub.segments:
                if seg[0]=="C": ch.append(f"C{f(seg[1])} {f(seg[2])} {f(seg[3])}")
                elif seg[0]=="Q": ch.append(f"Q{f(seg[1])} {f(seg[2])}")
                else: ch.append(f"L{f(seg[1])}")
            if sub.closed: ch.append("Z")
        else:
            start,segs=data; ch.append(f"M{f(start)}")
            for seg in segs:
                if seg[0]=="C": ch.append(f"C{f(seg[1])} {f(seg[2])} {f(seg[3])}")
                else: ch.append(f"L{f(seg[1])}")
            ch.append("Z")
    return "".join(ch)
