import sys; sys.path.insert(0,'/home/ali_d/codabench/analysis')
from sprint1114_kp import *
rows,d,sc,lr0,ptr,b,te,pte,keys=load(); y,Pn=d["y_test"],b["Pn"]
Ztr=sc.transform(d["F_train"]);Zte=sc.transform(d["F_test"]);ytr=d["y_train"]
Lb=(1-W)*lg(b["probas"])+W*lg(Pn)
rng=np.random.default_rng(0)
for C in (0.01,0.03,0.1):
    # control: head fitted on all OTHER-person rows only? (no) -> pooled head on all rows; and head on a random 190-row subset of other people
    P=LogisticRegression(C=C,class_weight="balanced",max_iter=3000).fit(Ztr,ytr).predict_proba(Zte)
    for v in (0.4,0.6):
        c=cells(rows,te,y,((1-v)*Lb+v*lg(P)).argmax(1))
        # control 2: head on another person's 190 rows (wrong person id)
        P2=np.zeros_like(P)
        for p,o in (("P04","P10"),("P10","P04")):
            m=ptr==o; t=pte==p
            P2[t]=LogisticRegression(C=C,class_weight="balanced",max_iter=3000).fit(Ztr[m],ytr[m]).predict_proba(Zte[t])
        c2=cells(rows,te,y,((1-v)*Lb+v*lg(P2)).argmax(1))
        f=lambda c:((c[c.person=="P04"].score.mean()-.6142)*100,(c[c.person=="P10"].score.mean()-.8549)*100,(c.score.mean()-.7346)*100)
        print(f"C{C} v{v} pooled-head d04/d10/d8 %+.2f %+.2f %+.2f | wrong-person head %+.2f %+.2f %+.2f"%(*f(c),*f(c2)))
