import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC
from scipy.linalg import eigh
CB = Path.home() / "codabench"
c = PC.load(); R = c.rows.reset_index(drop=True); X = c.X
people = sorted(R.person.unique())
def lg(P): return np.log(np.clip(P, 1e-12, 1))
def bacc(y, p):
    return np.mean([np.mean(p[y == k] == k) for k in range(3) if (y == k).any()])
def cm(y, p):
    M = np.zeros((3, 3), int)
    for a, b in zip(y, p): M[a, b] += 1
    return M
pred = {}
for k, pe in enumerate(people):
    d = np.load(CB / f"logs/sprint_1112/q5/q5_q0.85_l5.0_s20_t20_f16_C0.03_lopo{k}.npz")
    ti = d["test_idx"]; assert set(R.person.iloc[ti]) == {pe}, (k, set(R.person.iloc[ti]))
    L = 0.4 * lg(d["probas"]) + 0.6 * lg(d["Pn"])
    pb = L.argmax(1)
    pl = (0.5 * lg(d["Px"]) + 0.5 * lg(d["Pn"]))
    pred[pe] = dict(ti=ti, y=d["y"], p=pb, pl=pl.argmax(1), plconf=np.exp(pl - pl.max(1, keepdims=True)) / np.exp(pl - pl.max(1, keepdims=True)).sum(1, keepdims=True),
                    px=d["Px"].argmax(1), pn=d["Pn"].argmax(1), pa=d["probas"].argmax(1))
print("== (1) B3 balanced acc / pseudo-label acc / components per person")
print("person n  B3    X4f   X4f'  nets  PL(.5/.5) PL-acc-top85%conf  PL-acc-first40 PL-acc-after40")
for pe in people:
    q = pred[pe]; y = q["y"]; mx = q["plconf"].max(1)
    top = mx >= np.quantile(mx, 0.15)
    n = len(y); f = np.arange(n) < 40
    print(f"{pe} {n:4d} {bacc(y,q['p']):.3f} {bacc(y,q['px']):.3f} {bacc(y,q['pa']):.3f} {bacc(y,q['pn']):.3f} {bacc(y,q['pl']):.3f}  {np.mean(q['pl'][top]==y[top]):.3f}  {np.mean(q['pl'][f]==y[f]):.3f} {np.mean(q['pl'][~f]==y[~f]):.3f}")
print("\n== P04 confusion (rows true mi/sub/word, cols pred) B3:"); q = pred["P04"]; print(cm(q["y"], q["p"]))
print("pred class dist P04:", np.bincount(q["p"], minlength=3), " true:", np.bincount(q["y"], minlength=3))
print("P04 pseudo-label confusion:"); print(cm(q["y"], q["pl"]))
print("mean max-prob (PL) per person:", {pe: round(float(pred[pe]['plconf'].max(1).mean()), 3) for pe in people})
print("\n== P04 per session/run B3 bacc (n, acc) and others' mean per-session range")
for pe in ["P04"]:
    q = pred[pe]; r = R.iloc[q["ti"]].reset_index(drop=True)
    for s, g in r.groupby("session"):
        ix = g.index.values; print(f" {pe} s{s} n={len(ix)} bacc={bacc(q['y'][ix], q['p'][ix]):.3f} PLacc={np.mean(q['pl'][ix]==q['y'][ix]):.3f} iface={g.interface.iloc[0]}")
    for (s, ru), g in r.groupby(["session", "run"]):
        ix = g.index.values; print(f"   s{s} r{ru} n={len(ix)} acc={np.mean(q['p'][ix]==q['y'][ix]):.3f}")
allb = []
for pe in people:
    q = pred[pe]; r = R.iloc[q["ti"]].reset_index(drop=True)
    for s, g in r.groupby("session"):
        ix = g.index.values; allb.append((pe, s, len(ix), bacc(q["y"][ix], q["p"][ix])))
print("per-session bacc all people:", " ".join(f"{a}s{b}:{d:.2f}" for a, b, n, d in allb))
print("\n== (2) signal quality (per person; X in uV)")
ch = c.meta["ch_names"] if hasattr(c, "meta") else None
rms_ch = np.sqrt((X.astype(np.float64) ** 2).mean(2))  # (N,41)
med_ch_rms = np.median(rms_ch, 0)
win_rms = np.sqrt((rms_ch ** 2).mean(1))
gmed = np.median(win_rms)
# FFT line noise 50/60 Hz  (Nyquist 60)
from numpy.fft import rfft, rfftfreq
fr = rfftfreq(480, 1 / 120)
print("person  winRMS_med  frac_win>5xGlobalMed  frac(chan-win>5x chan med)  flat_chan_frac(rms<0.1xmed)  max|x|_p99  line50/(40-48 neighbours) line%  P95/P5 chan rms ratio")
chdev = {}
for pe in people:
    m = (R.person == pe).values; Xp = X[m].astype(np.float64)
    rp = np.sqrt((Xp ** 2).mean(2)); wr = np.sqrt((rp ** 2).mean(1))
    chmed = np.median(rp, 0)
    S = np.abs(rfft(Xp - Xp.mean(2, keepdims=True), axis=2)) ** 2
    ps = S.mean((0, 1))
    b50 = ps[(fr > 49) & (fr < 51)].mean() / ps[((fr > 42) & (fr < 47)) | ((fr > 53) & (fr < 58))].mean()
    chdev[pe] = chmed
    print(f"{pe} {np.median(wr):8.1f} {np.mean(wr > 5 * gmed):8.3f} {np.mean(rp > 5 * med_ch_rms):8.3f} {np.mean(rp < 0.1 * med_ch_rms):8.3f} {np.percentile(np.abs(Xp).max((1,2)), 99):9.0f} {b50:7.2f} {np.percentile(chmed,95)/np.percentile(chmed,5):7.1f}")
names = list(c.meta["ch_names"]) if hasattr(c, "meta") else [str(i) for i in range(41)]
lc = np.log(np.array([chdev[p] for p in people]))
z = (lc - lc.mean(0)) / lc.std(0)
i4 = people.index("P04")
o = np.argsort(-np.abs(z[i4]))[:6]
print("P04 channels with largest |z| of log-RMS vs 10-person dist:", [(names[j], round(float(z[i4, j]), 2)) for j in o])
print("\n== (3) log-Euclid covariance distance to the training mean (others; per-person mean cov, per-window)")
def logm(C):
    w, V = eigh(C); return (V * np.log(np.maximum(w, 1e-9))) @ V.T
def cov(x):
    x = x - x.mean(1, keepdims=True); C = x @ x.T / x.shape[1]; return C + 1e-3 * np.trace(C) / 41 * np.eye(41)
Lc = np.zeros((len(R), 41, 41), np.float32)
for i in range(len(R)): Lc[i] = logm(cov(X[i].astype(np.float64)))
pm = {pe: Lc[(R.person == pe).values].mean(0) for pe in people}
for pe in people:
    ti = pred[pe]["ti"]; tr = np.setdiff1d(np.where(R.person.isin([p for p in people if p != pe]))[0], [])
    mu = Lc[tr].mean(0)
    dw = np.linalg.norm((Lc[ti] - mu).reshape(len(ti), -1), axis=1)
    dm = np.linalg.norm(pm[pe] - mu)
    print(f"{pe} person-mean dist={dm:.2f}  window median dist={np.median(dw):.2f}  p90={np.percentile(dw,90):.2f}  bacc={bacc(pred[pe]['y'],pred[pe]['p']):.3f}")
ds = [np.linalg.norm(pm[pe] - Lc[(R.person != pe).values].mean(0)) for pe in people]
bs = [bacc(pred[pe]['y'], pred[pe]['p']) for pe in people]
from scipy.stats import spearmanr
print("spearman(person-mean cov dist, B3 bacc) =", spearmanr(ds, bs))
print("\n== (5) causal artefact rule probe: for P04, bacc on windows with winRMS below/above its own running-median x k")
q = pred["P04"]; wr = win_rms[q["ti"]]
for k in (2, 3, 5):
    rm = np.array([np.median(wr[max(0, i - 60):i]) if i >= 5 else np.nan for i in range(len(wr))])
    bad = wr > k * rm
    print(f" k={k}: flagged {np.nansum(bad)}/{len(wr)}  acc flagged={np.mean(q['p'][bad]==q['y'][bad]) if bad.sum() else float('nan'):.3f} acc rest={np.mean(q['p'][~bad]==q['y'][~bad]):.3f}")
for pe in people:
    q = pred[pe]; wr = win_rms[q["ti"]]; rm = np.array([np.median(wr[max(0, i - 60):i]) if i >= 5 else np.nan for i in range(len(wr))])
    bad = wr > 3 * rm
    print(f" {pe} flagged(k=3)={bad.sum()} accflag={np.mean(q['p'][bad]==q['y'][bad]) if bad.sum() else float('nan'):.2f} accrest={np.mean(q['p'][~bad]==q['y'][~bad]):.3f}", end=";")
print()
print("\n== (6) causal class-balance prior correction on the mixed log-probs (subtract running mean of past softmax-log-probs per class, a*bias; first 20 windows uncorrected)")
for a in (0.0, 0.5, 1.0):
    out = []
    for pe in people:
        d = np.load(CB / f"logs/sprint_1112/q5/q5_q0.85_l5.0_s20_t20_f16_C0.03_lopo{people.index(pe)}.npz")
        L = 0.4 * lg(d["probas"]) + 0.6 * lg(d["Pn"]); L = L - np.log(np.exp(L).sum(1, keepdims=True))
        Lc_ = L.copy()
        for i in range(20, len(L)):
            Lc_[i] = L[i] - a * (L[:i].mean(0) - L[:i].mean())
        out.append(bacc(d["y"], Lc_.argmax(1)))
    print(f" a={a}: P04 {out[3]:.3f} mean-all {np.mean(out):.3f} mean-others {np.mean(out[:3]+out[4:]):.3f}")
print("\n== (7) P04 within-person session shift: log-cov dist between session means (P04 vs P10)")
for pe in ("P04", "P10"):
    m = (R.person == pe).values; r = R[m]
    sm = {s: Lc[m][(r.session == s).values].mean(0) for s in sorted(r.session.unique())}
    print(pe, {f"{a}-{b}": round(float(np.linalg.norm(sm[a] - sm[b])), 2) for a in sm for b in sm if a < b})
