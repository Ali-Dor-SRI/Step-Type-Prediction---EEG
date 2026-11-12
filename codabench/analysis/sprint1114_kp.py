#!/usr/bin/env python
"""Sprint 11-14 KP: ceiling of person-specific modelling with ORACLE person id, `later` split only.
  st --k K : B3 self-training refit (copy of sprint1114_a.adapt_split, B3 config) with the stream's person's
             training rows weighted xK (others x1). K=1 must reproduce the B3 probas.
  final    : P1 personal head blend, P3 X4f refit xK (no self-training) + nets, collect P2, picked-on-other."""
import argparse, copy, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC
from proteus_splits import split
from sprint1106_tta_probe import cells

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
Q5 = CB / "logs/sprint_1112/q5"
OUT = CB / "logs/sprint_1114/kp"
B3 = Q5 / "q5_q0.85_l5.0_s20_t20_f16_C0.03_later.npz"
W = 0.6
def log(m): print(f"{time.strftime('%T')} {m}", flush=True)
def lg(P): return np.log(np.clip(P, 1e-12, 1))

def load():
    rows = PC.load().rows
    d = np.load(DUMP / "X4f__later__None.npz")
    clf = joblib.load(DUMP / "X4f__later__None.clf.joblib")
    sc, lr0 = clf.steps[0][1], clf.steps[-1][1]
    tr = np.asarray(split("later", None, rows).train)
    ptr = rows.person.iloc[tr].to_numpy()
    b = np.load(B3)
    assert (b["test_idx"] == d["test_idx"]).all()
    te = d["test_idx"]; r = rows.iloc[te]
    pte = r.person.to_numpy(); keys = (r.person.astype(str) + "_" + r.session.astype(str)).to_numpy()
    return rows, d, sc, lr0, ptr, b, te, pte, keys

def selftrain(k, q=0.85, lam=5.0, start=20, step=20, wpl=0.5, C=0.03):
    rows, d, sc, lr0, ptr, b, te, pte, keys = load()
    F, Px, ytr = d["F_test"], d["P_test"], d["y_train"]
    Pn = b["Pn"]
    Lpl = (1 - wpl) * lg(Px) + wpl * lg(Pn)
    Ppl = np.exp(Lpl - Lpl.max(1, keepdims=True)); Ppl /= Ppl.sum(1, keepdims=True)
    Ztr = sc.transform(d["F_train"]).astype(np.float16).astype(np.float32)
    Zte = sc.transform(F).astype(np.float32)
    Pa = Px.copy()
    for key in pd.unique(keys):
        idx = np.where(keys == key)[0]
        person = pte[idx[0]]
        wtr = np.where(ptr == person, float(k), 1.0)
        for bb in range(start, len(idx), step):
            past = idx[:bb]
            yp = Ppl[past].argmax(1); conf = Ppl[past].max(1)
            keep = []
            for c in range(3):
                m = np.where(yp == c)[0]
                if len(m): keep.extend(m[np.argsort(-conf[m])][:max(1, int(round(q * len(m))))])
            keep = np.asarray(sorted(keep))
            lr = copy.deepcopy(lr0); lr.set_params(warm_start=True, max_iter=200, C=C)
            lr.fit(np.vstack([Ztr, Zte[past[keep]]]), np.concatenate([ytr, yp[keep]]),
                   sample_weight=np.concatenate([wtr, np.full(len(keep), lam)]))
            nxt = idx[bb:bb + step]
            Pa[nxt] = lr.predict_proba(Zte[nxt])
    np.savez_compressed(OUT / f"st_k{k}.npz", probas=Pa)
    if k == 1:
        log(f"k=1 vs B3 probas max abs diff {np.abs(Pa - b['probas']).max():.2e}")
    log(f"st k={k} done")

def score(rows, te, y, P, Pn, w=W):
    c = cells(rows, te, y, ((1 - w) * lg(P) + w * lg(Pn)).argmax(1))
    return c

def final():
    rows, d, sc, lr0, ptr, b, te, pte, keys = load()
    y, Pn, Px = d["y_test"], b["Pn"], d["P_test"]
    ytr = d["y_train"]
    Ztr = sc.transform(d["F_train"]); Zte = sc.transform(d["F_test"])
    ppl = ["P04", "P10"]
    res = {}   # (arm, cfg) -> cells df
    def add(arm, cfg, c): res[(arm, cfg)] = c
    base = score(rows, te, y, b["probas"], Pn); add("B3", "w0.6", base)
    add("B1 X4f+nets", "w0.6", score(rows, te, y, Px, Pn))
    # P1
    Lb = (1 - W) * lg(b["probas"]) + W * lg(Pn)
    for C in (0.003, 0.01, 0.03, 0.1):
        Pp = np.zeros_like(Px)
        for p in ppl:
            m = ptr == p
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Ztr[m], ytr[m])
            t = pte == p
            Pp[t] = lr.predict_proba(Zte[t])
        for v in (0.2, 0.4, 0.6):
            add("P1", f"C{C}_v{v}", cells(rows, te, y, ((1 - v) * Lb + v * lg(Pp)).argmax(1)))
        add("P1 alone", f"C{C}", cells(rows, te, y, Pp.argmax(1)))
    # P3
    for k in (3, 10, 30):
        Pk = np.zeros_like(Px)
        for p in ppl:
            lr = copy.deepcopy(lr0); lr.set_params(warm_start=True, max_iter=3000)
            lr.fit(Ztr, ytr, sample_weight=np.where(ptr == p, float(k), 1.0))
            t = pte == p
            Pk[t] = lr.predict_proba(Zte[t])
        add("P3", f"k{k}", score(rows, te, y, Pk, Pn))
    # P2
    for k in (3, 10, 30):
        add("P2", f"k{k}", score(rows, te, y, np.load(OUT / f"st_k{k}.npz")["probas"], Pn))
    chk = np.load(OUT / "st_k1.npz")["probas"]
    add("P2 k=1 (check)", "k1", score(rows, te, y, chk, Pn))
    tab = []
    for (arm, cfg), c in res.items():
        s04 = c[c.person == "P04"].score.mean(); s10 = c[c.person == "P10"].score.mean()
        tab.append(dict(arm=arm, cfg=cfg, P04=s04, P10=s10, mean8=c.score.mean()))
    T = pd.DataFrame(tab)
    b04, b10, b8 = T[T.arm == "B3"].iloc[0][["P04", "P10", "mean8"]]
    T["d04"] = (T.P04 - b04) * 100; T["d10"] = (T.P10 - b10) * 100; T["d8"] = (T.mean8 - b8) * 100
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(T.round(4).to_string(index=False))
    T.to_csv(OUT / "kp_table.csv", index=False)
    print("\nPICKED ON THE OTHER PERSON (config chosen by the other person's delta; score on this person):")
    for arm in ("P1", "P2", "P3"):
        S = T[T.arm == arm]
        c04 = S.loc[S.d10.idxmax()]   # chosen on P10, applied to P04
        c10 = S.loc[S.d04.idxmax()]
        d04, d10 = c04.d04, c10.d10
        print(f"{arm}: P04 uses {c04.cfg} -> {d04:+.2f}; P10 uses {c10.cfg} -> {d10:+.2f}; 8-cell mean delta {(d04 + d10) / 2:+.2f} "
              f"(cells equal per person: {len(res[('B3','w0.6')][res[('B3','w0.6')].person=='P04'])}/{len(res[('B3','w0.6')][res[('B3','w0.6')].person=='P10'])})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--k", type=int, default=1)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    rows, d, *_ = load()
    log(f"data: n_train {len(d['y_train'])} n_test {len(d['y_test'])} test persons {sorted(set(rows.iloc[d['test_idx']].person))} train persons {len(set(load()[4]))}")
    selftrain(a.k) if a.mode == "st" else final()
