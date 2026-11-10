#!/usr/bin/env python
"""Sprint 2026-11-12 Q9+: time-domain (slow, ERP-like) features as a new probability source.

Per window (41 ch x 480 samples at 120 Hz = seconds 1-5 after the cue):
  x <- x - its per-channel mean (variant 'dm') or raw ('raw')
  bin means over `bin` samples (0.25 s = 30 samples -> 16 bins), channels in CH
  -> features (n_ch * n_bins). Running per-stream feature re-centring as X4f's
  fcenter: f_i - (prior * mu_train + sum_{j<=i} f_j) / (prior + i) (training
  windows centred by their own session mean, mu_train = 0 after that), then
  StandardScaler + LogisticRegression(C, balanced) [or shrinkage LDA] fitted on
  the training people. Causal, label-free at test time.
Writes logs/sprint_1112/td/<name>_<split>.npz (probas, y, test_idx) for every
lopo fold + later, prints the cell means; stack with sprint1112_stack.py.

    python sprint1112_td.py --name td_dm_all --mode dm --bin 30 --ch all --clf lr --C 0.01
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402
from sprint1106_tta_probe import cells  # noqa: E402

OUT = Path.home() / "codabench/logs/sprint_1112/td"
FRONT = ["AF4", "AF3", "F5", "F3", "F1", "Fz", "F2", "F4", "F6"]
SPLITS = [("lopo", k) for k in range(10)] + [("later", None)]


def log(m):
    print(f"{time.strftime('%T')} {m}", flush=True)


def feats(X, mode, b, chi, t0=0, t1=480):
    X = np.asarray(X[:, chi, t0:t1], dtype=np.float64)
    if mode == "dm":
        X = X - X.mean(-1, keepdims=True)
    n, c, t = X.shape
    nb = t // b
    return X[:, :, :nb * b].reshape(n, c, nb, b).mean(-1).reshape(n, -1)


def stream_keys(rows, idx):
    r = rows.iloc[idx]
    return (r.person.astype(str) + "_" + r.session.astype(str)).to_numpy()


def center_train(F, keys):
    out = F.copy()
    for k in np.unique(keys):
        m = keys == k
        out[m] -= F[m].mean(0)
    return out


def center_stream(F, keys, prior):
    out = np.empty_like(F)
    cur, s, n = None, None, 0
    for i, k in enumerate(keys):
        if k != cur:
            cur, s, n = k, np.zeros(F.shape[1]), 0
        s += F[i]
        n += 1
        out[i] = F[i] - s / (prior + n)          # prior * mu_train with mu_train = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="dm", choices=["dm", "raw"])
    ap.add_argument("--bin", type=int, default=30)
    ap.add_argument("--ch", default="all", choices=["all", "front", "nofront", "noaf"])
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=480)
    ap.add_argument("--clf", default="lr", choices=["lr", "lda"])
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--prior", type=float, default=8.0)
    a = ap.parse_args()
    c = PC.load()
    names = [str(n) for n in c.meta["ch_names"]]
    sel = {"all": names, "front": FRONT, "nofront": [n for n in names if n not in FRONT],
           "noaf": [n for n in names if n not in ("AF3", "AF4")]}[a.ch]
    chi = [names.index(n) for n in sel]
    Fall = feats(c.X[:], a.mode, a.bin, chi, a.t0, a.t1)
    log(f"[data] {a.name}: X {c.X.shape} -> F {Fall.shape} ({len(chi)} ch, bin {a.bin}, mode {a.mode}, "
        f"t {a.t0}-{a.t1}, clf {a.clf} C {a.C}, prior {a.prior})")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for proto, fold in SPLITS:
        sp = PS.split(proto, fold, c.rows)
        tr, te = np.asarray(sp.train), np.asarray(sp.test)
        Ftr = center_train(Fall[tr], stream_keys(c.rows, tr))
        Fte = center_stream(Fall[te], stream_keys(c.rows, te), a.prior)
        clf = (make_pipeline(StandardScaler(), LogisticRegression(C=a.C, class_weight="balanced", max_iter=3000))
               if a.clf == "lr" else make_pipeline(StandardScaler(),
                                                   LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")))
        clf.fit(Ftr, c.y[tr])
        P = clf.predict_proba(Fte)
        s = f"lopo{fold}" if proto == "lopo" else proto
        np.savez_compressed(OUT / f"{a.name}_{s}.npz", probas=P, y=c.y[te], test_idx=te)
        res[s] = cells(c.rows, te, c.y[te], P.argmax(1)).score.mean()
    lopo = np.mean([res[f"lopo{k}"] for k in range(10)])
    log(f"[row] {a.name} alone: lopo {lopo:.4f} later {res['later']:.4f} | per fold "
        + " ".join(f"{res[f'lopo{k}']:.3f}" for k in range(10)))


if __name__ == "__main__":
    main()
