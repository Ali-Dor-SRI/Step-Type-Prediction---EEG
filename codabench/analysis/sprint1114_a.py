#!/usr/bin/env python
"""Sprint 2026-11-14 A (copy-and-extend of sprint1112_q5.py; identical at default flags).

Extra flags: --gamma G (pseudo-row weight lam*conf**G, conf = pseudo-label ensemble max prob),
--agree (before the per-class top-q cut keep only past windows with argmax(Px)==argmax(Pn)),
--varnorm (per-stream variance normalisation of the scaled features, defined exactly as:
  training rows Ztr = sc.transform(F_train) are divided, per feature, by the std (ddof 0, floor 1e-3)
  of their own session (person+session of the training indices); at each refit at window b the stream's
  scaled test features Zte[idx[:b]] (past windows only, no centring) give std_b, and the past pseudo-rows
  AND the next `step` windows to be scored are divided by std_b. Windows before `start` keep Px.
  The warm start / scaler of X4f are unchanged.)  --name is required; --splits selects splits.
Output npz: logs/sprint_1112/q5/<name>_<split>.npz (probas, Px, Pn, y, test_idx).
Final print compares arm vs B3 (nested w over the grid) and vs B1 (X4f+nets).

Original docstring:
Sprint 2026-11-12 Q5: causal pseudo-label self-training of X4f's classifier inside a stream.

Per test stream (person x session, stream order), using the dumped streamed X4f
features: windows 1..start-1 are classified by X4f's own classifier; at windows
start, start+step, ... the logistic layer is refit (warm start from X4f's
coefficients, X4f's scaler kept) on the training features + the stream's PAST
windows, pseudo-labelled by the base ensemble (X4f + AdaBN nets, w 0.5), keeping
per predicted class the `q` most confident (class-balanced), each with sample
weight `lam`; the refit classifier scores the next `step` windows. Label-free,
causal (a window never sees its own or later windows' pseudo-labels).
The arm = X4f' (adapted) mixed with the same nets, w nested on the grid; the base
= X4f mixed with the nets, w nested; paired bootstrap over people.

    python sprint1112_q5.py --nets logs/sprint_1112/adabn/al_p32 --q 0.5 --lam 3 --start 40 --step 20
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
OUT = CB / "logs/sprint_1112/q5"
LOPO = [f"lopo{k}" for k in range(10)]
SPLITS = LOPO + ["later"]
B3NAME = "q5_q0.85_l5.0_s20_t20_f16_C0.03"


def log(m):
    print(f"{time.strftime('%T')} {m}", flush=True)


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def adapt_split(s, nets_dir, seeds, q, lam, start, step, w_pl, rows, maxit=200, ntrain=0, chain=False, zdtype="f64", plsrc="orig", pw=0.0, model="X4f", maxpast=0, pwcov=0.0, C=None, gamma=0.0, agree=False, varnorm=False):
    proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
    d = np.load(DUMP / f"{model}__{proto}__{fold}.npz")
    clf = joblib.load(DUMP / f"{model}__{proto}__{fold}.clf.joblib")
    sc, lr0 = clf.steps[0][1], clf.steps[-1][1]
    te, F, Px = d["test_idx"], d["F_test"], d["P_test"]
    Pn = np.mean([np.load(Path(nets_dir) / f"net_shallow_{s}_s{x}_o0.npz")["probas"] for x in seeds], axis=0)
    Lpl = (1 - w_pl) * lg(Px) + w_pl * lg(Pn)
    Ppl = np.exp(Lpl - Lpl.max(1, keepdims=True))
    Ppl /= Ppl.sum(1, keepdims=True)
    Ztr = sc.transform(d["F_train"])
    ytr = d["y_train"]
    from proteus_splits import split as _split
    ptr = rows.person.iloc[np.asarray(_split(proto, None if fold == "None" else int(fold), rows).train)].to_numpy()
    assert len(ptr) == len(ytr)
    if ntrain and ntrain < len(ytr):                 # fixed class-stratified subsample of training rows
        rng = np.random.default_rng(0)
        sub = np.concatenate([rng.choice(np.where(ytr == c)[0], ntrain // 3, replace=False) for c in range(3)])
        Ztr, ytr = Ztr[np.sort(sub)], ytr[np.sort(sub)]
    Zte = sc.transform(F)
    if varnorm:                                      # per-session std of training rows (per feature)
        assert not ntrain
        tri_ = np.asarray(_split(proto, None if fold == "None" else int(fold), rows).train)
        rt = rows.iloc[tri_]
        ktr = (rt.person.astype(str) + "_" + rt.session.astype(str)).to_numpy()
        Ztr = Ztr.copy()
        for k_ in pd.unique(ktr):
            m_ = ktr == k_
            Ztr[m_] = Ztr[m_] / np.maximum(Ztr[m_].std(0), 1e-3)
    if zdtype != "f64":                              # solver-side storage / compute precision
        Ztr = Ztr.astype(np.float16 if zdtype == "f16" else np.float32).astype(np.float32)
        Zte = Zte.astype(np.float32)
    r = rows.iloc[te]
    keys = (r.person.astype(str) + "_" + r.session.astype(str)).to_numpy()
    Pa = Px.copy()
    nfit, tfit = 0, 0.0
    if pwcov > 0:                                    # raw-covariance person similarity (log-Euclidean)
        cache = PC.load()
        tri = np.asarray(_split(proto, None if fold == "None" else int(fold), rows).train)
        def _logm(C):
            w_, V = np.linalg.eigh((C + C.T) / 2)
            return (V * np.log(np.maximum(w_, 1e-10))) @ V.T
        def _covs(idx):
            X = np.asarray(cache.X[idx], dtype=np.float64)
            X = X - X.mean(-1, keepdims=True)
            return np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
        pp = np.unique(ptr)
        Lp = {q_: _logm(_covs(tri[ptr == q_]).mean(0)) for q_ in pp}
        Cte = _covs(te)
    for k in pd.unique(keys):
        idx = np.where(keys == k)[0]                     # stream order
        prev = lr0
        for b in range(start, len(idx), step):
            past = idx[max(0, b - maxpast):b] if maxpast else idx[:b]
            yp = Ppl[past].argmax(1)
            conf = Ppl[past].max(1)
            keep = []
            ok = (Px[past].argmax(1) == Pn[past].argmax(1)) if agree else np.ones(len(past), bool)
            for c in range(3):
                m = np.where((yp == c) & ok)[0]
                if len(m):
                    keep.extend(m[np.argsort(-conf[m])][:max(1, int(round(q * len(m))))])
            keep = np.asarray(sorted(keep))
            lr = copy.deepcopy(prev if chain else lr0)
            lr.set_params(warm_start=True, max_iter=maxit, **({'C': C} if C else {}))
            if varnorm:
                sd = np.maximum(Zte[idx[:b]].std(0), 1e-3)
                Zp, Zn_ = Zte[past[keep]] / sd, Zte[idx[b:b + step]] / sd
            else:
                Zp, Zn_ = Zte[past[keep]], Zte[idx[b:b + step]]
            Xf = np.vstack([Ztr, Zp])
            yf = np.concatenate([ytr, yp[keep]])
            wtr = np.ones(len(ytr))
            if pwcov > 0:
                Ls = _logm(Cte[past].mean(0))
                d2 = np.array([((Lp[q_] - Ls) ** 2).sum() for q_ in pp])
                wp = np.exp(-d2 / (pwcov * np.median(d2)))
                wp = wp / wp.mean()
                wtr = wp[np.searchsorted(pp, ptr)]
            if pw > 0:                               # soft person weighting by feature-mean distance
                mu_s = Zte[past].mean(0)
                pp = np.unique(ptr)
                d2 = np.array([((Ztr[ptr == q_].mean(0) - mu_s) ** 2).sum() for q_ in pp])
                wp = np.exp(-d2 / (pw * np.median(d2)))
                wp = wp / wp.mean()
                wtr = wp[np.searchsorted(pp, ptr)]
            sw = np.concatenate([wtr, np.full(len(keep), lam) if not gamma else lam * conf[keep] ** gamma])
            t0 = time.perf_counter()
            lr.fit(Xf, yf, sample_weight=sw)
            tfit += time.perf_counter() - t0
            nfit += 1
            prev = lr
            nxt = idx[b:b + step]
            Pa[nxt] = lr.predict_proba(Zn_)
            if plsrc == "adapted":                   # later refits learn from the adapted ensemble
                L2 = (1 - w_pl) * lg(Pa[nxt]) + w_pl * lg(Pn[nxt])
                P2 = np.exp(L2 - L2.max(1, keepdims=True))
                Ppl[nxt] = P2 / P2.sum(1, keepdims=True)
    return te, d["y_test"], Px, Pa, Pn, (nfit, tfit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default=str(CB / "logs/sprint_1112/adabn/al_p32"))
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--wpl", type=float, default=0.5, help="w of the pseudo-labelling ensemble")
    ap.add_argument("--grid", default="0.4_0.5_0.6_0.7")
    ap.add_argument("--name", required=True)
    ap.add_argument("--maxit", type=int, default=200)
    ap.add_argument("--ntrain", type=int, default=0, help="training rows kept (0 = all)")
    ap.add_argument("--chain", action="store_true", help="warm start from the stream's previous refit")
    ap.add_argument("--C", type=float, default=None, help="refit C (default: X4f's own)")
    ap.add_argument("--pwcov", type=float, default=0.0, help="raw-covariance person weighting temperature (0 = off)")
    ap.add_argument("--maxpast", type=int, default=0, help="only the last N past windows as pseudo rows (0 = all)")
    ap.add_argument("--model", default="X4f", help="dump prefix (X4f, X4fc02, ...)")
    ap.add_argument("--pw", type=float, default=0.0, help="soft person weighting temperature (0 = off)")
    ap.add_argument("--plsrc", default="orig", choices=["orig", "adapted"])
    ap.add_argument("--zdtype", default="f64", choices=["f64", "f32", "f16"],
                    help="training features stored as (f16 -> computed in f32)")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--agree", action="store_true")
    ap.add_argument("--varnorm", action="store_true")
    ap.add_argument("--splits", default="all", help="all or comma list, e.g. lopo0,later")
    a = ap.parse_args()
    name = a.name
    grid = tuple(float(g) for g in a.grid.split("_"))
    seeds = a.seeds.split("_")
    rows = PC.load().rows
    OUT.mkdir(parents=True, exist_ok=True)
    cb, ca = {}, {}
    run_splits = SPLITS if a.splits == "all" else a.splits.split(",")
    for s in run_splits:
        f = OUT / f"{name}_{s}.npz"
        if f.exists():
            z = np.load(f)
            te, y, Px, Pa, Pn = z["test_idx"], z["y"], z["Px"], z["probas"], z["Pn"]
        else:
            te, y, Px, Pa, Pn, (nfit, tfit) = adapt_split(s, a.nets, seeds, a.q, a.lam, a.start, a.step, a.wpl,
                                                          rows, a.maxit, a.ntrain, a.chain, a.zdtype, a.plsrc, a.pw, a.model, a.maxpast, a.pwcov, a.C, a.gamma, a.agree, a.varnorm)
            np.savez_compressed(f, probas=Pa, Px=Px, Pn=Pn, y=y, test_idx=te)
            log(f"{name} {s}: {nfit} refits, {tfit / max(nfit, 1):.2f} s each; X4f alone {cells(rows, te, y, Px.argmax(1)).score.mean():.4f} "
                f"-> adapted {cells(rows, te, y, Pa.argmax(1)).score.mean():.4f}")
        for w in grid:
            cb[(s, w)] = cells(rows, te, y, ((1 - w) * lg(Px) + w * lg(Pn)).argmax(1))
            ca[(s, w)] = cells(rows, te, y, ((1 - w) * lg(Pa) + w * lg(Pn)).argmax(1))

    if run_splits != SPLITS:
        return                                       # smoke / partial run: no summary
    b3 = {}
    for s in SPLITS:
        z = np.load(OUT / f"{B3NAME}_{s}.npz")
        for w in grid:
            b3[(s, w)] = cells(rows, z["test_idx"], z["y"], ((1 - w) * lg(z["probas"]) + w * lg(z["Pn"])).argmax(1))

    def nested(cell):
        out, picks = [], []
        for s in LOPO:
            w = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO if t != s]).score.mean())
            out.append(cell[(s, w)])
            picks.append(w)
        wl = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO]).score.mean())
        return pd.concat(out), cell[("later", wl)].score.mean(), picks
    B, bl, bp = nested(cb)
    A, al, apk = nested(ca)
    B3s, b3l, b3p = nested(b3)

    def verdict(dd, lo, dl):
        return "ADOPT" if dd * 100 >= 1 and lo * 100 > 0 and dl >= -0.5 else (
            "lean" if dd > 0 and lo * 100 > -0.5 and dl >= -0.5 else "no gain")
    dd, lo, hi, up = paired(B, A)
    dl = (al - bl) * 100
    print(f"{name}: base B1 (X4f + nets) lopo {B.score.mean():.4f} later {bl:.4f} picks {bp} | "
          f"arm lopo {A.score.mean():.4f} later {al:.4f} picks {apk}\n  arm - B1: lopo {dd * 100:+.2f} "
          f"[{lo * 100:+.2f}, {hi * 100:+.2f}] up {up}; later {dl:+.2f}", flush=True)
    d3, lo3, hi3, up3 = paired(B3s, A)
    dl3 = (al - b3l) * 100
    print(f"  B3 lopo {B3s.score.mean():.4f} later {b3l:.4f} picks {b3p}\n  arm - B3: lopo {d3 * 100:+.2f} "
          f"[{lo3 * 100:+.2f}, {hi3 * 100:+.2f}] up {up3}; later {dl3:+.2f} -> {verdict(d3, lo3, dl3)}", flush=True)


if __name__ == "__main__":
    main()
