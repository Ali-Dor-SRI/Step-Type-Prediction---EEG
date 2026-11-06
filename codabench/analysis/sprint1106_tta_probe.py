#!/usr/bin/env python
"""Post-hoc label-free test-time adaptation probes on dumped streams
(sprint 2026-11-06, W2 screen). Reads ``logs/sprint_1106/dump/<name>__*.npz``
(+ the classifier joblib) from ``sprint1106_dump.py``.

Every probe is causal (window i uses windows <= i of its own stream only),
label-free, and reset at each stream start, as the contract's reset_state().
The classifier is the dumped pipeline StandardScaler -> LogisticRegression:
z = scaler(F), logits = z W^T + b.

- base   : the dumped probabilities (must reproduce the dump's cell mean)
- em     : Saerens EM prior adjustment, priors from the running mean of the
           adjusted posteriors (after n0 windows)
- ent    : bias-only entropy minimisation, a few gradient steps on the 3
           biases per window over the stream's windows so far (lr)
- t3a<M> : T3A (Iwasawa & Matsuo 2021): class supports start at the
           normalised W rows; each window joins the support of its predicted
           class; each support keeps its M lowest-entropy members; prediction
           = softmax(z_n . mean(support_c) / T)
- pm<l>  : pseudo-label class means: after n0 windows, the stream's mean z of
           the windows predicted c with p_max >= tau replaces a fraction l of
           the training class mean mu_c; the logits shift by W (mu_c - mu'_c)
           applied as a per-class translation of z (LDA-style bias update):
           logit_c += W_c . (mu_c - mu'_c)

Scores: cell mean (person, session, interface) over lopo (pooled folds) and
later; paired people bootstrap vs base. This is a screen on the test folds:
settings picked here need the weekend's pre-registered / nested check.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402

DUMP = Path.home() / "codabench/logs/sprint_1106/dump"


def softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def ent(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)


def per_stream(fn, Z, L, sid, **kw):
    """fn(Z_stream, L_stream, **kw) -> P_stream, applied per stream."""
    P = np.empty_like(L)
    for s in np.unique(sid):
        m = np.where(sid == s)[0]
        P[m] = fn(Z[m], L[m], **kw)
    return P


def p_em(Z, L, n0=16, iters=5):
    P0 = softmax(L)
    out = np.empty_like(P0)
    prior_tr = np.full(P0.shape[1], 1.0 / P0.shape[1])   # class_weight balanced
    for i in range(len(P0)):
        if i < n0:
            out[i] = P0[i]
            continue
        pi = prior_tr.copy()
        for _ in range(iters):
            Q = P0[:i + 1] * (pi / prior_tr)
            Q /= Q.sum(1, keepdims=True)
            pi = Q.mean(0)
        q = P0[i] * (pi / prior_tr)
        out[i] = q / q.sum()
    return out


def p_ent(Z, L, lr=0.5, steps=3, n0=8, div=True):
    """Bias-only information maximisation (SHOT / IM): minimise the mean
    window entropy minus (div) the entropy of the mean prediction, over the
    stream's windows so far, a few SGD steps on the 3 biases per window."""
    import torch
    b = torch.zeros(L.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([b], lr=lr)
    Lt = torch.as_tensor(L, dtype=torch.float64)
    out = np.empty_like(L)
    for i in range(len(L)):
        if i >= n0:
            for _ in range(steps):
                P = torch.softmax(Lt[:i + 1] + b, 1)
                h = -(P * torch.log(P.clamp_min(1e-12))).sum(1).mean()
                pm = P.mean(0)
                loss = h + (pm * torch.log(pm.clamp_min(1e-12))).sum() if div else h
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            out[i] = torch.softmax(Lt[i] + b, 0).numpy()
    return out


def p_t3a(Z, L, W=None, M=20, T=None):
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    K = W.shape[0]
    sup = [[(0.0, Wn[k])] for k in range(K)]        # (entropy, vector)
    out = np.empty_like(L)
    for i in range(len(Z)):
        p0 = softmax(L[i:i + 1])[0]
        c = int(p0.argmax())
        sup[c].append((float(ent(p0[None])[0]), Zn[i]))
        sup[c].sort(key=lambda t: t[0])
        sup[c] = sup[c][:M]
        Cm = np.stack([np.mean([v for _, v in sup[k]], axis=0) for k in range(K)])
        Cm /= np.linalg.norm(Cm, axis=1, keepdims=True)
        s = Zn[i] @ Cm.T
        out[i] = softmax((s / T)[None])[0]
    return out


def p_pm(Z, L, W=None, mu=None, lam=0.3, tau=0.5, n0=32):
    K = W.shape[0]
    out = np.empty_like(L)
    P0 = softmax(L)
    for i in range(len(Z)):
        if i < n0:
            out[i] = P0[i]
            continue
        P = P0[:i + 1]
        pred, conf = P.argmax(1), P.max(1)
        shift = np.zeros(K)
        for c in range(K):
            m = (pred == c) & (conf >= tau)
            if m.sum() < 3:
                continue
            mu_s = Z[:i + 1][m].mean(0)
            mu_new = (1 - lam) * mu[c] + lam * mu_s
            # LDA-style: score_c = W_c.z + b_c with W_c = S^-1 mu_c; moving mu_c
            # to mu_new keeps W and shifts b by -W_c.(mu_new - mu_c), i.e. the
            # window is compared with the stream's own class position
            shift[c] = -W[c] @ (mu_new - mu[c])
        out[i] = softmax(L[i:i + 1] + shift)[0]
    return out


def cells(rows, idx, y, p):
    sub = rows.iloc[idx].reset_index(drop=True)
    out = []
    for k, g in sub.assign(y=y, p=p).groupby(["person", "session", "interface"]):
        yy, pp = g["y"].to_numpy(), g["p"].to_numpy()
        out.append((*k, np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])))
    return pd.DataFrame(out, columns=["person", "session", "interface", "score"])


def paired(b, a, n=10000):
    m = b.merge(a, on=["person", "session", "interface"], suffixes=("_b", "_a"))
    m["d"] = m.score_a - m.score_b
    ppl = m.person.unique()
    per = {p: m.loc[m.person == p, "d"].to_numpy() for p in ppl}
    rng = np.random.default_rng(0)
    bs = [np.concatenate([per[p] for p in rng.choice(ppl, len(ppl))]).mean() for _ in range(n)]
    return m.d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), sum(per[p].mean() > 0 for p in ppl)


def main(name="X4f", protocols=("lopo", "later")):
    rows = PC.load().rows
    probes = {"base": None, "em": (p_em, {}), "ent0.5": (p_ent, {"lr": 0.5}),
              "ent0.1": (p_ent, {"lr": 0.1})}
    for M in (5, 20, 50):
        probes[f"t3a{M}"] = (p_t3a, {"M": M})
    for lam in (0.3, 0.5):
        for tau in (0.5, 0.6, 0.7):
            probes[f"pm{lam}_t{tau}"] = (p_pm, {"lam": lam, "tau": tau})
    res = {p: {k: [] for k in probes} for p in protocols}
    for proto in protocols:
        files = sorted(DUMP.glob(f"{name}__{proto}__*.npz"))
        for f in files:
            d = np.load(f)
            clf = joblib.load(str(f).replace(".npz", ".clf.joblib"))
            sc, lr = clf[0], clf[-1]
            Z = sc.transform(d["F_test"])
            W, b = lr.coef_, lr.intercept_
            L = Z @ W.T + b
            assert np.allclose(softmax(L), d["P_test"], atol=1e-8), f
            Ztr = sc.transform(d["F_train"])
            mu = np.stack([Ztr[d["y_train"] == c].mean(0) for c in range(W.shape[0])])
            # T3A temperature: the scale of the base logits' spread
            sid, idx, y = d["stream_id"], d["test_idx"], d["y_test"]
            for k, v in probes.items():
                if v is None:
                    P = d["P_test"]
                else:
                    fn, kw = v
                    kw = dict(kw)
                    if fn is p_t3a:
                        kw.update(W=W, T=0.05)
                    if fn is p_pm:
                        kw.update(W=W, mu=mu)
                    P = per_stream(fn, Z, L, sid, **kw)
                res[proto][k].append(cells(rows, idx, y, P.argmax(1)))
        if not files:
            print(f"no dumps for {proto}")
    for proto in protocols:
        if not res[proto]["base"]:
            continue
        base = pd.concat(res[proto]["base"])
        print(f"\n{name} {proto}: base {base.score.mean():.4f} ({len(res[proto]['base'])} files)")
        for k in probes:
            if k == "base":
                continue
            a = pd.concat(res[proto][k])
            d, lo, hi, up = paired(base, a)
            print(f"  {k:12s} {a.score.mean():.4f}  {100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] up {up}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["X4f"]))
