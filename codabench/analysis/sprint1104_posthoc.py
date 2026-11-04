#!/usr/bin/env python
"""Sprint 2026-11-04 probe: causal, label-free post-hoc corrections of streamed
probabilities (no refit). Reads probs/<arm>__<protocol>__<fold>__<hash>.npz.

Correction "bal": per stream, before window t, m = (K0/3 + sum_{s<t} p_s) / (K0 + t-1)
(running mean of the model's own probabilities, pulled to uniform); the window's
probabilities are divided by m**alpha and renormalised. Uses only earlier windows
of the same stream (allowed: within-session label-free adaptation).

    python analysis/sprint1104_posthoc.py X4f 48a95c1a
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

P = Path.home() / "codabench/logs/proteus_w1016/probs"


def load(arm, h, protocol, fold):
    f = P / f"{arm}__{protocol}__{'' if fold is None else fold}__{h}.npz"
    return np.load(f, allow_pickle=True)


def correct(p, sid, alpha, k0):
    if alpha == 0:
        return p
    out = np.empty_like(p)
    K = p.shape[1]
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        acc = np.full(K, k0 / K)
        n = k0
        for i in idx:
            m = acc / n
            q = p[i] / m ** alpha
            out[i] = q / q.sum()
            acc += p[i]
            n += 1
    return out


def cells(d, p):
    y = d["y_true"]
    key = np.char.add(d["stream_id"].astype(str), "_" + d["context_id"].astype(str))
    yhat = p.argmax(1)
    return [balanced_accuracy_score(y[key == c], yhat[key == c]) for c in np.unique(key)]


GRID = [(0, 0)] + [(a, k) for a in (0.5, 1.0) for k in (4, 16, 64)]


def score(arm, h, protocol, folds, a, k):
    cs = []
    per = []
    for f in folds:
        d = load(arm, h, protocol, f)
        c = cells(d, correct(d["probas"], d["stream_id"], a, k))
        cs += c
        per.append(np.mean(c))
    return np.mean(cs), np.array(per)


if __name__ == "__main__":
    arm, h = sys.argv[1], sys.argv[2]
    base_lopo, base_per = score(arm, h, "lopo", range(10), 0, 0)
    base_later, _ = score(arm, h, "later", [None], 0, 0)
    print(f"{arm}: base lopo {base_lopo:.4f} later {base_later:.4f}")
    for a, k in GRID[1:]:
        lo, per = score(arm, h, "lopo", range(10), a, k)
        la, _ = score(arm, h, "later", [None], a, k)
        print(f"alpha={a} K0={k:3d}: lopo {lo:.4f} ({100*(lo-base_lopo):+.2f}, up {int((per>base_per).sum())}/10) "
              f"later {la:.4f} ({100*(la-base_later):+.2f})")
