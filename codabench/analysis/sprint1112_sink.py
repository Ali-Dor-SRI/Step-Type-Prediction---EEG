#!/usr/bin/env python
"""Sprint 2026-11-12 Q5b: causal Sinkhorn class balancing of the ensemble's probabilities.

Per stream (order kept), window i: take the base ensemble probabilities of the
last N windows (i included), find per-class scalings c (Sinkhorn, `it`
iterations) so that the column sums of rows-normalised P * c are uniform, and
predict window i by argmax of P_i * c. Uses only the dataset's published design
(3 balanced classes); no label, count or order of test labels. N = 0 = off.
Base = X4f + AdaBN nets (--nets), w nested; arm = the same with balancing, (N, w)
chosen nested. Paired bootstrap over people vs the base.

    python sprint1112_sink.py --nets logs/sprint_1112/adabn/al_p32 --Ns 20_40_80
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
LOPO = [f"lopo{k}" for k in range(10)]


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def balance(P, keys, N, it=30):
    out = np.empty(len(P), int)
    for k in pd.unique(keys):
        idx = np.where(keys == k)[0]
        for j, i in enumerate(idx):
            B = P[idx[max(0, j - N + 1):j + 1]]
            c = np.ones(P.shape[1])
            for _ in range(it):
                Q = B * c
                Q /= Q.sum(1, keepdims=True)
                c /= Q.sum(0) / (len(B) / P.shape[1])
            out[i] = int(np.argmax(P[i] * c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default=str(CB / "logs/sprint_1112/adabn/al_p32"))
    ap.add_argument("--model", default="shallow")
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--Ns", default="20_40_80")
    ap.add_argument("--grid", default="0.3_0.4_0.5_0.6_0.7")
    ap.add_argument("--x4f", default=None, help="sprint1112_q5 run name: use its adapted X4f probabilities")
    a = ap.parse_args()
    rows = PC.load().rows
    Ns = [int(n) for n in a.Ns.split("_")]
    grid = tuple(float(g) for g in a.grid.split("_"))
    cell = {}
    for s in LOPO + ["later"]:
        proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        te, y = d["test_idx"], d["y_test"]
        Pn = np.mean([np.load(Path(a.nets) / f"net_{a.model}_{s}_s{x}_o0.npz")["probas"]
                      for x in a.seeds.split("_")], axis=0)
        r = rows.iloc[te]
        keys = (r.person.astype(str) + "_" + r.session.astype(str)).to_numpy()
        for w in grid:
            Px = (np.load(CB / f"logs/sprint_1112/q5/{a.x4f}_{s}.npz")["probas"] if a.x4f else d["P_test"])
            L = (1 - w) * lg(Px) + w * lg(Pn)
            P = np.exp(L - L.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            cell[(s, 0, w)] = cells(rows, te, y, P.argmax(1))
            for N in Ns:
                cell[(s, N, w)] = cells(rows, te, y, balance(P, keys, N))

    def nested(opts):
        out, picks = [], []
        for s in LOPO:
            o = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO if t != s]).score.mean())
            out.append(cell[(s, *o)])
            picks.append(o)
        ol = max(opts, key=lambda o: pd.concat([cell[(t, *o)] for t in LOPO]).score.mean())
        return pd.concat(out), cell[("later", *ol)].score.mean(), picks, ol
    B, bl, bp, _ = nested([(0, w) for w in grid])
    for N in Ns:
        A, al, ap_, ol = nested([(N, w) for w in grid])
        dd, lo, hi, up = paired(B, A)
        print(f"sink N={N}: lopo {A.score.mean():.4f} later {al:.4f} vs base {B.score.mean():.4f} / {bl:.4f}: "
              f"{dd * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] up {up}; later {(al - bl) * 100:+.2f}", flush=True)
    A, al, ap_, ol = nested([(N, w) for N in Ns for w in grid])
    dd, lo, hi, up = paired(B, A)
    print(f"sink nested over N and w: lopo {A.score.mean():.4f} later {al:.4f} (all-ten pick {ol}): "
          f"{dd * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] up {up}; later {(al - bl) * 100:+.2f}", flush=True)


if __name__ == "__main__":
    main()
