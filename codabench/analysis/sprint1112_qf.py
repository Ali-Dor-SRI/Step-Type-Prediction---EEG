#!/usr/bin/env python
"""Sprint 2026-11-12 QF (FLAGGED track, measure only): what exact per-run class balance is worth.

Both arms use run boundaries and run lengths from the metadata, which predict()
never receives, plus the design fact that runs are (near) exactly balanced
(D3): they are flagged ceilings, never part of a main candidate.
  (a) per-run balanced assignment, non-causal: Hungarian assignment of the run's
      windows to classes with quotas round(n/3), maximising the sum of log-probs;
  (b) causal quota decoding: window j of a run takes the most probable class whose
      quota (ceil(n/3)) is not yet used up by the run's earlier predictions.
Base = X4f + AdaBN nets (w 0.5, the B1 deployment weight).

    python sprint1112_qf.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
NETS = CB / "logs/sprint_1112/adabn/al_p32"
LOPO = [f"lopo{k}" for k in range(10)]
W = 0.5


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def main():
    rows = PC.load().rows
    res = {k: {} for k in ("base", "a", "b")}
    for s in LOPO + ["later"]:
        proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        te, y = d["test_idx"], d["y_test"]
        Pn = np.mean([np.load(NETS / f"net_shallow_{s}_s{x}_o0.npz")["probas"] for x in range(33, 38)], axis=0)
        L = (1 - W) * lg(d["P_test"]) + W * lg(Pn)
        r = rows.iloc[te]
        runkey = (r.person.astype(str) + "_" + r.session.astype(str) + "_" + r.run.astype(str)).to_numpy()
        pa, pb = L.argmax(1).copy(), L.argmax(1).copy()
        for k in np.unique(runkey):
            idx = np.where(runkey == k)[0]
            n = len(idx)
            q = [n // 3 + (1 if c < n % 3 else 0) for c in range(3)]
            cols = np.concatenate([[c] * q[c] for c in range(3)])
            ri, ci = linear_sum_assignment(-L[idx][:, cols])
            pa[idx[ri]] = cols[ci]
            used = np.zeros(3, int)
            cap = int(np.ceil(n / 3))
            for i in idx:
                order = np.argsort(-L[i])
                c = next(c for c in order if used[c] < cap)
                pb[i] = c
                used[c] += 1
        for name, p in (("base", L.argmax(1)), ("a", pa), ("b", pb)):
            res[name][s] = cells(rows, te, y, p)
    import pandas as pd
    B = pd.concat([res["base"][s] for s in LOPO])
    for name, lab in (("a", "(a) per-run Hungarian, non-causal"), ("b", "(b) causal per-run quota")):
        A = pd.concat([res[name][s] for s in LOPO])
        dd, lo, hi, up = paired(B, A)
        print(f"FLAGGED {lab}: lopo {A.score.mean():.4f} vs base {B.score.mean():.4f}: {dd * 100:+.2f} "
              f"[{lo * 100:+.2f}, {hi * 100:+.2f}] up {up}; later {res[name]['later'].score.mean():.4f} vs "
              f"{res['base']['later'].score.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
