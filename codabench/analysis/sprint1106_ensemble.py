#!/usr/bin/env python
"""X4f (dumped streamed probabilities) + a braindecode net (sprint1106_netsmoke
npz) log-probability average, weight w for the net. Fixed-w rows for every w,
and a nested row: for lopo test person k, w maximises the mean cell score over
the other nine folds (never the test person); `later` uses the w picked on all
ten lopo folds (labelled). Paired people bootstrap vs X4f.

    python sprint1106_ensemble.py shallow 33
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
from sprint1106_tta_probe import cells, paired  # noqa: E402

DUMP = Path.home() / "codabench/logs/sprint_1106/dump"
ROWS = Path.home() / "codabench/logs/sprint_1106/rows"
WS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def load(model, seed, split):
    """seed: '33' or '33_34_35' (the seeds' probabilities averaged)."""
    proto, fold = ("lopo", split[4:]) if split.startswith("lopo") else ("later", "None")
    d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
    Ps = []
    for s in str(seed).split("_"):
        n = np.load(ROWS / f"net_{model}_{split}_s{s}_o0.npz")
        assert np.array_equal(d["test_idx"], n["test_idx"]), split
        Ps.append(n["probas"])
    return d["test_idx"], d["y_test"], d["P_test"], np.mean(Ps, axis=0)


def mix(Px, Pn, w):
    L = (1 - w) * np.log(np.clip(Px, 1e-12, 1)) + w * np.log(np.clip(Pn, 1e-12, 1))
    return L.argmax(1)


def main(model="shallow", seed="33"):
    rows = PC.load().rows
    splits = [f"lopo{k}" for k in range(10)] + ["later"]
    data = {s: load(model, seed, s) for s in splits}
    cell = {(s, w): cells(rows, data[s][0], data[s][1], mix(data[s][2], data[s][3], w))
            for s in splits for w in WS}
    net = {s: cells(rows, data[s][0], data[s][1], data[s][3].argmax(1)) for s in splits}
    lopo = [f"lopo{k}" for k in range(10)]
    base_l = pd.concat([cell[(s, 0.0)] for s in lopo])
    base_t = cell[("later", 0.0)]
    print(f"X4f base lopo {base_l.score.mean():.4f} later {base_t.score.mean():.4f}")
    nl = pd.concat([net[s] for s in lopo])
    print(f"{model} alone lopo {nl.score.mean():.4f} later {net['later'].score.mean():.4f}")
    for w in WS[1:]:
        a = pd.concat([cell[(s, w)] for s in lopo])
        d, lo, hi, up = paired(base_l, a)
        dt = cell[("later", w)].score.mean() - base_t.score.mean()
        print(f"  w={w:.1f} lopo {a.score.mean():.4f} {100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] "
              f"up {up}  later {cell[('later', w)].score.mean():.4f} ({100*dt:+.2f})")
    picks, nest = [], []
    for k, s in enumerate(lopo):
        others = [t for t in lopo if t != s]
        w = max(WS, key=lambda w: pd.concat([cell[(t, w)] for t in others]).score.mean())
        picks.append(w)
        nest.append(cell[(s, w)])
    a = pd.concat(nest)
    d, lo, hi, up = paired(base_l, a)
    wl = max(WS, key=lambda w: pd.concat([cell[(t, w)] for t in lopo]).score.mean())
    dt = cell[("later", wl)].score.mean() - base_t.score.mean()
    print(f"  nested: picks {picks} lopo {a.score.mean():.4f} {100*d:+.2f} "
          f"[{100*lo:+.2f}, {100*hi:+.2f}] up {up}; later w={wl} "
          f"{cell[('later', wl)].score.mean():.4f} ({100*dt:+.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
