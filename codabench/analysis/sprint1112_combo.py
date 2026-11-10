#!/usr/bin/env python
"""Sprint 2026-11-12: compare net sides on top of a Q5-adapted X4f part (base B2 by default).

X4f part = the adapted probabilities of a sprint1112_q5 run (`--x4f NAME`, or `orig` for
the dumped X4f). Net part = mean over members of each member's seed-mean probabilities;
a member is DIR[:MODEL[:SEEDS]] (model default shallow, seeds default 33_34_35_36_37).
Base net part default = AdaBN p32 aligned Shallow. w nested per lopo fold for both;
paired bootstrap over people.

    python sprint1112_combo.py --arm DIR1+DIR2:shallow:38_39 [--base DIR] [--x4f q5_name]
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
Q5 = CB / "logs/sprint_1112/q5"
LOPO = [f"lopo{k}" for k in range(10)]


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def nets(spec, s):
    out = []
    for m in spec.split("+"):
        parts = m.split(":")
        d = parts[0]
        model = parts[1] if len(parts) > 1 and parts[1] else "shallow"
        seeds = parts[2].split("_") if len(parts) > 2 else "33_34_35_36_37".split("_")
        out.append(np.mean([np.load(Path(d) / f"net_{model}_{s}_s{x}_o0.npz")["probas"] for x in seeds], axis=0))
    return np.mean(out, axis=0)


def x4f(name, s):
    if name == "orig":
        proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
        return np.load(DUMP / f"X4f__{proto}__{fold}.npz")["P_test"]
    return np.load(Q5 / f"{name}_{s}.npz")["probas"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", default=str(CB / "logs/sprint_1112/adabn/al_p32"))
    ap.add_argument("--x4f", default="q5_q0.7_l5.0_s40_t20")
    ap.add_argument("--x4f-base", default=None, help="X4f part of the base (default = --x4f)")
    ap.add_argument("--grid", default="0.4_0.5_0.6_0.7_0.8")
    ap.add_argument("--label", default="arm")
    a = ap.parse_args()
    grid = tuple(float(g) for g in a.grid.split("_"))
    rows = PC.load().rows
    cb, ca = {}, {}
    for s in LOPO + ["later"]:
        proto, fold = ("lopo", s[4:]) if s.startswith("lopo") else ("later", "None")
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        te, y = d["test_idx"], d["y_test"]
        Xa, Xb = x4f(a.x4f, s), x4f(a.x4f_base or a.x4f, s)
        Na, Nb = nets(a.arm, s), nets(a.base, s)
        for w in grid:
            cb[(s, w)] = cells(rows, te, y, ((1 - w) * lg(Xb) + w * lg(Nb)).argmax(1))
            ca[(s, w)] = cells(rows, te, y, ((1 - w) * lg(Xa) + w * lg(Na)).argmax(1))

    def nested(cell):
        out, picks = [], []
        for s in LOPO:
            w = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO if t != s]).score.mean())
            out.append(cell[(s, w)])
            picks.append(w)
        wl = max(grid, key=lambda w: pd.concat([cell[(t, w)] for t in LOPO]).score.mean())
        return pd.concat(out), cell[("later", wl)].score.mean(), picks, wl
    B, bl, bp, bw = nested(cb)
    A, al, apk, aw = nested(ca)
    dd, lo, hi, up = paired(B, A)
    dl = (al - bl) * 100
    v = "ADOPT" if dd * 100 >= 1 and lo > 0 and dl >= 0 else ("lean" if dd > 0 and lo * 100 > -0.5 and dl >= -0.5 else "no gain")
    print(f"{a.label}: base lopo {B.score.mean():.4f} later {bl:.4f} (w {bw}) | arm lopo {A.score.mean():.4f} "
          f"later {al:.4f} (w {aw}; picks {apk})\n  arm - base: lopo {dd * 100:+.2f} [{lo * 100:+.2f}, "
          f"{hi * 100:+.2f}] up {up}; later {dl:+.2f} -> {v}", flush=True)


if __name__ == "__main__":
    main()
