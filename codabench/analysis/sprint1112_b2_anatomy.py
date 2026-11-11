#!/usr/bin/env python
"""Sprint 2026-11-12: where B2 gains vs slot 2 (D1-style breakdown).

slot 2 = X4f + plain Shallow (logs/sprint_1106/rows, seeds 33-37), w 0.3;
B1 = X4f + AdaBN aligned Shallow, w 0.5; B2 = Q5-adapted X4f (q5_q0.85_l5.0_s20_t20_f16) + the
same nets, w 0.6 (the deployed weights). Pooled balanced accuracy by run type, position bin,
interface, person; lopo and later.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402

CB = Path.home() / "codabench"
SPLITS = [f"lopo{k}" for k in range(10)] + ["later"]


def lg(P):
    return np.log(np.clip(P, 1e-12, 1))


def bacc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


rows = PC.load().rows
out = []
for s in SPLITS:
    z = np.load(CB / f"logs/sprint_1112/q5/q5_q0.85_l5.0_s20_t20_f16_{s}.npz")
    te, y = z["test_idx"], z["y"]
    Pp = np.mean([np.load(CB / f"logs/sprint_1106/rows/net_shallow_{s}_s{x}_o0.npz")["probas"] for x in range(33, 38)], axis=0)
    r = rows.iloc[te].reset_index(drop=True).copy()
    r["proto"] = "later" if s == "later" else "lopo"
    r["y"] = y
    r["slot2"] = (0.7 * lg(z["Px"]) + 0.3 * lg(Pp)).argmax(1)
    r["B1"] = (0.5 * lg(z["Px"]) + 0.5 * lg(z["Pn"])).argmax(1)
    r["B2"] = (0.4 * lg(z["probas"]) + 0.6 * lg(z["Pn"])).argmax(1)
    r["runtype"] = np.where(r.task.str.startswith("Acquisition"), "calib", "online")
    r["pos"] = r.groupby(["person", "session"]).loader_pos.rank(method="first").astype(int)
    r["posbin"] = pd.cut(r.pos, [0, 20, 40, 70, 120, 1000], labels=["1-20", "21-40", "41-70", "71-120", "121+"])
    out.append(r)
df = pd.concat(out, ignore_index=True)
for proto in ("lopo", "later"):
    d = df[df.proto == proto]
    print(f"### {proto}\n")
    for by in ("runtype", "posbin", "interface", "person"):
        g = [dict(**{by: k}, n=len(x), **{m: round(bacc(x.y.to_numpy(), x[m].to_numpy()), 4) for m in ("slot2", "B1", "B2")})
             for k, x in d.groupby(by, observed=True)]
        t = pd.DataFrame(g)
        t["B2-slot2"] = (t.B2 - t.slot2).round(4)
        print(t.to_markdown(index=False), "\n")
