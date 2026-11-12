#!/usr/bin/env python
"""Sprint A: bag existing self-trained X4f heads (geometric mean of adapted probas), saved as a new probe name.
usage: python sprint1114_bag.py OUTNAME NAME1 NAME2 ...   (Pn / y / test_idx must agree; Px taken from NAME1)"""
import sys
from pathlib import Path
import numpy as np
Q5 = Path.home() / "codabench/logs/sprint_1112/q5"
out, names = sys.argv[1], sys.argv[2:]
for s in [f"lopo{k}" for k in range(10)] + ["later"]:
    zs = [np.load(Q5 / f"{n}_{s}.npz") for n in names]
    for z in zs[1:]:
        assert np.array_equal(z["test_idx"], zs[0]["test_idx"]) and np.allclose(z["Pn"], zs[0]["Pn"], atol=1e-6)
    L = np.mean([np.log(np.clip(z["probas"], 1e-12, 1)) for z in zs], axis=0)
    P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    np.savez_compressed(Q5 / f"{out}_{s}.npz", probas=P, Px=zs[0]["Px"], Pn=zs[0]["Pn"], y=zs[0]["y"], test_idx=zs[0]["test_idx"])
print("saved", out, "from", names)
