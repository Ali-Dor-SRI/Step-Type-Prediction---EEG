"""Sprint 11-04 probe 2: (a) probability averages of existing arms, (b) accuracy vs stream position."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "codabench/analysis"))
from sprint1104_posthoc import load, cells, P

hashes = {}
for f in P.glob("*__lopo__0__*.npz"):
    arm, _, _, h = f.stem.split("__")
    hashes[arm] = h
def arm_scores(arms, w=None):
    out = {}
    for prot, folds in (("lopo", range(10)), ("later", [None])):
        cs, per = [], []
        for f in folds:
            ds = [load(a, hashes[a], prot, f) for a in arms]
            p = sum((w[i] if w else 1) * np.log(d["probas"] + 1e-9) for i, d in enumerate(ds))
            c = cells(ds[0], p); cs += c; per.append(np.mean(c))
        out[prot] = (np.mean(cs), np.array(per))
    return out
b = arm_scores(["X4f"])
print("arms with probs:", sorted(a for a in hashes if "X4f" in a or "C2f" in a or a in ("F64","fb6","FI64")))
for combo in (["X4f","C2f"], ["X4f","X4f+asym"], ["X4f","X4f+roi"], ["X4f","X4f+burst"], ["X4f","X4f+asym","X4f+roi","X4f+burst"], ["X4f","C2f","FI64"]):
    if not all(a in hashes for a in combo): print("skip", combo); continue
    s = arm_scores(combo)
    print(f"{'+'.join(combo):40s} lopo {s['lopo'][0]:.4f} ({100*(s['lopo'][0]-b['lopo'][0]):+.2f}, up {(s['lopo'][1]>b['lopo'][1]).sum()}/10) later {s['later'][0]:.4f} ({100*(s['later'][0]-b['later'][0]):+.2f})")
# position curve
pos_ok = {}
for f in range(10):
    d = load("X4f", hashes["X4f"], "lopo", f)
    for s in np.unique(d["stream_id"]):
        m = d["stream_id"] == s
        ok = (d["probas"][m].argmax(1) == d["y_true"][m])
        for i, o in enumerate(ok):
            pos_ok.setdefault(min(i // 16, 12), []).append(o)
print("X4f lopo accuracy by stream position (bins of 16 windows):")
for k in sorted(pos_ok): print(f"  {16*k:4d}-{16*k+15:4d}{'+' if k==12 else ' '}: {np.mean(pos_ok[k]):.3f} (n={len(pos_ok[k])})")
