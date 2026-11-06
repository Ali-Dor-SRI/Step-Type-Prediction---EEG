#!/usr/bin/env python
"""Sprint 2026-11-09 W0 gate: ShallowNetT (plain torch) == braindecode ShallowFBCSPNet.

Loads the seed-33 weights retrained by sprint1106_netsmoke --save-weights into
both nets (eval, float32, 1 thread), predicts lopo0 + later, and checks
max |P_T - P_bd| <= 1e-6, and against the saved probabilities (the retrain's
npz and the 11-06 npz). Prints both key lists. rc 1 on any FAIL.
"""

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402
from shallownet_t import ShallowNetT  # noqa: E402
from sprint1106_netsmoke import make, zscore  # noqa: E402

W0 = Path.home() / "codabench/logs/sprint_1109/w0"
OLD = Path.home() / "codabench/logs/sprint_1106/rows"


def main():
    c = PC.load()
    fails = 0
    for split in ("lopo0", "later"):
        sd = torch.load(W0 / f"w_shallow_{split}_s33.pt")
        bd = make("shallow", 41, 480)
        bd.load_state_dict(sd)
        nt = ShallowNetT()
        print("bd keys", list(bd.state_dict().keys()))
        print("T  keys", list(nt.state_dict().keys()))
        nt.load_state_dict(sd, strict=True)
        bd.eval(); nt.eval()
        proto, fold = ("lopo", 0) if split == "lopo0" else ("later", None)
        te = np.asarray(PS.split(proto, fold).test)
        X = torch.from_numpy(zscore(c.X[te]))
        with torch.no_grad():
            Pb = torch.softmax(bd(X), -1).numpy()
            Pt = torch.softmax(nt(X), -1).numpy()
            Pt1 = np.concatenate([torch.softmax(nt(X[i:i + 1]), -1).numpy() for i in range(len(X))])
            Pb1 = np.concatenate([torch.softmax(bd(X[i:i + 1]), -1).numpy() for i in range(len(X))])
        new = np.load(W0 / f"net_shallow_{split}_s33_o0.npz")["probas"]
        for name, A, B, tol in [("T vs braindecode", Pt, Pb, 1e-6),
                                ("T at B=1 vs braindecode at B=1", Pt1, Pb1, 1e-6),
                                ("(info) T at B=1 vs batched braindecode", Pt1, Pb, 1e-5),
                                ("T vs retrain npz", Pt, new, 1e-6),
                                ]:
            d = float(np.abs(A - B).max())
            same = bool((A.argmax(1) == B.argmax(1)).all())
            ok = d <= tol and same
            fails += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'} {split} {name}: max|d|={d:.2e} classes equal={same}")
    print("ALL PASS" if not fails else f"{fails} FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
