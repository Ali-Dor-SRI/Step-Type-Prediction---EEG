#!/usr/bin/env python
"""Sprint 2026-11-06 diagnostics on the X4f dumps: accuracy by position in the
stream, per-class recall / confusion, calibration vs online runs."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402

DUMP = Path.home() / "codabench/logs/sprint_1106/dump"
rows = PC.load().rows
for proto in ("lopo", "later"):
    pos, cor, Y, Pr, calib = [], [], [], [], []
    for f in sorted(DUMP.glob(f"X4f__{proto}__*.npz")):
        d = np.load(f)
        sid, y, p = d["stream_id"], d["y_test"], d["P_test"].argmax(1)
        task = rows.iloc[d["test_idx"]]["task"].to_numpy()
        for s in np.unique(sid):
            m = np.where(sid == s)[0]
            pos.append(np.arange(len(m)))
            cor.append(p[m] == y[m])
        Y.append(y)
        Pr.append(p)
        calib.append(np.char.startswith(task.astype(str), "Acquisition"))
    pos, cor = np.concatenate(pos), np.concatenate(cor)
    Y, Pr, calib = np.concatenate(Y), np.concatenate(Pr), np.concatenate(calib)
    print(f"\n{proto}: windows {len(Y)}, accuracy {np.mean(Y == Pr):.4f}")
    for lo, hi in ((0, 16), (16, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 400)):
        m = (pos >= lo) & (pos < hi)
        print(f"  position {lo:3d}-{hi:3d}: n={m.sum():4d} acc={cor[m].mean():.4f}")
    C = np.zeros((3, 3), int)
    for a, b in zip(Y, Pr):
        C[a, b] += 1
    print("  confusion (rows true mi/sub/word):\n", C, "\n  recall", (C.diagonal() / C.sum(1)).round(4))
    print(f"  calib runs acc {np.mean(Y[calib] == Pr[calib]):.4f}  online runs acc {np.mean(Y[~calib] == Pr[~calib]):.4f}")
