#!/usr/bin/env python
"""Sprint 2026-11-12: per-component ms/window of a submission folder (X4f part vs net part),
B = 1, one stream of the official test split, after a 20-window warm-up.

    python sprint1112_timing.py <folder> [n]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_eval as PE  # noqa: E402
import proteus_splits as PS  # noqa: E402

folder, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150
c = PC.load()
m = PE.load_submission(folder, PE.official_meta())
te = np.asarray(PS.split("official", None, c.rows).test)[:n + 20]
X = [torch.from_numpy(np.asarray(c.X[i:i + 1])) for i in te]
m.reset_state()
for x in X[:20]:
    m.predict(x)
tx = tn = tt = 0.0
for x in X[20:]:
    t0 = time.perf_counter()
    m.x4f.predict_proba(x)
    t1 = time.perf_counter()
    m.net_proba(x)
    t2 = time.perf_counter()
    m.predict(x)
    t3 = time.perf_counter()
    tx, tn, tt = tx + t1 - t0, tn + t2 - t1, tt + t3 - t2
k = 1000 / n
print(f"{type(m).__name__}: x4f {tx * k:.1f} ms, nets {tn * k:.1f} ms, full predict {tt * k:.1f} ms/window "
      f"(torch threads {torch.get_num_threads()})", flush=True)
