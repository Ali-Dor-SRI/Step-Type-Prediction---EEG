#!/usr/bin/env python
"""Thread-pinning check (2026-10-27): load an exported submission folder as the
platform does, stream it over the official test (B = 1), save its predictions and
print the cell mean, ms/window and the thread pools in force after load_model.

    python ~/codabench/analysis/proteus_threads_check.py --submission DIR --out y.npy [--ref ref.npy]

With --ref, also reports whether the predictions equal a reference run window
for window (the pinned F64T must equal F64).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_eval as PE  # noqa: E402
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    PC.require_env040()
    c = PC.load()
    model = PE.load_submission(a.submission, PE.official_meta())
    import torch
    from threadpoolctl import threadpool_info
    pools = sorted({(p["internal_api"], p["num_threads"]) for p in threadpool_info()})
    print(f"torch threads {torch.get_num_threads()}; pools {pools}", flush=True)
    sp = PS.split("official", None, c.rows)
    sc, _, y_pred, _ = PE.evaluate(model, PE.test_loader(c, sp.test))
    np.save(a.out, y_pred)
    print(f"RESULT {a.submission}: cell mean {sc['balanced_accuracy_cell_mean']:.6f} "
          f"bal {sc['balanced_accuracy']:.6f} acc {sc['accuracy']:.6f}; "
          f"{sc['seconds']:.1f} s, {sc['ms_per_window']:.1f} ms/window", flush=True)
    if a.ref:
        ref = np.load(a.ref)
        print(f"EQUAL to {a.ref}: {np.array_equal(ref, y_pred)} "
              f"({int((ref != y_pred).sum())} of {len(ref)} differ)", flush=True)


if __name__ == "__main__":
    main()
