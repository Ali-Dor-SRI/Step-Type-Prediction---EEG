#!/usr/bin/env python
"""Gate G-c of sprint 2026-10-16 (brief § 5 Phase 0): align="stream" streams correctly.

Fits Riemann-Sealed with align="stream" (bpt4, buffer=0) on `later`, then checks:
  1. B = 1 streamed (the organisers' predict_streams) = one batched predict per
     stream on a fresh copy, in stream order (the running reference is causal
     inside a batch, so both must agree window for window);
  2. isolation: stream k predicted alone on a fresh copy = stream k inside the
     full streamed run (nothing crosses a stream);
  3. without reset_state() a reused model differs (the state really is used),
     and the deep copy + reset restores the first prediction exactly;
  4. ms per window of the streamed run (projected sealed-test time).
Prints PASS / FAIL per check; rc 1 on any FAIL.

    python ~/codabench/analysis/proteus_w1016_gates.py [--solver riemann_sealed] [--set k=v ...]
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_sweep as SW  # noqa: E402
import proteus_eval as PE  # noqa: E402
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="riemann_sealed")
    ap.add_argument("--set", nargs="*", default=["personal=pooled", "align=stream",
                                                 "adapt=online", "buffer=0", "xblocks=bpt4"])
    a = ap.parse_args()
    PC.require_env040()
    params = {k: SW.parse_value(v) for k, _, v in (kv.partition("=") for kv in a.set)}
    Solver = SW.load_solver_class(a.solver)
    model, sc, y_true, y_pred, info, fit_s = SW.fit_and_score(Solver, params, "later", None)
    print(f"fit {fit_s:.1f} s; streamed cell mean {sc['balanced_accuracy_cell_mean']:.6f}; "
          f"{sc['ms_per_window']:.1f} ms/window", flush=True)
    c = PC.load()
    sp = PS.split("later", None, c.rows)
    idx = np.asarray(sp.test)
    sid = np.asarray(info["stream_id"])
    X = np.asarray(c.X[idx])
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    batched = np.empty_like(y_pred)
    alone = {}
    for s in np.unique(sid):
        m = sid == s
        f = copy.deepcopy(model)
        f.reset_state()
        batched[m] = np.asarray(f.predict(X[m])).reshape(-1)
        g = copy.deepcopy(model)
        g.reset_state()
        alone[s] = np.asarray([int(np.asarray(g.predict(X[m][i:i + 1])).reshape(-1)[0])
                               for i in range(m.sum())])
    check("1 B=1 streamed == batched per stream", np.array_equal(batched, y_pred),
          f"{int((batched != y_pred).sum())} of {len(y_pred)} differ")
    ok2 = all(np.array_equal(alone[s], y_pred[sid == s]) for s in alone)
    check("2 each stream alone == inside the streamed run", ok2)
    s0, s1 = np.unique(sid)[:2]
    h = copy.deepcopy(model)
    h.reset_state()
    h.predict(X[sid == s0])                      # leaves stream s0's state behind
    dirty = np.asarray(h.predict(X[sid == s1])).reshape(-1)
    h.reset_state()
    clean = np.asarray(h.predict(X[sid == s1])).reshape(-1)
    check("3a without reset the state is used (predictions change)",
          not np.array_equal(dirty, y_pred[sid == s1]),
          f"{int((dirty != y_pred[sid == s1]).sum())} differ")
    check("3b reset_state() restores the clean prediction", np.array_equal(clean, y_pred[sid == s1]))
    ms = sc["ms_per_window"]
    proj = ms * 106 * 199 / 60000
    check("4 projected sealed test < 30 min (60 allowed)", proj < 30,
          f"{ms:.1f} ms/window -> {proj:.1f} min for 106 sessions x 199 windows")
    print("ALL PASS" if fails == 0 else f"{fails} FAIL", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
