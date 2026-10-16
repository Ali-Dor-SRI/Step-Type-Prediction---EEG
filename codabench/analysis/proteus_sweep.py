#!/usr/bin/env python
"""Fit solver configs in-process on the PROTEUS protocols; one JSON row each.

Sprint 2026-10-16 (brief ``prompts/2026-10-16_proteus_weekend_model.md``,
Phase 0). A config is a solver file plus parameter overrides; it is fitted on a
protocol's training set exactly as benchopt would (route B's ``get_data()``,
the solver's ``set_objective`` -> ``run`` -> ``get_result()["model"]``) and
scored by route A (the organisers' ``predict_streams``: B = 1, fresh deep copy
+ ``reset_state()`` per stream).

    python ~/codabench/analysis/proteus_sweep.py --lane A --name S64 \
        --solver riemann_sealed --set personal=pooled align=stream adapt=online \
        buffer=64 xblocks=bpt4 --protocols lopo later

Rows go to ``~/codabench/logs/proteus_w1016/<lane>.jsonl`` (one file per lane:
never two writers on one file); y_true / y_pred / probabilities and the test
rows' ids to ``probs/<key>.npz``. Resumable: a (config, protocol, fold) key
already in the lane file is skipped. ``--protocols lopo`` expands to folds
0..9. The first line of every fit is the data-shape line.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_eval as PE  # noqa: E402  (sets the competition path)
import proteus_splits as PS  # noqa: E402
import proteus_cache as PC  # noqa: E402

SOLVERS = Path.home() / "codabench/solvers/bci_decoding"
OUT = Path(os.environ.get("SWEEP_OUT", Path.home() / "codabench/logs/proteus_w1016"))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def parse_value(v):
    if v in ("None", "none_"):
        return None
    if v in ("True", "False"):
        return v == "True"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def load_solver_class(name):
    path = SOLVERS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sweep_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # inspect.getfile (the export) needs it
    spec.loader.exec_module(mod)
    return mod.Solver


def config_key(name, solver, params, protocol, fold):
    blob = json.dumps(dict(solver=solver, params=params), sort_keys=True)
    h = hashlib.sha1(blob.encode()).hexdigest()[:8]
    return f"{name}|{protocol}|{'' if fold is None else fold}|{h}"


def done_keys(path):
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except Exception:
            pass
    return keys


def fit_and_score(Solver, params, protocol, fold, seed=42):
    """(model, scores, y_true, y_pred, info, fit_seconds) for one split."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = PE.route_b_data(protocol, fold)
    tr, te = data["train_loader"], data["test_loader"]
    meta = {k: v for k, v in data.items() if not k.endswith("_loader")}
    n_tr, n_te = len(tr.dataset), len(te.dataset)
    log(f"[data] protocol={protocol} fold={fold} train={n_tr} test={n_te} "
        f"chans={meta.get('n_chans')} times={meta.get('n_times')} "
        f"sfreq={meta.get('sfreq')} classes={meta.get('n_classes')}")
    with tempfile.TemporaryDirectory(dir=str(Path.home() / "neuralbench")) as sub:
        os.environ["COMPET_SUBMISSION_DIR"] = sub
        solver = Solver.get_instance(**params)
        t0 = time.time()
        solver.set_objective(train_loader=tr, **meta)
        solver.run(None)
        model = solver.get_result()["model"]
        fit_s = time.time() - t0
    scores, y_true, y_pred, info = PE.evaluate(model, te)
    return model, scores, y_true, y_pred, info, fit_s


def stream_probas(model, X, stream_id, y_pred):
    """(n, K) probabilities replayed exactly as streamed: each stream on a fresh
    deep copy + reset_state(), one window per call. Checked: argmax = y_pred."""
    from benchmark_utils.streaming import fresh_model
    P = None
    for s in np.unique(stream_id):
        m = np.where(stream_id == s)[0]
        f = fresh_model(model)
        for i in m:
            p = np.asarray(f.predict_proba(X[i:i + 1])).reshape(1, -1)
            if P is None:
                P = np.zeros((len(X), p.shape[1]))
            P[i] = p
    if not np.array_equal(P.argmax(1), np.asarray(y_pred).reshape(-1)):
        raise RuntimeError("probability replay: argmax != streamed y_pred "
                           f"({int((P.argmax(1) != y_pred).sum())} windows)")
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--name", required=True, help="arm name (row label)")
    ap.add_argument("--solver", default="riemann_sealed")
    ap.add_argument("--set", nargs="*", default=[], help="param=value overrides")
    ap.add_argument("--protocols", nargs="+", default=["lopo", "later"])
    ap.add_argument("--folds", default="0-9", help="lopo folds, e.g. 0-9 or 0,3")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    PC.require_env040()

    params = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        params[k] = parse_value(v)
    if "-" in a.folds:
        lo, hi = a.folds.split("-")
        folds = list(range(int(lo), int(hi) + 1))
    else:
        folds = [int(f) for f in a.folds.split(",")]
    jobs = [(p, f) for p in a.protocols for f in (folds if p == "lopo" else [None])]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probs").mkdir(exist_ok=True)
    lane_file = OUT / f"{a.lane}.jsonl"
    done = done_keys(lane_file)
    Solver = load_solver_class(a.solver)
    cache = PC.load()
    log(f"[config] lane={a.lane} name={a.name} solver={a.solver} params={params} "
        f"jobs={len(jobs)} seed={a.seed}")
    for protocol, fold in jobs:
        key = config_key(a.name, a.solver, {**params, "_seed": a.seed}, protocol, fold)
        if key in done:
            log(f"skip {key} (done)")
            continue
        t0 = time.time()
        model, sc, y_true, y_pred, info, fit_s = fit_and_score(
            Solver, params, protocol, fold, a.seed)
        sp = PS.split(protocol, fold)
        test_idx = np.asarray(sp.test)
        t1 = time.time()
        probas = stream_probas(model, np.asarray(cache.X[test_idx]),
                               np.asarray(info["stream_id"]), y_pred)
        log(f"[probas] replayed {len(probas)} windows in {time.time() - t1:.1f}s, argmax = y_pred")
        row = dict(key=key, lane=a.lane, name=a.name, solver=a.solver, params=params,
                   seed=a.seed, protocol=protocol, fold=fold,
                   cell_mean=sc["balanced_accuracy_cell_mean"],
                   session_mean=sc["balanced_accuracy_session_mean"],
                   bacc=sc["balanced_accuracy"], n_cells=sc["n_cells"],
                   cell_scores=sc["cell_scores"], fit_s=round(fit_s, 1),
                   eval_s=sc["seconds"], ms_per_window=sc["ms_per_window"],
                   n_test=int(len(y_pred)), time=time.strftime("%F %T"),
                   solver_sha1=hashlib.sha1((SOLVERS / f"{a.solver}.py").read_bytes()).hexdigest()[:10])
        np.savez_compressed(
            OUT / "probs" / (key.replace("|", "__") + ".npz"),
            y_true=y_true, y_pred=y_pred, test_idx=test_idx, probas=probas,
            stream_id=np.asarray(info["stream_id"]),
            context_id=np.asarray(info.get("context_id", np.zeros(len(y_pred)))))
        with open(lane_file, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"[row] {a.name} {protocol} fold={fold} cell_mean={row['cell_mean']:.6f} "
            f"n_cells={row['n_cells']} fit={fit_s:.1f}s eval={sc['seconds']:.1f}s "
            f"total={time.time() - t0:.1f}s")
    log(f"[done] {a.name}")


if __name__ == "__main__":
    main()
