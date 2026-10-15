#!/usr/bin/env python
"""Route A: a streamed local evaluation of PROTEUS protocols that calls the
organisers' own evaluator and metric, plus the way the platform loads a
submission folder, the test doubles and the gates G5e-G8.

Setup session 2026-10-15 (brief ``prompts/2026-10-13_proteus_train_test_setup.md``,
Phase 4). Nothing of the organisers' is re-implemented: the test rows of a
protocol are wrapped with ``benchmark_utils.data.make_loader`` (batch 1,
unshuffled, ids from the stable keys through ``proteus_splits.ids_of``,
``onset`` = window start as a sample index) and scored with
``benchmark_utils.streaming.predict_streams``, ``metrics.balanced_accuracy``
and ``metrics.group_scores`` exactly as ``tracks/bci_decoding/objective.py``
does. Results map back to cache rows by position, never by a returned onset.

    source ~/codabench/env040.sh
    # per-window predictions of a submission folder on a protocol (saves
    # y_pred.npy, y_true.npy and scores.json under --out):
    python ~/codabench/analysis/proteus_eval.py score <submission_dir> <protocol> [fold] --out <dir>
    # the gates G5e, G6, G7, G8 (G6's two CLI runs write under --out):
    python ~/codabench/analysis/proteus_eval.py gates --out ~/codabench/logs/proteus_setup/g6

Every score is for information only; the gates are exact comparisons.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

COMPET_ROOT = os.environ.get("COMPET_ROOT", str(Path.home() / "neuralbench/competition-040"))
sys.path.insert(0, COMPET_ROOT + "/tracks/bci_decoding")
sys.path.insert(0, str(Path.home() / "codabench/analysis"))

import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402
from benchmark_utils.data import make_loader, to_numpy  # noqa: E402
from benchmark_utils.metrics import balanced_accuracy, group_scores  # noqa: E402
from benchmark_utils.streaming import predict_streams  # noqa: E402

ROUTE_B = str(Path.home() / "codabench/datasets/proteus_protocols.py")
OFFICIAL_DATASET = "BCI[study=dreyer2026proteus,subset=full]"
SCORE_KEYS = ("balanced_accuracy_cell_mean", "balanced_accuracy_session_mean",
              "balanced_accuracy", "accuracy")


# -- route A -------------------------------------------------------------------

def test_loader(c, idx, device="cpu"):
    """The organisers' ``make_loader`` over the cache rows ``idx`` (a test set
    in stream order), B = 1, unshuffled, ids from the stable keys."""
    idx = np.asarray(idx, dtype=np.int64)
    ids = PS.ids_of(c.rows, idx)
    return make_loader(np.asarray(c.X[idx]), c.y[idx], batch_size=1, shuffle=False,
                       record_id=ids["record_id"], onset=ids["onset"],
                       stream_id=ids["stream_id"], context_id=ids["context_id"], device=device)


def scores_from(y_true, y_pred, info):
    """The objective's ``evaluate_result`` arithmetic on streamed predictions."""
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if len(y_pred) != len(info["stream_id"]):
        raise ValueError("predict(X) must return one class per window.")
    context = info.get("context_id", np.zeros(len(y_true), np.int64))
    cells = np.stack([info["stream_id"], context], axis=1)
    cell_scores = group_scores(balanced_accuracy, y_true, y_pred, cells)
    session_scores = group_scores(balanced_accuracy, y_true, y_pred, info["stream_id"])
    return dict(
        balanced_accuracy_cell_mean=float(cell_scores.mean()),
        balanced_accuracy_session_mean=float(session_scores.mean()),
        balanced_accuracy=balanced_accuracy(y_true, y_pred),
        accuracy=float(np.mean(y_true == y_pred)),
        n_cells=int(len(cell_scores)),
        cell_scores=[float(v) for v in cell_scores],
    )


def evaluate(model, loader):
    """``(scores, y_true, y_pred, info)`` of ``model`` streamed over ``loader``
    by the organisers' ``predict_streams``."""
    t0 = time.time()
    y_true, y_pred, info = predict_streams(model, loader)
    scores = scores_from(y_true, y_pred, info)
    scores["seconds"] = round(time.time() - t0, 3)
    scores["ms_per_window"] = round(1000.0 * (time.time() - t0) / max(len(y_pred), 1), 3)
    return scores, np.asarray(y_true), np.asarray(y_pred), info


# -- loading a submission folder the way the platform does -----------------------

def official_meta(loaders=None, meta=None):
    """The dataset dict of ``bci_studies.Dataset.get_data`` without its two
    loaders: what ``set_objective`` receives (``n_classes``, ``sfreq``,
    ``ch_names``, ``chs_info``, ``n_chans``, ``n_times``, ``device``)."""
    if loaders is None:
        loaders, meta = PC.official_loaders()
    d = PC.official_data_dict(loaders, meta)
    return {k: v for k, v in d.items() if not k.endswith("_loader")}


def load_submission(folder, meta):
    """Import ``<folder>/submission.py`` by path with
    ``COMPET_SUBMISSION_DIR=<folder>``, instantiate its ``Solver`` with the
    file's default parameters (``get_instance()``, as on Codabench), call
    ``set_objective(train_loader=None, **meta)`` and return the model."""
    folder = Path(folder).resolve()
    os.environ["COMPET_SUBMISSION_DIR"] = str(folder)
    name = "submission_" + "".join(ch if ch.isalnum() else "_" for ch in str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver = mod.Solver.get_instance()
    solver.set_objective(train_loader=None, **meta)
    return solver.get_result()["model"]


def route_b_data(protocol, fold=None):
    """Route B's ``Dataset.get_instance(protocol=..., fold=...).get_data()``,
    in-process (outside a benchopt run its seed is 0)."""
    spec = importlib.util.spec_from_file_location("proteus_protocols_module", ROUTE_B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Dataset.get_instance(protocol=protocol, fold=fold).get_data()


# -- test doubles ------------------------------------------------------------------

class ConstantModel:
    """Always class 0."""

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)


class LookupModel:
    """The true label of each window, found by the SHA-1 of its tensor."""

    def __init__(self, labels):
        self.labels = dict(labels)          # sha1 -> label

    def predict(self, X):
        return np.array([self.labels[PC.sha1_of(x)] for x in to_numpy(X)], dtype=np.int64)


class CellMarkerModel:
    """The look-up on Graz windows, (label + 1) mod 3 on BrainHero windows."""

    def __init__(self, labels, interfaces):
        self.labels, self.interfaces = dict(labels), dict(interfaces)

    def predict(self, X):
        out = []
        for x in to_numpy(X):
            h = PC.sha1_of(x)
            y = int(self.labels[h])
            out.append(y if self.interfaces[h] == "Graz" else (y + 1) % 3)
        return np.array(out, dtype=np.int64)


class CounterModel:
    """The number of windows seen so far in this stream (no modulo)."""

    def __init__(self):
        self.n = 0

    def predict(self, X):
        k = len(X)
        out = np.arange(self.n + 1, self.n + k + 1, dtype=np.int64)
        self.n += k
        return out


class MemoriserModel:
    """Hash -> label of the rows it was given (the training rows); class 0
    for an unknown tensor."""

    def __init__(self, labels):
        self.labels = dict(labels)

    def predict(self, X):
        return np.array([self.labels.get(PC.sha1_of(x), 0) for x in to_numpy(X)], dtype=np.int64)


# -- gates ---------------------------------------------------------------------------

def _r6(x):
    return round(float(x), 6)


def _hash_pass(loader):
    """(hashes, labels) served by a loader, in its order."""
    hs, ys = [], []
    for X, y, info in loader:
        for x, yy in zip(to_numpy(X), to_numpy(y)):
            hs.append(PC.sha1_of(x))
            ys.append(int(yy))
    return hs, ys


def check_served(c, official):
    """G5e: what route B's loaders serve, every protocol and fold."""
    out = []
    exp = {k: official[k] for k in ("n_classes", "sfreq", "ch_names", "chs_info", "n_chans",
                                    "n_times", "device")}
    for name, sp in PS.every_split(c.rows):
        d = route_b_data(sp.protocol, sp.fold)
        htr, ytr = _hash_pass(d["train_loader"])
        hte, yte = _hash_pass(d["test_loader"])
        set_tr, set_te = set(c.sha1[sp.train]), set(c.sha1[sp.test])
        out.append((f"G5e {name}: train serves the protocol's train set (set and count)",
                    set(htr) == set_tr and len(htr) == len(sp.train), f"{len(htr)} windows"))
        out.append((f"G5e {name}: test serves the protocol's test set (set and count)",
                    set(hte) == set_te and len(hte) == len(sp.test), f"{len(hte)} windows"))
        out.append((f"G5e {name}: every served label = cache label of its hash",
                    all(c.y[c.by_sha1[h]] == y for h, y in zip(htr + hte, ytr + yte)), "ok"))
        if sp.protocol == "all":
            out.append((f"G5e {name}: test hashes inside training", set(hte) <= set(htr), "ok"))
        else:
            out.append((f"G5e {name}: train and test hashes disjoint", not (set(htr) & set(hte)),
                        f"{len(set(htr) & set(hte))} shared"))
        out.append((f"G5e {name}: test hash sequence = protocol's test order",
                    hte == list(c.sha1[sp.test]), "ok"))
        got = {k: d[k] for k in exp}
        out.append((f"G5e {name}: n_classes, sfreq, ch_names, chs_info, n_chans, n_times, device = organisers'",
                    got == exp, f"n_classes {d['n_classes']} sfreq {d['sfreq']} n_chans {d['n_chans']} "
                                f"n_times {d['n_times']} device {d['device']}"))
    return out


def benchopt_inference(submission_dir, dataset, out_parquet, log_path, timeout=900):
    """One inference-only benchopt run with the brief's flags; returns the
    parquet's last row as a dict (and writes the log)."""
    out_parquet, log_path = Path(out_parquet), Path(log_path)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, COMPET_SUBMISSION_DIR=str(Path(submission_dir).resolve()))
    cmd = ["benchopt", "run", "tracks/bci_decoding", "-o", "BCI-decoding[training=False]",
           "-d", dataset, "-s", str(Path(submission_dir).resolve() / "submission.py"),
           "--seed", "42", "--no-cache", "--no-plot", "--no-html", "--no-display",
           "--output", str(out_parquet)]
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=COMPET_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                            timeout=timeout).returncode
    log = log_path.read_text(errors="replace")
    import pandas as pd
    row = {}
    if out_parquet.exists():
        df = pd.read_parquet(out_parquet)
        row = {k: df[k].iloc[-1] for k in df.columns if k.startswith("objective_")}
        row["dataset_name"] = df["dataset_name"].iloc[-1]
        row["solver_name"] = df["solver_name"].iloc[-1]
    row["rc"] = rc
    row["seconds"] = round(time.time() - t0, 1)
    # evidence that the exported model, not a fresh one, predicted: MeanLogReg
    # prints "[loading] ... weights.joblib into MeanLogReg" when it loads its
    # joblib; Riemann-Sealed prints "[Riemann-Sealed] predict: ..." lines, and
    # without its joblib it raises on the first window (brief section 4), so for
    # it the predict lines are the evidence. Neither is a "weights loaded" line
    # for the other model.
    row["inference_evidence"] = ("weights.joblib into MeanLogReg" in log
                                 or "[Riemann-Sealed] predict:" in log)
    return row


def check_routes_agree(c, ref_dir, out_dir, loaders, meta):
    """G6: the Phase 1 MeanLogReg export through the CLI on the organisers'
    dataset and on route B official (both = G1), and in-process y_pred from
    the organisers' test loader, route B's test loader and route A's loader."""
    import pandas as pd
    ref_dir, out_dir = Path(ref_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g1 = pd.read_parquet(ref_dir / "train.parquet").iloc[-1]
    g1v = {k: _r6(g1["objective_" + k]) for k in SCORE_KEYS}
    out = []
    for tag, dataset in (("official_loader", OFFICIAL_DATASET),
                         ("route_b_official", f"{ROUTE_B}[protocol=official]")):
        row = benchopt_inference(ref_dir / "submission", dataset, out_dir / f"g6_{tag}.parquet",
                                 out_dir / f"g6_{tag}.log")
        got = {k: _r6(row.get("objective_" + k, float("nan"))) for k in SCORE_KEYS}
        ok = (row["rc"] == 0 and row["inference_evidence"] and got == g1v
              and int(row.get("objective_n_cells", -1)) == 8)
        out.append((f"G6 CLI inference-only on {tag} = G1 (weights loaded, 8 cells)", ok,
                    f"{got} n_cells {row.get('objective_n_cells')} rc {row['rc']} "
                    f"loaded {row['inference_evidence']} {row['seconds']} s dataset {row.get('dataset_name')}"))
    model = load_submission(ref_dir / "submission", official_meta(loaders, meta))
    sp = PS.split("official", rows=c.rows)
    sc_a, _, yp_official, _ = evaluate(model, loaders["test"])
    sc_b, _, yp_b, _ = evaluate(model, route_b_data("official")["test_loader"])
    sc_c, yt_c, yp_c, _ = evaluate(model, test_loader(c, sp.test))
    np.save(out_dir / "g6_y_pred_official_loader.npy", yp_official)
    np.save(out_dir / "g6_y_pred_route_b.npy", yp_b)
    np.save(out_dir / "g6_y_pred_route_a.npy", yp_c)
    same = (len(yp_official) == len(yp_b) == len(yp_c) == 755 and np.array_equal(yp_official, yp_b)
            and np.array_equal(yp_official, yp_c))
    out.append(("G6 in-process y_pred: organisers' test loader = route B = route A, element by element",
                same, f"{len(yp_official)} / {len(yp_b)} / {len(yp_c)} windows"))
    out.append(("G6 in-process predict_streams on the organisers' loader = G1 scores",
                {k: _r6(sc_a[k]) for k in SCORE_KEYS} == g1v and sc_a["n_cells"] == 8,
                f"{ {k: _r6(sc_a[k]) for k in SCORE_KEYS} }"))
    out.append(("G6 route A scores = route B scores = organisers' loader scores (exact floats)",
                all(sc_a[k] == sc_b[k] == sc_c[k] for k in SCORE_KEYS), "ok"))
    out.append(("G6 route A y_true = cache labels of the test rows", np.array_equal(yt_c, c.y[sp.test]), "ok"))
    return out


EXPECTED_LENGTHS = {"official": [189, 189, 188, 189], "all": [189, 189, 188, 189],
                    "later": [189, 188, 186, 199]}


def check_doubles(c):
    """G7: the four test doubles on route A and route B, every protocol and fold."""
    out = []
    labels = dict(zip(c.sha1, c.y.tolist()))
    interfaces = dict(zip(c.sha1, c.rows["interface"]))
    for name, sp in PS.every_split(c.rows):
        lengths = PS.stream_lengths(c.rows, sp.test)
        if sp.protocol in EXPECTED_LENGTHS:
            out.append((f"G7 {name}: stream lengths from the cache keys = brief", lengths == EXPECTED_LENGTHS[sp.protocol],
                        str(lengths)))
        expected_counter = np.concatenate([np.arange(1, n + 1) for n in lengths])
        for route, loader in (("A", test_loader(c, sp.test)), ("B", route_b_data(sp.protocol, sp.fold)["test_loader"])):
            sc, _, yp, _ = evaluate(ConstantModel(), loader)
            out.append((f"G7 {name} route {route}: constant model cell mean exactly 0.333333",
                        _r6(sc["balanced_accuracy_cell_mean"]) == 0.333333, f"{sc['balanced_accuracy_cell_mean']:.6f}"))
            sc, _, yp, _ = evaluate(LookupModel(labels), loader)
            out.append((f"G7 {name} route {route}: look-up model exactly 1.000000",
                        sc["balanced_accuracy_cell_mean"] == 1.0 and sc["accuracy"] == 1.0, f"{sc['balanced_accuracy_cell_mean']:.6f}"))
            sc, _, yp, _ = evaluate(CellMarkerModel(labels, interfaces), loader)
            cs = sc["cell_scores"]
            n_exp = PS.LOPO_CELLS[sp.fold] if sp.protocol == "lopo" else 8
            ok = (len(cs) == n_exp and all(v in (0.0, 1.0) for v in cs)
                  and cs.count(1.0) == cs.count(0.0) == len(cs) // 2
                  and _r6(sc["balanced_accuracy_cell_mean"]) == 0.5)
            out.append((f"G7 {name} route {route}: cell-marker: {n_exp} cells exactly 1.0 / 0.0, half each, mean 0.500000",
                        ok, f"{len(cs)} cells {sorted(cs)} mean {sc['balanced_accuracy_cell_mean']:.6f}"))
            base = CounterModel()
            _, _, yp, _ = evaluate(base, loader)
            out.append((f"G7 {name} route {route}: counter = 1..n_s per stream {lengths}; base counter still 0",
                        np.array_equal(yp, expected_counter) and base.n == 0,
                        f"max {int(yp.max())} base.n {base.n}"))
    return out


def check_no_shared(c):
    """G8: a model memorising the training rows recognises 0 test windows in
    official, later and every lopo fold, and all 755 in all."""
    out = []
    for name, sp in PS.every_split(c.rows):
        mem = {c.sha1[i]: int(c.y[i]) for i in sp.train}
        hits = sum(h in mem for h in c.sha1[sp.test])
        for route, loader in (("A", test_loader(c, sp.test)), ("B", route_b_data(sp.protocol, sp.fold)["test_loader"])):
            sc, _, _, _ = evaluate(MemoriserModel(mem), loader)
            cm = _r6(sc["balanced_accuracy_cell_mean"])
            if sp.protocol == "all":
                ok = hits == 755 == len(sp.test) and cm == 1.0
                out.append((f"G8 {name} route {route}: memoriser recognises all 755 (exactly 1.000000)", ok,
                            f"{hits} recognised, cell mean {cm:.6f}"))
            else:
                ok = hits == 0 and cm == 0.333333
                out.append((f"G8 {name} route {route}: memoriser recognises 0 windows (exactly 0.333333)", ok,
                            f"{hits} recognised, cell mean {cm:.6f}"))
    return out


def run_gates(out_dir, ref_dir=None, c=None, loaders=None, meta=None, which=("G5e", "G6", "G7", "G8")):
    """Run the Phase 4 gates; returns the list of (name, ok, value)."""
    c = c or PC.load()
    ref_dir = Path(ref_dir or Path.home() / "codabench/logs/proteus_setup/ref_meanlogreg")
    if loaders is None:
        loaders, meta = PC.official_loaders()
    checks = []
    t = {}
    if "G5e" in which:
        t0 = time.time()
        checks += check_served(c, official_meta(loaders, meta))
        t["G5e"] = time.time() - t0
    if "G6" in which:
        t0 = time.time()
        checks += check_routes_agree(c, ref_dir, out_dir, loaders, meta)
        t["G6"] = time.time() - t0
    if "G7" in which:
        t0 = time.time()
        checks += check_doubles(c)
        t["G7"] = time.time() - t0
    if "G8" in which:
        t0 = time.time()
        checks += check_no_shared(c)
        t["G8"] = time.time() - t0
    print("timings: " + ", ".join(f"{k} {v:.1f} s" for k, v in t.items()), flush=True)
    return checks


# -- G9 / G10: the exported solver ---------------------------------------------------

def check_export_replay(c, export, replay, unzipped, export2, loaders, meta, out_dir=None):
    """G9 in-process: the exported folder, its read-only copy, the unzipped
    folder and a second training run give identical ``y_pred`` on the
    ``official`` test rows (route A); the loaded model has ``personal ==
    "pooled"`` and ``align == "none"``."""
    out = []
    sp = PS.split("official", rows=c.rows)
    loader = test_loader(c, sp.test)
    preds, base, base_scores = {}, None, None
    for tag, folder in (("export", export), ("replay", replay), ("unzipped", unzipped), ("export2", export2)):
        if folder is None:
            continue
        model = load_submission(folder, official_meta(loaders, meta))
        sc, _, yp, _ = evaluate(model, loader)
        preds[tag] = yp
        if out_dir is not None:
            np.save(Path(out_dir) / f"g9_y_pred_{tag}.npy", yp)
        if base is None:
            base, base_scores = model, sc
    for tag, yp in preds.items():
        if tag == "export":
            continue
        out.append((f"G9 y_pred {tag} == export (route A, official test, {len(yp)} windows)",
                    np.array_equal(yp, preds["export"]), f"{int((yp != preds['export']).sum())} differ"))
    out.append(("G9 loaded model personal == 'pooled' and align == 'none'",
                getattr(base, "personal", None) == "pooled" and getattr(base, "align", None) == "none",
                f"personal={getattr(base, 'personal', None)!r} align={getattr(base, 'align', None)!r} "
                f"adapt={getattr(base, 'adapt', None)!r}"))
    out.append(("G9 (info) route A scores of the export", True,
                " ".join(f"{k}={base_scores[k]:.6f}" for k in SCORE_KEYS) + f" n_cells={base_scores['n_cells']} "
                f"{base_scores['ms_per_window']} ms/window"))
    return out


def check_streaming_contract(c, export, loaders, meta, fit=False):
    """G10 on the fitted ``official`` model: B = 1 vs one batch, deepcopy,
    reset_state, prediction shape and values, the 20 highest-amplitude
    windows, counters after a reload (and after one in-process fit)."""
    import copy
    out = []
    m = official_meta(loaders, meta)
    model = load_submission(export, m)
    out.append(("G10 after reload: _pred['calls'] == 0 and _buf == {}",
                model._pred["calls"] == 0 and model._buf == {}, f"{model._pred} {model._buf}"))
    sp = PS.split("official", rows=c.rows)
    X = torch.as_tensor(np.asarray(c.X[sp.test]))
    P_batch = np.asarray(model.predict_proba(X))
    t0 = time.time()
    P_one = np.concatenate([np.asarray(model.predict_proba(X[i:i + 1])) for i in range(len(X))])
    ms = 1000.0 * (time.time() - t0) / len(X)
    same = bool((P_batch.argmax(1) == P_one.argmax(1)).all())
    maxdiff = float(np.abs(P_batch - P_one).max())
    out.append((f"G10 B = 1 vs one batch: same class on every window ({len(X)})", same,
                f"{int((P_batch.argmax(1) != P_one.argmax(1)).sum())} differ"))
    out.append(("G10 (report) largest probability difference B = 1 vs batch, flag above 1e-9",
                True, f"{maxdiff:.3e}" + (" FLAG > 1e-9" if maxdiff > 1e-9 else " ok")))
    try:
        clone = copy.deepcopy(model)
        # same call shape on both (a batch of 20 and 20 single windows differ
        # by ~1e-11 in probability): exact equality expected
        P_clone = np.asarray(clone.predict_proba(X[:20]))
        P_same = np.asarray(model.predict_proba(X[:20]))
        out.append(("G10 copy.deepcopy works and the copy predicts the same (20 windows, exact)",
                    np.array_equal(P_clone, P_same), "ok"))
    except Exception as e:
        out.append(("G10 copy.deepcopy works", False, repr(e)[:100]))
    model.reset_state()
    P_after = np.concatenate([np.asarray(model.predict_proba(X[i:i + 1])) for i in range(50)])
    out.append(("G10 predictions unchanged after reset_state() (50 windows, exact)",
                np.array_equal(P_after, P_one[:50]), "ok"))
    model.reset_state()
    out.append(("G10 reset_state(): _buf == {} and _pred zeroed", model._buf == {} and model._pred["calls"] == 0,
                f"{model._pred}"))
    shapes_ok, vals_ok = True, True
    for i in range(len(X)):
        p = np.asarray(to_numpy(model.predict(X[i:i + 1])))
        shapes_ok &= p.shape == (1,)
        vals_ok &= bool(np.issubdtype(p.dtype, np.integer)) and int(p[0]) in (0, 1, 2)
    out.append((f"G10 every prediction has shape (1,) and is an integer in {{0, 1, 2}} ({len(X)} windows)",
                shapes_ok and vals_ok, "ok"))
    amax = np.abs(np.asarray(c.X)).reshape(len(c.X), -1).max(axis=1)
    top = np.argsort(-amax)[:20]
    P_top = np.concatenate([np.asarray(model.predict_proba(torch.as_tensor(np.asarray(c.X[j:j + 1]))))
                            for j in top])
    out.append(("G10 the 20 highest-amplitude windows give finite outputs", bool(np.isfinite(P_top).all()),
                f"max |x| {amax[top].max():.1f} .. {amax[top].min():.1f} uV"))
    out.append(("G10 (report) ms per window at B = 1, in-process route A loop", True,
                f"{ms:.2f} ms/window; projected 106 sessions x 199 windows = {ms * 106 * 199 / 1000:.0f} s"))
    if fit:
        fresh = load_submission(export, m)
        t0 = time.time()
        fresh.fit(route_b_data("official")["train_loader"])
        out.append(("G10 right after one in-process fit (route B official train rows): _pred['calls'] == 0 and _buf == {}",
                    fresh._pred["calls"] == 0 and fresh._buf == {}, f"{fresh._pred} {fresh._buf} fit {time.time() - t0:.1f} s"))
    return out


def equal_parquets(out_md, named, zip_path=None):
    """Exact equality of the ``objective_*`` columns across parquets; writes a
    small RESULTS.md and returns True when all equal."""
    import hashlib
    import zipfile
    import pandas as pd
    rows, keys = {}, []
    for name, path in named:
        df = pd.read_parquet(path)
        r = df.iloc[-1]
        # the scores only: objective_name reads training=True / False
        cols = [k for k in df.columns if k.startswith("objective_") and k != "objective_name"]
        keys += [k for k in cols if k not in keys]          # the union over all parquets
        rows[name] = {k: r[k] for k in cols}
        rows[name]["dataset_name"] = r["dataset_name"]
        rows[name]["solver_name"] = r["solver_name"]
    base = next(iter(rows))
    # a column missing from one parquet counts as different
    equal = all(all(k in rows[n] and k in rows[base] and rows[n][k] == rows[base][k] for k in keys)
                for n in rows)
    for n in rows:
        for k in keys:
            rows[n].setdefault(k, float("nan"))
    lines = [f"# {Path(out_md).parent.name}: benchopt scores (for information only; exact comparison)", "",
             "| run | dataset | solver | " + " | ".join(k.replace("objective_", "") for k in keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for n, r in rows.items():
        lines.append(f"| {n} | `{r['dataset_name']}` | {r['solver_name']} | " + " | ".join(
            f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
    lines.append("")
    lines.append(f"**{'EQUAL' if equal else 'DIFFERENT'}** across {len(rows)} runs ({', '.join(rows)}).")
    if zip_path and not Path(zip_path).exists():
        equal = False
        lines += ["", f"**ZIP MISSING**: `{zip_path}` does not exist."]
    if zip_path and Path(zip_path).exists():
        zp = Path(zip_path)
        sha = hashlib.sha256(zp.read_bytes()).hexdigest()
        names = zipfile.ZipFile(zp).namelist()
        lines += ["", f"Zip `{zp}`: {zp.stat().st_size} bytes ({zp.stat().st_size / 2**20:.2f} MiB), sha256 `{sha}`, "
                  f"entries {names} ({'files at the zip root' if all('/' not in n for n in names) else 'NOT at the root'})"]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return equal


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gates")
    g.add_argument("--out", required=True)
    g.add_argument("--only", nargs="*", default=None)
    s = sub.add_parser("score")
    s.add_argument("submission_dir")
    s.add_argument("protocol")
    s.add_argument("fold", nargs="?", type=int, default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--official-loader", action="store_true",
                   help="score on the organisers' test loader instead of route A")
    q = sub.add_parser("g9g10")
    q.add_argument("--export", required=True)
    q.add_argument("--replay")
    q.add_argument("--unzipped")
    q.add_argument("--export2")
    q.add_argument("--out", required=True)
    q.add_argument("--fit", action="store_true")
    q.add_argument("--no-g9", action="store_true")
    e = sub.add_parser("equal")
    e.add_argument("out_md")
    e.add_argument("named", nargs="+", help="name=path.parquet")
    e.add_argument("--zip")
    a = ap.parse_args()
    PC.require_env040()
    if a.cmd == "score" and a.official_loader and a.protocol != "official":
        sys.exit("--official-loader scores the organisers' official test set: use protocol 'official'")
    if a.cmd == "equal":
        ok = equal_parquets(a.out_md, [n.split("=", 1) for n in a.named], a.zip)
        sys.exit(0 if ok else 1)
    if a.cmd == "gates":
        checks = run_gates(a.out, which=tuple(a.only) if a.only else ("G5e", "G6", "G7", "G8"))
        ok = PC.report(checks)
        print(f"G5e-G8 {'PASS' if ok else 'FAIL'} ({len(checks)} checks)", flush=True)
        sys.exit(0 if ok else 1)
    c = PC.load()
    loaders, meta = PC.official_loaders()
    if a.cmd == "g9g10":
        checks = []
        if not a.no_g9:
            checks += check_export_replay(c, a.export, a.replay, a.unzipped, a.export2, loaders, meta, a.out)
        checks += check_streaming_contract(c, a.export, loaders, meta, fit=a.fit)
        ok = PC.report(checks)
        print(f"G9/G10 {'PASS' if ok else 'FAIL'} ({len(checks)} checks)", flush=True)
        sys.exit(0 if ok else 1)
    model = load_submission(a.submission_dir, official_meta(loaders, meta))
    sp = PS.split(a.protocol, a.fold, rows=c.rows)
    loader = loaders["test"] if a.official_loader else test_loader(c, sp.test)
    print(f"[proteus_eval] score {a.submission_dir} on {sp.name}: test X=({len(sp.test)}, 41, 480)"
          + (" via the organisers' test loader" if a.official_loader else " via route A")
          + (" SCORE MEANINGLESS (test inside train)" if sp.score_meaningless else ""), flush=True)
    scores, y_true, y_pred, info = evaluate(model, loader)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_pred.npy", y_pred)
    np.save(out / "y_true.npy", y_true)
    scores.update(protocol=sp.name, submission_dir=str(Path(a.submission_dir).resolve()),
                  score_meaningless=sp.score_meaningless, n_windows=int(len(y_pred)),
                  note="for information only")
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    print(json.dumps({k: v for k, v in scores.items() if k != "cell_scores"}), flush=True)


if __name__ == "__main__":
    main()
