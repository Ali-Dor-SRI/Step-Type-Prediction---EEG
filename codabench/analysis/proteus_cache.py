#!/usr/bin/env python
"""One window cache of all 2,655 PROTEUS windows, built from the official loader.

Setup session 2026-10-15 (brief ``prompts/2026-10-13_proteus_train_test_setup.md``,
Phase 2). The cache is what every protocol, loader and gate of ours reads; it is
built by calling the organisers' ``load_task`` exactly as
``$COMPET_ROOT/tracks/bci_decoding/datasets/bci_studies.py`` does and reading
the three official datasets (train, val, test) item by item, each item keyed by
its trigger-table row (``ds.seg_ds.triggers.iloc[int(ds.index[i])]``: positional,
the table's index labels do not start at 0).

Layout of ``$PROTEUS_CACHE_ROOT`` (default ``~/neuralbench/proteus_cache``):

- ``X.npy``: ``(2655, 41, 480)`` float32, microvolts, 120 Hz, exactly the
  tensors the loader serves (no scaling, no clamping), in **cache order** =
  sorted by (person, session, run, cue): every stream contiguous, its runs by
  run number, each run's windows by cue time.
- ``rows.parquet``: one row per window, cache order. Stable keys ``person``
  (``'P04'``), ``session`` (int), ``run`` (int), ``interface`` (``'Graz'`` /
  ``'BrainHero'``), ``task`` (the BIDS task), ``cue`` (s, the trigger table's
  ``start``), ``start`` (s, the window start = ``info["onset"]`` = cue + 1.0);
  ``label`` (0 mi / 1 sub / 2 word), ``code`` and ``description`` from the
  table, the organisers' ``split`` (train / val / test), ``loader_pos`` (the
  item's position in that official dataset) and ``sha1`` of the tensor bytes.
- ``keys.csv``: the stable keys + label, for the ``-I`` events check (G4).
- ``meta.json``: channel names, units, sampling rate, neuralbench version,
  upstream commit, task name, build time, counts.

    source ~/codabench/env040.sh
    python ~/codabench/analysis/proteus_cache.py build     # ~15 s
    python ~/codabench/analysis/proteus_cache.py check     # G2 + G3

The integer ids of the official loaders' ``info`` (``record_id``,
``context_id``, ``stream_id``, ``subject_id``) are numbered by first
appearance in each loader and are **not** stored as keys (brief § 1 item 7).
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

COMPET_ROOT = os.environ.get("COMPET_ROOT", str(Path.home() / "neuralbench/competition-040"))
sys.path.insert(0, COMPET_ROOT + "/tracks/bci_decoding")
sys.path.insert(0, str(Path.home() / "codabench/analysis"))

CACHE_ROOT = Path(os.environ.get("PROTEUS_CACHE_ROOT", Path.home() / "neuralbench/proteus_cache"))
DATA_DIR = Path(os.environ.get("BENCHOPT_DATA_HOME",
                               Path.home() / "neuralbench/benchopt_data_040")) / "neural_compet"
TASK = "_motor_imagery_stream"
STUDY = "dreyer2026proteus"

N_WINDOWS, N_CHANS, N_TIMES, SFREQ = 2655, 41, 480, 120.0
CLASSES = ("mi", "sub", "word")          # label 0, 1, 2
CLASS_COUNTS = (885, 886, 884)
INTERFACES = ("Graz", "BrainHero")       # context_id 0, 1 (ours, from the stable keys)
CH_NAMES = ["C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1", "P1", "P3", "PO7", "P5", "TP7",
            "CP5", "CP3", "C3", "PO8", "P6", "P4", "TP8", "CP6", "CP4", "C6", "C4", "FC6", "FC4",
            "FC2", "F6", "F4", "F2", "AF4", "AF3", "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1",
            "FCz", "C5"]
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
# (person, session) -> windows, brief § 1
SESSION_COUNTS = {("P01", 1): 190, ("P02", 1): 189, ("P03", 1): 189, ("P04", 1): 189,
                  ("P04", 2): 189, ("P04", 3): 188, ("P05", 1): 190, ("P06", 1): 187,
                  ("P07", 1): 189, ("P08", 1): 189, ("P09", 1): 189, ("P10", 1): 192,
                  ("P10", 2): 186, ("P10", 3): 199}
OFFICIAL_SPLIT = {"train": ("P01", "P02", "P03", "P06", "P07", "P10"),
                  "val": ("P05", "P08"), "test": ("P04", "P09")}
SPLIT_SIZES = {"train": 1521, "val": 379, "test": 755}
KEY_COLUMNS = ["person", "session", "run", "cue"]


def require_env040():
    """Refuse to run outside the 0.4.0 environment (G0 in four lines): the old
    `env.sh` world would otherwise run the 0.3.1 benchopt against this
    checkout. Called by every CLI entry of the PROTEUS tooling."""
    import importlib.metadata as m
    problems = []
    try:
        v = m.version("neuralbench")
    except Exception:            # noqa: BLE001
        v = "missing"
    if v != "0.4.0":
        problems.append(f"neuralbench {v} != 0.4.0")
    if not sys.prefix.endswith(".venv040"):
        problems.append(f"sys.prefix {sys.prefix} is not .venv040")
    if not os.environ.get("BENCHOPT_DATA_HOME", "").endswith("benchopt_data_040"):
        problems.append("BENCHOPT_DATA_HOME does not end with benchopt_data_040")
    if "/competition-040/" not in COMPET_ROOT + "/":
        problems.append(f"COMPET_ROOT {COMPET_ROOT} is not competition-040")
    if problems:
        sys.exit("wrong environment (source ~/codabench/env040.sh first): " + "; ".join(problems))


def sha1_of(x):
    """SHA-1 of one window's float32 bytes, ``x`` of shape (41, 480) (a torch
    tensor, a numpy array, or a (1, 41, 480) batch of one)."""
    x = np.asarray(_to_numpy(x), dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.shape != (N_CHANS, N_TIMES):
        raise ValueError(f"sha1_of expects a (41, 480) window, got {x.shape}")
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cue_ms(cue):
    """The cue time as an integer millisecond count (an exact dict key)."""
    return int(np.round(np.asarray(cue, dtype=np.float64) * 1000.0)) if np.ndim(cue) == 0 \
        else np.round(np.asarray(cue, dtype=np.float64) * 1000.0).astype(np.int64)


# -- the official loader ----------------------------------------------------

def official_loaders(batch_size=64, seed=0, device="cpu", num_workers=0):
    """``load_task`` exactly as ``bci_studies.Dataset._load`` calls it."""
    from benchmark_utils.nb_task import load_task
    from datasets.bci_studies import _proteus_interface, _proteus_run
    return load_task(
        "eeg", TASK, data_dir=DATA_DIR, dataset=None, device=device,
        batch_size=batch_size, seed=seed, num_workers=num_workers, subset="full",
        target_transform=lambda y: y.argmax(-1),
        context_of=_proteus_interface, order_of=_proteus_run)


def official_data_dict(loaders, meta):
    """The dict ``bci_studies.Dataset.get_data`` returns, built from
    ``load_task``'s output as its lines 157-165 do."""
    meta = dict(meta)
    return dict(train_loader=loaders["train"], test_loader=loaders["test"],
                n_classes=int(meta.pop("raw_target_shape")[-1]),
                **{k: v for k, v in meta.items() if k != "target_shape"})


def interface_of(task):
    from datasets.bci_studies import _PROTEUS_INTERFACES
    return _PROTEUS_INTERFACES[task]


def iter_items(ds):
    """``(i, X, y, info, trigger_row)`` for every item of an official dataset,
    read as ``ds[i]`` (never by iterating a loader: train is shuffled)."""
    trig = ds.seg_ds.triggers
    for i in range(len(ds)):
        X, y, info = ds[i]
        yield i, X, y, info, trig.iloc[int(ds.index[i])]


def key_of(row):
    """Stable key of a trigger-table row: (person, session, run, cue_ms)."""
    return (str(row["subject"]).split("/")[-1], int(row["session"]), int(row["run"]),
            cue_ms(float(row["start"])))


# -- build -------------------------------------------------------------------

def build(root=CACHE_ROOT):
    t0 = time.time()
    loaders, meta = official_loaders()
    print(f"[proteus_cache] load_task {time.time() - t0:.1f} s; meta sfreq={meta['sfreq']} "
          f"n_chans={meta['n_chans']} n_times={meta['n_times']}", flush=True)
    rows, Xs = [], []
    for split in ("train", "val", "test"):
        ds = loaders[split].dataset
        t1 = time.time()
        for i, X, y, info, r in iter_items(ds):
            x = np.ascontiguousarray(_to_numpy(X), dtype=np.float32)
            person, session, run, _ = key_of(r)
            rows.append(dict(
                person=person, session=session, run=run, interface=interface_of(r["task"]),
                task=str(r["task"]), cue=float(r["start"]), start=float(info["onset"]),
                label=int(_to_numpy(y)), code=int(r["code"]), description=str(r["description"]),
                split=split, loader_pos=int(i), sha1=sha1_of(x)))
            Xs.append(x)
        print(f"[proteus_cache] {split}: {len(ds)} items in {time.time() - t1:.1f} s", flush=True)
    df = pd.DataFrame(rows)
    X = np.stack(Xs)
    order = np.lexsort((df["cue"].to_numpy(), df["run"].to_numpy(),
                        df["session"].to_numpy(), df["person"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    X = np.ascontiguousarray(X[order])
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X.npy", X)
    df.to_parquet(root / "rows.parquet", index=False)
    df[["person", "session", "run", "task", "cue", "label"]].to_csv(root / "keys.csv", index=False)
    import importlib.metadata as m
    try:
        commit = subprocess.run(["git", "-C", COMPET_ROOT, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:            # not a git checkout: record what we can
        commit = "unknown"
    meta_out = dict(
        ch_names=list(meta["ch_names"]), units="microvolts (uV); no scaling, no clamping",
        sfreq=float(meta["sfreq"]), n_chans=int(meta["n_chans"]), n_times=int(meta["n_times"]),
        n_windows=int(len(df)), classes=list(CLASSES), interfaces=list(INTERFACES),
        neuralbench_version=m.version("neuralbench"), upstream_commit=commit,
        compet_root=COMPET_ROOT, data_dir=str(DATA_DIR), task=f"eeg/{TASK}", study=STUDY,
        loader_call="load_task(dataset=None, device='cpu', batch_size=64, seed=0, num_workers=0, "
                    "subset='full', target_transform=argmax, context_of=_proteus_interface, "
                    "order_of=_proteus_run)",
        order="sorted by (person, session, run, cue)",
        built=time.strftime("%Y-%m-%d %H:%M:%S %Z"), build_seconds=round(time.time() - t0, 1))
    (root / "meta.json").write_text(json.dumps(meta_out, indent=1))
    print(f"[proteus_cache] built {root}: X {X.shape} {X.dtype}, {len(df)} rows, "
          f"{time.time() - t0:.1f} s", flush=True)
    return root


# -- load --------------------------------------------------------------------

class Cache:
    """The cache as arrays: ``X`` (memory-mapped), ``rows``, ``meta`` and the
    key -> row and sha1 -> row maps."""

    def __init__(self, root=CACHE_ROOT):
        root = Path(root)
        if not (root / "meta.json").exists():
            raise RuntimeError(f"no PROTEUS cache at {root}: build it with "
                               "`python ~/codabench/analysis/proteus_cache.py build`")
        self.root = root
        self.meta = json.loads((root / "meta.json").read_text())
        self.X = np.load(root / "X.npy", mmap_mode="r")
        self.rows = pd.read_parquet(root / "rows.parquet")
        self.y = self.rows["label"].to_numpy(dtype=np.int64)
        self.sha1 = self.rows["sha1"].to_numpy()
        self.ms = cue_ms(self.rows["cue"].to_numpy())
        self.by_key = {(p, int(s), int(r), int(m)): i for i, (p, s, r, m) in enumerate(zip(
            self.rows["person"], self.rows["session"], self.rows["run"], self.ms))}
        self.by_sha1 = {h: i for i, h in enumerate(self.sha1)}
        if len(self.by_key) != len(self.rows) or len(self.by_sha1) != len(self.rows):
            raise RuntimeError("cache keys or hashes are not unique")

    def __len__(self):
        return len(self.rows)

    def row_of(self, trigger_row):
        return self.by_key[key_of(trigger_row)]

    def keys(self):
        """The (person, session, run, cue_ms) key of every row, cache order."""
        return list(zip(self.rows["person"], self.rows["session"].astype(int),
                        self.rows["run"].astype(int), self.ms.astype(int)))


def load(root=CACHE_ROOT):
    return Cache(root)


# -- gates ---------------------------------------------------------------------

def check_content(c):
    """G2: one (name, ok, value) per check."""
    out = []
    d = c.rows
    X = c.X
    out.append(("shape (2655, 41, 480) float32", X.shape == (N_WINDOWS, N_CHANS, N_TIMES)
                and X.dtype == np.float32, f"{X.shape} {X.dtype}"))
    nan = int(np.isnan(np.asarray(X)).sum())
    out.append(("no NaN", nan == 0, f"{nan} NaN"))
    cnt = tuple(int((c.y == k).sum()) for k in range(3))
    out.append(("classes 885 / 886 / 884", cnt == CLASS_COUNTS, str(cnt)))
    out.append(("labels in {0,1,2} and == code and description", bool(
        set(np.unique(c.y).tolist()) == {0, 1, 2} and (d["label"] == d["code"]).all()
        and all(CLASSES[k] == s for k, s in zip(d["label"], d["description"]))), "ok"))
    npeople = d["person"].nunique()
    nsess = d.groupby(["person", "session"]).ngroups
    nruns = d.groupby(["person", "session", "run"]).ngroups
    out.append(("10 people, 14 sessions, 84 runs", (npeople, nsess, nruns) == (10, 14, 84),
                f"{npeople}, {nsess}, {nruns}"))
    out.append(("people P01..P10", tuple(sorted(d["person"].unique())) == PEOPLE,
                ",".join(sorted(d["person"].unique()))))
    sc = {(p, int(s)): int(n) for (p, s), n in d.groupby(["person", "session"]).size().items()}
    out.append(("per-session counts (brief § 1)", sc == SESSION_COUNTS,
                " ".join(str(sc[k]) for k in sorted(sc))))
    cells = d.groupby(["person", "session", "interface"]).size()
    graz = cells.xs("Graz", level="interface")
    bh = cells.xs("BrainHero", level="interface")
    out.append(("28 cells", len(cells) == 28, str(len(cells))))
    out.append(("Graz cells exactly 96 windows", bool((graz == 96).all()),
                f"{graz.min()}..{graz.max()} over {len(graz)} cells"))
    out.append(("BrainHero cells 90..103", bool(bh.min() >= 90 and bh.max() <= 103),
                f"{bh.min()}..{bh.max()} over {len(bh)} cells"))
    cls_cell = d.groupby(["person", "session", "interface"])["label"].nunique()
    cls_run = d.groupby(["person", "session", "run"])["label"].nunique()
    out.append(("every cell and run holds all 3 classes",
                bool((cls_cell == 3).all() and (cls_run == 3).all()),
                f"min classes cell {cls_cell.min()} run {cls_run.min()}"))
    runs_per_sess = d.groupby(["person", "session"])["run"].nunique()
    runset = d.groupby(["person", "session"])["run"].agg(lambda s: tuple(sorted(set(s))))
    out.append(("every session has runs 1..6", bool((runs_per_sess == 6).all()
                and all(t == (1, 2, 3, 4, 5, 6) for t in runset)), str(sorted(set(runset)))))
    sp = {k: int(v) for k, v in d["split"].value_counts().items()}
    out.append(("organisers' split 1521 / 379 / 755", sp == SPLIT_SIZES, str(sp)))
    sp_people = {k: tuple(sorted(d.loc[d["split"] == k, "person"].unique())) for k in SPLIT_SIZES}
    out.append(("split people", all(sp_people[k] == tuple(sorted(OFFICIAL_SPLIT[k]))
                                    for k in SPLIT_SIZES), str(sp_people)))
    out.append(("channel names == brief § 1 in order", list(c.meta["ch_names"]) == CH_NAMES,
                "ok" if list(c.meta["ch_names"]) == CH_NAMES else str(c.meta["ch_names"])))
    out.append(("sfreq 120, units microvolts", c.meta["sfreq"] == SFREQ
                and c.meta["units"].startswith("microvolts"), f"{c.meta['sfreq']} {c.meta['units']}"))
    dt = np.abs(d["start"].to_numpy() - (d["cue"].to_numpy() + 1.0))
    out.append(("window start = cue + 1.000 s", bool(dt.max() < 1e-6), f"max |diff| {dt.max():.2e} s"))
    out.append(("2655 distinct SHA-1", len(set(c.sha1)) == N_WINDOWS, str(len(set(c.sha1)))))
    out.append(("keys (person, session, run, cue) unique", len(c.by_key) == N_WINDOWS,
                str(len(c.by_key))))
    amax = np.abs(np.asarray(X)).reshape(len(X), -1).max(axis=1)
    n200 = int((amax > 200).sum())
    xmin, xmax = float(np.asarray(X).min()), float(np.asarray(X).max())
    out.append(("925 windows with |x| > 200 uV", n200 == 925, str(n200)))
    out.append(("min -4417.8, max 3511.5 (1 dp)", (round(xmin, 1), round(xmax, 1)) == (-4417.8, 3511.5),
                f"{xmin:.1f} / {xmax:.1f}"))
    ordered = bool((d.index == np.lexsort((d["cue"], d["run"], d["session"], d["person"]))).all())
    out.append(("cache order = (person, session, run, cue)", ordered, "ok"))
    return out


def check_equal_loader(c, loaders=None):
    """G3: every item of every official dataset equals its cached row."""
    if loaders is None:
        loaders, _ = official_loaders()
    out = []
    for split in ("train", "val", "test"):
        ds = loaders[split].dataset
        n, maxdiff, lab_ok, onset_ok, code_ok, split_ok = 0, 0.0, True, True, True, True
        for i, X, y, info, r in iter_items(ds):
            j = c.row_of(r)
            x = np.asarray(_to_numpy(X), dtype=np.float32)
            maxdiff = max(maxdiff, float(np.abs(x - c.X[j]).max()))
            lab_ok &= int(_to_numpy(y)) == int(c.y[j])
            code_ok &= int(r["code"]) == int(c.y[j])
            onset_ok &= abs(float(info["onset"]) - (float(r["start"]) + 1.0)) < 1e-6
            split_ok &= c.rows["split"].iat[j] == split and c.rows["loader_pos"].iat[j] == i
            n += 1
        out.append((f"G3 {split}: {n} items, max |X - cache| == 0.0", n == len(ds) and maxdiff == 0.0,
                    f"max diff {maxdiff}"))
        out.append((f"G3 {split}: labels == cache == table code", bool(lab_ok and code_ok), "ok"))
        out.append((f"G3 {split}: info onset == start + 1.0", bool(onset_ok), "ok"))
        out.append((f"G3 {split}: split and loader position recorded", bool(split_ok), "ok"))
    return out


def report(checks, prefix=""):
    ok_all = True
    for name, ok, value in checks:
        ok_all &= bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {prefix}{name}: {value}", flush=True)
    return ok_all


if __name__ == "__main__":
    require_env040()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build()
    elif cmd == "check":
        c = load()
        t0 = time.time()
        ok = report(check_content(c), "G2 ")
        print(f"G2 {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)", flush=True)
        t0 = time.time()
        ok3 = report(check_equal_loader(c))
        print(f"G3 {'PASS' if ok3 else 'FAIL'} ({time.time() - t0:.1f} s)", flush=True)
        sys.exit(0 if ok and ok3 else 1)
    else:
        sys.exit(f"usage: proteus_cache.py build|check")
