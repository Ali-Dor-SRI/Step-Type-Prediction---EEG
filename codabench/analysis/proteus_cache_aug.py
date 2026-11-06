#!/usr/bin/env python
"""Window caches of PROTEUS at other cue offsets, from the official loader.

Sprint 2026-11-06 (brief ``prompts/2026-11-06_next_session_plan.md`` § 2 P1).
The official stream config cuts each trial at ``[cue + 1.0, cue + 5.0)``
(``data.start: 1.0``, ``duration: 4.0``; the trial ends at cue + 8 s). This
module calls ``load_task`` exactly as ``proteus_cache.official_loaders`` does,
plus the one data-section override ``{"start": o}``, so the preprocessing
(120 Hz, 0.1-75 Hz, notch 50/60, x1e6) is the organisers' own.

The frozen caches are never touched: the neuralset / exca cache folder is a
**copy** (``$AUG_ROOT/nbcache``) of ``benchopt_data_040/neural_compet/cache``,
and every offset is written to ``$AUG_ROOT/off_<o>/`` (``X.npy``,
``rows.parquet`` with the official stable keys plus ``offset``).

    python ~/codabench/analysis/proteus_cache_aug.py build 1.0 0.5 1.5 ...
    python ~/codabench/analysis/proteus_cache_aug.py check 1.0 0.5 1.5 ...

Gate A0: offset 1.0 equals the official cache bit for bit (X, keys, labels).
Gate A1: each offset has 2,655 rows, keys and labels equal to the official
cache row by row, start == cue + o, no NaN / inf.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as pc  # noqa: E402

AUG_ROOT = Path(__import__("os").environ.get(
    "PROTEUS_AUG_ROOT", Path.home() / "neuralbench/proteus_cache_aug"))
NB_CACHE = AUG_ROOT / "nbcache"
KEYS = ["person", "session", "run", "interface", "task", "cue", "label", "code"]


def tag(o):
    return f"{float(o):.2f}"


def off_dir(o, root=AUG_ROOT):
    return Path(root) / f"off_{tag(o)}"


def loaders_at(start):
    from benchmark_utils.nb_task import load_task
    from datasets.bci_studies import _proteus_interface, _proteus_run
    if not NB_CACHE.exists():
        src = pc.DATA_DIR / "cache"
        print(f"[aug] copying {src} -> {NB_CACHE}", flush=True)
        shutil.copytree(src, NB_CACHE)
    return load_task(
        "eeg", pc.TASK, data_dir=pc.DATA_DIR, dataset=None, device="cpu",
        batch_size=64, seed=0, num_workers=0, subset="full",
        target_transform=lambda y: y.argmax(-1),
        context_of=_proteus_interface, order_of=_proteus_run,
        overrides={"start": float(start),
                   "study.source.infra.folder": str(NB_CACHE),
                   "neuro.infra.folder": str(NB_CACHE)})


def build_one(o):
    t0 = time.time()
    loaders, meta = loaders_at(o)
    assert int(meta["n_times"]) == pc.N_TIMES and int(meta["n_chans"]) == pc.N_CHANS, meta
    rows, Xs = [], []
    for split in ("train", "val", "test"):
        for i, X, y, info, r in pc.iter_items(loaders[split].dataset):
            x = np.ascontiguousarray(pc._to_numpy(X), dtype=np.float32)
            person, session, run, _ = pc.key_of(r)
            rows.append(dict(
                person=person, session=session, run=run, interface=pc.interface_of(r["task"]),
                task=str(r["task"]), cue=float(r["start"]), start=float(info["onset"]),
                label=int(pc._to_numpy(y)), code=int(r["code"]),
                description=str(r["description"]), split=split, loader_pos=int(i),
                sha1=pc.sha1_of(x), offset=float(o)))
            Xs.append(x)
    df = pd.DataFrame(rows)
    X = np.stack(Xs)
    order = np.lexsort((df["cue"].to_numpy(), df["run"].to_numpy(),
                        df["session"].to_numpy(), df["person"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    X = np.ascontiguousarray(X[order])
    d = off_dir(o)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", X)
    df.to_parquet(d / "rows.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(dict(
        offset=float(o), n_windows=len(df), shape=list(X.shape),
        sfreq=float(meta["sfreq"]), built=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        seconds=round(time.time() - t0, 1)), indent=1))
    print(f"[aug] offset {tag(o)}: X {X.shape}, {len(df)} rows, {time.time() - t0:.1f} s",
          flush=True)


def check(offsets):
    off = pc.Cache()
    ok_all = True
    for o in offsets:
        d = off_dir(o)
        X = np.load(d / "X.npy", mmap_mode="r")
        df = pd.read_parquet(d / "rows.parquet")
        res = []
        res.append(("A1 rows", len(df) == pc.N_WINDOWS and X.shape == (pc.N_WINDOWS, pc.N_CHANS, pc.N_TIMES),
                    f"{len(df)} {X.shape}"))
        keq = all((df[k].to_numpy() == off.rows[k].to_numpy()).all() for k in KEYS)
        res.append(("A1 keys+labels == official", bool(keq), ""))
        dt = np.abs(df["start"].to_numpy() - (df["cue"].to_numpy() + float(o)))
        res.append(("A1 start == cue + o", bool(dt.max() < 1e-6), f"max {dt.max():.2e}"))
        res.append(("A1 finite", bool(np.isfinite(np.asarray(X)).all()), ""))
        if abs(float(o) - 1.0) < 1e-9:
            diff = float(np.abs(np.asarray(X, dtype=np.float64) - np.asarray(off.X, dtype=np.float64)).max())
            sha = bool((df["sha1"].to_numpy() == off.rows["sha1"].to_numpy()).all())
            res.append(("A0 X == official bit for bit", diff == 0.0 and sha, f"max |diff| {diff}, sha1 {sha}"))
        else:
            same = int((df["sha1"].to_numpy() == off.rows["sha1"].to_numpy()).sum())
            res.append(("windows differ from official", same == 0, f"{same} identical"))
        for name, ok, info in res:
            ok_all &= ok
            print(f"[check] off {tag(o)} {'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    print(f"[check] {'ALL PASS' if ok_all else 'SOME FAIL'}", flush=True)
    return ok_all


if __name__ == "__main__":
    cmd, offs = sys.argv[1], [float(a) for a in sys.argv[2:]]
    if cmd == "build":
        for o in offs:
            if (off_dir(o) / "meta.json").exists():
                print(f"[aug] offset {tag(o)} exists, skipped", flush=True)
                continue
            build_one(o)
    elif cmd == "check":
        sys.exit(0 if check(offs) else 1)
