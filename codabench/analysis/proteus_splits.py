#!/usr/bin/env python
"""The four PROTEUS training / test protocols, defined once on the stable keys.

Setup session 2026-10-15 (brief ``prompts/2026-10-13_proteus_train_test_setup.md``,
Phase 3). Every protocol returns cache row indices (``proteus_cache.Cache``,
cache order = (person, session, run, cue)) for the training set and the test
set, the test set in **stream order**: streams (person, session) one after
another, inside a stream by run number, then cue time.

| Protocol   | Train                                             | Test                                   | Streams / cells |
|------------|---------------------------------------------------|----------------------------------------|-----------------|
| ``official`` | P01 P02 P03 P06 P07 P10 (1,521); val P05 P08 (379) kept apart | P04 x 3 sessions + P09 (755)   | 4 / 8 |
| ``lopo``   | fold k in 0..9: all sessions of the other nine people | all sessions of P(k+1)              | 14 / 28 over the folds |
| ``later``  | session 1 of all ten people (1,893)               | sessions 2-3 of P04 and P10 (762)      | 4 / 8 |
| ``all``    | all 14 sessions (2,655)                           | the ``official`` test again (755), inside the training set: ``score_meaningless`` | 4 / 8 |

The integer ids handed to an evaluator (``stream_id``, ``record_id``,
``context_id``, ``subject_id``) come from the stable keys, in one function
(:func:`ids_of`), never from an official loader's ``info``.

    from proteus_splits import split, ids_of
    sp = split("lopo", fold=3)          # holds out P04
    sp.train, sp.test                    # cache row indices
    ids = ids_of(cache.rows, sp.test)    # dict of int64 arrays, one per test row
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("COMPET_ROOT", str(Path.home() / "neuralbench/competition-040"))
                + "/tracks/bci_decoding")
sys.path.insert(0, str(Path.home() / "codabench/analysis"))

PROTOCOLS = ("official", "lopo", "later", "all")
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
OFFICIAL_TRAIN = ("P01", "P02", "P03", "P06", "P07", "P10")
OFFICIAL_VAL = ("P05", "P08")
OFFICIAL_TEST = ("P04", "P09")
LATER_TEST_PEOPLE = ("P04", "P10")
INTERFACE_ID = {"Graz": 0, "BrainHero": 1}
# expected sizes (brief § 3)
EXPECTED = {
    "official": dict(train=1521, val=379, test=755, streams=4, cells=8),
    "later": dict(train=1893, val=0, test=762, streams=4, cells=8),
    "all": dict(train=2655, val=0, test=755, streams=4, cells=8),
}
LOPO_TEST_SIZES = (190, 189, 189, 566, 190, 187, 189, 189, 189, 577)
LOPO_CELLS = (2, 2, 2, 6, 2, 2, 2, 2, 2, 6)
N_LOPO_CELLS = 28


@dataclass
class Split:
    protocol: str
    fold: object                      # int for lopo, None otherwise
    train: np.ndarray                 # cache row indices, stream order
    test: np.ndarray                  # cache row indices, stream order
    val: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    score_meaningless: bool = False

    @property
    def name(self):
        return self.protocol if self.fold is None else f"{self.protocol}[fold={self.fold}]"


def _rows(rows):
    if rows is None:
        import proteus_cache
        rows = proteus_cache.load().rows
    return rows


def stream_order(rows, idx):
    """``idx`` ordered by (person, session, run, cue): streams one after
    another, each stream's runs by run number, each run's windows by cue."""
    idx = np.asarray(idx, dtype=np.int64)
    sub = rows.iloc[idx]
    order = np.lexsort((sub["cue"].to_numpy(), sub["run"].to_numpy(dtype=np.int64),
                        sub["session"].to_numpy(dtype=np.int64), sub["person"].to_numpy()))
    return idx[order]


def split(protocol, fold=None, rows=None):
    """Training and test row indices of ``protocol`` (``fold`` for ``lopo``).

    Raises ``ValueError`` on an unknown protocol, a missing or out-of-range
    ``lopo`` fold, or a fold given for another protocol.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}: expected one of {PROTOCOLS}")
    rows = _rows(rows)
    person = rows["person"].to_numpy()
    session = rows["session"].to_numpy(dtype=np.int64)
    all_idx = np.arange(len(rows), dtype=np.int64)
    val = np.zeros(0, dtype=np.int64)
    meaningless = False
    if protocol == "lopo":
        if fold is None or int(fold) != fold or not 0 <= int(fold) < len(PEOPLE):
            raise ValueError(f"lopo needs fold in 0..{len(PEOPLE) - 1} (0 = P01), got {fold!r}")
        fold = int(fold)
        held = PEOPLE[fold]
        test = all_idx[person == held]
        train = all_idx[person != held]
    else:
        if fold is not None:
            raise ValueError(f"protocol {protocol!r} has no folds, got fold={fold!r}")
        if protocol == "official":
            train = all_idx[np.isin(person, OFFICIAL_TRAIN)]
            val = all_idx[np.isin(person, OFFICIAL_VAL)]
            test = all_idx[np.isin(person, OFFICIAL_TEST)]
        elif protocol == "later":
            train = all_idx[session == 1]
            test = all_idx[np.isin(person, LATER_TEST_PEOPLE) & (session >= 2)]
        else:                                   # "all"
            train = all_idx
            test = all_idx[np.isin(person, OFFICIAL_TEST)]
            meaningless = True
    return Split(protocol, fold, stream_order(rows, train), stream_order(rows, test),
                 stream_order(rows, val), meaningless)


def ids_of(rows, idx):
    """The integer ids of the rows ``idx``, from the stable keys only:
    ``subject_id`` = rank of person, ``stream_id`` = rank of (person, session),
    ``record_id`` = rank of (person, session, run), ``context_id`` = 0 Graz /
    1 BrainHero. Ranks are dense over the **whole** cache, so an id means the
    same thing in every protocol. ``onset`` = window start as a sample index
    at 120 Hz (``round(start_s * 120)``)."""
    idx = np.asarray(idx, dtype=np.int64)
    p = rows["person"].to_numpy()
    s = rows["session"].to_numpy(dtype=np.int64)
    r = rows["run"].to_numpy(dtype=np.int64)
    people = sorted(set(p.tolist()))
    sessions = sorted(set(zip(p.tolist(), s.tolist())))
    records = sorted(set(zip(p.tolist(), s.tolist(), r.tolist())))
    prank = {k: i for i, k in enumerate(people)}
    srank = {k: i for i, k in enumerate(sessions)}
    rrank = {k: i for i, k in enumerate(records)}
    sub = rows.iloc[idx]
    return dict(
        subject_id=np.array([prank[k] for k in sub["person"]], dtype=np.int64),
        stream_id=np.array([srank[k] for k in zip(sub["person"], sub["session"].astype(int))],
                           dtype=np.int64),
        record_id=np.array([rrank[k] for k in zip(sub["person"], sub["session"].astype(int),
                                                  sub["run"].astype(int))], dtype=np.int64),
        context_id=np.array([INTERFACE_ID[k] for k in sub["interface"]], dtype=np.int64),
        onset=np.round(sub["start"].to_numpy(dtype=np.float64) * 120.0).astype(np.int64),
    )


def cell_key(rows, idx):
    """(person, session, interface) of each row: the stable cell key."""
    sub = rows.iloc[np.asarray(idx, dtype=np.int64)]
    return list(zip(sub["person"], sub["session"].astype(int), sub["interface"]))


def stream_lengths(rows, idx):
    """Length of each test stream, in stream order."""
    sub = rows.iloc[np.asarray(idx, dtype=np.int64)]
    key = list(zip(sub["person"], sub["session"].astype(int)))
    out, last = [], None
    for k in key:
        if k != last:
            out.append(0)
            last = k
        out[-1] += 1
    return out


def every_split(rows=None):
    """(name, Split) for official, later, all and the ten lopo folds."""
    rows = _rows(rows)
    out = [(p, split(p, rows=rows)) for p in ("official", "later", "all")]
    out += [(f"lopo[fold={k}]", split("lopo", k, rows=rows)) for k in range(len(PEOPLE))]
    return out


def lopo_summary(fold_cell_means, fold_n_cells):
    """The lopo figure: the mean over the 28 cells of all folds (sum of cells x
    cell mean over folds, divided by 28), and the mean of the fold means."""
    cm = np.asarray(fold_cell_means, dtype=np.float64)
    nc = np.asarray(fold_n_cells, dtype=np.int64)
    if nc.sum() != N_LOPO_CELLS:
        raise ValueError(f"lopo folds hold {nc.sum()} cells, expected {N_LOPO_CELLS}")
    return float((cm * nc).sum() / N_LOPO_CELLS), float(cm.mean())


# -- gates (G5a-d) --------------------------------------------------------------

def check_sizes(rows=None):
    rows = _rows(rows)
    out = []
    for p, e in EXPECTED.items():
        sp = split(p, rows=rows)
        got = (len(sp.train), len(sp.val), len(sp.test), len(set(zip(
            rows["person"].iloc[sp.test], rows["session"].iloc[sp.test]))), len(set(cell_key(rows, sp.test))))
        exp = (e["train"], e["val"], e["test"], e["streams"], e["cells"])
        out.append((f"G5a {p} train/val/test/streams/cells", got == exp, f"{got} (expected {exp})"))
    sizes, cells, trains = [], [], []
    for k in range(10):
        sp = split("lopo", k, rows=rows)
        sizes.append(len(sp.test))
        cells.append(len(set(cell_key(rows, sp.test))))
        trains.append(len(sp.train))
    out.append(("G5a lopo test sizes", tuple(sizes) == LOPO_TEST_SIZES, str(sizes)))
    out.append(("G5a lopo train = 2655 - test", all(t == 2655 - s for t, s in zip(trains, sizes)),
                str(trains)))
    out.append(("G5a lopo cells per fold (28 in all)", tuple(cells) == LOPO_CELLS and sum(cells) == 28,
                str(cells)))
    out.append(("G5a lopo fold k holds out PEOPLE[k] (fold=3 is P04)", all(
        set(rows["person"].iloc[split("lopo", k, rows=rows).test]) == {PEOPLE[k]} for k in range(10)),
        "ok"))
    for bad in [("nope", None), ("lopo", None), ("lopo", 10), ("lopo", -1), ("official", 0)]:
        try:
            split(*bad, rows=rows)
            out.append((f"G5a raises on {bad}", False, "no error"))
        except ValueError as e:
            out.append((f"G5a raises on {bad}", True, str(e)[:60]))
    return out


def check_leaks(rows=None):
    rows = _rows(rows)
    person = rows["person"].to_numpy()
    sess = list(zip(rows["person"], rows["session"].astype(int)))
    out = []
    for name, sp in every_split(rows):
        tr, te = set(sp.train.tolist()), set(sp.test.tolist())
        if sp.protocol == "all":
            out.append((f"G5b {name}: test inside train (stated exception), score_meaningless",
                        te <= tr and len(te) == 755 and sp.score_meaningless, f"{len(te & tr)} shared"))
            continue
        out.append((f"G5b {name}: train and test disjoint", not (tr & te) and not sp.score_meaningless,
                    f"{len(tr & te)} shared"))
        if sp.protocol in ("official", "lopo"):
            pt, pr = set(person[sp.test]), set(person[sp.train])
            out.append((f"G5b {name}: no test person in training", not (pt & pr), f"{sorted(pt & pr)}"))
        if sp.protocol == "official":
            va = set(sp.val.tolist())
            out.append((f"G5b {name}: validation apart from both", not (va & tr) and not (va & te)
                        and set(person[sp.val]) == set(OFFICIAL_VAL), f"{len(va)} val rows"))
        if sp.protocol == "later":
            st, sr = set(sess[i] for i in sp.test), set(sess[i] for i in sp.train)
            out.append((f"G5b {name}: no (person, session) on both sides", not (st & sr), str(st & sr)))
            out.append((f"G5b {name}: every test person has session 1 in training",
                        all((p, 1) in sr for p, _ in st), str(sorted(set(p for p, _ in st)))))
    return out


def _ordered(rows, idx):
    """True when the test rows are stream by stream (each contiguous) and,
    within a stream, by (run, cue)."""
    sub = rows.iloc[idx]
    key = list(zip(sub["person"], sub["session"].astype(int)))
    seen, last = set(), None
    for k in key:
        if k != last:
            if k in seen:
                return False
            seen.add(k)
            last = k
    rc = list(zip(key, sub["run"].astype(int), sub["cue"].astype(float)))
    return all(rc[i] <= rc[i + 1] for i in range(len(rc) - 1))


def check_order(rows=None, loaders=None, cache=None):
    """G5c. With ``loaders`` (the official ones) also compares ``official``'s
    test sequence with the organisers' test dataset order and its val streams
    with their validation dataset order, element by element."""
    import proteus_cache
    c = cache if cache is not None else proteus_cache.load()
    rows = c.rows if rows is None else rows
    out = []
    for name, sp in every_split(rows):
        out.append((f"G5c {name}: test streams contiguous, ordered by (run, cue)", _ordered(rows, sp.test), "ok"))
    if loaders is None:
        loaders, _ = proteus_cache.official_loaders()
    sp = split("official", rows=rows)
    for split_name, idx in (("test", sp.test), ("val", sp.val)):
        ds = loaders[split_name].dataset
        theirs = [c.row_of(r) for _, _, _, _, r in proteus_cache.iter_items(ds)]
        same = len(theirs) == len(idx) and all(int(a) == int(b) for a, b in zip(theirs, idx))
        out.append((f"G5c official {split_name}: key sequence == organisers' {split_name} dataset order",
                    same, f"{len(theirs)} items"))
    return out


def check_ids(rows=None):
    rows = _rows(rows)
    out = []
    expected_cells = {"official": 8, "later": 8, "all": 8}
    for name, sp in every_split(rows):
        ids = ids_of(rows, sp.test)
        cells = cell_key(rows, sp.test)
        pairs = set(zip(ids["stream_id"].tolist(), ids["context_id"].tolist(), cells))
        a = len(set((s, cx) for s, cx, _ in pairs))
        b = len(set(k for _, _, k in pairs))
        bij = a == b == len(pairs)
        exp = expected_cells.get(sp.protocol, LOPO_CELLS[sp.fold] if sp.fold is not None else None)
        out.append((f"G5d {name}: (stream_id, context_id) <-> (person, session, interface) both ways, "
                    f"{exp} cells", bij and a == exp, f"{a} id cells, {b} key cells"))
        sub = rows.iloc[sp.test]
        ctx_ok = all((cx == 0) == (i == "Graz") for cx, i in zip(ids["context_id"], sub["interface"]))
        rec_ok = len(set(zip(ids["record_id"].tolist(), sub["person"], sub["session"].astype(int),
                             sub["run"].astype(int)))) == len(set(ids["record_id"].tolist()))
        sub_ok = len(set(zip(ids["subject_id"].tolist(), sub["person"]))) == len(set(sub["person"]))
        out.append((f"G5d {name}: context 0 Graz / 1 BrainHero; record_id <-> (person, session, run); "
                    "subject_id <-> person", ctx_ok and rec_ok and sub_ok, "ok"))
    return out


if __name__ == "__main__":
    import proteus_cache
    proteus_cache.require_env040()
    c = proteus_cache.load()
    checks = check_sizes(c.rows) + check_leaks(c.rows) + check_order(cache=c) + check_ids(c.rows)
    ok = proteus_cache.report(checks)
    print("G5a-d", "PASS" if ok else "FAIL", f"({len(checks)} checks)")
    sys.exit(0 if ok else 1)
