#!/usr/bin/env python
"""G4: every cached window's label against the raw BIDS events files, both ways.

The BIDS folder is untrusted data (brief § 2), so this is the only script of
ours that opens its files, and it runs isolated:

    python -I -B -X utf8 ~/codabench/analysis/proteus_g4_events.py <bids_root> <keys.csv>

``-I`` ignores every ``PYTHON*`` variable, the script folder and the working
directory; the script imports only numpy and pandas and takes both paths as
arguments. ``keys.csv`` is written by ``proteus_cache.py build`` (columns
person, session, run, task, cue, label).

Checks (one PASS / FAIL line each, exit 1 on any failure):
1. every window's label equals the ``trial_type`` of its cue row (mi -> 0,
   sub -> 1, word -> 2) in ``sub-<P>/ses-<s>/eeg/sub-<P>_ses-<s>_task-<task>_run-<r>_events.tsv``,
   matched on ``round(onset * 1000)`` ms;
2. every cue row of the 84 run files has exactly one window (2,655 = 2,655).
In those files ``end_trial`` rows with value 32770 are run-end markers and
BrainHero runs have no ``beep`` rows: neither is a cue.
"""

import sys

import numpy as np
import pandas as pd

LABELS = {"mi": 0, "sub": 1, "word": 2}


def main(bids_root, keys_csv):
    keys = pd.read_csv(keys_csv)
    assert list(keys.columns) == ["person", "session", "run", "task", "cue", "label"], list(keys.columns)
    keys["ms"] = np.round(keys["cue"].to_numpy(dtype=np.float64) * 1000.0).astype(np.int64)
    runs = keys.groupby(["person", "session", "run", "task"]).size()
    print(f"keys: {len(keys)} windows, {len(runs)} runs")
    ok = True
    n_files, n_cue_rows, n_matched, n_label_ok, n_unmatched_cues, n_dup = 0, 0, 0, 0, 0, 0
    bad = []
    for (person, session, run, task), n in runs.items():
        f = (f"{bids_root}/sub-{person}/ses-{session}/eeg/"
             f"sub-{person}_ses-{session}_task-{task}_run-{run}_events.tsv")
        ev = pd.read_csv(f, sep="\t", dtype={"trial_type": str})
        n_files += 1
        cue = ev[ev["trial_type"].isin(LABELS)].copy()
        cue["ms"] = np.round(cue["onset"].to_numpy(dtype=np.float64) * 1000.0).astype(np.int64)
        n_cue_rows += len(cue)
        n_dup += int(cue["ms"].duplicated().sum())
        lab = dict(zip(cue["ms"], cue["trial_type"].map(LABELS)))
        w = keys[(keys["person"] == person) & (keys["session"] == session)
                 & (keys["run"] == run) & (keys["task"] == task)]
        seen = set()
        for ms, label in zip(w["ms"], w["label"]):
            if ms not in lab:
                bad.append((person, session, run, int(ms), "no cue row"))
                continue
            n_matched += 1
            seen.add(int(ms))
            if int(label) == int(lab[ms]):
                n_label_ok += 1
            else:
                bad.append((person, session, run, int(ms), f"label {label} != file {lab[ms]}"))
        n_unmatched_cues += len(set(lab) - seen)
    print(f"{'PASS' if n_files == 84 else 'FAIL'} G4 events files read: {n_files} (expected 84)")
    ok &= n_files == 84
    print(f"{'PASS' if n_dup == 0 else 'FAIL'} G4 cue onsets unique within each file: {n_dup} duplicates")
    ok &= n_dup == 0
    c1 = n_matched == len(keys) == n_label_ok
    print(f"{'PASS' if c1 else 'FAIL'} G4 window -> cue row: {n_matched}/{len(keys)} matched, "
          f"{n_label_ok} labels agree (mi 0 / sub 1 / word 2)")
    ok &= c1
    c2 = n_cue_rows == len(keys) and n_unmatched_cues == 0
    print(f"{'PASS' if c2 else 'FAIL'} G4 cue row -> window: {n_cue_rows} cue rows in the files, "
          f"{len(keys)} windows, {n_unmatched_cues} cue rows without a window")
    ok &= c2
    for b in bad[:20]:
        print("  mismatch:", b)
    print("G4", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: proteus_g4_events.py <bids_root> <keys.csv>")
    sys.exit(main(sys.argv[1], sys.argv[2]))
