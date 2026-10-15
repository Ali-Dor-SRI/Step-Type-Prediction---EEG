"""ProteusProtocols: the four PROTEUS protocols served through the Track 2
dataset contract (route B of the setup session 2026-10-15, brief
``prompts/2026-10-13_proteus_train_test_setup.md`` Phase 4).

Serves any protocol and fold of ``analysis/proteus_splits.py`` from the window
cache of ``analysis/proteus_cache.py`` (built from the official loader, proven
identical to it: G2/G3), exactly like ``BCI[study=dreyer2026proteus]`` does
through ``benchmark_utils.nb_task``, so a solver can be trained and scored on
``official``, ``lopo[fold=k]``, ``later`` or ``all`` with the organisers' own
objective and streamed evaluator. Used by path, with parameters
(benchopt 1.10 loads a dataset straight from a file):

    source ~/codabench/env040.sh
    benchopt run tracks/bci_decoding \\
        -d "$HOME/codabench/datasets/proteus_protocols.py[protocol=lopo,fold=3]" \\
        -s <solver file> -o "BCI-decoding[training=True]" --seed 42 \\
        --no-cache --no-plot --no-html --no-display --output <abs>.parquet

- ``protocol``: official | lopo | later | all (``proteus_splits.PROTOCOLS``);
  ``fold``: 0..9 for ``lopo`` (fold k holds out P(k+1)), ``None`` otherwise
  (a fold given for another protocol, or missing for ``lopo``, raises).
- Batches are ``(X, y, info)``: X float32 ``(B, 41, 480)`` microvolts, y long
  ``(B,)``, info with the organisers' five keys ``subject_id`` (rank of the
  person), ``record_id`` (rank of (person, session, run)), ``onset`` (window
  start in samples at 120 Hz), ``stream_id`` (rank of (person, session)),
  ``context_id`` (0 Graz, 1 BrainHero), all from the stable keys
  (``proteus_splits.ids_of``), never from an official loader's ``info``.
- Train loader: batch 64, shuffled by ``torch.Generator().manual_seed(seed)``
  with benchopt's seed (``get_seed()``; 0 outside a benchopt run). Test loader:
  batch 1, unshuffled, in stream order (streams one after another, runs by
  number, windows by cue time).
- As on NeuralBench, each loader's dataset exposes ``.seg_ds.triggers``: a
  DataFrame in dataset index order with columns subject, session, run,
  context, start (window start, s), which ``Riemann-Sealed._collect`` reads
  session ids from.
- The other values equal the organisers' ``get_data()``: ``n_classes`` 3,
  ``sfreq`` 120.0, ``ch_names``, ``chs_info``, ``n_chans`` 41, ``n_times``
  480, ``device``.
- ``get_data`` prints the protocol, the fold and both shapes in its first line.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from benchopt import BaseDataset
from torch.utils.data import DataLoader, default_collate

from benchmark_utils.data import chs_info_from_names, get_device

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache  # noqa: E402
import proteus_splits  # noqa: E402


class _Segments:
    """Stand-in for the neuralset ``SegmentDataset`` behind a NeuralBench
    loader: only the per-window ``triggers`` table (dataset index order)."""

    def __init__(self, triggers):
        self.triggers = triggers

    def __len__(self):
        return len(self.triggers)


class _Windows(torch.utils.data.Dataset):
    """Cache rows ``rows`` as ``(X, y, info)`` items (X read lazily from the
    memory-mapped cache)."""

    def __init__(self, cache, rows):
        self.X, self.rows = cache.X, np.asarray(rows, dtype=np.int64)
        self.y = torch.as_tensor(cache.y[self.rows], dtype=torch.long)
        self.ids = proteus_splits.ids_of(cache.rows, self.rows)
        import pandas as pd
        sub = cache.rows.iloc[self.rows]
        self.seg_ds = _Segments(pd.DataFrame({
            "subject": sub["person"].to_numpy(), "session": sub["session"].to_numpy(dtype=np.int64),
            "run": sub["run"].to_numpy(dtype=np.int64), "context": sub["interface"].to_numpy(),
            "start": sub["start"].to_numpy(dtype=np.float64)}))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        X = torch.from_numpy(np.array(self.X[self.rows[i]], dtype=np.float32))
        info = {k: int(self.ids[k][i]) for k in
                ("subject_id", "record_id", "onset", "stream_id", "context_id")}
        return X, self.y[i], info


def _collate(device):
    """nb_task's collate: default collate, X / y moved onto ``device``."""
    def collate(batch):
        X, y, info = default_collate(batch)
        return X.to(device), y.to(device), info
    return collate


class Dataset(BaseDataset):

    name = "ProteusProtocols"

    requirements = []

    parameters = {
        "protocol": ["official"],      # official | lopo | later | all
        "fold": [None],                # 0..9 for lopo (fold k holds out P(k+1))
        "batch_size": [64],            # training batches; the test loader is B = 1
    }

    test_parameters = {
        "protocol": ["official"],
        "fold": [None],
        "batch_size": [64],
    }

    def prepare(self):
        # Nothing to download: the cache is built by analysis/proteus_cache.py.
        # Fail early if it is missing.
        proteus_cache.load()

    def get_data(self):
        c = proteus_cache.load()
        fold = None if self.fold is None else int(self.fold)
        sp = proteus_splits.split(self.protocol, fold, rows=c.rows)
        device = get_device()
        try:
            seed = self.get_seed()
        except Exception:        # get_data called outside a benchopt run
            seed = 0
        train, test = _Windows(c, sp.train), _Windows(c, sp.test)
        print(f"[ProteusProtocols] protocol={sp.protocol} fold={sp.fold} "
              f"train X=({len(train)}, {c.X.shape[1]}, {c.X.shape[2]}) "
              f"test X=({len(test)}, {c.X.shape[1]}, {c.X.shape[2]}) seed={seed}"
              + (" score_meaningless (test inside train)" if sp.score_meaningless else ""),
              flush=True)
        ch_names = list(c.meta["ch_names"])
        return dict(
            train_loader=DataLoader(
                train, batch_size=int(self.batch_size), shuffle=True,
                generator=torch.Generator().manual_seed(int(seed)),
                collate_fn=_collate(device)),
            test_loader=DataLoader(test, batch_size=1, shuffle=False,
                                   collate_fn=_collate(device)),
            n_classes=len(c.meta["classes"]),
            sfreq=float(c.meta["sfreq"]),
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=int(c.X.shape[1]),
            n_times=int(c.X.shape[2]),
            device=device,
        )
