#!/usr/bin/env python
"""Sprint 2026-11-06 P3: deep-net smoke on PROTEUS (lopo fold k + later).

braindecode ShallowFBCSPNet / EEGNetv4 on the cached windows, per-window
per-channel z-score (the E0f setting), Adam, fixed epochs, one seed. The nets
are stateless, so batch prediction equals the streamed contract; scored as
the ranking metric (balanced accuracy per (person, session, interface) cell,
averaged over cells). Optional ``--offsets`` adds training windows from
``proteus_cache_aug`` (training rows only). One JSON row per (model, split)
to ``logs/sprint_1106/rows/NET.jsonl``; heartbeat every 5 epochs.

    python sprint1106_netsmoke.py --model shallow --epochs 60 --seed 33 --splits lopo0 later
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402

OUT = Path.home() / "codabench/logs/sprint_1106/rows"


def log(m):
    print(f"{time.strftime('%T')} {m}", flush=True)


def zscore(X):
    X = np.asarray(X, dtype=np.float32)
    m = X.mean(-1, keepdims=True)
    s = X.std(-1, keepdims=True) + 1e-6
    return (X - m) / s


def make(model, n_chans, n_times):
    from braindecode import models as M
    if model == "shallow":
        return M.ShallowFBCSPNet(n_chans=n_chans, n_outputs=3, n_times=n_times,
                                 final_conv_length="auto")
    if model == "eegnet":
        cls = getattr(M, "EEGNetv4", None) or getattr(M, "EEGNet")
        return cls(n_chans=n_chans, n_outputs=3, n_times=n_times)
    raise ValueError(model)


def cell_mean(rows, idx, y, p):
    keys = PS.cell_key(rows, idx)
    out = {}
    for k in sorted(set(keys)):
        m = np.array([kk == k for kk in keys])
        cls = np.unique(y[m])
        out["|".join(map(str, k))] = float(np.mean([(p[m][y[m] == c] == c).mean() for c in cls]))
    return float(np.mean(list(out.values()))), out


def train_rows(c, idx, offsets):
    Xs, ys = [zscore(c.X[idx])], [c.y[idx]]
    if offsets:
        import proteus_cache_aug as PA
        for o in offsets:
            Xo = np.load(PA.off_dir(o) / "X.npy", mmap_mode="r")
            Xs.append(zscore(Xo[idx]))
            ys.append(c.y[idx])
    return np.concatenate(Xs), np.concatenate(ys)


def run(model, split, epochs, seed, offsets, lr, bs):
    c = PC.load()
    proto, fold = ("lopo", int(split[4:])) if split.startswith("lopo") else (split, None)
    sp = PS.split(proto, fold)
    tr, te = np.asarray(sp.train), np.asarray(sp.test)
    Xtr, ytr = train_rows(c, tr, offsets)
    Xte, yte = zscore(c.X[te]), c.y[te]
    log(f"[data] {model} {split} train={Xtr.shape} test={Xte.shape} offsets={offsets} "
        f"epochs={epochs} seed={seed} threads={torch.get_num_threads()}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = make(model, Xtr.shape[1], Xtr.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g)
        tl = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tl += float(loss) * len(b)
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"[epoch] {ep + 1}/{epochs} loss={tl / len(yt):.4f} "
                f"{(time.time() - t0) / (ep + 1):.1f} s/epoch")
    fit_s = time.time() - t0
    net.eval()
    with torch.no_grad():
        P = torch.softmax(net(torch.from_numpy(Xte)), -1).numpy()
    score, cells = cell_mean(c.rows, te, yte, P.argmax(1))
    row = dict(model=model, split=split, epochs=epochs, seed=seed, offsets=offsets, lr=lr,
               cell_mean=score, cells=cells, fit_s=round(fit_s, 1),
               s_per_epoch=round(fit_s / epochs, 2), n_train=int(len(ytr)),
               time=time.strftime("%F %T"))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "NET.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    np.savez_compressed(OUT / f"net_{model}_{split}_s{seed}_o{len(offsets or [])}.npz",
                        probas=P, y=yte, test_idx=te)
    log(f"[row] {model} {split} cell_mean={score:.4f} fit={fit_s:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", nargs="+", default=["lopo0", "later"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--offsets", default="")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    offs = [float(v) for v in a.offsets.split("_")] if a.offsets else []
    for s in a.splits:
        run(a.model, s, a.epochs, a.seed, offs, a.lr, a.bs)
