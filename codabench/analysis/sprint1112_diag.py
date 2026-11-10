#!/usr/bin/env python
"""Sprint 2026-11-12 P2: diagnostics of the slot-2 ensemble from the existing dumps.

Inputs: logs/sprint_1106/dump/X4f__<proto>__<fold>.npz (streamed X4f features,
probabilities, training features), the slot-2 nets' saved probabilities
(logs/sprint_1106/rows, seeds 33-37), w = 0.3 (the nested pick in 10/10 folds).

  D1  error anatomy: balanced accuracy by run type, interface, stream position,
      class (confusion), person; X4f / nets / ensemble.
  D2  ceilings (ORACLES, labelled): (a) per-session leave-one-run-out with the
      session's own labels, pooled + own; (b) feature re-centring by the
      session's final (whole-stream) mean instead of the running one (label-free
      but non-causal); (c) same with the class-balanced session mean (labels).
  D3  label sequence structure of the whole dataset (counts per run, transitions,
      longest repeat, is the order shared across people / sessions).
  D4  calibration vs online as two tasks (X4f features): train on one run type,
      test on the other and on itself.
  D5  sealed - local offset: per-cell scores by run type and position, what mix
      of them would give 0.736.

    python sprint1112_diag.py [D1 D2 D3 D4 D5]  -> stdout (markdown tables)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
import joblib

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import proteus_cache as PC  # noqa: E402
import proteus_splits as PS  # noqa: E402
from sprint1106_tta_probe import cells  # noqa: E402

CB = Path.home() / "codabench"
DUMP = CB / "logs/sprint_1106/dump"
NETS = CB / "logs/sprint_1106/rows"
SEEDS = ["33", "34", "35", "36", "37"]
W = 0.3
SPLITS = [("lopo", k) for k in range(10)] + [("later", None)]


def sname(proto, fold):
    return f"lopo{fold}" if proto == "lopo" else proto


def logmix(Px, Pn, w=W):
    return (1 - w) * np.log(np.clip(Px, 1e-12, 1)) + w * np.log(np.clip(Pn, 1e-12, 1))


def bacc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def load_all(rows):
    """One frame per test window (all 11 splits): meta + X4f / net / ens predictions."""
    out = []
    for proto, fold in SPLITS:
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        te = d["test_idx"]
        Pn = np.mean([np.load(NETS / f"net_shallow_{sname(proto, fold)}_s{s}_o0.npz")["probas"]
                      for s in SEEDS], axis=0)
        r = rows.iloc[te].reset_index(drop=True).copy()
        r["idx"] = te
        r["proto"], r["fold"] = proto, -1 if fold is None else fold
        r["y"] = d["y_test"]
        r["p_x4f"] = d["P_test"].argmax(1)
        r["p_net"] = Pn.argmax(1)
        r["p_ens"] = logmix(d["P_test"], Pn).argmax(1)
        out.append(r)
    df = pd.concat(out, ignore_index=True)
    df["runtype"] = np.where(df.task.str.startswith("Acquisition"), "calib", "online")
    # position inside the stream (person x session), by loader order
    df["pos"] = df.groupby(["proto", "fold", "person", "session"]).loader_pos.rank(method="first").astype(int)
    df["posbin"] = pd.cut(df.pos, [0, 35, 70, 1000], labels=["1-35", "36-70", "71+"])
    return df


def table(df, by, cols=("p_x4f", "p_net", "p_ens")):
    g = []
    for k, s in df.groupby(by, observed=True):
        row = dict(zip(by if isinstance(by, list) else [by], k if isinstance(k, tuple) else (k,)))
        row["n"] = len(s)
        for c in cols:
            row[c[2:]] = round(bacc(s.y.to_numpy(), s[c].to_numpy()), 4)
        g.append(row)
    return pd.DataFrame(g)


def cellscore(df, col):
    return cells(PC.load().rows, df.idx.to_numpy(), df.y.to_numpy(), df[col].to_numpy()).score.mean()


def d1(df):
    print("## D1 — error anatomy of the slot-2 ensemble (w 0.3; pooled balanced accuracy per group)\n")
    for proto in ("lopo", "later"):
        s = df[df.proto == proto]
        print(f"### {proto}: cell means X4f {cellscore(s, 'p_x4f'):.4f}, nets {cellscore(s, 'p_net'):.4f}, "
              f"ens {cellscore(s, 'p_ens'):.4f}\n")
        for by in (["runtype"], ["interface"], ["runtype", "interface"], ["posbin"], ["run"],
                   ["person"]):
            print(table(s, by).to_markdown(index=False), "\n")
        cm = pd.crosstab(s.y, s.p_ens, normalize="index").round(3)
        print("confusion (rows = true 0 mi / 1 sub / 2 word, row-normalised):\n")
        print(cm.to_markdown(), "\n")


def d2(df, rows):
    print("## D2 — ceilings (ORACLES unless stated; X4f features, X4f's own classifier)\n")
    res = []
    for proto, fold in SPLITS:
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        clf = joblib.load(DUMP / f"X4f__{proto}__{fold}.clf.joblib")
        te = d["test_idx"]
        F, y = d["F_test"], d["y_test"]
        r = rows.iloc[te].reset_index(drop=True)
        key = r.person.astype(str) + "_" + r.session.astype(str)
        tr_mean = d["F_train"].mean(0)
        tr_cm = np.mean([d["F_train"][d["y_train"] == c].mean(0) for c in range(3)], axis=0)
        p_b = np.empty(len(y), int)
        p_c = np.empty(len(y), int)
        p_a = np.empty(len(y), int)
        for k in key.unique():
            m = (key == k).to_numpy()
            Fs = F[m]
            p_b[m] = clf.predict(Fs - Fs.mean(0) + tr_mean)
            cm = np.mean([Fs[y[m] == c].mean(0) for c in np.unique(y[m])], axis=0)
            p_c[m] = clf.predict(Fs - cm + tr_cm)
            # (a) leave-one-run-out inside the session: pooled training + the session's other runs
            runs = r.run[m].to_numpy()
            pa = np.empty(m.sum(), int)
            for rr in np.unique(runs):
                o = runs != rr
                c2 = clone(clf).fit(np.vstack([d["F_train"], np.repeat(Fs[o], 3, axis=0)]),
                                    np.concatenate([d["y_train"], np.repeat(y[m][o], 3)]))
                pa[~o] = c2.predict(Fs[~o])
            p_a[m] = pa
        for name, p in (("now (running)", d["P_test"].argmax(1)), ("b final-mean (label-free, non-causal)", p_b),
                        ("c class-balanced mean (labels)", p_c), ("a pooled + own other runs ×3 (labels)", p_a)):
            sc = cells(rows, te, y, p).score.mean()
            res.append(dict(proto=proto, fold=fold, arm=name, cell_mean=sc))
        print(f"  {proto} {fold} done", file=sys.stderr, flush=True)
    t = pd.DataFrame(res)
    lopo = t[t.proto == "lopo"].groupby("arm").cell_mean.mean()
    later = t[t.proto == "later"].set_index("arm").cell_mean
    print(pd.DataFrame({"lopo": lopo.round(4), "later": later.round(4)}).to_markdown(), "\n")


def d3(rows):
    print("## D3 — label sequence structure (all 2,655 windows; measure only, feeds the flagged track)\n")
    g = rows.sort_values(["person", "session", "loader_pos"]).groupby(["person", "session", "run"])
    cnt = g.label.apply(lambda s: tuple(np.bincount(s, minlength=3))).value_counts()
    print("class counts per run (mi, sub, word) -> number of runs:\n")
    print(cnt.to_frame("runs").to_markdown(), "\n")
    seqs = g.label.apply(lambda s: "".join(map(str, s)))
    T = np.zeros((3, 3))
    longest = []
    for s in seqs:
        a = np.array(list(map(int, s)))
        for i, j in zip(a[:-1], a[1:]):
            T[i, j] += 1
        L, cur = 1, 1
        for i in range(1, len(a)):
            cur = cur + 1 if a[i] == a[i - 1] else 1
            L = max(L, cur)
        longest.append(L)
    print("transition matrix P(next | current), all runs:\n")
    print(pd.DataFrame(T / T.sum(1, keepdims=True), index=["mi", "sub", "word"],
                       columns=["mi", "sub", "word"]).round(3).to_markdown(), "\n")
    print(f"same-class repeat rate {np.trace(T) / T.sum():.3f} (iid balanced: 0.333); longest repeat per run: "
          f"median {np.median(longest):.0f}, max {max(longest)}\n")
    by_run = seqs.reset_index()
    shared = by_run.groupby("run").label.agg(lambda s: f"{s.nunique()} distinct of {len(s)}")
    print("is the cue order shared? distinct label sequences per run number:\n")
    print(shared.to_frame("sequences").to_markdown(), "\n")
    # first-half vs second-half balance inside a run
    def halfbal(s):
        a = np.array(list(map(int, s)))
        h = len(a) // 2
        return np.abs(np.bincount(a[:h], minlength=3) - np.bincount(a[h:], minlength=3)).sum()
    hb = seqs.apply(halfbal)
    print(f"|count(first half) - count(second half)| summed over classes, per run: median {hb.median():.0f}, "
          f"max {hb.max()} (block-balanced design would give ≈ 0)\n")
    # same sequence across sessions / people for the same task?
    tseq = rows.sort_values(["person", "session", "loader_pos"]).groupby(["person", "session", "task"]).label.apply(
        lambda s: "".join(map(str, s)))
    print("distinct label sequences per task (across person × session):\n")
    print(tseq.reset_index().groupby("task").label.agg(lambda s: f"{s.nunique()} distinct of {len(s)}")
          .to_frame("sequences").to_markdown(), "\n")


def d4(rows):
    print("## D4 — calibration vs online as two tasks (X4f features, lopo folds, X4f's classifier settings)\n")
    res = []
    for proto, fold in SPLITS:
        if proto != "lopo":
            continue
        d = np.load(DUMP / f"X4f__{proto}__{fold}.npz")
        clf = joblib.load(DUMP / f"X4f__{proto}__{fold}.clf.joblib")
        tr = np.asarray(PS.split(proto, fold).train)
        rt_tr = rows.task.iloc[tr].str.startswith("Acquisition").to_numpy()
        te = d["test_idx"]
        rt_te = rows.task.iloc[te].str.startswith("Acquisition").to_numpy()
        for trn, mtr in (("calib", rt_tr), ("online", ~rt_tr), ("all", np.ones_like(rt_tr))):
            c2 = clf if trn == "all" else clone(clf).fit(d["F_train"][mtr], d["y_train"][mtr])
            for tst, mte in (("calib", rt_te), ("online", ~rt_te)):
                p = c2.predict(d["F_test"][mte])
                res.append(dict(fold=fold, train=trn, test=tst, n_train=int(mtr.sum()),
                                bacc=bacc(d["y_test"][mte], p)))
        print(f"  lopo {fold} done", file=sys.stderr, flush=True)
    t = pd.DataFrame(res).groupby(["train", "test"]).agg(n_train=("n_train", "mean"), bacc=("bacc", "mean"))
    print(t.round(4).to_markdown(), "\n")


def d5(df):
    print("## D5 — what would explain sealed 0.736 vs lopo 0.644 / later 0.701 (slot-2 ens)\n")
    for proto in ("lopo", "later"):
        s = df[df.proto == proto]
        a = {}
        for (rt), g in s.groupby("runtype"):
            a[rt] = cells(PC.load().rows, g.idx.to_numpy(), g.y.to_numpy(), g.p_ens.to_numpy()).score.mean()
        late = s[s.pos > 70]
        a["windows 71+ only"] = cells(PC.load().rows, late.idx.to_numpy(), late.y.to_numpy(),
                                      late.p_ens.to_numpy()).score.mean()
        print(f"- {proto}: cell means restricted to calib {a['calib']:.4f}, online {a['online']:.4f}, "
              f"windows 71+ {a['windows 71+ only']:.4f}")
    print()


def main(which):
    rows = PC.load().rows
    df = load_all(rows) if set(which) & {"D1", "D5"} else None
    if "D1" in which:
        d1(df)
    if "D3" in which:
        d3(rows)
    if "D5" in which:
        d5(df)
    if "D4" in which:
        d4(rows)
    if "D2" in which:
        d2(df, rows)


if __name__ == "__main__":
    main(sys.argv[1:] or ["D1", "D3", "D5", "D4", "D2"])
