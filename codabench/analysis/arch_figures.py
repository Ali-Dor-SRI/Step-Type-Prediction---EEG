#!/usr/bin/env python
"""Figures for codabench/MODEL_ARCHITECTURE.md (written to codabench/figures/).

PROTEUS figures (rewrite of 2026-10-26, for the F64 sealed candidate). Real data
from the window cache (~/neuralbench/proteus_cache) and the real sealed model
(the F64 `all` zip) unless a figure says "schematic":
  timeline    schematic of one trial and the window, one real stream's run layout,
              and the people x sessions grid with the local protocols
  shift       session-to-session distance of mean covariances, and how fast the
              F64 running reference settles on a new session
  recentre    PCA of the F64 feature vectors: one fixed reference / + per-stream
              covariance re-centring / + per-stream feature re-centring
  bands       one window's tangent-space input per band, as the F64 model sees it
  features    feature-vector composition (block sizes measured from the joblibs)
  signal      where the class signal is: Fisher ratio of log band power per channel
  results     the sprint 2026-10-16 ladder (numbers: LOG.md "2026-10-16 sprint")
Kept from the 2026-09-29 version (warm-up data, Dreyer 2023):
  eegnet      learned temporal responses + activation patterns of the uploaded
              EEGNet-StepType-WU1 (weights in submissions/, Dreyer windows in eda_cache)

Run in WSL:
    source ~/codabench/env040.sh
    python ~/codabench/analysis/arch_figures.py            # all PROTEUS figures
    python ~/codabench/analysis/arch_figures.py shift recentre
    python ~/codabench/analysis/arch_figures.py eegnet     # the kept warm-up figure

The 2026-09-29 figures (router, alignment, proxy results on the stand-in datasets)
are in git at commit bfbeea6.
"""
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "figures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "solvers/bci_decoding"))
if os.environ.get("COMPET_ROOT"):                  # benchmark_utils, for the solver import
    sys.path.insert(0, os.environ["COMPET_ROOT"])

F64_ZIP = ROOT / "logs/proteus_w1016/proteus_f64_cand_all/riemann_sealed_proteus_f64_all_2026-10-16.zip"
FI64_ZIP = ROOT / "logs/proteus_w1016/proteus_fi64_cand_all/riemann_sealed_proteus_fi64_all_2026-10-16.zip"
BUFFER, K = 64, 8                                  # F64's baked buffer and stream_k

BLUE, ORANGE, GREEN, GREY, PURPLE, LBLUE, RED = ("#2a78d4", "#f06a3c", "#12a679", "#8d8d88",
                                                 "#7a5cc7", "#9cc3f0", "#c8414b")
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 160, "axes.titleweight": "bold", "axes.titlesize": 11,
})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# shared PROTEUS state (loaded once)
# ---------------------------------------------------------------------------
_S = {}


def cache():
    if "c" not in _S:
        import proteus_cache as PC
        import proteus_splits as PS
        c = PC.load()
        keys = list(c.rows.groupby(["person", "session"]).groups)
        streams = {k: PS.stream_order(c.rows, np.where((c.rows.person == k[0])
                                                       & (c.rows.session == k[1]))[0])
                   for k in keys}
        _S["c"], _S["keys"], _S["streams"] = c, keys, streams
        log(f"cache: X={c.X.shape}, {len(keys)} sessions, {c.rows.person.nunique()} people")
    return _S["c"], _S["keys"], _S["streams"]


def model_parts(zpath=F64_ZIP):
    import joblib
    if zpath not in _S:
        _S[zpath] = joblib.load(io.BytesIO(zipfile.ZipFile(zpath).read("riemann_sealed.joblib")))
    return _S[zpath]


def f64_model():
    """The sealed model object exactly as load_model builds it from the baked defaults."""
    import riemann_sealed_proteus_f64 as RS
    parts = model_parts()
    m = RS.RiemannSealedModel(parts, None, align="stream", adapt="online", personal="pooled",
                              buffer=BUFFER, stream_k=K, fcenter=True, xblocks="bpt4")
    return RS, m


def affine_dist(A, B):
    w, V = np.linalg.eigh(A)
    Ai = (V / np.sqrt(w)) @ V.T
    m = np.linalg.eigvalsh(Ai @ B @ Ai)
    return float(np.sqrt(np.sum(np.log(np.maximum(m, 1e-12)) ** 2)))


def expm_sym(L):
    w, V = np.linalg.eigh((L + L.T) / 2)
    return (V * np.exp(w)) @ V.T


# ---------------------------------------------------------------------------
def fig_timeline():
    """Schematic trial timeline + one real stream's run layout + protocol grid."""
    c, keys, streams = cache()
    rows = c.rows
    fig = plt.figure(figsize=(12, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], width_ratios=[1.25, 1], hspace=0.45, wspace=0.25)

    # A: trial timeline (RELEASE_DAY.md § 1.3)
    a = fig.add_subplot(gs[0, :])
    a.set_xlim(-2.3, 12.6); a.set_ylim(-0.2, 3.1); a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.add_patch(Rectangle((1, 0.35), 4, 1.1, color=BLUE, alpha=0.18, lw=0))
    a.add_patch(Rectangle((1, 0.35), 2, 1.1, fill=False, ec=BLUE, lw=1.6))
    a.add_patch(Rectangle((3, 0.35), 2, 1.1, fill=False, ec=ORANGE, lw=1.6, ls="--"))
    a.text(2, 0.9, "task only\n(window s 0-2)", ha="center", va="center", fontsize=9, color=BLUE)
    a.text(4, 0.9, "task + feedback\n(window s 2-4)", ha="center", va="center", fontsize=9, color=ORANGE)
    a.text(3, 1.55, "model input: 4 s, 41 ch x 480 samples, µV",
           ha="center", fontsize=9, fontweight="bold", zorder=5,
           bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    for t, lab, col, y in [(-1.5, "trial start,\nfixation cross", GREY, 2.45), (-0.5, "beep\n(Graz only)", GREY, 1.85),
                           (0, "CUE\nmi / sub / word", "k", 2.45), (3, "feedback starts\n(sham in calibration runs)", ORANGE, 2.1),
                           (8, "trial end", GREY, 2.1), (10.5, "next cue\n(10.5-12.5 s)", GREY, 2.1)]:
        a.plot([t, t], [0, y - 0.05], color=col, lw=1.4 if col != GREY else 1.0)
        a.text(t, y, lab, ha="center", va="bottom", fontsize=8.5, color=col)
    a.set_xticks(range(-2, 13)); a.set_xlabel("time from the cue (s)")
    a.set_title("A  One trial (schematic, from the BIDS events)", loc="left")

    # B: one real stream's composition (P04 session 1)
    b = fig.add_subplot(gs[1, 0])
    k0 = ("P04", 1) if ("P04", 1) in streams else keys[0]
    idx = streams[k0]
    sub = rows.iloc[idx]
    x = 0
    calib_end = None
    for run, g in sub.groupby("run", sort=True):
        n = len(g)
        iface = g["interface"].iloc[0]
        calib = str(g["task"].iloc[0]).startswith("Acquisition")
        col = BLUE if iface == "Graz" else PURPLE
        b.add_patch(Rectangle((x, 0), n, 1, color=col, alpha=0.35 if calib else 0.85, lw=0.6, ec="white"))
        b.text(x + n / 2, 0.5, f"run {run}\n{iface}\n{'calib.' if calib else 'online'}\n{n}",
               ha="center", va="center", fontsize=7.8, color="k" if calib else "white")
        x += n
        if calib:
            calib_end = x
    b.set_xlim(0, x); b.set_ylim(-0.25, 1.5); b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.axvline(BUFFER, color=GREEN, lw=1.5, ls=":")
    b.text(BUFFER - 1.5, -0.2, f"buffer full ({BUFFER})", color=GREEN, fontsize=8, ha="right")
    if calib_end:
        b.text(calib_end + 1.5, 1.08, "online runs start", fontsize=8, ha="left")
        b.axvline(calib_end, color="k", lw=0.8, ls="--", ymax=0.7)
    b.set_xlabel("window number in the stream (arrival order)")
    b.set_title(f"B  One real stream ({k0[0]} session {k0[1]}, {x} windows)", loc="left")
    b.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE, label="Graz (bar)"),
                      Rectangle((0, 0), 1, 1, color=PURPLE, label="BrainHero (game)"),
                      Rectangle((0, 0), 1, 1, color=GREY, alpha=0.35, label="pale = calibration (sham)")],
             loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False, fontsize=8.3)

    # C: people x sessions grid with the protocols
    cc = fig.add_subplot(gs[1, 1])
    people = sorted(rows.person.unique())
    have = {k: len(v) for k, v in streams.items()}
    for i, p in enumerate(people):
        for s in (1, 2, 3):
            if (p, s) in have:
                later_test = s > 1
                cc.add_patch(Rectangle((s - 1, -i), 0.9, 0.82, color=ORANGE if later_test else BLUE, lw=0))
                cc.text(s - 1 + 0.45, -i + 0.41, str(have[(p, s)]), ha="center", va="center",
                        fontsize=7.5, color="white")
        cc.text(-0.12, -i + 0.41, p, ha="right", va="center", fontsize=8.5)
    for s in (1, 2, 3):
        cc.text(s - 1 + 0.45, 1.05, f"S{s}", ha="center", fontsize=9)
    cc.add_patch(Rectangle((-0.06, -3 - 0.06), 3.0, 0.94, fill=False, ec=RED, lw=1.6, ls="--"))
    cc.text(3.05, -3 + 0.41, "lopo: one person out\n(all their sessions);\n10 folds, 28 cells",
            fontsize=8, va="center", color=RED)
    cc.text(3.05, -6.6, "later: train on every S1,\ntest S2-S3 of P04, P10\n(8 cells)",
            fontsize=8, va="center", color=ORANGE)
    cc.set_xlim(-0.9, 5.4); cc.set_ylim(-len(people) + 0.4, 1.6); cc.axis("off")
    cc.set_title("C  The released data (windows per session)", loc="left")
    fig.text(0.56, 0.015, "Sealed test (private): later sessions of P01-P10  +  10 unseen people; "
             "no labels, one window at a time per session.", fontsize=8.8, color="#444")
    save(fig, "proteus_trial_and_data.png")


# ---------------------------------------------------------------------------
def fig_shift():
    """A: affine-invariant distance between session mean covariances. B: how fast a
    new session's reference settles, plain running mean vs F64's (buffer 64, K = 8)."""
    import riemann_sealed_proteus_f64 as RS
    c, keys, streams = cache()
    t0 = time.time()
    Ls, means = {}, {}
    for k in keys:
        C = RS._window_covs(np.asarray(c.X[streams[k]], dtype=np.float64))
        L = np.stack([RS._logm_spd(x) for x in C])
        Ls[k] = L
        means[k] = expm_sym(L.mean(0))                         # log-Euclidean session mean
    log(f"shift: covariances + logs of 14 sessions in {time.time() - t0:.0f} s")
    n = len(keys)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = affine_dist(means[keys[i]], means[keys[j]])
    same = [D[i, j] for i in range(n) for j in range(i + 1, n) if keys[i][0] == keys[j][0]]
    diff = [D[i, j] for i in range(n) for j in range(i + 1, n) if keys[i][0] != keys[j][0]]
    log(f"shift: different people median {np.median(diff):.2f} ({min(diff):.2f}-{max(diff):.2f}); "
        f"same person median {np.median(same):.2f} ({min(same):.2f}-{max(same):.2f})")

    curves = {"plain": [], "f64": []}
    for k in keys:
        Lg = np.mean([Ls[j].mean(0) for j in keys if j[0] != k[0]], 0)   # other people's mean (log domain)
        full = means[k]
        base = affine_dist(expm_sym(Lg), full)
        L = Ls[k]
        cs = np.cumsum(L, 0)
        pl, fr = [], []
        for m in range(1, len(L) + 1):
            pl.append(affine_dist(expm_sym(cs[m - 1] / m), full) / base)
            lo = max(0, m - BUFFER)
            win = L[lo:m]
            t = len(win) / (len(win) + K)
            fr.append(affine_dist(expm_sym((1 - t) * Lg + t * win.mean(0)), full) / base)
        curves["plain"].append(pl); curves["f64"].append(fr)
    log(f"shift: settling curves in {time.time() - t0:.0f} s")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})
    im = a.imshow(D, cmap="magma_r", vmin=0)
    lab = [f"{p} s{s}" for p, s in keys]
    a.set_xticks(range(n)); a.set_yticks(range(n))
    a.set_xticklabels(lab, rotation=90, fontsize=7.5); a.set_yticklabels(lab, fontsize=7.5)
    for i in range(n):
        for j in range(n):
            if i != j and keys[i][0] == keys[j][0]:
                a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=GREEN, lw=1.6))
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label="Riemannian distance of session mean covariances")
    a.set_title(f"A  Every session is its own domain\nsame person (green boxes) median {np.median(same):.1f}, "
                f"different people {np.median(diff):.1f}", loc="left", fontsize=10)

    L_max = min(len(x) for x in curves["f64"])          # every session contributes to every n
    for name, col, lw in (("plain", GREY, 1.0), ("f64", GREEN, 1.0)):
        curves[name] = [cv[:L_max] for cv in curves[name]]
        for cv in curves[name]:
            b.plot(np.arange(1, len(cv) + 1), cv, color=col, lw=0.5, alpha=0.35)
        M = np.full((len(curves[name]), L_max), np.nan)
        for i, cv in enumerate(curves[name]):
            M[i, :len(cv)] = cv
        med = np.nanmedian(M, 0)
        b.plot(np.arange(1, L_max + 1), med, color=col if name == "plain" else "#0a7a58", lw=2.4,
               label=("running mean of every window so far" if name == "plain"
                      else f"F64's reference: last {BUFFER} windows, pulled to the\ntraining mean with weight {K}/(n+{K})"))
        for q in (8, 16, 32, 64, 128):
            if name == "f64":
                log(f"shift: F64 reference after {q} windows: median relative distance {med[q - 1]:.2f}")
            else:
                log(f"shift: plain running mean after {q} windows: median {med[q - 1]:.2f}")
    b.axhline(1, color="k", ls=":", lw=1)
    b.text(L_max, 1.02, "= the training mean (no adaptation)", ha="right", fontsize=8.5)
    b.axvline(BUFFER, color=GREEN, ls=":", lw=1)
    b.set_xlim(1, L_max); b.set_ylim(0, 1.15)
    b.set_xlabel("windows of the new session seen so far (n)")
    b.set_ylabel("distance to the session's own centre\n(relative to the training mean's distance)")
    b.legend(frameon=False, fontsize=8.5, loc="center right", bbox_to_anchor=(1, 0.66))
    b.set_title("B  A new session's reference settles within tens of windows\n"
                "(thin lines: the 14 sessions, each treated as unseen; bold: median)", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "session_shift.png")


# ---------------------------------------------------------------------------
def _share(V, groups):
    mu = V.mean(0)
    tot = ((V - mu) ** 2).sum()
    ss = sum((groups == g).sum() * ((V[groups == g].mean(0) - mu) ** 2).sum() for g in np.unique(groups))
    return float(ss / tot)


def fig_recentre():
    """PCA of the F64 feature vectors under three re-centrings, the real model's code."""
    from sklearn.decomposition import PCA
    c, keys, streams = cache()
    RS, model = f64_model()
    parts = model.parts
    t0 = time.time()
    Fn, Fs, Ff, sess, yy = [], [], [], [], []
    n_equal = 0
    for i, k in enumerate(keys):
        X = np.asarray(c.X[streams[k]], dtype=np.float64)
        Fn.append(RS._features(parts, np.einsum("ij,njt->nit", parts["W_global"], X)))
        model.reset_state()
        F_s = RS._features(parts, model._stream_whiten(X))
        F_f = model._stream_fcenter(F_s)
        Fs.append(F_s); Ff.append(F_f)
        # the path drawn here = the model's own prediction (fresh state, B = 1)
        model.reset_state()
        P_model = np.concatenate([model.predict_proba(X[j:j + 1].astype(np.float32)) for j in range(len(X))])
        n_equal += int((P_model.argmax(1) == parts["lda"].predict_proba(F_f).argmax(1)).sum())
        sess += [i] * len(X); yy += list(c.y[streams[k]])
        log(f"recentre: {k} ({len(X)} windows) {time.time() - t0:.0f} s")
    sess, yy = np.array(sess), np.array(yy)
    log(f"recentre: figure path = model's streamed prediction on {n_equal}/{len(sess)} windows")
    assert n_equal == len(sess), "the figure's feature path differs from the model"
    person = np.array([keys[s][0] for s in sess])
    snum = np.array([keys[s][1] for s in sess])
    people = sorted(set(person))
    cmap = plt.get_cmap("tab10")
    panels = [("A  one fixed reference for everyone\n(no per-session re-centring)", np.concatenate(Fn)),
              ("B  + covariance re-centring per stream\n(the S64 step)", np.concatenate(Fs)),
              ("C  + feature re-centring per stream\n(F64, the candidate)", np.concatenate(Ff))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    rng = np.random.default_rng(0)
    for ax, (title, F) in zip(axes, panels):
        Z = (F - F.mean(0)) / (F.std(0) + 1e-12)
        vs, vc = _share(Z, sess), _share(Z, yy)
        P2 = PCA(2, random_state=0).fit_transform(Z)
        order = rng.permutation(len(Z))
        for pi, p in enumerate(people):
            for s_, mk in ((1, "o"), (2, "^"), (3, "s")):
                m = (person[order] == p) & (snum[order] == s_)
                if m.any():
                    ax.scatter(P2[order][m, 0], P2[order][m, 1], s=7, marker=mk, color=cmap(pi), alpha=0.55, lw=0)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.text(0.02, 0.02, f"variance explained by\n  session: {100 * vs:.0f} %\n  class: {100 * vc:.1f} %",
                transform=ax.transAxes, fontsize=8.5, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
        log(f"recentre: {title.splitlines()[0]}: session share {vs:.4f}, class share {vc:.4f}")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=cmap(i), label=p) for i, p in enumerate(people)]
               + [Line2D([], [], marker=mk, ls="", color="#555", label=f"session {s_}") for s_, mk in ((1, "o"), (2, "^"), (3, "s"))],
               loc="lower center", ncol=13, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"The 5,302-number feature vector of all {len(sess):,} windows (each feature z-scored), "
                 "streamed session by session through the sealed model's own code", fontsize=10.5, y=1.01)
    fig.tight_layout()
    save(fig, "recentring_pca.png")


# ---------------------------------------------------------------------------
def fig_bands():
    """One window as the F64 model sees it: raw, then its tangent-space input per band."""
    from pyriemann.estimation import Covariances
    c, keys, streams = cache()
    RS, model = f64_model()
    parts = model.parts
    k0 = ("P06", 1) if ("P06", 1) in streams else keys[0]
    idx = streams[k0]
    j = 100                                                   # an online-run window
    X = np.asarray(c.X[idx[:j + 1]], dtype=np.float64)
    model.reset_state()
    Xw = model._stream_whiten(X)[j:j + 1]                   # whitened with the stream's reference
    names = list(c.meta["ch_names"])
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1.25], hspace=0.32, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    t = 1 + np.arange(X.shape[-1]) / 120.0
    w = X[j]
    sc = np.percentile(np.abs(w), 99) * 1.0
    for i in range(w.shape[0]):
        ax.plot(t, w[i] / sc + i * 1.6, color="#333", lw=0.5)
    ax.set_yticks(np.arange(w.shape[0]) * 1.6); ax.set_yticklabels(names, fontsize=5.6)
    ax.axvline(3, color=ORANGE, lw=1.2, ls="--")
    ax.text(3.02, w.shape[0] * 1.6, "feedback starts", color=ORANGE, fontsize=8.5, va="bottom")
    ax.set_xlim(1, 5); ax.set_xlabel("time after the cue (s)"); ax.spines["left"].set_visible(False)
    lab = c.meta["classes"][int(c.y[idx[j]])]
    ax.set_title(f"One 4-s window as delivered ({k0[0]} session {k0[1]}, window {j + 1} of the stream, class '{lab}'; "
                 "41 channels, µV, per-channel scale for display)", loc="left", fontsize=10)
    mats = []
    cov = Covariances(estimator=parts["estimator"])
    C = cov.transform(Xw)
    mats.append(("broadband", parts["broad_ts"].reference_, C[0]))
    for band, ts in zip(parts["fb_bands"], parts["fb_ts"]):
        Cb = RS._band_covs(parts, Xw, band)[0]
        mats.append((f"{band.replace('to', '-')} Hz", ts.reference_, Cb))
    logs = []
    for label, R, Cw in mats:
        Rm = RS._inv_sqrtm(R)
        lam, V = np.linalg.eigh(Rm @ Cw @ Rm)
        logs.append((label, (V * np.log(np.maximum(lam, 1e-12))) @ V.T))
    vmax = np.percentile(np.abs(np.stack([m for _, m in logs])), 99)
    for jj, (label, S) in enumerate(logs):
        a = fig.add_subplot(gs[1, jj])
        im = a.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(label, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        if jj == 0:
            a.set_ylabel("re-centred coordinate x\nre-centred coordinate")
    fig.colorbar(im, ax=fig.axes[1:], fraction=0.018, pad=0.02, label="log-map (deviation from the\nre-centred average)")
    fig.text(0.5, 0.02, "Each panel: the window's covariance in that band after re-centring on its stream's running reference, "
             "mapped to the tangent space at the training mean (matrix log), i.e. what the LDA receives from that block.\n"
             "Each 41 x 41 symmetric matrix gives 861 numbers. Axes are re-centred coordinates (near, but not identical to, "
             "the channels): diagonal = change in power, off-diagonal = change in coupling.", ha="center", fontsize=8.5, color="#444")
    save(fig, "band_covariances.png")


# ---------------------------------------------------------------------------
def fig_features():
    """Block sizes measured by running the joblibs' own feature code on two windows."""
    c, keys, streams = cache()
    import riemann_sealed_proteus_f64 as RS
    X = np.asarray(c.X[:2], dtype=np.float64)
    rows = []
    for name, z in (("F64 (the candidate)\n41 ch, 3 classes", F64_ZIP), ("FI64 (variant)\n+ icoh", FI64_ZIP)):
        blocks = RS._feature_blocks(model_parts(z), X)
        sizes = {k: v.shape[1] for k, v in blocks.items()}
        n_lda = model_parts(z)["lda"].coef_.shape[1]
        assert sum(sizes.values()) == n_lda, (sizes, n_lda)
        rows.append((name, sizes))
        log(f"features: {name.splitlines()[0]}: {sizes} = {n_lda}")
    labels = {"xdawn": "xDAWN", "broad": "broadband", "logvar": "log-variance", "fb": "filter bank (4 bands)",
              "x:bpt4": "bpt4 (band power, 4 x 1 s)", "x:icoh": "icoh (imaginary coherence)"}
    cols = {"xdawn": ORANGE, "broad": BLUE, "logvar": GREEN, "fb": PURPLE, "x:bpt4": RED, "x:icoh": GREY}
    fig, ax = plt.subplots(figsize=(11.5, 2.9))
    seen = set()
    for r, (name, sizes) in enumerate(rows):
        left = 0
        for k, v in sizes.items():
            ax.barh(r, v, left=left, color=cols[k], height=0.6, label=None if k in seen else labels[k])
            seen.add(k)
            if v > 400:
                ax.text(left + v / 2, r, f"{v:,}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            else:
                ax.text(left + v / 2, r - 0.38, f"{v}", ha="center", va="bottom", color=cols[k], fontsize=8)
            left += v
        ax.text(left + 80, r, f"{left:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 9600); ax.set_xlabel("features per window (input to the LDA)")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.42), fontsize=9)
    ax.set_title("Feature-vector composition (measured from the sealed zips)", loc="left", pad=44)
    fig.tight_layout()
    save(fig, "feature_composition.png")


# ---------------------------------------------------------------------------
def fig_signal():
    """Median-over-people Fisher ratio of log band power per channel (reports/proteus_eda § 3)."""
    import mne
    from scipy.signal import welch
    c, keys, streams = cache()
    names = list(c.meta["ch_names"])
    sf = float(c.meta["sfreq"])
    f, P = welch(np.asarray(c.X, dtype=np.float64), fs=sf, nperseg=int(sf), axis=-1)
    rows, y = c.rows, c.y
    bands = {"theta 4-8 Hz": (4, 8), "alpha 8-13 Hz": (8, 13), "beta 13-30 Hz": (13, 30)}
    info = mne.create_info(names, sf, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for ax, (bname, (lo, hi)) in zip(axes, bands.items()):
        bp = np.log(P[:, :, (f >= lo) & (f < hi)].mean(-1))
        per = []
        for p in sorted(rows.person.unique()):
            m = (rows.person == p).to_numpy()
            b, yy = bp[m], y[m]
            mu = b.mean(0)
            between = sum(((b[yy == k].mean(0) - mu) ** 2) * (yy == k).mean() for k in range(3))
            within = sum(b[yy == k].var(0) * (yy == k).mean() for k in range(3))
            per.append(between / within)
        fr = np.median(np.stack(per), 0)
        best = int(np.argmax(fr))
        im, _ = mne.viz.plot_topomap(fr, info, axes=ax, show=False, contours=0, sensors=True,
                                     cmap="Reds", vlim=(0, 0.32), sphere=(0, 0, 0, 0.095),
                                     names=[n if i == best else "" for i, n in enumerate(names)])
        ax.set_title(f"{bname}\nbest {names[best]} {fr[best]:.3f}", fontsize=10)
        log(f"signal: {bname}: best {names[best]} {fr[best]:.3f}")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="Fisher ratio (between-class / within-class),\nmedian over the 10 people")
    fig.suptitle("Where the three classes differ: log band power per channel (no eye channels in the release; "
                 "AF3/AF4 are the eye-nearest)", fontsize=10, y=0.98)
    save(fig, "class_signal.png")


# ---------------------------------------------------------------------------
def fig_results():
    # LOG.md "2026-10-16 sprint" (Phases 1, 3, 6, 7b) and logs/proteus_setup/RESULTS.md (MeanLogReg)
    arms = [("Mean-\nLogReg\n(baseline)", 0.4428, 0.4288, GREY, None),
            ("pooled\nLDA", 0.4185, 0.4924, LBLUE, None),
            ("+ bpt4", 0.4268, 0.5279, LBLUE, None),
            ("+ cov.\nre-centring\n(S64)", 0.4839, 0.5302, BLUE, None),
            ("+ feature\nre-centring\n(F64)", 0.5672, 0.6005, GREEN, None),
            ("+ icoh\n(FI64,\nvariant)", 0.5838, 0.6240, GREEN, "//"),
            ("EEGNet\n(5 seeds)", 0.4852, 0.4764, ORANGE, None)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, (pi, title) in zip(axes, ((1, "lopo: unseen people (28 cells)"), (2, "later: later sessions of P04, P10 (8 cells)"))):
        for i, a in enumerate(arms):
            v = a[pi]
            ax.bar(i, v, 0.72, color=a[3], hatch=a[4], edgecolor="white" if a[4] is None else "#0a7a58")
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5,
                    fontweight="bold" if "(F64)" in a[0] else None)
        ax.axhline(1 / 3, color="k", ls=":", lw=1.2)
        ax.text(len(arms) - 0.5, 1 / 3 + 0.006, "chance", ha="right", fontsize=8.5)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[0] for a in arms], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_ylim(0.3, 0.68)
    axes[0].set_ylabel("cell-averaged balanced accuracy")
    fig.suptitle("Sprint 2026-10-16 on PROTEUS: re-centring per session carries the model", fontsize=11,
                 fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "proteus_results.png")


# ---------------------------------------------------------------------------
def fig_eegnet():
    import mne
    import torch
    HOME = Path.home()
    z = zipfile.ZipFile(HOME / "codabench/submissions/eegnet_steptype_wu1_2026-09-24.zip")
    sd = torch.load(io.BytesIO(z.read("weights.pt")), map_location="cpu")
    Wt = sd["temporal_conv.weight"].numpy()[:, 0, 0, :]          # (8, 60)
    Ws = sd["spatial_depthwise.weight"].numpy()[:, 0, :, 0]      # (16, 27)
    cache_dir = HOME / "neuralbench/eda_cache"
    import json
    meta = json.loads((cache_dir / "meta.json").read_text())
    names, sf = meta["ch_names"], meta["sfreq"]
    Xall = np.load(cache_dir / "X.npy", mmap_mode="r")
    split = np.load(cache_dir / "split.npy")
    rng = np.random.default_rng(0)
    rows = rng.choice(np.where(split == 0)[0], 1500, replace=False)
    Xd = np.asarray(Xall[np.sort(rows)], dtype=np.float64)       # (n, 27, 480)
    log(f"eegnet: windows {Xd.shape}, temporal {Wt.shape}, spatial {Ws.shape}")
    Kk = Wt.shape[1]
    left = (Kk - 1) // 2                                         # TF "same" padding
    Xp = np.pad(Xd, ((0, 0), (0, 0), (left, Kk - 1 - left)))
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(Xp, Kk, axis=-1)                   # (n, C, T, K)
    freqs = np.fft.rfftfreq(2048, 1 / sf)
    resp, peak, pats = [], [], []
    for k in range(Wt.shape[0]):
        y = win @ Wt[k]                                          # conv2d is cross-correlation: (n, C, T)
        y = y - y.mean(-1, keepdims=True)
        S = np.einsum("nct,ndt->cd", y, y) / (y.shape[0] * y.shape[-1])
        h = np.abs(np.fft.rfft(Wt[k], 2048))
        resp.append(h / h.max()); peak.append(float(freqs[h.argmax()]))
        pk = []
        for dd in range(2):
            w = Ws[2 * k + dd]
            a = S @ w / (w @ S @ w)
            pk.append(a / np.abs(a).max())
        pats.append(pk)
    order = np.argsort(peak)
    info = mne.create_info(names, sf, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
    fig, axes = plt.subplots(8, 3, figsize=(7.4, 13.2), gridspec_kw={"width_ratios": [1.9, 1, 1], "wspace": 0.12, "hspace": 0.5})
    for r, k in enumerate(order):
        a = axes[r, 0]
        a.fill_between(freqs, resp[k], color=LBLUE, alpha=0.7); a.plot(freqs, resp[k], color=BLUE, lw=1.2)
        a.set_xlim(0, 60); a.set_ylim(0, 1.05); a.set_yticks([])
        a.axvline(peak[k], color=ORANGE, lw=0.8, ls="--")
        a.text(58, 0.85, f"peak {peak[k]:.1f} Hz", ha="right", fontsize=8.5, color=ORANGE)
        a.set_ylabel(f"temporal\nfilter {r + 1}", rotation=0, ha="right", va="center", fontsize=8.5)
        if r < 7:
            a.set_xticklabels([])
        else:
            a.set_xlabel("frequency (Hz)")
        for dd in range(2):
            ax = axes[r, 1 + dd]
            mne.viz.plot_topomap(pats[k][dd], info, axes=ax, show=False, contours=0, sensors=True,
                                 cmap="RdBu_r", vlim=(-1, 1), sphere=(0, 0, 0, 0.095))
            if r == 0:
                ax.set_title(f"spatial pattern {dd + 1}", fontsize=9)
    fig.suptitle("Uploaded EEGNet-StepType-WU1 (Dreyer warm-up): learned temporal filters and\nspatial activation patterns "
                 "(sign arbitrary), filters sorted by peak frequency", fontsize=10, y=0.94)
    save(fig, "eegnet_learned_filters.png")
    log("eegnet: peak frequencies (Hz), sorted: " + ", ".join(f"{peak[k]:.1f}" for k in order))


FIGS = {"timeline": fig_timeline, "shift": fig_shift, "recentre": fig_recentre, "bands": fig_bands,
        "features": fig_features, "signal": fig_signal, "results": fig_results, "eegnet": fig_eegnet}
DEFAULT = ["timeline", "shift", "recentre", "bands", "features", "signal", "results"]

if __name__ == "__main__":
    want = sys.argv[1:] or DEFAULT
    t0 = time.time()
    for n in want:
        log(f"=== {n}")
        FIGS[n]()
    log(f"done in {time.time() - t0:.0f} s")
