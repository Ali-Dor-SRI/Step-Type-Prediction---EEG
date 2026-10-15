#!/usr/bin/env python
"""Self-test of the PROTEUS setup: G0, G2-G8 and G10 in one command, under
5 minutes, training nothing.

    source ~/codabench/env040.sh
    python ~/codabench/analysis/proteus_selftest.py

Reads the window cache (``~/neuralbench/proteus_cache``), the official loader,
the Phase 1 ``MeanLogReg`` export (``logs/proteus_setup/ref_meanlogreg/submission``)
and the ``official`` Riemann export of Phase 5
(``logs/proteus_setup/cand_official/submission``). G6's two CLI runs are
subprocesses whose parquets and logs land under ``logs/proteus_setup/selftest/``;
G4 runs ``proteus_g4_events.py`` as ``python -I -B -X utf8`` (the only script
that opens the BIDS files). Prints one PASS / FAIL line per check and exits
non-zero on any failure. Setup session 2026-10-15, brief
``prompts/2026-10-13_proteus_train_test_setup.md`` Phase 6.
"""

import importlib.metadata
import os
import subprocess
import sys
import time
from pathlib import Path

COMPET_ROOT = os.environ.get("COMPET_ROOT", str(Path.home() / "neuralbench/competition-040"))
sys.path.insert(0, COMPET_ROOT + "/tracks/bci_decoding")
sys.path.insert(0, str(Path.home() / "codabench/analysis"))

LOGS = Path.home() / "codabench/logs/proteus_setup"
REF = LOGS / "ref_meanlogreg"
EXPORT = LOGS / "cand_official/submission"
BIDS = (Path(os.environ.get("BENCHOPT_DATA_HOME", Path.home() / "neuralbench/benchopt_data_040"))
        / "neural_compet/Dreyer2026Proteus/download/nm000290")


def g0():
    out = []
    try:
        v = importlib.metadata.version("neuralbench")
    except Exception as e:                      # noqa: BLE001
        v = repr(e)
    out.append(("G0 neuralbench == 0.4.0", v == "0.4.0", v))
    try:
        import benchmark_utils
        f = benchmark_utils.__file__
    except Exception as e:                      # noqa: BLE001
        f = repr(e)
    out.append(("G0 benchmark_utils under competition-040", "/competition-040/" in str(f), str(f)))
    out.append(("G0 sys.prefix ends with .venv040", sys.prefix.endswith(".venv040"), sys.prefix))
    home = os.environ.get("BENCHOPT_DATA_HOME", "")
    out.append(("G0 BENCHOPT_DATA_HOME ends with benchopt_data_040", home.endswith("benchopt_data_040"), home))
    return out


def g4(c):
    keys = c.root / "keys.csv"
    cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(Path.home() / "codabench/analysis/proteus_g4_events.py"),
           str(BIDS), str(keys)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = []
    for line in r.stdout.splitlines():
        if line.startswith(("PASS ", "FAIL ")):
            name, _, value = line[5:].partition(": ")
            out.append((name, line.startswith("PASS "), value))
    out.append(("G4 script exit 0", r.returncode == 0, f"rc {r.returncode}" + ("" if r.returncode == 0 else " " + r.stderr[-300:])))
    return out


def main():
    t0 = time.time()
    checks, timings = [], []

    def lap(name, items):
        checks.extend(items)
        timings.append((name, time.time() - t0 - sum(t for _, t in timings)))

    lap("G0", g0())
    if not all(ok for _, ok, _ in checks):
        import proteus_cache as PC
        PC.report(checks)
        print("SELFTEST FAIL: the environment is wrong (G0); nothing else can be trusted")
        sys.exit(1)
    import proteus_cache as PC
    import proteus_splits as PS
    import proteus_eval as PE
    try:
        c = PC.load()
    except RuntimeError as e:
        print(f"FAIL G2 cache: {e}")
        sys.exit(1)
    loaders, meta = PC.official_loaders()
    lap("G2", [("G2 " + n, ok, v) for n, ok, v in PC.check_content(c)])
    lap("G3", PC.check_equal_loader(c, loaders))
    lap("G4", g4(c))
    lap("G5a-d", PS.check_sizes(c.rows) + PS.check_leaks(c.rows) + PS.check_order(loaders=loaders, cache=c)
        + PS.check_ids(c.rows))
    for need, what in ((REF / "submission" / "weights.joblib", "the Phase 1 MeanLogReg export"),
                       (REF / "train.parquet", "the Phase 1 reference parquet"),
                       (EXPORT / "riemann_sealed.joblib", "the Phase 5 official Riemann export")):
        if not need.exists():
            checks.append((f"input present: {what}", False, str(need)))
    if all(ok for _, ok, _ in checks):
        lap("G5e-G8", PE.run_gates(LOGS / "selftest", REF, c, loaders, meta))
        lap("G10", PE.check_streaming_contract(c, EXPORT, loaders, meta, fit=False))
    ok = PC.report(checks)
    n_fail = sum(1 for _, k, _ in checks if not k)
    print("timings: " + ", ".join(f"{n} {t:.1f} s" for n, t in timings) + f"; total {time.time() - t0:.1f} s")
    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}: {len(checks) - n_fail} / {len(checks)} checks pass, "
          f"{n_fail} fail", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
