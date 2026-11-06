#!/usr/bin/env python
"""Sprint 2026-11-09 W5: bake the all-trained X4f-Shallow-ens submission zip.

folder logs/sprint_1109/ens_cand_all/submission/ =
  submission.py          = solvers/bci_decoding/ens_x4f_shallow.py
  riemann_sealed.joblib  = byte-identical copy from the sealed X4f all zip (60a0148c...)
  shallow_nets.pt        = {"w", "seeds", "state_dicts"} of the K nets trained on `all`
zip -> logs/sprint_1109/ens_cand_all/ens_x4f_shallow_all_<date>.zip, sha256 printed
and written next to it. Gates are run separately on the unzipped folder.

    python sprint1109_bake.py --rows logs/sprint_1109/rows_all --seeds 33_34_35_36_37 --w 0.3
"""

import argparse
import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
from sprint1109_w1_gates import make_folder  # noqa: E402

CB = Path.home() / "codabench"
X4F_ZIP = CB / "logs/proteus_w1016/proteus_x4f_cand_all/riemann_sealed_proteus_x4f_all_2026-10-27.zip"
X4F_SHA = "60a0148c"
OUT = CB / "logs/sprint_1109/ens_cand_all"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--w", type=float, default=0.3)
    a = ap.parse_args()
    assert sha256(X4F_ZIP).startswith(X4F_SHA), "sealed X4f zip is not 60a0148c"
    sub = OUT / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    with zipfile.ZipFile(X4F_ZIP) as z:
        (sub / "riemann_sealed.joblib").write_bytes(z.read("riemann_sealed.joblib"))
        jl_sha = hashlib.sha256(z.read("riemann_sealed.joblib")).hexdigest()
    make_folder(sub, sub / "riemann_sealed.joblib", a.rows, a.seeds.split("_"), a.w, "all")
    assert sha256(sub / "riemann_sealed.joblib") == jl_sha
    zp = OUT / f"ens_x4f_shallow_all_{time.strftime('%Y-%m-%d')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("submission.py", "riemann_sealed.joblib", "shallow_nets.pt"):
            z.write(sub / f, f)
    h = sha256(zp)
    (OUT / "SHA256.txt").write_text(f"{h}  {zp.name}\n")
    unz = OUT / "unzipped"
    if unz.exists():
        shutil.rmtree(unz)
    with zipfile.ZipFile(zp) as z:
        z.extractall(unz)
        names = z.namelist()
    print(f"zip {zp} {zp.stat().st_size} bytes, files {names}")
    print(f"sha256 {h}")
    print(f"joblib sha256 {jl_sha} (byte-identical to the sealed X4f all zip's)")


if __name__ == "__main__":
    main()
