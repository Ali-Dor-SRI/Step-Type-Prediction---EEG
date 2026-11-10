#!/usr/bin/env python
"""Sprint 2026-11-12 P1: bake the all-trained X4f + aligned-Shallow ensemble zip.

As sprint1109_bake.py, with ens_x4f_shallow_al.py and stream-aligned nets
(sprint1112_al_gates.make_folder_al: "align" = {Cg, buffer 64, prior 8} in
shallow_nets.pt). The X4f joblib is the byte-identical copy from the sealed
X4f all zip (60a0148c). Output: logs/sprint_1112/<out>/ (zip, SHA256.txt,
submission/, unzipped/).

    python sprint1112_bake_al.py --rows logs/sprint_1112/rows_align_all --w 0.5 --out al_cand_all
"""

import argparse
import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path.home() / "codabench/analysis"))
from sprint1112_al_gates import make_folder_al  # noqa: E402

CB = Path.home() / "codabench"
X4F_ZIP = CB / "logs/proteus_w1016/proteus_x4f_cand_all/riemann_sealed_proteus_x4f_all_2026-10-27.zip"
X4F_SHA = "60a0148c"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--seeds", default="33_34_35_36_37")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--out", default="al_cand_all")
    ap.add_argument("--name", default="ens_x4f_shallow_al_all")
    ap.add_argument("--adabn", type=float, default=None, help="AdaBN prior (opt-in)")
    a = ap.parse_args()
    assert sha256(X4F_ZIP).startswith(X4F_SHA), "sealed X4f zip is not 60a0148c"
    out = CB / "logs/sprint_1112" / a.out
    sub = out / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    with zipfile.ZipFile(X4F_ZIP) as z:
        (sub / "riemann_sealed.joblib").write_bytes(z.read("riemann_sealed.joblib"))
        jl_sha = hashlib.sha256(z.read("riemann_sealed.joblib")).hexdigest()
    make_folder_al(sub, sub / "riemann_sealed.joblib", a.rows, a.seeds.split("_"), a.w, "all", adabn=a.adabn)
    assert sha256(sub / "riemann_sealed.joblib") == jl_sha
    zp = out / f"{a.name}_{time.strftime('%Y-%m-%d')}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("submission.py", "riemann_sealed.joblib", "shallow_nets.pt"):
            z.write(sub / f, f)
    h = sha256(zp)
    (out / "SHA256.txt").write_text(f"{h}  {zp.name}\n")
    unz = out / "unzipped"
    if unz.exists():
        shutil.rmtree(unz)
    with zipfile.ZipFile(zp) as z:
        z.extractall(unz)
        names = z.namelist()
    print(f"zip {zp} {zp.stat().st_size} bytes, files {names}, w {a.w}, seeds {a.seeds}, adabn {a.adabn}")
    print(f"sha256 {h}")
    print(f"joblib sha256 {jl_sha} (byte-identical to the sealed X4f all zip's)")


if __name__ == "__main__":
    main()
