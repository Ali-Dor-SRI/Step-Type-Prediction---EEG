#!/usr/bin/env bash
# usage: sprint1114_eval.sh NAME... : per-arm verdict vs B3 (arm alone, w nested)
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUTF8=1
for n in "$@"; do echo "== $n"; python sprint1114_vsb3.py "$n" --no-b3-option 2>&1 | grep -v "fold picks"; done
