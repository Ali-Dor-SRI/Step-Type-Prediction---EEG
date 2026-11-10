#!/usr/bin/env bash
# Sprint 11-12 Q4: classifier kinds on the X4f dumps, then nested stacking on slot 2.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=${TH:-4} OPENBLAS_NUM_THREADS=${TH:-4} MKL_NUM_THREADS=${TH:-4} OMP_WAIT_POLICY=PASSIVE
Q=~/codabench/logs/sprint_1112/q4
for k in "$@"; do
  python sprint1112_q4.py --kind "$k" 2>&1 | grep -v Warn
  python sprint1112_stack.py --new "$Q:$k" --label "q4_$k" 2>&1 | grep -v Warn
done
