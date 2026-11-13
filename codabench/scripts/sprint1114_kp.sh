#!/usr/bin/env bash
source ~/codabench/env040.sh >/dev/null
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUTF8=1
O=~/codabench/logs/sprint_1114/kp.out
mkdir -p ~/codabench/logs/sprint_1114/kp
: > $O
for k in 1 3 10 30; do python sprint1114_kp.py st --k $k >> $O 2>&1 & done
wait
python sprint1114_kp.py final >> $O 2>&1
echo DONE >> $O
