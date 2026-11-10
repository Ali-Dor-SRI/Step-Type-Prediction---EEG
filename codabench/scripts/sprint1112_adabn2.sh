#!/usr/bin/env bash
# Sprint 11-12 AB2: AdaBN priors 2 / 4 / 16 on aligned nets; compare all on a wider w grid.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs
O=$L/sprint_1112/adabn
for p in 2 4 16; do
  python sprint1112_adabn.py --rows $L/sprint_1109/rows_align --out $O/al_p$p --prior $p 2>&1 | grep -v Warn | tail -1 &
done
wait
for p in 2 4 8 16 32; do
  echo "== prior $p"
  python sprint1109_compare.py $L/sprint_1106/rows $O/al_p$p 33_34_35_36_37 0.3_0.4_0.5_0.6_0.7_0.8_0.9 2>&1 | grep -v Warn | tail -3
done
