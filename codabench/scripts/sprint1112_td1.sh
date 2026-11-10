#!/usr/bin/env bash
# Sprint 11-12 Q9+ first pass: time-domain variants, then each stacked on slot 2 (nested).
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
T=~/codabench/logs/sprint_1112/td
python sprint1112_td.py --name td_raw_all --mode raw --ch all 2>&1 | grep row
python sprint1112_td.py --name td_dm_front --mode dm --ch front 2>&1 | grep row
python sprint1112_td.py --name td_dm_b15 --mode dm --bin 15 --ch all 2>&1 | grep row
for n in td_dm_all td_raw_all td_dm_front td_dm_b15; do
  python sprint1112_stack.py --new "$T:$n" --label "$n" 2>&1 | grep -v Warn
done
