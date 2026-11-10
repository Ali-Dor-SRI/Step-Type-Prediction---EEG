#!/usr/bin/env bash
# Sprint 11-12: rebuild ens_x4f_shallow_al.py; gates (i)/(ii) without AdaBN (regression) and with AdaBN p32 w 0.5.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs
python sprint1112_build_ens_al.py
python sprint1112_al_gates.py --w 0.5 --out $L/sprint_1112/al_gates 2>&1 | grep -E "PASS|FAIL|cell mean" > $L/sprint_1112/al_gates_i_ii_rebuild.log &
python sprint1112_al_gates.py --w 0.5 --adabn 32 --saved $L/sprint_1112/adabn/al_p32 --out $L/sprint_1112/ab_gates 2>&1 | grep -E "PASS|FAIL|cell mean|loaded" > $L/sprint_1112/ab_gates_i_ii.log &
wait
cat $L/sprint_1112/al_gates_i_ii_rebuild.log $L/sprint_1112/ab_gates_i_ii.log
