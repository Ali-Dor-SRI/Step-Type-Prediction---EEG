#!/usr/bin/env bash
# Sprint 11-12: rebuild the solver; regression gates (AdaBN, no selftrain) + self-training gates.
# Usage: sprint1112_stgates.sh <probe> <q> <lam> <step>
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs/sprint_1112
python sprint1112_build_ens_al.py
python sprint1112_al_gates.py --w 0.5 --adabn 32 --saved $L/adabn/al_p32 --out $L/ab_gates 2>&1 | grep -E "PASS|FAIL" > $L/ab_gates_i_ii_rebuild2.log
tail -1 $L/ab_gates_i_ii_rebuild2.log
python sprint1112_st_gates.py --probe "$1" --q "$2" --lam "$3" --step "$4" --w 0.5 2>&1 | grep -E "PASS|FAIL|loaded|Error|Trace" | tee $L/st_gates_$1.log
