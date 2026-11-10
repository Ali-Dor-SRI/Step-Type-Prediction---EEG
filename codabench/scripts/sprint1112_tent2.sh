#!/usr/bin/env bash
# Sprint 11-12 TT2: IM at lr 1e-4 (rules out a step-size artefact of TT1), compare vs B1 nets.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs; R=$L/sprint_1109/rows_align; O=$L/sprint_1112/tent
python sprint1112_tent.py --rows $R --out $O/im_lr1e-4 --lr 1e-4 --beta 1 2>&1 | tail -1
python sprint1109_compare.py $L/sprint_1112/adabn/al_p32 $O/im_lr1e-4 33_34_35_36_37 0.3_0.4_0.5_0.6_0.7_0.8 2>&1 | grep -v Warn | tail -3
