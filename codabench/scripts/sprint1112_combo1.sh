#!/usr/bin/env bash
# Sprint 11-12: Q5-net on top of B2 (X4f = Q5 q0.7/l5/t20), both mu; and AdaBN + Q5-net members together.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=1
O=~/codabench/logs/sprint_1112
python sprint1112_combo.py --arm $O/q5net/q7_lr1e-2_mu1 --label "B2 + q5net mu1" 2>&1 | grep -v Warn
python sprint1112_combo.py --arm $O/q5net/q7_lr1e-2_mu0.1 --label "B2 + q5net mu0.1" 2>&1 | grep -v Warn
