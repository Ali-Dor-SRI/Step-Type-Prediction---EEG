#!/usr/bin/env bash
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUTF8=1
B=q5_q0.85_l5.0_s20_t20_f16
python sprint1114_bag.py BAGC ${B}_C0.003 ${B} ${B}_C0.03 ${B}_C0.1 ${B}_C0.3
python sprint1114_bag.py BAGQ q5_q0.7_l5.0_s20_t20_f16 ${B}_C0.03 q5_q1.0_l5.0_s20_t20_f16
python sprint1114_bag.py BAGROB ${B}_C0.03 ${B}_X4fhub_C0.03 ${B}_X4ftyl_C0.03
python sprint1114_bag.py BAGALL ${B}_C0.003 ${B} ${B}_C0.03 ${B}_C0.1 ${B}_C0.3 ${B}_X4fhub_C0.03 ${B}_X4ftyl_C0.03 A1a2 A1d a2a_t10
