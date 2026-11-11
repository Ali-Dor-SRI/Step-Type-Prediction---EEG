#!/usr/bin/env bash
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2
O=~/codabench/logs/sprint_1112
B3=q5_q0.85_l5.0_s20_t20_f16_C0.03
H2=q5_q0.85_l5.0_s20_t20_f16_X4fc02_C0.03; H4=q5_q0.85_l5.0_s20_t20_f16_X4fc24_C0.03
for X in "$B3+$H2" "$B3+$H4" "$B3+$H2+$H4"; do
  python sprint1112_combo.py --x4f "$X" --x4f-base $B3 --arm $O/adabn/al_p32 --base $O/adabn/al_p32 --label "x4f-part = ${X//q5_q0.85_l5.0_s20_t20_f16_/}" 2>&1 | grep -v Warn
done
