#!/usr/bin/env bash
# Sprint 11-12 Wed night: (a) AdaBN on aligned FBCNet, stack on B3; (b) Q5 (B3 config) on the window halves,
# stack as extra X4f-type members on B3.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE
O=~/codabench/logs/sprint_1112; S=~/codabench/scripts/sprint1112_job.sh
B3=q5_q0.85_l5.0_s20_t20_f16_C0.03
OMP_NUM_THREADS=1 python sprint1112_adabn_gen.py --model fbcnet --rows $O/rows_fbc_al --out $O/adabn/fbc_p32 --prior 32 2>&1 | tail -n 1 &
bash $S q5_c02 3 python sprint1112_q5.py --model X4fc02 --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --C 0.03 &
bash $S q5_c24 3 python sprint1112_q5.py --model X4fc24 --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --C 0.03 &
wait
python sprint1112_combo.py --x4f $B3 --arm "$O/adabn/al_p32+$O/adabn/fbc_p32:fbcnet" --label "B3 + FBCNet aligned AdaBN" 2>&1 | grep -v Warn
for X in "$B3+${B3}_X4fc02" "$B3+${B3}_X4fc24" "$B3+${B3}_X4fc02+${B3}_X4fc24"; do
  python sprint1112_combo.py --x4f "$X" --x4f-base $B3 --arm $O/adabn/al_p32 --label "B3 x4f-part = $X" 2>&1 | grep -v Warn
done
