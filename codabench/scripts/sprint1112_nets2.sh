#!/usr/bin/env bash
# Sprint 11-12: AdaBN p32 on plain Shallow 33-37 (rows_pl) and aligned 38-42 (rows_al2); stack on B2.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
O=~/codabench/logs/sprint_1112
python sprint1112_adabn_gen.py --model shallow --rows $O/rows_pl --out $O/adabn/pl_p32 --prior 32 2>&1 | tail -1 &
python sprint1112_adabn_gen.py --model shallow --rows $O/rows_al2 --out $O/adabn/al2_p32 --prior 32 --seeds 38_39_40_41_42 2>&1 | tail -1 &
wait
X=q5_q0.85_l5.0_s20_t20_f16
python sprint1112_combo.py --x4f $X --arm "$O/adabn/al_p32+$O/adabn/al2_p32:shallow:38_39_40_41_42" --label "B2 + 5 more aligned seeds" 2>&1 | grep -v Warn
python sprint1112_combo.py --x4f $X --arm "$O/adabn/al_p32+$O/adabn/pl_p32" --label "B2 + plain AdaBN nets" 2>&1 | grep -v Warn
python sprint1112_combo.py --x4f $X --arm "$O/adabn/al_p32+$O/adabn/al2_p32:shallow:38_39_40_41_42+$O/adabn/pl_p32" --label "B2 + both" 2>&1 | grep -v Warn
