#!/usr/bin/env bash
# Sprint 11-12: AdaBN probe. smoke (frozen = saved, lopo0 s33) then adabn prior 8 / 32 on aligned and plain nets.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs
O=$L/sprint_1112/adabn
python sprint1112_adabn.py --rows $L/sprint_1109/rows_align --out $O/smoke_frozen --mode frozen --splits lopo0 --seeds 33 2>&1 | grep -v Warn
[ "$1" = smoke ] && exit 0
python sprint1112_adabn.py --rows $L/sprint_1109/rows_align --out $O/al_p8 --prior 8 2>&1 | grep -v Warn | tail -1 &
python sprint1112_adabn.py --rows $L/sprint_1109/rows_align --out $O/al_p32 --prior 32 2>&1 | grep -v Warn | tail -1 &
wait
for d in al_p8 al_p32; do python sprint1109_compare.py $L/sprint_1106/rows $O/$d 33_34_35_36_37 0.2_0.3_0.4_0.5_0.6 2>&1 | grep -v Warn | tail -3; done
