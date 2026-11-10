#!/usr/bin/env bash
# Sprint 11-12 TT1: online Tent / IM on the aligned Shallow nets (AdaBN p32 statistics).
# smoke: lr 0 must equal AdaBN p32; then 3 configs in parallel (1 thread each), then nested compare.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs; R=$L/sprint_1109/rows_align; O=$L/sprint_1112/tent
python sprint1112_tent.py --rows $R --out $O/smoke_lr0 --lr 0 --splits lopo0 --seeds 33 2>&1 | tail -1
python -c "
import numpy as np
a=np.load('$O/smoke_lr0/net_shallow_lopo0_s33_o0.npz')['probas']; b=np.load('$L/sprint_1112/adabn/al_p32/net_shallow_lopo0_s33_o0.npz')['probas']
print('tent lr0 vs adabn p32 max|d|', abs(a-b).max())"
[ "$1" = smoke ] && exit 0
python sprint1112_tent.py --rows $R --out $O/im_lr1e-3 --lr 1e-3 --beta 1 2>&1 | tail -1 &
python sprint1112_tent.py --rows $R --out $O/tent_lr1e-3 --lr 1e-3 --beta 0 2>&1 | tail -1 &
python sprint1112_tent.py --rows $R --out $O/im_lr1e-2 --lr 1e-2 --beta 1 2>&1 | tail -1 &
wait
for d in im_lr1e-3 tent_lr1e-3 im_lr1e-2; do
  echo "== $d"
  python sprint1109_compare.py $L/sprint_1112/adabn/al_p32 $O/$d 33_34_35_36_37 0.3_0.4_0.5_0.6_0.7_0.8 2>&1 | grep -v Warn | tail -3
done
