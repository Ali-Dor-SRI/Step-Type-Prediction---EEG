#!/usr/bin/env bash
# Sprint 11-12 Q5-net: smoke (iters 0 must equal AdaBN p32), then two configs, compare vs B1 nets (X4f orig).
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs; O=$L/sprint_1112/q5net
python sprint1112_q5net.py --out $O/smoke_it0 --iters 0 --splits lopo0 --seeds 33 2>&1 | tail -1
python -c "
import numpy as np
a=np.load('$O/smoke_it0/net_shallow_lopo0_s33_o0.npz')['probas']; b=np.load('$L/sprint_1112/adabn/al_p32/net_shallow_lopo0_s33_o0.npz')['probas']
print('q5net iters0 vs adabn p32 max|d|', abs(a-b).max())"
[ "$1" = smoke ] && exit 0
python sprint1112_q5net.py --out $O/q7_lr1e-2_mu1 --q 0.7 --lr 1e-2 --mu 1 2>&1 | tail -1 &
python sprint1112_q5net.py --out $O/q7_lr1e-2_mu0.1 --q 0.7 --lr 1e-2 --mu 0.1 2>&1 | tail -1 &
wait
for d in q7_lr1e-2_mu1 q7_lr1e-2_mu0.1; do
  echo "== $d"
  python sprint1109_compare.py $L/sprint_1112/adabn/al_p32 $O/$d 33_34_35_36_37 0.3_0.4_0.5_0.6_0.7_0.8 2>&1 | grep -v Warn | tail -3
done
