source ~/codabench/env040.sh
cd ~/codabench/analysis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
B3=q5_q0.85_l5.0_s20_t20_f16_C0.03
COMMON="--q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --C 0.03 --splits lopo0"
python sprint1114_a.py $COMMON --name smoke_b3 2>&1 | tail -3
python - <<PY
import numpy as np
a=np.load("$HOME/codabench/logs/sprint_1112/q5/smoke_b3_lopo0.npz")["probas"]
b=np.load("$HOME/codabench/logs/sprint_1112/q5/${B3}_lopo0.npz")["probas"]
print("B3 reproduction max abs diff", np.abs(a-b).max())
PY
python sprint1114_a.py $COMMON --gamma 2 --name smoke_g 2>&1 | tail -2
python sprint1114_a.py $COMMON --agree --name smoke_ag 2>&1 | tail -2
python sprint1114_a.py $COMMON --varnorm --name smoke_vn 2>&1 | tail -2
python sprint1114_vsb3.py $B3
rm -f ~/codabench/logs/sprint_1112/q5/smoke_*.npz
