#!/usr/bin/env bash
# Sprint 11-12: the B3 deployment config (C 0.03, q 0.85, lam 5, start 20, step 20, wpl 0.5, w 0.6, AdaBN 32):
# solver gates (i) vs its probe on lopo0 + later, then bake on `all` + gates iii / vi / CLI.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE
L=~/codabench/logs/sprint_1112
python sprint1112_build_ens_al.py
echo "$(date +%T) START b3_stgates" >> $L/STATUS_jobs.md
python sprint1112_st_gates.py --probe q5_q0.85_l5.0_s20_t20_f16_C0.03 --C 0.03 --q 0.85 --lam 5 --start 20 --step 20 --w 0.6 \
  --out $L/st_gates_b3 2>&1 | grep -E "PASS|FAIL|loaded|Error|Trace" > $L/st_gates_b3.log
echo "$(date +%T) END b3_stgates rc=$(grep -c FAIL $L/st_gates_b3.log)" >> $L/STATUS_jobs.md
bash ~/codabench/scripts/sprint1112_p1b.sh 0.6 b3_cand_all ~/codabench/logs/sprint_1112/rows_align_all 32 bake 0.85,5,20,20,0.5,1500,0.03
