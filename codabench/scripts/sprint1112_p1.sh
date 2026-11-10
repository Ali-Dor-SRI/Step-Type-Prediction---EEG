#!/usr/bin/env bash
# Sprint 11-12 P1: 5 stream-aligned ShallowNets on `all` (seeds 33-37, 60 ep, 4 thr),
# weights + cg_*.npy for the safety-net zip. Resumable (--skip-done skips saved weights).
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE PYTHONUTF8=1
D=~/codabench/logs/sprint_1112; mkdir -p $D/rows_align_all
for s in 33 34 35 36 37; do
  echo "$(date +%T) START p1_align_all s$s" >> $D/STATUS_jobs.md
  python analysis/sprint1106_netsmoke.py --model shallow --epochs 60 --seed $s --splits all \
    --align stream --save-weights --outdir $D/rows_align_all --skip-done >> $D/p1_align_all.log 2>&1
  echo "$(date +%T) END p1_align_all s$s rc=$?" >> $D/STATUS_jobs.md
done
echo "$(date +%T) ALL DONE p1_align_all" >> $D/STATUS_jobs.md
