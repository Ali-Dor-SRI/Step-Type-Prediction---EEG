#!/usr/bin/env bash
# Sprint 11-12 H2 / Q7: stream-aligned nets, seeds x 11 splits, 60 ep, 4 thr, weights saved.
# Usage: sprint1112_fam_al.sh <model> <outdir-name> <seeds...>   (resumable: --skip-done)
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE PYTHONUTF8=1
m=$1; od=$2; shift 2
D=~/codabench/logs/sprint_1112; mkdir -p "$D/$od"
SPLITS="lopo0 lopo1 lopo2 lopo3 lopo4 lopo5 lopo6 lopo7 lopo8 lopo9 later"
for s in "$@"; do
  echo "$(date +%T) START fam_al $m s$s ($od)" >> $D/STATUS_jobs.md
  python analysis/sprint1106_netsmoke.py --model $m --epochs 60 --seed $s --splits $SPLITS \
    --align stream --save-weights --outdir "$D/$od" --skip-done >> "$D/fam_al_${m}_${od}.log" 2>&1
  echo "$(date +%T) END fam_al $m s$s rc=$?" >> $D/STATUS_jobs.md
done
echo "$(date +%T) ALL DONE fam_al $m $od" >> $D/STATUS_jobs.md
