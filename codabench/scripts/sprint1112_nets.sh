#!/usr/bin/env bash
# Sprint 11-12 net lane: <model> <align none|stream> <outdir-name> <seeds...>; 11 splits + `all`,
# 60 ep, 4 thr, weights saved, resumable (--skip-done). STATUS rows in STATUS_jobs.md.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE PYTHONUTF8=1
m=$1; al=$2; od=$3; shift 3
D=~/codabench/logs/sprint_1112; mkdir -p "$D/$od"
SPLITS="lopo0 lopo1 lopo2 lopo3 lopo4 lopo5 lopo6 lopo7 lopo8 lopo9 later all"
for s in "$@"; do
  echo "$(date +%T) START nets $m $al s$s ($od)" >> $D/STATUS_jobs.md
  python analysis/sprint1106_netsmoke.py --model $m --epochs 60 --seed $s --splits $SPLITS \
    --align $al --save-weights --outdir "$D/$od" --skip-done >> "$D/nets_${od}.log" 2>&1
  echo "$(date +%T) END nets $m $al s$s rc=$?" >> $D/STATUS_jobs.md
done
echo "$(date +%T) ALL DONE nets $od" >> $D/STATUS_jobs.md
