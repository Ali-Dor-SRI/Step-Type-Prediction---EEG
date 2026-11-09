#!/usr/bin/env bash
# Midweek 11-10 W2: FBCNet / Deep4 / ATCNet, 5 seeds x 11 splits, 60 ep, 4 thr each.
# Resumable (--skip-done skips finished npz). LANES=fbcnet,deep4,atcnet subset via args.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE PYTHONUTF8=1
D=~/codabench/logs/sprint_1110; mkdir -p $D/rows_fam
SPLITS="lopo0 lopo1 lopo2 lopo3 lopo4 lopo5 lopo6 lopo7 lopo8 lopo9 later"
lane() { m=$1
  for s in 33 34 35 36 37; do
    echo "$(date +%T) START $m s$s" >> $D/STATUS.md
    python analysis/sprint1106_netsmoke.py --model $m --epochs 60 --seed $s --splits $SPLITS \
      --outdir $D/rows_fam --skip-done >> $D/fam_$m.log 2>&1
    echo "$(date +%T) END $m s$s rc=$?" >> $D/STATUS.md
  done; }
for m in ${@:-fbcnet deep4 atcnet}; do lane $m & done
wait; echo "$(date +%T) ALL DONE" >> $D/STATUS.md
