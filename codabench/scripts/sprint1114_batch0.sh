#!/usr/bin/env bash
# Sprint A batch 0: flag-only arms on the existing probe (B3 config), 1 thread each.
S=~/codabench/scripts/sprint1114_job.sh
B="python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --zdtype f16 --C 0.03"
bash $S a1c1 1 $B --step 20 --wpl 0.3 --name a1c1_wpl0.3 &
bash $S a1c2 1 $B --step 20 --wpl 0.7 --name a1c2_wpl0.7 &
bash $S a1c3 1 $B --step 20 --wpl 1.0 --name a1c3_wpl1.0 &
bash $S a2a 1 $B --step 10 --name a2a_t10 &
bash $S a2b 1 $B --step 5 --name a2b_t5 &
wait
echo "$(date +%T) ALL DONE batch0" >> ~/codabench/logs/sprint_1114/STATUS_jobs.md
