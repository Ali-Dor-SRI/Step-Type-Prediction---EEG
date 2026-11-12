#!/usr/bin/env bash
# Sprint A batch 1: A1a/A1b/A1d arms on sprint1114_a.py (B3 config), 1 thread each; staggered starts (concurrent appends on /mnt/c lose rows).
S=~/codabench/scripts/sprint1114_job.sh
B="python sprint1114_a.py --lam 5 --start 20 --step 20 --zdtype f16 --C 0.03"
bash $S A1a1 1 $B --q 0.85 --gamma 1 --name A1a1 & sleep 2
bash $S A1a2 1 $B --q 0.85 --gamma 2 --name A1a2 & sleep 2
bash $S A1a3 1 $B --q 1.0 --gamma 2 --name A1a3 & sleep 2
bash $S A1b1 1 $B --q 0.85 --agree --name A1b1 & sleep 2
bash $S A1b2 1 $B --q 1.0 --agree --name A1b2 & sleep 2
bash $S A1d 1 $B --q 0.85 --varnorm --name A1d &
wait
echo "$(date +%T) ALL DONE batch1" >> ~/codabench/logs/sprint_1114/STATUS_jobs.md
