#!/usr/bin/env bash
# Sprint 11-12 Q5 + sliding past window (last 60 / 100 windows), deployment config, 1 thread each.
S=~/codabench/scripts/sprint1112_job.sh
bash $S q5_mp60 1 python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --maxpast 60 &
bash $S q5_mp100 1 python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --maxpast 100 &
wait
