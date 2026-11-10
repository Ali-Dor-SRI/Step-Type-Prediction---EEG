#!/usr/bin/env bash
# Sprint 11-12 Q5 neighbours of the adopted config (q 0.7 / w 5 / every 20, f16).
S=~/codabench/scripts/sprint1112_job.sh
bash $S q5_s20 2 python sprint1112_q5.py --q 0.7 --lam 5 --start 20 --step 20 --zdtype f16 &
bash $S q5_q85 2 python sprint1112_q5.py --q 0.85 --lam 5 --step 20 --zdtype f16 &
wait
