#!/usr/bin/env bash
# Sprint 11-12 Q5: combinations of the best neighbours.
S=~/codabench/scripts/sprint1112_job.sh
bash $S q5_s20q85 2 python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 &
bash $S q5_s20q10 2 python sprint1112_q5.py --q 1.0 --lam 5 --start 20 --step 20 --zdtype f16 &
wait
