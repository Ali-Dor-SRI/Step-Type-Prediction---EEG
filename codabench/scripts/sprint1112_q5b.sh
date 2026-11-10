#!/usr/bin/env bash
# Sprint 11-12 Q5 deeper: three configs in parallel (2 thr each), f16-stored features.
S=~/codabench/scripts/sprint1112_job.sh
bash $S q5_q7l5t40 2 python sprint1112_q5.py --q 0.7 --lam 5 --step 40 --zdtype f16 &
bash $S q5_q10l10t20 2 python sprint1112_q5.py --q 1.0 --lam 10 --step 20 --zdtype f16 &
bash $S q5_q7l10t40 2 python sprint1112_q5.py --q 0.7 --lam 10 --step 40 --zdtype f16 &
wait
