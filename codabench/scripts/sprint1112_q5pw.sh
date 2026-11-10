#!/usr/bin/env bash
# Sprint 11-12 Q5 + soft person weighting of training rows (deployment config q 0.85 / w 5 / s20 / t20).
S=~/codabench/scripts/sprint1112_job.sh
bash $S q5_pw1 2 python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --pw 1.0 &
bash $S q5_pw03 2 python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --pw 0.3 &
wait
