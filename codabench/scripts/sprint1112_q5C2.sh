#!/usr/bin/env bash
# Sprint 11-12 Q5 + raw-covariance person weighting (tau 1.0 / 0.3), deployment config, 1 thread each.
S=~/codabench/scripts/sprint1112_job.sh
bash $S q5_C01 1 python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --C 0.1 &
bash $S q5_C3 1 python sprint1112_q5.py --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --C 0.3 &
wait
