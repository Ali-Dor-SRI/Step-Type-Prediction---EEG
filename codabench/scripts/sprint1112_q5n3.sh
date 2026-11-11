#!/usr/bin/env bash
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_NUM_THREADS=2
python sprint1112_q5_nested.py $(cat ~/codabench/logs/sprint_1112/q5_nested3_list.txt) 2>&1 | grep -v Warn
