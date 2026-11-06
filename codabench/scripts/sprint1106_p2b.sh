#!/usr/bin/env bash
# Sprint 2026-11-06 P2b (deviation from the brief, logged in LOG / RESULTS): after
# fold 0 of aug4 (X4f -5.9, C2f -13.7), the mechanism test: near offsets
# {0.75, 1.25} (small shifts keep the cue-locked xDAWN / bpt4 features) with
# per-(session, offset) centring (:sep, matching the test-time stream centring).
# lopo 0-9 + later, one lane x 4 threads. Rows: logs/sprint_1106/rows/AGN.jsonl.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1106/p2"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE
export SWEEP_OUT="$HOME/codabench/logs/sprint_1106/rows"
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SWP="python $HOME/codabench/analysis/proteus_sweep.py"
FB6="fbands=1to4_4to8_8to13_13to20_20to30_30to45"
X4F="personal=pooled align=stream adapt=online buffer=64 fcenter=True $FB6 xblocks=bpt4_icoh fscale=True clf=logreg0.01"
step x_near_sep 85 180m $SWP --lane AGN --name X4f+near2s --set $X4F --train-offsets 0.75_1.25:sep
finish_if_done x_near_sep
