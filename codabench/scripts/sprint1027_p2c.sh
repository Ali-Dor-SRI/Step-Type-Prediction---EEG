#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 2 (pre-registered rule; arms fixed 12:28, brief § 4):
# C2f = F64 + fb6 + icoh (lane J), X4f = C2f + fscale + logreg0.01 (lane K); switched from
# fbd to fb6 at 16:21 when fb6 was adopted, before any C2 / X4 number existed.
# Rows: logs/proteus_w1016/{J2,K2}.jsonl.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1027/p2c"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE   # 12:45: two 10-thread lanes spin-waited 10x slower (logs/sprint_1027 LOG)
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
B="personal=pooled align=stream adapt=online buffer=64 fcenter=True xblocks=bpt4_icoh fbands=1to4_4to8_8to13_13to20_20to30_30to45"
[[ ${LANES:-JK} == *J* ]] && step C2f 25 75m python $HOME/codabench/analysis/proteus_sweep.py --lane J2 --name C2f --set $B
[[ ${LANES:-JK} == *K* ]] && step X4f 25 75m python $HOME/codabench/analysis/proteus_sweep.py --lane K2 --name X4f --set $B fscale=True clf=logreg0.01
true
