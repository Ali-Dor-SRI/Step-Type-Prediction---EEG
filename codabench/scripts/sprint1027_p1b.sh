#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 1b (pre-registered 11:40, brief § 4): delta in the
# filter-bank tangent space. Lane G rows: logs/proteus_w1016/G1.jsonl.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1027/p1b"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=10
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
B="personal=pooled align=stream adapt=online buffer=64 fcenter=True xblocks=bpt4"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane G1"
step fbd 18 60m $SW --name fbd --set $B fbands=1to4_4to8_8to13_13to30_30to45
step fb6 20 60m $SW --name fb6 --set $B fbands=1to4_4to8_8to13_13to20_20to30_30to45
finish_if_done fbd fb6
