#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 1: single arms vs F64 on lopo 0-9 + later (brief
# prompts/2026-10-27_sealed_accuracy.md). Resumable by <step>.done and by the
# sweep's per-lane keys. Lane E / F rows: logs/proteus_w1016/{E1,F1}.jsonl.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1027/p1"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=10
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
B="personal=pooled align=stream adapt=online buffer=64 fcenter=True"

laneE() {
  local SW="python $HOME/codabench/analysis/proteus_sweep.py --lane E1"
  step fscale 18 60m $SW --name fscale --set $B xblocks=bpt4 fscale=True
  step dbpt8 18 60m $SW --name dbpt8 --set $B xblocks=bpt4_dbpt8
}
laneF() {
  local SW="python $HOME/codabench/analysis/proteus_sweep.py --lane F1"
  step logreg 18 60m $SW --name logreg --set $B xblocks=bpt4 clf=logreg0.01
  step bpt8 18 60m $SW --name bpt8 --set $B xblocks=bpt8
  step slow 18 60m $SW --name slow --set $B xblocks=bpt4 slow_block=True
}
LANES=${LANES:-EF}
[[ $LANES == *E* ]] && { laneE & }
[[ $LANES == *F* ]] && { laneF & }
wait
finish_if_done fscale dbpt8 logreg bpt8 slow
