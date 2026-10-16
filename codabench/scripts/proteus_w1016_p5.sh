#!/usr/bin/env bash
# proteus_w1016_p5.sh: sprint 2026-10-16 Phase 5, the nested-selection check.
# For each outer lopo fold k, the Phase-1 choice (B1 vs stream re-centring with
# N in {16, 64, all}) is re-made on the nine other people by an inner lopo
# (protocol "nest", folds "k-j"); analysis/proteus_w1016_nested.py then scores
# the inner choice on person k with the outer rows already in A.jsonl.
# One writer: logs/proteus_w1016/N.jsonl. Resumable by step and by row.
#   wsl: bash ~/codabench/scripts/proteus_w1016_p5.sh
LOGDIR=$HOME/codabench/logs/proteus_w1016/p5
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane N --protocols nest --outer 0-9"
declare -A ARM
ARM[B1]="personal=pooled align=none adapt=none xblocks=bpt4"
for N in 16 64 0; do
  ARM[S$N]="personal=pooled align=stream adapt=online buffer=$N xblocks=bpt4"
done
ARMS=${ARMS:-"B1 S64 S0 S16"}
for a in $ARMS; do
  step "nest_$a" 150 420m $SW --name "$a" --set ${ARM[$a]}
done
finish_if_done $(for a in $ARMS; do echo "nest_$a"; done)
