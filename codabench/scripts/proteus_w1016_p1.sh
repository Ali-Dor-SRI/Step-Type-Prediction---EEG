#!/usr/bin/env bash
# proteus_w1016_p1.sh: sprint 2026-10-16 Phase 1, the decision-7 base arms on
# lopo (10 folds) + later through analysis/proteus_sweep.py (lane A, one
# writer: logs/proteus_w1016/A.jsonl). Resumable by step (.done) and by row.
# Brief: prompts/2026-10-16_proteus_weekend_model.md § 5 Phase 1.
#   wsl: bash ~/codabench/scripts/proteus_w1016_p1.sh            (all arms)
#        ARMS="B0 B1" bash ~/codabench/scripts/proteus_w1016_p1.sh
LOGDIR=$HOME/codabench/logs/proteus_w1016/p1
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane A"
POOL="personal=pooled align=none adapt=none"

declare -A ARM
ARM[B0]="$POOL"
ARM[B1]="$POOL xblocks=bpt4"
ARM[R]="personal=blend align=subject adapt=none xblocks=bpt4 blend_w=0.5"
# stream re-centring arms (need align="stream", Phase 0 step 1)
for N in 16 32 64 128 0; do
  ARM[S$N]="personal=pooled align=stream adapt=online buffer=$N xblocks=bpt4"
done
ARMS=${ARMS:-"B0 B1 R"}
for a in $ARMS; do
  step "arm_$a" 30 120m $SW --name "$a" --set ${ARM[$a]}
done
finish_if_done $(for a in $ARMS; do echo "arm_$a"; done)
