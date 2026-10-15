#!/usr/bin/env bash
# proteus_w1016_p2.sh: sprint 2026-10-16 Phase 2, one-at-a-time ablations against
# the Phase-1 base on lopo + later (lane A: logs/proteus_w1016/A.jsonl).
# BASE = the base's parameter string (decided by Phase 1, written in
# logs/proteus_w1016/p2/BASE.txt and the LOG). Each arm = BASE + one change
# (a later key=value overrides the base's). Resumable by step and by row.
# Brief § 5 Phase 2 (rules 2, 3, 3c, 8 of RELEASE_DAY § 5 + exploratory rows).
#   wsl: bash ~/codabench/scripts/proteus_w1016_p2.sh
LOGDIR=$HOME/codabench/logs/proteus_w1016/p2
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
BASE=${BASE:-$(cat "$LOGDIR/BASE.txt")}
[ -n "$BASE" ] || { status_row "ERROR: no BASE"; exit 1; }
status_row "BASE = $BASE"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane A"
XB=$(echo "$BASE" | grep -o 'xblocks=[^ ]*' | cut -d= -f2)
declare -A ARM
ARM[X_icoh]="xblocks=${XB:+${XB}_}icoh"      # rule 3: icoh
ARM[X_noxd]="use_xdawn=False"                # rule 3c: xDAWN
ARM[X_car]="reference=car"                   # rule 8: reference
ARM[X_clip]="clip=0.999"                     # rule 8: amplitude handling
ARM[X_euclid]="kind=euclid"                  # exploratory
ARM[X_crop]="crop=0-2"                       # exploratory: 1-3 s after the cue
ARMS=${ARMS:-"X_noxd X_car X_clip X_crop X_euclid X_icoh"}
for a in $ARMS; do
  step "arm_$a" 40 150m $SW --name "$a" --set $BASE ${ARM[$a]}
done
finish_if_done $(for a in $ARMS; do echo "arm_$a"; done)
