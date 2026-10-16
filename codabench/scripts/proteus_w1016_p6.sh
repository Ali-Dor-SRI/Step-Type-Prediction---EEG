#!/usr/bin/env bash
# proteus_w1016_p6.sh: sprint 2026-10-16 Phase 6 (pre-registered 10-16 06:45):
# F64 = S64 + fcenter (per-stream feature re-centring) on lopo + later, lane B
# (B.jsonl). WAIT_FOR=<STATUS.md>: start only after that file shows ALL DONE or a
# "not done" row (lane B's nested arms). Resumable.
#   wsl: WAIT_FOR=~/codabench/logs/proteus_w1016/p5b/STATUS.md bash ~/codabench/scripts/proteus_w1016_p6.sh
if [ -n "${WAIT_FOR:-}" ]; then
  until grep -qE "ALL DONE|not done" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
fi
LOGDIR=$HOME/codabench/logs/proteus_w1016/p6
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
step arm_F64 40 150m python "$HOME/codabench/analysis/proteus_sweep.py" --lane B --name F64 \
  --set personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4 fcenter=True
finish_if_done arm_F64
