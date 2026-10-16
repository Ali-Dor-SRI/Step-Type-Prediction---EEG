#!/usr/bin/env bash
# proteus_w1016_p7.sh <a|b>: sprint 2026-10-16 Phase 7 (pre-registered 10-16 09:10).
#   a: nest arm F64 (rows to N3.jsonl)    b: FI64 = F64 + icoh on lopo + later (rows to C.jsonl)
# WAIT_FOR=<STATUS.md> waits for ALL DONE / "not done" there first. Resumable.
PART=${1:?a|b}
if [ -n "${WAIT_FOR:-}" ]; then
  until grep -qE "ALL DONE|not done" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
fi
LOGDIR=$HOME/codabench/logs/proteus_w1016/p7$PART
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
F64="personal=pooled align=stream adapt=online buffer=64 fcenter=True"
if [ "$PART" = a ]; then
  step nest_F64 150 420m python "$HOME/codabench/analysis/proteus_sweep.py" --lane N3 --protocols nest \
    --outer 0-9 --name F64 --set $F64 xblocks=bpt4
  finish_if_done nest_F64
else
  step arm_FI64 40 150m python "$HOME/codabench/analysis/proteus_sweep.py" --lane C --name FI64 \
    --set $F64 xblocks=bpt4_icoh
  finish_if_done arm_FI64
fi
