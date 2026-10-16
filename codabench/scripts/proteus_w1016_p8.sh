#!/usr/bin/env bash
# proteus_w1016_p8.sh <2|32>: Phase 8 (pre-registered 10-16 18:15): F64 with stream_k K,
# lopo + later, information only (rows to lane K<K>). Resumable.
K=${1:?2|32}
LOGDIR=$HOME/codabench/logs/proteus_w1016/p8_$K
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
step arm_K$K 40 150m python "$HOME/codabench/analysis/proteus_sweep.py" --lane K$K --name K$K \
  --set personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4 fcenter=True stream_k=$K
finish_if_done arm_K$K
