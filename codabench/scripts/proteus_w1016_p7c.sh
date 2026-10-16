#!/usr/bin/env bash
# proteus_w1016_p7c.sh <16|0>: Phase 7c (pre-registered 10-16 13:35): F<N> = fcenter on
# buffer N, on lopo + later (rows to C<N>... lane C16 / C0 jsonl) then the nested inner folds
# (N4 for 16, N5 for 0). One lane per N. Resumable.
N=${1:?16|0}
case $N in 16) NL=N4 ;; 0) NL=N5 ;; *) echo "N must be 16 or 0"; exit 2 ;; esac
LOGDIR=$HOME/codabench/logs/proteus_w1016/p7c_$N
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SET="personal=pooled align=stream adapt=online buffer=$N xblocks=bpt4 fcenter=True"
step arm_F$N 40 150m python "$HOME/codabench/analysis/proteus_sweep.py" --lane C$N --name F$N --set $SET
step nest_F$N 180 420m python "$HOME/codabench/analysis/proteus_sweep.py" --lane $NL --protocols nest \
  --outer 0-9 --name F$N --set $SET
finish_if_done arm_F$N nest_F$N
