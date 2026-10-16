#!/usr/bin/env bash
# proteus_w1016_p5b.sh: Phase 5 split across lanes (sprint 2026-10-16): the nest
# arms S0 and S16 on lane B (rows to N2.jsonl, one writer), while lane A's
# p5 script runs B1 and S64. Lane A's script still lists S0 / S16: they are
# pre-marked done there so it skips them. Resumable.
LOGDIR=$HOME/codabench/logs/proteus_w1016/p5b
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-8}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane N2 --protocols nest --outer 0-9"
for N in 0 16; do
  step "nest_S$N" 150 420m $SW --name "S$N" --set personal=pooled align=stream adapt=online buffer=$N xblocks=bpt4
done
finish_if_done nest_S0 nest_S16
