#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 2x (pre-registered 11:58, brief § 4): exploratory
# combinations of the positive-leaning parts (icoh, fscale, logreg0.01).
# LANES=H runs X1 then X2, LANES=I runs X3. Rows: logs/proteus_w1016/{H2,I2}.jsonl.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1027/p2"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE   # 12:45: two 10-thread lanes spin-waited 10x slower (logs/sprint_1027 LOG)
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
B="personal=pooled align=stream adapt=online buffer=64 fcenter=True xblocks=bpt4_icoh"
laneH() {
  local SW="python $HOME/codabench/analysis/proteus_sweep.py --lane H2"
  step X1 20 60m $SW --name X1 --set $B fscale=True clf=logreg0.01
  step X2 20 60m $SW --name X2 --set $B clf=logreg0.01
}
laneI() {
  local SW="python $HOME/codabench/analysis/proteus_sweep.py --lane I2"
  step X3 20 60m $SW --name X3 --set $B fscale=True
}
LANES=${LANES:-HI}
[[ $LANES == *H* ]] && { laneH & }
[[ $LANES == *I* ]] && { laneI & }
wait
