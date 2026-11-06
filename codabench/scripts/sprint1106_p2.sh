#!/usr/bin/env bash
# Sprint 2026-11-06 P2 (brief prompts/2026-11-06_next_session_plan.md § 2):
# augmented training windows (official loader at cue offsets 0.5/1.5/2.0/2.5 s,
# training rows only) on X4f and C2f. lopo 0-9 + later. 2 lanes x 4 threads.
# Rows: logs/sprint_1106/rows/AG{X,C}.jsonl. Resumable by <step>.done; LANES=XC.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1106/p2"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE
export SWEEP_OUT="$HOME/codabench/logs/sprint_1106/rows"
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SWP="python $HOME/codabench/analysis/proteus_sweep.py"
FB6="fbands=1to4_4to8_8to13_13to20_20to30_30to45"
C2F="personal=pooled align=stream adapt=online buffer=64 fcenter=True $FB6 xblocks=bpt4_icoh"
X4F="$C2F fscale=True clf=logreg0.01"
AUG4="0.5_1.5_2.0_2.5"

laneX() {
  step x_aug4 70 160m $SWP --lane AGX --name X4f+aug4 --set $X4F --train-offsets $AUG4
}
laneC() {
  step c_aug4 70 160m $SWP --lane AGC --name C2f+aug4 --set $C2F --train-offsets $AUG4
}
LANES=${LANES:-XC}
[[ $LANES == *X* ]] && { laneX & }
[[ $LANES == *C* ]] && { laneC & }
wait
finish_if_done x_aug4 c_aug4
