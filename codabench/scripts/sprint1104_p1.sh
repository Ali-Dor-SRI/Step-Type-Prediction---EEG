#!/usr/bin/env bash
# Sprint 2026-11-04 Phase 1 (brief prompts/2026-11-04_train_as_streamed.md § 5):
# trainstream arms + logreg C side arms on X4f (decide), C2f+tscf (reported).
# lopo 0-9 + later. 3 lanes x 4 threads, PASSIVE waits. Rows: logs/proteus_w1016/T1{P,Q,R}.jsonl.
# Resumable by <step>.done; LANES=PQR selects lanes.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1104/p1"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SWP="python $HOME/codabench/analysis/proteus_sweep.py"
FB6="fbands=1to4_4to8_8to13_13to20_20to30_30to45"
C2F="personal=pooled align=stream adapt=online buffer=64 fcenter=True $FB6 xblocks=bpt4_icoh"
X4FB="$C2F fscale=True"
X4F="$X4FB clf=logreg0.01"

laneP() {
  step x_tsf 40 100m $SWP --lane T1P --name X4f+tsf --set $X4F trainstream=f
  step x_c003 40 100m $SWP --lane T1P --name X4f+C0.003 --set $X4FB clf=logreg0.003
}
laneQ() {
  step x_tsc 45 110m $SWP --lane T1Q --name X4f+tsc --set $X4F trainstream=c
  step x_c03 40 100m $SWP --lane T1Q --name X4f+C0.03 --set $X4FB clf=logreg0.03
}
laneR() {
  step x_tscf 45 110m $SWP --lane T1R --name X4f+tscf --set $X4F trainstream=cf
  step c_tscf 45 110m $SWP --lane T1R --name C2f+tscf --set $C2F trainstream=cf
}
LANES=${LANES:-PQR}
[[ $LANES == *P* ]] && { laneP & }
[[ $LANES == *Q* ]] && { laneQ & }
[[ $LANES == *R* ]] && { laneR & }
wait
finish_if_done x_tsf x_c003 x_tsc x_c03 x_tscf c_tscf
