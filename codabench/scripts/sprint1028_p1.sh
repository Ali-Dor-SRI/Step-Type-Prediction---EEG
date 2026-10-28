#!/usr/bin/env bash
# Sprint 2026-10-28 Phase 1 (brief prompts/2026-10-28_feature_engineering.md § 5):
# each task-driven block alone, on C2f (primary, decides adoption) and on X4f
# (secondary, reported). lopo 0-9 + later. 3 lanes x 4 threads, PASSIVE waits.
# Lane R1 has one arm fewer and starts Phase 3's C2f inner folds afterwards
# (needed whatever Phase 1 shows). Rows: logs/proteus_w1016/{P1,Q1,R1,NF1}.jsonl.
# Resumable by <step>.done; LANES=PQR selects lanes.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1028/p1"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SWP="python $HOME/codabench/analysis/proteus_sweep.py"
FB6="fbands=1to4_4to8_8to13_13to20_20to30_30to45"
C2F="personal=pooled align=stream adapt=online buffer=64 fcenter=True $FB6"
X4F="$C2F fscale=True clf=logreg0.01"

laneP() {
  step c_roi 40 100m $SWP --lane P1 --name C2f+roi --set $C2F xblocks=bpt4_icoh_roi
  step c_asymabs 40 100m $SWP --lane P1 --name C2f+asymabs --set $C2F xblocks=bpt4_icoh_asym_asymabs
  step x_burst 40 100m $SWP --lane P1 --name X4f+burst --set $X4F xblocks=bpt4_icoh_burst
}
laneQ() {
  step c_asym 40 100m $SWP --lane Q1 --name C2f+asym --set $C2F xblocks=bpt4_icoh_asym
  step x_roi 40 100m $SWP --lane Q1 --name X4f+roi --set $X4F xblocks=bpt4_icoh_roi
  step x_asymabs 40 100m $SWP --lane Q1 --name X4f+asymabs --set $X4F xblocks=bpt4_icoh_asym_asymabs
}
laneR() {
  step c_burst 40 100m $SWP --lane R1 --name C2f+burst --set $C2F xblocks=bpt4_icoh_burst
  step x_asym 40 100m $SWP --lane R1 --name X4f+asym --set $X4F xblocks=bpt4_icoh_asym
  step nest_c2f 110 300m $SWP --lane NF1 --protocols nest --outer 0-9 --name C2f --set $C2F xblocks=bpt4_icoh
}
LANES=${LANES:-PQR}
[[ $LANES == *P* ]] && { laneP & }
[[ $LANES == *Q* ]] && { laneQ & }
[[ $LANES == *R* ]] && { laneR & }
wait
finish_if_done c_roi c_asymabs x_burst c_asym x_roi x_asymabs c_burst x_asym
