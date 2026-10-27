#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 0: harness + new opt-ins smoke test (brief
# prompts/2026-10-27_sealed_accuracy.md). Resumable by <step>.done.
#   lane A: F64 re-run on lopo fold 0 + later (must equal the 10-16 rows: defaults
#           and the one-thread scoring change nothing), then each new arm on later
#   lane B: old-world zhou regression (defaults bit-identical)
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1027/p0"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=10
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane D0"
F64="personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4 fcenter=True"

laneA() {
  step f64r 5 30m $SW --name F64r --set $F64 --protocols lopo later --folds 0
  step s_fscale 3 20m $SW --name s_fscale --set $F64 fscale=True --protocols later
  step s_dbpt8 3 20m $SW --name s_dbpt8 --set personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4_dbpt8 fcenter=True --protocols later
  step s_logreg 3 20m $SW --name s_logreg --set $F64 clf=logreg0.01 --protocols later
}
laneB() {
  step zhou 4 30m bash -c 'source ~/codabench/env.sh >/dev/null; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8; RUN_NAME=regress_sprint1027 bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data zhou2016_xsess; grep -a "gate:" ~/codabench/logs/regress_sprint1027/STATUS.md'
}
laneA &
laneB &
wait
finish_if_done f64r s_fscale s_dbpt8 s_logreg zhou
