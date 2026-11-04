#!/usr/bin/env bash
# Sprint 2026-11-04 Phase 0 (brief prompts/2026-11-04_train_as_streamed.md § 5):
#   lane A: X4f re-run on lopo fold 0 + later under the edited solver (must equal the
#           X4f rows of logs/proteus_w1016/K2.jsonl to 6 dp), then each trainstream arm on later
#   lane B: old-world zhou regression (defaults bit-identical), then the stateful
#           streaming gates on X4f + trainstream=cf
# Resumable by <step>.done. Rows: logs/proteus_w1016/T0.jsonl.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1104/p0"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane T0"
FB6="fbands=1to4_4to8_8to13_13to20_20to30_30to45"
X4F="personal=pooled align=stream adapt=online buffer=64 fcenter=True $FB6 xblocks=bpt4_icoh fscale=True clf=logreg0.01"

laneA() {
  step x4fr 5 30m $SW --name X4fr --set $X4F --protocols lopo later --folds 0
  step s_f 3 20m $SW --name s_tsf --set $X4F trainstream=f --protocols later
  step s_c 3 20m $SW --name s_tsc --set $X4F trainstream=c --protocols later
  step s_cf 3 20m $SW --name s_tscf --set $X4F trainstream=cf --protocols later
}
laneB() {
  step zhou 4 30m bash -c 'source ~/codabench/env.sh >/dev/null; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; RUN_NAME=regress_sprint1104 bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data zhou2016_xsess; grep -a "gate:" ~/codabench/logs/regress_sprint1104/STATUS.md'
  step gates 5 30m python $HOME/codabench/analysis/proteus_w1016_gates.py --set $X4F trainstream=cf
}
laneA &
laneB &
wait
finish_if_done x4fr s_f s_c s_cf zhou gates
