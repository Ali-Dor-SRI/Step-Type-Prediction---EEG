#!/usr/bin/env bash
# Sprint 2026-10-28 Phase 0 (brief prompts/2026-10-28_feature_engineering.md § 5):
#   lane A: C2f re-run on lopo fold 0 + later under the edited solver (must equal
#           logs/proteus_w1016/J2.jsonl to 6 dp), then each new block on later
#   lane B: old-world zhou regression (defaults bit-identical), then the stateful
#           streaming gates on C2f + every new block (asymabs uses the stream path)
# Resumable by <step>.done. Rows: logs/proteus_w1016/P0.jsonl.
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1028/p0"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane P0"
FB6="fbands=1to4_4to8_8to13_13to20_20to30_30to45"
BASE="personal=pooled align=stream adapt=online buffer=64 fcenter=True $FB6"

laneA() {
  step c2fr 6 30m $SW --name C2fr --set $BASE xblocks=bpt4_icoh --protocols lopo later --folds 0
  step s_roi 3 20m $SW --name s_roi --set $BASE xblocks=bpt4_icoh_roi --protocols later
  step s_asym 3 20m $SW --name s_asym --set $BASE xblocks=bpt4_icoh_asym --protocols later
  step s_asymabs 3 20m $SW --name s_asymabs --set $BASE xblocks=bpt4_icoh_asym_asymabs --protocols later
  step s_burst 3 20m $SW --name s_burst --set $BASE xblocks=bpt4_icoh_burst --protocols later
}
laneB() {
  step zhou 4 30m bash -c 'source ~/codabench/env.sh >/dev/null; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; RUN_NAME=regress_sprint1028 bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data zhou2016_xsess; grep -a "gate:" ~/codabench/logs/regress_sprint1028/STATUS.md'
  step gates 5 30m python $HOME/codabench/analysis/proteus_w1016_gates.py --set $BASE xblocks=bpt4_icoh_roi_asym_asymabs_burst
}
laneA &
laneB &
wait
finish_if_done c2fr s_roi s_asym s_asymabs s_burst zhou gates
