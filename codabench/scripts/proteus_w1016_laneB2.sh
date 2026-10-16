#!/usr/bin/env bash
# proteus_w1016_laneB2.sh: sprint 2026-10-16, lane B after the EEGNet seeds.
#  1. Phase 4: bake the candidate (S64) and run the full chain on official, then
#     all (the sealed zip: trained on all 14 sessions).
#  2. EEGNet E0f seeds 36-37 (the 3-seed winner, brief amendment).
#  3. Probability re-runs for the ensemble: E0f_s33p (its first run predates the
#     probability replay) and S64p (= S64, deterministic, re-run only to save
#     probabilities). Resumable (each part skips what is done).
#   wsl: bash ~/codabench/scripts/proteus_w1016_laneB2.sh
set -u
export CANDNAME=proteus_s64
export BAKE="personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4"
export THREADS_PER_JOB=8
for p in official all; do
  bash "$HOME/codabench/scripts/proteus_w1016_cand.sh" "$p"
  echo "== candidate $p rc=$?"
done
LOGDIR=$HOME/codabench/logs/proteus_w1016/p3
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=10
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SWB="python $HOME/codabench/analysis/proteus_sweep.py --lane B --solver eegnet_stream"
for seed in 36 37; do
  step "E0f_s$seed" 130 480m $SWB --name "E0f_s$seed" --seed $seed --set align=none seed=$seed
done
step "E0f_s33p" 130 480m $SWB --name "E0f_s33p" --seed 33 --set align=none seed=33
THREADS_PER_JOB=8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
step "S64p" 40 150m python "$HOME/codabench/analysis/proteus_sweep.py" --lane B --name S64p \
  --set personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4
finish_if_done E0f_s36 E0f_s37 E0f_s33p S64p
