#!/usr/bin/env bash
# proteus_w1016_p3.sh: sprint 2026-10-16 Phase 3, the second model on lane B
# (EEGNet-Stream, solvers/bci_decoding/eegnet_stream.py) on lopo (10 folds) +
# later through analysis/proteus_sweep.py (one writer: logs/proteus_w1016/B.jsonl).
# Resumable by step (.done) and by row. Brief § 5 Phase 3.
#   E1 = align stream (training EA per person-session, running EA per stream)
#   E0 = align none (the control: does the alignment help the network?)
#   seeds 33-35 for both; then E1 seeds 36-37 (a 5-seed average for deployment)
#   wsl: bash ~/codabench/scripts/proteus_w1016_p3.sh
LOGDIR=$HOME/codabench/logs/proteus_w1016/p3
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-10}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SW="python $HOME/codabench/analysis/proteus_sweep.py --lane B --solver eegnet_stream"
STEPS=""
for seed in 33 34 35; do
  for arm in E1 E0; do
    case $arm in E1) A=stream ;; E0) A=none ;; esac
    step "${arm}_s$seed" 100 300m $SW --name "${arm}_s$seed" --seed $seed --set align=$A seed=$seed
    STEPS="$STEPS ${arm}_s$seed"
  done
done
for seed in 36 37; do
  step "E1_s$seed" 100 300m $SW --name "E1_s$seed" --seed $seed --set align=stream seed=$seed
  STEPS="$STEPS E1_s$seed"
done
finish_if_done $STEPS
