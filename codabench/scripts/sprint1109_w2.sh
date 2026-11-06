#!/usr/bin/env bash
# Sprint 2026-11-09 W2 (brief prompts/2026-11-06_weekend_ensemble.md § 4): two ways to
# strengthen ShallowNet, judged only by the ensemble with X4f (nested w).
#   lane E: 120 epochs (cosine over 120), seeds 33-37 x 11 splits
#   lane A: Euclidean stream alignment (--align stream), 60 epochs; smoke first
# Rows/npz/weights: logs/sprint_1109/rows_{e120,align}/. Resumable (.done + --skip-done).
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1109/w2"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=${THREADS_PER_JOB:-3}
export OMP_WAIT_POLICY=PASSIVE
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
NS="python $HOME/codabench/analysis/sprint1106_netsmoke.py --model shallow --save-weights --skip-done"
SPL="lopo0 lopo1 lopo2 lopo3 lopo4 lopo5 lopo6 lopo7 lopo8 lopo9 later"
R=$HOME/codabench/logs/sprint_1109
laneE() {
  for s in 33 34 35 36 37; do
    step e120_s$s 110 300m $NS --epochs 120 --seed $s --splits $SPL --outdir $R/rows_e120
  done
}
laneA() {
  # baseline 60-epoch weights for gate (i) (seeds 34-37 on lopo0 + later; 33 is in w0/)
  for s in 34 35 36 37; do
    step base_w_s$s 10 60m $NS --epochs 60 --seed $s --splits lopo0 later --outdir $R/w0
  done
  step align_smoke 3 20m $NS --epochs 2 --seed 99 --splits lopo0 --align stream --outdir $R/rows_align_smoke
  for s in 33 34 35 36 37; do
    step align_s$s 60 200m $NS --epochs 60 --seed $s --splits $SPL --align stream --outdir $R/rows_align
  done
}
LANES=${LANES:-EA}
[[ $LANES == *E* ]] && { laneE & }
[[ $LANES == *A* ]] && { laneA & }
wait
finish_if_done base_w_s34 base_w_s35 base_w_s36 base_w_s37 e120_s33 e120_s34 e120_s35 e120_s36 e120_s37 align_smoke align_s33 align_s34 align_s35 align_s36 align_s37
