#!/usr/bin/env bash
# Resume sprint 2026-10-27 after the user's CPU pause (stopped 13:12 at checkpoints).
# Everything below is resumable: finished folds (per-lane jsonl keys) and steps
# (.done) are skipped. Two lanes, 4 threads each, passive OpenMP waits.
#   lane 1: X1 bake (official + all chains, gates, zhou), then X3, then X4
#   lane 2: fb6 (1 fold left), then X2, then C2
# ~2 h in total. Launch: wsl.exe bash -lc 'bash ~/codabench/scripts/sprint1027_resume.sh'
exec 2>&1
S=~/codabench/scripts
L=~/codabench/logs
lane1() {
  bash $S/sprint1027_bake_x1.sh > $L/proteus_w1016/proteus_x1_p4.out 2>&1
  LANES=I bash $S/sprint1027_p2.sh > $L/sprint1027_p2I.out 2>&1
  LANES=K bash $S/sprint1027_p2c.sh > $L/sprint1027_p2cK.out 2>&1
}
lane2() {
  bash $S/sprint1027_p1b.sh > $L/sprint1027_p1b.out 2>&1
  LANES=H bash $S/sprint1027_p2.sh > $L/sprint1027_p2H.out 2>&1
  LANES=J bash $S/sprint1027_p2c.sh > $L/sprint1027_p2cJ.out 2>&1
}
echo "resume start $(date +%T)"
lane1 &
lane2 &
wait
echo "resume end $(date +%T)"
