#!/usr/bin/env bash
# proteus_w1016_p4v3.sh: Phase 4 for the Phase-6 candidate F64 (S64 + fcenter):
# the same chain / gates / regression as p4v2 (CANDNAME=proteus_f64), then the
# clean timing of the all-trained export with lane B paused.
export CANDNAME=proteus_f64
export BAKE="personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4 fcenter=True"
bash "$HOME/codabench/scripts/proteus_w1016_p4v2.sh"
echo "== clean timing"
bash "$HOME/codabench/scripts/w1016_timing.sh" "$HOME/codabench/logs/proteus_w1016/${CANDNAME}_cand_all/submission"
