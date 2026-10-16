#!/usr/bin/env bash
# proteus_w1016_p4fi.sh: a ready-made VARIANT zip (not the candidate): FI64 = F64 +
# icoh (Phase 7b: +1.65 lopo, below rule 3's bar), through the same chain, gates
# and regression as p4v2 (CANDNAME=proteus_fi64), for a later sealed upload.
export CANDNAME=proteus_fi64
export BAKE="personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4_icoh fcenter=True"
bash "$HOME/codabench/scripts/proteus_w1016_p4v2.sh"
