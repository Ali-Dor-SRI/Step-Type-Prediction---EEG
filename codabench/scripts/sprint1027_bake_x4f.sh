#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 3: bake X4f (C2f + fscale + logreg0.01, threads pinned) -
# the best arm (lopo +6.02 [+4.33, +7.78], 10/10; later +7.31), exploratory: needs nested.
export CANDNAME=proteus_x4f
export BAKE="personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4_icoh fcenter=True fbands=1to4_4to8_8to13_13to20_20to30_30to45 fscale=True clf=logreg0.01 threads=1"
bash ~/codabench/scripts/sprint1027_bake.sh
