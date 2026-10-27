#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 3: bake C2f (F64 + fb6 filter bank + icoh, threads pinned) -
# the rule-selected candidate (lopo +4.11 [+3.11, +5.22], 10/10; later +4.19).
export CANDNAME=proteus_c2f
export BAKE="personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4_icoh fcenter=True fbands=1to4_4to8_8to13_13to20_20to30_30to45 threads=1"
bash ~/codabench/scripts/sprint1027_bake.sh
