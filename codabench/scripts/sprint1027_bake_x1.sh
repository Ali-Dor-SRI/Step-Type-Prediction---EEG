#!/usr/bin/env bash
# Sprint 2026-10-27 Phase 3: bake X1 (F64 + icoh + fscale + logreg0.01, threads pinned).
export CANDNAME=proteus_x1
export BAKE="personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4_icoh fcenter=True fscale=True clf=logreg0.01 threads=1"
bash ~/codabench/scripts/sprint1027_bake.sh
