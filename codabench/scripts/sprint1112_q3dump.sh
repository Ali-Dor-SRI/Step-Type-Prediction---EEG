#!/usr/bin/env bash
# Sprint 11-12 Q3 (split-window, in addition): X4f with crop 0-2 s (task half) / 2-4 s (feedback half) of the
# window, streamed features dumped per split (sprint1106_dump.py), 4 threads each lane.
S=~/codabench/scripts/sprint1112_job.sh
bash $S q3_dump_c02 4 python sprint1106_dump.py --name X4fc02 --solver riemann_sealed_proteus_x4f --set crop=0-2 --protocols lopo later &
bash $S q3_dump_c24 4 python sprint1106_dump.py --name X4fc24 --solver riemann_sealed_proteus_x4f --set crop=2-4 --protocols lopo later &
wait
