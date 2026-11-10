#!/usr/bin/env bash
# Sprint 11-12 generic lane runner: sprint1112_job.sh <name> <threads> <cmd...>
# Runs <cmd> in ~/codabench/analysis under env040 with capped threads; stdout -> <name>.out,
# stderr -> <name>.err in logs/sprint_1112; START / END rows in STATUS_jobs.md; <name>.done on rc 0.
source ~/codabench/env040.sh >/dev/null 2>&1
name=$1; th=$2; shift 2
export OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th OMP_WAIT_POLICY=PASSIVE PYTHONUTF8=1
D=~/codabench/logs/sprint_1112; mkdir -p "$D"
cd ~/codabench/analysis
echo "$(date +%T) START $name" >> "$D/STATUS_jobs.md"
"$@" > "$D/$name.out" 2> "$D/$name.err"
rc=$?
echo "$(date +%T) END $name rc=$rc" >> "$D/STATUS_jobs.md"
[ $rc -eq 0 ] && touch "$D/$name.done"
exit $rc
