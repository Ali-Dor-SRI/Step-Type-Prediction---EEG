#!/usr/bin/env bash
# w1016_timing.sh <submission dir>: clean timing of an exported submission on the
# official test streams (route A, B = 1), with lane B's running python paused
# (SIGSTOP / SIGCONT by PID from the newest .pid file under PAUSE_DIR), at 1 and
# 4 BLAS threads. Sprint 2026-10-16 Phase 4.
SUB=${1:?submission dir}
PAUSE_DIR=${PAUSE_DIR:-$HOME/codabench/logs/proteus_w1016/p5b}
pidf=$(ls -t "$PAUSE_DIR"/*.pid 2>/dev/null | head -1)
TO=""
if [ -n "$pidf" ]; then
  T=$(cat "$pidf")                     # the `timeout` wrapper
  TO=$(pgrep -P "$T" | head -1)        # its python child
fi
[ -n "$TO" ] && kill -STOP "$TO" && echo "paused pid $TO ($(date +%T))"
trap '[ -n "$TO" ] && kill -CONT "$TO" && echo "resumed pid $TO ($(date +%T))"' EXIT
source ~/codabench/env040.sh >/dev/null
export PYTHONUTF8=1
uptime
for th in 1 4; do
  export OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th
  echo "== threads $th"
  python ~/codabench/analysis/proteus_w1016_gates.py --submission "$SUB" 2>&1 | grep -E "PASS  4|FAIL  4|loaded"
done
