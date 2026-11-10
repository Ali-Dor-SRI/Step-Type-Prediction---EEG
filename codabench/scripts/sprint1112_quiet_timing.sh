#!/usr/bin/env bash
# Sprint 11-12: quiet timing. Pause (SIGSTOP) the training lanes, time submission folders at 1 thread
# (sprint1112_timing.py: x4f / nets / full predict ms per window, incl. self-training refits),
# then resume (SIGCONT) - always, via trap.
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
export OMP_WAIT_POLICY=PASSIVE
PIDS=$(pgrep -f "sprint1106_netsmoke.py" | tr '\n' ' ')
trap 'kill -CONT $PIDS 2>/dev/null; echo "resumed: $PIDS"' EXIT
echo "pausing: $PIDS"; kill -STOP $PIDS
sleep 5; uptime
for f in "$@"; do
  echo "== $f"
  python sprint1112_timing.py ~/codabench/logs/sprint_1112/$f/unzipped 170 2>&1 | grep -E "Model:|Error"
done
