#!/usr/bin/env bash
# w1016_watch.sh <logdir>...: one-shot watchdog for sprint 2026-10-16. Exits (so
# the background Bash call notifies once) on the first of:
#   - a new END / ALL DONE row in any STATUS.md (rows present at start are ignored),
#   - a crash signature in a step log written after start,
#   - a STALLED / DEAD / FAILED verdict from watch_run.sh (SLOW is ignored: the
#     ETAs are single-lane, two lanes run at ~2x),
#   - MAX_S seconds (default 4 h): a periodic check-in.
# Run it again after handling the event.
INTERVAL=${INTERVAL:-120}; MAX_S=${MAX_S:-14400}
WATCH="/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/watch_run.sh"
declare -A seen
for d in "$@"; do seen[$d]=$(wc -l < "$d/STATUS.md" 2>/dev/null || echo 0); done
t0=$(date +%s); stamp=$(mktemp); touch "$stamp"
while true; do
  for d in "$@"; do
    n=$(wc -l < "$d/STATUS.md" 2>/dev/null || echo 0)
    if [ "$n" -gt "${seen[$d]}" ]; then
      new=$(tail -n +$((seen[$d] + 1)) "$d/STATUS.md" | grep -E "END|ALL DONE")
      seen[$d]=$n
      [ -n "$new" ] && { echo "EVENT $(basename "$d"): $new"; exit 0; }
    fi
    f=$(find "$d" -maxdepth 1 -name "*.log" -newer "$stamp" -exec grep -lE "Traceback|Killed|OOM|MemoryError" {} + 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "CRASH $(basename "$f"): $(grep -E "Traceback|Error|Killed|OOM" "$f" | tail -1 | cut -c1-200)"; exit 0; }
    bad=$(bash "$WATCH" "$d" 2>/dev/null | grep -E "verdict=(STALLED|DEAD|FAILED)")
    [ -n "$bad" ] && { echo "PROBLEM $(basename "$d"): $bad"; exit 0; }
  done
  [ $(( $(date +%s) - t0 )) -gt "$MAX_S" ] && { echo "CHECK-IN after $MAX_S s: no events"; exit 0; }
  sleep "$INTERVAL"
done
