#!/usr/bin/env bash
# Sprint 11-12: wait until every given job name has an END row in STATUS_jobs.md.
# Usage: sprint1112_wait.sh <name> [<name> ...]   (matches "END <name>" followed by space or EOL)
F=~/codabench/logs/sprint_1112/STATUS_jobs.md
for n in "$@"; do
  until grep -qE "END $n( |$)" "$F"; do sleep 30; done
done
grep -E "END ($(IFS='|'; echo "$*"))( |$)" "$F"
