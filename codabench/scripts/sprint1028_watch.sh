#!/usr/bin/env bash
# Sprint 2026-10-28: one Monitor stream over the phase log dirs (run inside WSL).
M="/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/monitor_loop.sh"
for d in "$@"; do
  bash "$M" ~/codabench/logs/sprint_1028/$d 300 | sed -u "s/^/[$d] /" &
done
wait
