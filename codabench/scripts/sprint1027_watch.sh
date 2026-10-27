#!/usr/bin/env bash
# Sprint 2026-10-27: one Monitor stream over every phase log dir.
M="/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/monitor_loop.sh"
for d in "$@"; do
  mkdir -p ~/codabench/logs/sprint_1027/$d
  bash "$M" ~/codabench/logs/sprint_1027/$d 120 | sed -u "s/^/[$d] /" &
done
wait
