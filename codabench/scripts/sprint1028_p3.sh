#!/usr/bin/env bash
# Sprint 2026-10-28 Phase 3 (brief § 5 + addendum 18:40): inner "nest" folds
# (10 outer x 9 inner = 90 fits per config) for the configs of the nested walk.
# F64's rows exist (N3); C2f's come from Phase 1 lane R1 (NF1). Here: fb6 (NF2),
# X4f (NF3), and the Phase 2 configs when given:
#   CF_NAME / CF_XB  = the C2f+F config's name and xblocks   (lane NF4)
#   XF_NAME / XF_XB  = the X4f+F config's name and xblocks   (lane NF5)
# 4 threads per lane, PASSIVE waits; LANES picks lanes (default: all defined).
# Resumable: steps by <step>.done, rows by key (a killed step resumes its folds).
exec 2>&1
LOGDIR="$HOME/codabench/logs/sprint_1028/p3"
ENV_SETUP='source ~/codabench/env040.sh >/dev/null'
THREADS_PER_JOB=4
export OMP_WAIT_POLICY=PASSIVE
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
SWN="python $HOME/codabench/analysis/proteus_sweep.py --protocols nest --outer 0-9"
FB6="fbands=1to4_4to8_8to13_13to20_20to30_30to45"
BASE="personal=pooled align=stream adapt=online buffer=64 fcenter=True $FB6"

lane2() { step nest_fb6 100 300m $SWN --lane NF2 --name fb6 --set $BASE xblocks=bpt4; }
lane3() { step nest_x4f 110 300m $SWN --lane NF3 --name X4f --set $BASE xblocks=bpt4_icoh fscale=True clf=logreg0.01; }
lane4() { step nest_cf 120 330m $SWN --lane NF4 --name "$CF_NAME" --set $BASE xblocks=$CF_XB; }
lane5() { step nest_xf 120 330m $SWN --lane NF5 --name "$XF_NAME" --set $BASE xblocks=$XF_XB fscale=True clf=logreg0.01; }

LANES=${LANES:-23${CF_NAME:+4}${XF_NAME:+5}}
[[ $LANES == *2* ]] && { lane2 & }
[[ $LANES == *3* ]] && { lane3 & }
[[ $LANES == *4* ]] && { lane4 & }
[[ $LANES == *5* ]] && { lane5 & }
wait
true
