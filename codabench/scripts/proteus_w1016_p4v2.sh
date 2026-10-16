#!/usr/bin/env bash
# proteus_w1016_p4v2.sh: sprint 2026-10-16 Phase 4, final pass of the candidate
# chain (CANDNAME, default proteus_s64v2; BAKE = the decided settings) on
# official and all, then the stateful streaming contract
# (proteus_w1016_gates.py --submission) on each exported and unzipped folder,
# then the old-world zhou regression (defaults bit-identical after every solver
# edit of this sprint). Output: logs/proteus_w1016/<CANDNAME>_p4.out
set -u
export CANDNAME=${CANDNAME:-proteus_s64v2}
export BAKE=${BAKE:-"personal=pooled align=stream adapt=online buffer=64 xblocks=bpt4"}
export THREADS_PER_JOB=8
for p in official all; do
  bash "$HOME/codabench/scripts/proteus_w1016_cand.sh" "$p"
  echo "== candidate $p rc=$?"
  tail -2 "$HOME/codabench/logs/proteus_w1016/${CANDNAME}_cand_$p/STATUS.md"
done
(
  source ~/codabench/env040.sh >/dev/null
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUTF8=1
  for p in official all; do
    d=$HOME/codabench/logs/proteus_w1016/${CANDNAME}_cand_$p
    for sub in "$d/submission" "$HOME/neuralbench/proteus_runs/${CANDNAME}_cand_$p/unzipped"; do
      echo "== stateful contract $sub"
      python ~/codabench/analysis/proteus_w1016_gates.py --submission "$sub" 2>&1 | grep -E "PASS|FAIL|loaded|Error|Trace"
    done
  done
)
(
  source ~/codabench/env.sh >/dev/null
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  RUN_NAME=regress_w1016_${CANDNAME} bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data zhou2016_xsess > /dev/null 2>&1
  echo "== zhou regression rc=$?"
  grep -a "gate:" ~/codabench/logs/regress_w1016_${CANDNAME}/STATUS.md
)
