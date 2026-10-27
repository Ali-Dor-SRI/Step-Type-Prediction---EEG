#!/usr/bin/env bash
# threads_check_1027.sh: F64 vs thread-pinned F64T on the official test streams
# (2026-10-27, decision 12). Builds the F64T submission folders (same joblib, new
# submission.py), then times and compares predictions:
#   all20  = every core visible, pools at their defaults (20 threads)
#   quota2 = 2 cores, pools still asked for 20 threads (a CPU-quota'd container)
exec 2>&1
source ~/codabench/env040.sh
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS   # server-like defaults
OUT=~/codabench/logs/threads_1027
CHK="python $HOME/codabench/analysis/proteus_threads_check.py"
mkdir -p "$OUT"
for p in official all; do
  src=~/codabench/logs/proteus_w1016/proteus_f64_cand_$p/submission
  dst=$OUT/f64t_$p/submission
  mkdir -p "$dst"
  cp "$src/riemann_sealed.joblib" "$dst/"
  cp ~/codabench/solvers/bci_decoding/riemann_sealed_proteus_f64t.py "$dst/submission.py"
  cmp "$src/riemann_sealed.joblib" "$dst/riemann_sealed.joblib" && echo "joblib $p identical"
done
run() {  # name, cpus, submission dir, ref
  local name=$1 cpus=$2 sub=$3 ref=$4
  echo "== $name ($(date +%T))"
  if [ "$cpus" = all ]; then
    $CHK --submission "$sub" --out "$OUT/$name.npy" ${ref:+--ref "$OUT/$ref.npy"}
  else
    OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20 \
      taskset -c "$cpus" $CHK --submission "$sub" --out "$OUT/$name.npy" ${ref:+--ref "$OUT/$ref.npy"}
  fi | grep -E "torch threads|RESULT|EQUAL|Error|error"
}
O=~/codabench/logs/proteus_w1016/proteus_f64_cand_official/submission
A=~/codabench/logs/proteus_w1016/proteus_f64_cand_all/submission
run f64_off_all20   all  "$O"
run f64t_off_all20  all  "$OUT/f64t_official/submission" f64_off_all20
run f64_all_all20   all  "$A"
run f64t_all_all20  all  "$OUT/f64t_all/submission"      f64_all_all20
run f64t_off_quota2 0,1  "$OUT/f64t_official/submission" f64_off_all20
run f64_off_quota2  0,1  "$O"                            f64_off_all20
echo "done $(date +%T)"
