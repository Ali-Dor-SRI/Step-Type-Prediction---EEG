#!/usr/bin/env bash
# threads_check_1027_unzip.sh: the F64T zips unzipped and replayed (= the F64 runs,
# window for window), plus the stateful streaming gates on the official folder.
exec 2>&1
source ~/codabench/env040.sh
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS   # server-like defaults
OUT=~/codabench/logs/threads_1027
for p in official all; do
  u=$OUT/f64t_$p/unzipped
  rm -rf "$u"; mkdir -p "$u"
  python -I -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$OUT/f64t_$p/riemann_sealed_proteus_f64t_${p}_2026-10-27.zip" "$u"
  echo "== unzipped $p ($(date +%T))"
  python ~/codabench/analysis/proteus_threads_check.py --submission "$u" \
    --out "$OUT/f64t_${p}_unzip.npy" --ref "$OUT/f64_${p:0:3}_all20.npy" | grep -E "torch threads|RESULT|EQUAL|Error"
done
echo "== gates official ($(date +%T))"
python ~/codabench/analysis/proteus_w1016_gates.py --submission "$OUT/f64t_official/unzipped" | grep -E "PASS|FAIL|loaded"
echo "done $(date +%T)"
