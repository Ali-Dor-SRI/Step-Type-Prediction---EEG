#!/usr/bin/env bash
# Sprint 11-12 P1b: bake the aligned-Shallow ensemble zip (w 0.5) on `all` + gates.
#   bake (sprint1112_bake_al.py) -> gate iii (proteus_w1016_gates.py --submission, 1-4 incl. timing)
#   -> gate vi (no braindecode, python -I) -> benchopt CLI inference-only on the official dataset.
# Usage: sprint1112_p1b.sh [w] [out] [rows]
source ~/codabench/env040.sh
W=${1:-0.5}; OUT=${2:-al_cand_all}; ROWS=${3:-$HOME/codabench/logs/sprint_1112/rows_align_all}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE PYTHONUTF8=1
D=~/codabench/logs/sprint_1112; C=$D/$OUT; mkdir -p "$C"
A=~/codabench/analysis
echo "$(date +%T) START p1b $OUT w=$W" >> $D/STATUS_jobs.md
python $A/sprint1112_bake_al.py --rows "$ROWS" --w "$W" --out "$OUT" > "$C/bake.log" 2>&1 || { echo "$(date +%T) END p1b bake FAIL" >> $D/STATUS_jobs.md; exit 1; }
python $A/proteus_w1016_gates.py --submission "$C/unzipped" > "$C/gate_iii.log" 2>&1; r3=$?
python -I $A/sprint1109_nobd.py "$C/unzipped" 50 > "$C/gate_vi.log" 2>&1; r6=$?
mkdir -p "$C/cli"
benchopt run tracks/bci_decoding -o "BCI-decoding[training=False]" -d "BCI[study=dreyer2026proteus,subset=full]" \
  -s "$C/unzipped/submission.py" --seed 42 --no-cache --no-plot --no-html --no-display \
  --output "$C/cli/cli_official.parquet" > "$C/cli/cli_official.log" 2>&1; rc=$?
echo "$(date +%T) END p1b $OUT gate_iii=$r3 gate_vi=$r6 cli=$rc" >> $D/STATUS_jobs.md
