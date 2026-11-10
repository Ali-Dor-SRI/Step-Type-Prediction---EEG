#!/usr/bin/env bash
# Sprint 11-12: bake an aligned-Shallow ensemble zip on `all` + gates.
#   bake (sprint1112_bake_al.py) -> gate iii (proteus_w1016_gates.py --submission, 1-4 incl. timing)
#   -> gate vi (no braindecode, python -I) -> benchopt CLI inference-only on the official dataset
#   (weights staged in tracks/bci_decoding/outputs/X4f-Shallow-ens-al/ for the run, removed after).
# Usage: sprint1112_p1b.sh [w] [out] [rows] [adabn-prior|none] [skip-bake|bake] [selftrain q,lam,start,step,wpl,budget]
source ~/codabench/env040.sh
W=${1:-0.5}; OUT=${2:-al_cand_all}; ROWS=${3:-$HOME/codabench/logs/sprint_1112/rows_align_all}; AB=${4:-none}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE PYTHONUTF8=1
D=~/codabench/logs/sprint_1112; C=$D/$OUT; mkdir -p "$C"
A=~/codabench/analysis
echo "$(date +%T) START p1b $OUT w=$W adabn=$AB" >> $D/STATUS_jobs.md
if [ "$5" != skip-bake ]; then
  ABARG=""; [ "$AB" != none ] && ABARG="--adabn $AB"
  [ -n "$6" ] && ABARG="$ABARG --st $6"
  python $A/sprint1112_bake_al.py --rows "$ROWS" --w "$W" --out "$OUT" $ABARG > "$C/bake.log" 2>&1 || { echo "$(date +%T) END p1b bake FAIL" >> $D/STATUS_jobs.md; exit 1; }
  python $A/proteus_w1016_gates.py --submission "$C/unzipped" > "$C/gate_iii.log" 2>&1; r3=$?
  python -I $A/sprint1109_nobd.py "$C/unzipped" 50 > "$C/gate_vi.log" 2>&1; r6=$?
fi
mkdir -p "$C/cli"
STG=$COMPET_ROOT/tracks/bci_decoding/outputs/X4f-Shallow-ens-al
rm -rf "$STG"; mkdir -p "$STG"; cp "$C/unzipped/"* "$STG/"
benchopt run tracks/bci_decoding -o "BCI-decoding[training=False]" -d "BCI[study=dreyer2026proteus,subset=full]" \
  -s "$C/unzipped/submission.py" --seed 42 --no-cache --no-plot --no-html --no-display \
  --output "$C/cli/cli_official.parquet" > "$C/cli/cli_official.log" 2>&1; rc=$?
rm -rf "$STG"
echo "$(date +%T) END p1b $OUT gate_iii=${r3:-skip} gate_vi=${r6:-skip} cli=$rc" >> $D/STATUS_jobs.md
