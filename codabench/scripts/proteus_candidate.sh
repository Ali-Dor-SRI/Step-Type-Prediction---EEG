#!/usr/bin/env bash
# proteus_candidate.sh <official|lopo|later|all> [fold]
# One solver through the whole path on PROTEUS: bake -> train -> replay ->
# equal -> zip. Setup session 2026-10-15 (brief
# prompts/2026-10-13_proteus_train_test_setup.md, Phase 5). The vehicle is the
# existing Riemann-Sealed solver in its simplest valid configuration for people
# it has never seen: personal="pooled", align="none", adapt="none", everything
# else at the file's defaults, baked into
# solvers/bci_decoding/riemann_sealed_proteus_pooled.py (Codabench runs a
# solver file with its default parameters). It is a plumbing vehicle, not a
# candidate model: scores are for information only. NOTHING IS UPLOADED.
#
#   official : train on the organisers' dataset exactly as the server runs it,
#              replay from a read-only copy, zip, replay from the unzipped
#              folder, replay through route B official, a second training run
#              (the audit re-runs training), in-process G9 + G10.
#   all      : the same chain through route B all (its test = the official
#              test, inside the training set: score meaningless), plus a
#              replay on the organisers' dataset. Zip kept, no SUBMISSIONS row.
#   later, lopo <fold> : training with evaluation only, through route B.
#
# Resumable by step (step_lib.sh): re-running skips the steps whose .done
# exists. Logs: ~/codabench/logs/proteus_setup/cand_<protocol>[_f<fold>]/;
# copies: ~/neuralbench/proteus_runs/<run>/ (never /tmp).
#   wsl: bash ~/codabench/scripts/proteus_candidate.sh official
set -u
PROTOCOL=${1:?usage: proteus_candidate.sh <official|lopo|later|all> [fold]}
FOLD=${2:-}
case "$PROTOCOL" in
  official|later|all) [ -z "$FOLD" ] || { echo "protocol $PROTOCOL takes no fold"; exit 2; }; RUN=cand_$PROTOCOL ;;
  lopo) [ -n "$FOLD" ] || { echo "lopo needs a fold 0..9"; exit 2; }; RUN=cand_lopo_f$FOLD ;;
  *) echo "unknown protocol $PROTOCOL"; exit 2 ;;
esac
LOGDIR=$HOME/codabench/logs/proteus_setup/$RUN
ENV_SETUP='source ~/codabench/env040.sh'
THREADS_PER_JOB=${THREADS_PER_JOB:-6}
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"
status_row "proteus_candidate.sh $PROTOCOL $FOLD (run $RUN, neuralbench $(python -c 'import importlib.metadata as m; print(m.version("neuralbench"))'), cwd $(pwd))"

# -- bake ---------------------------------------------------------------------
SRC=$HOME/codabench/solvers/bci_decoding/riemann_sealed.py
CAND=$HOME/codabench/solvers/bci_decoding/riemann_sealed_proteus_pooled.py
TMP=$LOGDIR/riemann_sealed_proteus_pooled.py.new
sed -e 's/"personal": \["blend"\]/"personal": ["pooled"]/' \
    -e 's/"align": \["subject"\]/"align": ["none"]/' \
    -e 's/name = "Riemann-Sealed"$/name = "Riemann-Sealed-Proteus-Pooled"/' \
    "$SRC" > "$TMP"
for pat in '"personal": ["pooled"]' '"align": ["none"]' '"adapt": ["none"]' 'name = "Riemann-Sealed-Proteus-Pooled"'; do
  grep -qF "$pat" "$TMP" || { status_row "ERROR: bake: $pat not in the copy"; exit 1; }
done
if [ -f "$CAND" ] && cmp -s "$TMP" "$CAND"; then
  rm -f "$TMP"; status_row "bake: $CAND unchanged"
else
  mv "$TMP" "$CAND"; status_row "bake: wrote $CAND"
fi
status_row "bake check: $(grep -cF -e '"personal": ["pooled"]' -e '"align": ["none"]' -e '"adapt": ["none"]' "$CAND") of 3 baked lines present"

# -- datasets -----------------------------------------------------------------
OFFICIAL="BCI[study=dreyer2026proteus,subset=full]"
ROUTEB=$HOME/codabench/datasets/proteus_protocols.py
case "$PROTOCOL" in
  official) DATASET="$OFFICIAL" ;;
  lopo)     DATASET="$ROUTEB[protocol=lopo,fold=$FOLD]" ;;
  *)        DATASET="$ROUTEB[protocol=$PROTOCOL]" ;;
esac
export LOGDIR RUN DATASET OFFICIAL ROUTEB CAND
export RUNS=$HOME/neuralbench/proteus_runs/$RUN
mkdir -p "$RUNS"
export OUT=$LOGDIR/submission            # fresh, absolute, never the solver's folder
export OUT2=$LOGDIR/submission2
export REPLAY=$RUNS/replay
export UNZ=$RUNS/unzipped
export ZIP=$LOGDIR/riemann_sealed_proteus_pooled_${PROTOCOL}${FOLD:+_f$FOLD}_$(date +%F).zip

FLAGS='--seed 42 --no-cache --no-plot --no-html --no-display'
export FLAGS

# -- train --------------------------------------------------------------------
[ -f "$LOGDIR/train.done" ] || rm -rf "$OUT"
COMPET_SUBMISSION_DIR="$OUT" step train 6 60m bash -c \
  'benchopt run tracks/bci_decoding -d "$DATASET" -s "$CAND" -o "BCI-decoding[training=True]" $FLAGS --output "$LOGDIR/train.parquet"'
[ -f "$LOGDIR/train.done" ] || { status_row "train failed: see train.log"; exit 1; }
status_row "fit line: $(grep -ao '\[Riemann-Sealed\] fitting on X=([0-9, ]*), subjects=[0-9]*' "$LOGDIR/train.log" | head -1)"
ls "$OUT/submission.py" "$OUT/riemann_sealed.joblib" >/dev/null || { status_row "export missing"; exit 1; }

if [ "$PROTOCOL" = later ] || [ "$PROTOCOL" = lopo ]; then
  python "$HOME/codabench/analysis/proteus_eval.py" equal "$LOGDIR/RESULTS.md" train="$LOGDIR/train.parquet" \
    && status_row "RESULTS.md written" || status_row "equal: failed"
  status_row "ALL DONE $RUN (training with evaluation only)"
  exit 0
fi

# -- replay from a read-only copy (copy made in the step that uses it) ---------
COMPET_SUBMISSION_DIR="$REPLAY" step replay 3 30m bash -c '
  set -e
  [ -d "$REPLAY" ] && chmod -R u+w "$REPLAY"; rm -rf "$REPLAY"
  cp -r "$OUT" "$REPLAY"; chmod -R a-w "$REPLAY"; ls -la "$REPLAY"
  benchopt run tracks/bci_decoding -d "$DATASET" -s "$REPLAY/submission.py" -o "BCI-decoding[training=False]" $FLAGS --output "$LOGDIR/replay.parquet"'
[ -f "$LOGDIR/replay.done" ] || { status_row "replay failed"; exit 1; }

# -- zip (files at the zip root), extract into an empty folder, replay ---------
step zip 1 10m bash -c '
  set -e
  rm -f "$ZIP"
  python -m zipfile -c "$ZIP" "$OUT/submission.py" "$OUT/riemann_sealed.joblib"
  python -m zipfile -l "$ZIP"
  ls -l "$ZIP"; sha256sum "$ZIP"'
[ -f "$LOGDIR/zip.done" ] || { status_row "zip failed"; exit 1; }
COMPET_SUBMISSION_DIR="$UNZ" step replay_unzip 3 30m bash -c '
  set -e
  rm -rf "$UNZ"; mkdir -p "$UNZ"
  python -m zipfile -e "$ZIP" "$UNZ"; ls -la "$UNZ"
  benchopt run tracks/bci_decoding -d "$DATASET" -s "$UNZ/submission.py" -o "BCI-decoding[training=False]" $FLAGS --output "$LOGDIR/replay_unzip.parquet"'
[ -f "$LOGDIR/replay_unzip.done" ] || { status_row "replay_unzip failed"; exit 1; }

# -- the cross replay: official -> route B official; all -> the organisers' dataset
if [ "$PROTOCOL" = official ]; then CROSS="$ROUTEB[protocol=official]"; else CROSS="$OFFICIAL"; fi
export CROSS
COMPET_SUBMISSION_DIR="$REPLAY" step replay_cross 3 30m bash -c '
  benchopt run tracks/bci_decoding -d "$CROSS" -s "$REPLAY/submission.py" -o "BCI-decoding[training=False]" $FLAGS --output "$LOGDIR/replay_cross.parquet"'
[ -f "$LOGDIR/replay_cross.done" ] || { status_row "replay_cross failed"; exit 1; }

if [ "$PROTOCOL" = official ]; then
  # -- a second training run with seed 42 into a fresh folder (the audit re-runs training)
  [ -f "$LOGDIR/train2.done" ] || rm -rf "$OUT2"
  COMPET_SUBMISSION_DIR="$OUT2" step train2 6 60m bash -c \
    'benchopt run tracks/bci_decoding -d "$DATASET" -s "$CAND" -o "BCI-decoding[training=True]" $FLAGS --output "$LOGDIR/train2.parquet"'
  [ -f "$LOGDIR/train2.done" ] || { status_row "train2 failed"; exit 1; }
  # -- in-process G9 (y_pred across folders, personal / align) and G10 (streaming contract, with one in-process fit)
  step g9_g10 8 60m python "$HOME/codabench/analysis/proteus_eval.py" g9g10 --export "$OUT" --replay "$REPLAY" \
    --unzipped "$UNZ" --export2 "$OUT2" --out "$LOGDIR" --fit
  [ -f "$LOGDIR/g9_g10.done" ] || status_row "g9_g10 reported a failure: see g9_g10.log"
  PARQUETS="train=$LOGDIR/train.parquet replay=$LOGDIR/replay.parquet unzipped=$LOGDIR/replay_unzip.parquet routeB=$LOGDIR/replay_cross.parquet train2=$LOGDIR/train2.parquet"
else
  PARQUETS="train=$LOGDIR/train.parquet replay=$LOGDIR/replay.parquet unzipped=$LOGDIR/replay_unzip.parquet official_loader=$LOGDIR/replay_cross.parquet"
fi
python "$HOME/codabench/analysis/proteus_eval.py" equal "$LOGDIR/RESULTS.md" $PARQUETS --zip "$ZIP" \
  && status_row "equal: EQUAL (RESULTS.md written)" || status_row "equal: DIFFERENT or failed (see RESULTS.md)"
status_row "predict timing: $(grep -ao '\[Riemann-Sealed\] predict: [0-9]* calls, [0-9]* windows, [0-9.]* s ([0-9.]* ms/window)' "$LOGDIR/replay.log" | tail -1)"
status_row "ALL DONE $RUN"
