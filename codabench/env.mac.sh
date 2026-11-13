# Source from bash or zsh: source ~/codabench/env.mac.sh
# Native Apple Silicon environment; the Windows/WSL entry points stay unchanged.
if [ -n "${BASH_VERSION:-}" ]; then
    _cb_source="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    eval '_cb_source=${(%):-%x}'
else
    echo 'env.mac.sh requires bash or zsh' >&2
    return 1
fi
export CODABENCH_ROOT="$(cd "$(dirname "$_cb_source")" && pwd -P)"
export CODABENCH_RUNTIME="$CODABENCH_ROOT/logs/mac_setup/neuralbench"
export CODABENCH_NAS_ROOT="${CODABENCH_NAS_ROOT:-/Volumes/Beast_PC/Projects/codabench}"
export CODABENCH_ARTIFACT_ROOT="$CODABENCH_NAS_ROOT/windows-transfer/codabench"
if [ ! -d "$CODABENCH_NAS_ROOT/neural_compet" ]; then
    echo "NAS data folder is unavailable: $CODABENCH_NAS_ROOT/neural_compet" >&2
    return 1
fi
if [ ! -f "$CODABENCH_RUNTIME/.venv040/bin/activate" ]; then
    echo "Missing native environment: $CODABENCH_RUNTIME/.venv040" >&2
    return 1
fi
source "$CODABENCH_RUNTIME/.venv040/bin/activate"
export COMPET_ROOT="$CODABENCH_RUNTIME/competition-040"
export BENCHOPT_DATA_HOME="$CODABENCH_RUNTIME/benchopt_data_040"
if [ ! -L "$BENCHOPT_DATA_HOME" ] || [ "$(cd "$BENCHOPT_DATA_HOME" && pwd -P)" != "$(cd "$CODABENCH_NAS_ROOT" && pwd -P)" ]; then
    echo "Expected $BENCHOPT_DATA_HOME to link to $CODABENCH_NAS_ROOT; refusing a local data download" >&2
    return 1
fi
export PROTEUS_CACHE_ROOT="$CODABENCH_NAS_ROOT/proteus_cache"
export NEURALBENCH_CONFIG="$CODABENCH_RUNTIME/neuralbench_config.json"
export XDG_CACHE_HOME="$CODABENCH_RUNTIME/cache"
export MPLCONFIGDIR="$CODABENCH_RUNTIME/cache/matplotlib"
export PYTHONPATH="$CODABENCH_ROOT/analysis:$COMPET_ROOT/tracks/bci_decoding${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::FutureWarning MNE_LOGGING_LEVEL=ERROR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
mkdir -p "$BENCHOPT_DATA_HOME" "$CODABENCH_RUNTIME/data" "$CODABENCH_RUNTIME/cache" "$CODABENCH_RUNTIME/results"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['CODABENCH_RUNTIME'])
cfg = dict(USER=os.environ.get('USER', 'local'), ENTITY_NAME='local',
           PROJECT_NAME='neuralbench', CLUSTER=None)
cfg.update({key: str(root / name) for key, name in
            [('DATA_DIR', 'data'), ('CACHE_DIR', 'cache'), ('SAVE_DIR', 'results')]})
(root / 'neuralbench_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
PY
cd "$COMPET_ROOT" || return 1
python -c 'import importlib.metadata as m, os; print("Native Mac: neuralbench", m.version("neuralbench"), "cwd", os.getcwd())'
unset _cb_source
