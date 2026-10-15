#!/usr/bin/env bash
# The 0.4.0 (PROTEUS) environment. Source this inside WSL before any work on
# the released data:
#
#     source ~/codabench/env040.sh
#
# Activates ~/neuralbench/.venv040 (Python 3.12, torch+cpu, neuralbench 0.4.0),
# points benchopt at the 0.4.0 data home, exposes the upstream checkout as
# COMPET_ROOT and moves into it. It prints the neuralbench version and the
# working directory on purpose: never source it with >/dev/null, the print is
# the proof that a run is in this environment and not the frozen 0.3.1 one
# (env.sh). See PROTEUS_SETUP.md and SETUP.md ("2026-10-13").

export PATH="$HOME/.local/bin:$PATH"                       # uv
source "$HOME/neuralbench/.venv040/bin/activate"           # Python 3.12, neuralbench 0.4.0
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore::FutureWarning MNE_LOGGING_LEVEL=ERROR
export NEURALBENCH_CONFIG="$HOME/codabench/config/neuralbench_config.json"
export BENCHOPT_DATA_HOME="$HOME/neuralbench/benchopt_data_040"
export BENCHOPT_DEBUG=true
export COMPET_ROOT="$HOME/neuralbench/competition-040"     # upstream 0.4.0 worktree (442edcd)
export PROTEUS_CACHE_ROOT="$HOME/neuralbench/proteus_cache"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$COMPET_ROOT" || return 1
python -c "import importlib.metadata as m, os; print('neuralbench', m.version('neuralbench'), 'cwd', os.getcwd())"
