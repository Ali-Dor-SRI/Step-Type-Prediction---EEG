#!/usr/bin/env python
"""Gate (vi), sprint 2026-11-09: load a submission folder and predict with
`braindecode` made unimportable (a meta-path finder raises on it), run under
`python -I`. Usage: python -I sprint1109_nobd.py <folder> [n_windows]"""
import importlib.abc
import sys


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "braindecode" or name.startswith("braindecode."):
            raise ImportError(f"blocked for gate (vi): {name}")
        return None


sys.meta_path.insert(0, _Block())
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path.home() / "codabench/analysis"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import proteus_cache as PC  # noqa: E402
import proteus_eval as PE  # noqa: E402
import proteus_splits as PS  # noqa: E402

folder = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
c = PC.load()
m = PE.load_submission(folder, PE.official_meta())
te = np.asarray(PS.split("official", None, c.rows).test)[:n]
m.reset_state()
y = [int(np.asarray(m.predict(torch.from_numpy(np.asarray(c.X[i:i + 1])))).reshape(-1)[0]) for i in te]
print("braindecode in sys.modules:", any(k.startswith("braindecode") for k in sys.modules))
ok = not any(k.startswith("braindecode") for k in sys.modules) and len(y) == n
print(f"{'PASS' if ok else 'FAIL'} (vi) loaded + predicted {len(y)} windows without braindecode; classes {np.bincount(y, minlength=3).tolist()}")
sys.exit(0 if ok else 1)
