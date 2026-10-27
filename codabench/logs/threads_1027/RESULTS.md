# F64 vs thread-pinned F64T (2026-10-27, decision 12)

Why: the 10-26 F64 warm-up upload matched exactly (0.507944 / 0.506931 / 0.507285) but took
198.3 s = 263 ms/window on the server vs 8.2 ms locally. F64T = F64 with
`torch.set_num_threads(1)` + `threadpoolctl.threadpool_limits(1)` at `load_model`
(`solvers/bci_decoding/riemann_sealed_proteus_f64t.py`); same joblib (byte-identical).

Scripts: `scripts/threads_check_1027.sh` (10:19:34-10:27:32, 8 min, est. 6-8 min),
`scripts/threads_check_1027_unzip.sh` (10:27:37-10:28:36, 1 min). Thread env vars unset
(server-like defaults: torch 10, OpenBLAS 20, OpenMP 10/20 threads on this 20-core WSL).
Official test streams, B = 1, 755 windows.

| run | cores | threads | official cell mean / bal / acc | time | ms/window | preds vs F64 |
|---|---|---|---|---|---|---|
| F64 official | 20 | default | 0.507944 / 0.506931 / 0.507285 | 199.8 s | **264.7** (server: 263) | ref |
| F64T official | 20 | 1 | 0.507944 / 0.506931 / 0.507285 | 7.1 s | **9.4** | 0 of 755 differ |
| F64T official, unzipped | 20 | 1 | same | 6.5 s | 8.6 | 0 differ |
| F64 all | 20 | default | 0.987856 (meaningless: test in training) | 202.3 s | 267.9 | ref |
| F64T all | 20 | 1 | 0.987856 | 6.5 s | 8.7 | 0 differ |
| F64T all, unzipped | 20 | 1 | same | 7.0 s | 9.3 | 0 differ |
| F64T official | 2 (taskset) | 20 requested | same | 6.5 s | 8.6 | 0 differ |
| F64 official | 2 (taskset) | 20 requested | same | 8.5 s | 11.3 | 0 differ |

- Local default-thread F64 reproduces the server time (199.8 vs 198.3 s): the slowdown is
  thread-pool overhead on many cores for tiny B = 1 work, not the server's CPU. On 2 cores
  it mostly disappears (11.3 ms), so it is many-core spin/sync, not a CPU quota.
- Pinning removes it everywhere and changes no prediction.
- Stateful gates on the unzipped F64T official folder: ALL PASS (B=1 = batched, streams
  isolated, state used, reset restores; 8.5 ms/window -> 3.0 min for 106 x 199 windows).

Zips:
- `f64t_official/riemann_sealed_proteus_f64t_official_2026-10-27.zip` 705,086 B, sha256
  `fcfd7c621e0666e59e72b3311dcc588427e69b705ce82b68643d7bd4780d7430` (warm-up re-time;
  expect 0.507944 / 0.506931 / 0.507285 in well under a minute of scoring)
- `f64t_all/riemann_sealed_proteus_f64t_all_2026-10-27.zip` 856,946 B, sha256
  `a774135adbcc669d96dfbfba5635aed093ef6a7b6caf212b7c759b83cb2a0db8` (sealed candidate)
