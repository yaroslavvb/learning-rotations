"""Short equivalence test + timing benchmark: dense O(d^3) vs fast O(d^2)."""
import multiprocessing as mp
import os
import time

import numpy as np

from large_d_collapse import simulate_dense, simulate_fast

D = 128
STEPS = 160  # short run; per-step cost is constant, so results extrapolate


def bench(label, fn):
    t0 = time.perf_counter()
    fn()
    wall = time.perf_counter() - t0
    print(f"{label:44s} {wall:6.2f}s")
    return wall


if __name__ == "__main__":
    workers = os.cpu_count()

    dist_dense, _ = simulate_dense(D, 4, 60, seed=0)
    dist_fast, drift = simulate_fast(D, 4, 60, seed=0)
    print(f"equivalence (d={D}, 4 trials, 60 steps, same seed):")
    for m in dist_dense:
        rel = np.max(np.abs(dist_dense[m] - dist_fast[m]) / dist_dense[m])
        print(f"  {m:4s}: max relative curve difference {rel:.2e}")
    print(f"  orthogonality drift of fast W: {drift:.2e}\n")

    print(f"benchmark (d={D}, {STEPS} steps):")
    base = bench("dense (SVD polar), 32 trials, sequential",
                 lambda: simulate_dense(D, 32, STEPS))
    t_seq = bench("fast (closed-form), 32 trials, sequential",
                  lambda: simulate_fast(D, 32, STEPS))
    with mp.get_context("spawn").Pool(workers) as pool:
        pool.map(abs, range(workers))  # warm up workers before timing
        t_par = bench(f"fast, 256 trials, parallel ({workers} workers)",
                      lambda: simulate_fast(D, 256, STEPS, pool=pool,
                                            nchunks=workers))

    print(f"\nspeedup vs dense at 32 trials: {base / t_seq:.0f}x (sequential)")
    print(f"full figure config (1280 steps, 256 trials) projected: "
          f"{t_par * 1280 / STEPS:.1f}s fast-parallel vs "
          f"{base * 8 * 1280 / STEPS:.0f}s dense")
