"""Large-d regime at d = 128: normalized error vs rescaled time tau = t/d.

The per-step contraction (1 - c/d) of E||W - A||_F^2 becomes exp(-c tau) in
rescaled time, so Frobenius distance follows exp(-c tau / 2) with c = 1 (GD),
3/2 (projected GD), 2 (exact geodesic).
W_0 = I, A = pi/2 isoclinic; initial distance is sqrt(2d), divided out.

Fast path (no SVD): every update is a rank-<=2 correction of W, O(trials*d^2)
per step. For W in SO(d) the polar projection of the GD step W + r x^T has a
closed form: it rotates W in the plane span{Wx, y} by HALF the angle between
u = Wx and y, while the geodesic update rotates by the full angle. Error
norms use <W, A> via A's two-nonzeros-per-row sparsity, O(trials*d).
Independent trials are chunked across cores with multiprocessing.
"""
import multiprocessing as mp
import os
import time

import numpy as np

THEORY_C = {"gd": 1.0, "pgd": 1.5, "geo": 2.0}
LABELS = {"gd": "GD (Unconstrained)", "pgd": "Projected GD", "geo": "Exact Geodesic"}
COLORS = {"gd": "tab:blue", "pgd": "tab:orange", "geo": "tab:green"}
D = 128
TRIALS = 256
TAU_MAX = 10.0


def aligned_isoclinic(n, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.kron(np.eye(n // 2), np.array([[c, -s], [s, c]]))


def batched_polar_so(Z):
    """Reference SVD-based projection to SO(d); kept for tests/benchmarks."""
    u, _, vt = np.linalg.svd(Z)
    # det of a large orthogonal matrix can spuriously over/underflow inside
    # LAPACK's pivot product; the returned values are still correct (+-1)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        sign = np.linalg.det(u @ vt)
    u[sign < 0, :, -1] *= -1
    return u @ vt


def simulate_dense(d, trials, steps, seed=0):
    """Original O(d^3)-per-step implementation (SVD polar, dense Rodrigues)."""
    rng = np.random.default_rng(seed)
    A = aligned_isoclinic(d, np.pi / 2)
    I = np.eye(d)
    Ws = {m: np.tile(I, (trials, 1, 1)) for m in THEORY_C}
    dist = {m: np.zeros(steps + 1) for m in THEORY_C}
    for m in THEORY_C:
        dist[m][0] = np.sqrt(2 * d)
    timing = dict.fromkeys(THEORY_C, 0.0)
    for t in range(steps):
        x = rng.standard_normal((trials, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        y = x @ A.T
        for m, W in Ws.items():
            tic = time.perf_counter()
            Wx = np.einsum("nij,nj->ni", W, x)
            if m in ("gd", "pgd"):
                Z = W + (y - Wx)[:, :, None] * x[:, None, :]
                Ws[m] = Z if m == "gd" else batched_polar_so(Z)
            else:
                c = np.clip((Wx * y).sum(1), -1.0, 1.0)
                K = y[:, :, None] * Wx[:, None, :] - Wx[:, :, None] * y[:, None, :]
                Ws[m] = (I + K + K @ K / (1 + c)[:, None, None]) @ W
            dist[m][t + 1] = np.sqrt(((Ws[m] - A) ** 2).sum(axis=(1, 2))).mean()
            timing[m] += time.perf_counter() - tic
    return dist, timing


def _trace_against_A(W, i2):
    """<W, A> for the pi/2 isoclinic A: only entries (2i,2i+1)=-1, (2i+1,2i)=+1."""
    return (W[:, i2 + 1, i2] - W[:, i2, i2 + 1]).sum(1)


def _simulate_chunk(args):
    """Fast O(trials*d^2)-per-step run of all three methods on one trial chunk.

    Returns (sums, drift): sums[k, t] = sum over chunk trials of ||W_t - A||_F
    for method k in THEORY_C order; drift = max orthogonality error of the
    final PGD/geodesic W (sanity check on the rank-2 rotation updates).
    """
    d, trials, steps, seed = args
    rng = np.random.default_rng(seed)
    A = aligned_isoclinic(d, np.pi / 2)
    i2 = np.arange(0, d, 2)
    I = np.eye(d)
    Ws = {m: np.tile(I, (trials, 1, 1)) for m in THEORY_C}
    sums = {m: np.zeros(steps + 1) for m in THEORY_C}
    for m in THEORY_C:
        sums[m][0] = trials * np.sqrt(2.0 * d)
    e2_gd = np.full(trials, 2.0 * d)  # exact recursion: e2' = e2 - ||r||^2

    for t in range(steps):
        x = rng.standard_normal((trials, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        y = x @ A.T

        # GD: rank-1 update; error via the Kaczmarz identity
        W = Ws["gd"]
        r = y - np.einsum("nij,nj->ni", W, x)
        W += r[:, :, None] * x[:, None, :]
        e2_gd -= (r * r).sum(1)
        np.maximum(e2_gd, 0.0, out=e2_gd)
        sums["gd"][t + 1] = np.sqrt(e2_gd).sum()

        # PGD: closed-form polar of W + r x^T = rotate by half the u->y angle
        # in span{u, y}; degenerate plane (c = +-1) leaves W unchanged
        W = Ws["pgd"]
        u = np.einsum("nij,nj->ni", W, x)
        c = np.clip((u * y).sum(1), -1.0, 1.0)
        wt = np.einsum("nij,ni->nj", W, y)  # W^T y
        beta = np.sqrt(np.maximum(1.0 - c * c, 0.0))
        good = beta > 1e-12
        safe = np.where(good, beta, 1.0)[:, None]
        f2 = (wt - c[:, None] * x) / safe   # domain-side 2nd basis vector
        g2 = (y - c[:, None] * u) / safe    # range-side (= W f2)
        theta = np.arctan2(beta, 1.0 + c)   # half-angle: atan2(sin, 1+cos)
        cs = (np.cos(theta) - 1.0)[:, None]
        sn = np.sin(theta)[:, None]
        P = np.stack([u, g2], axis=2)
        Q = np.stack([cs * x - sn * f2, sn * x + cs * f2], axis=2)
        Q[~good] = 0.0
        W += P @ Q.transpose(0, 2, 1)
        sums["pgd"][t + 1] = np.sqrt(np.maximum(
            2.0 * d - 2.0 * _trace_against_A(W, i2), 0.0)).sum()

        # Geodesic: Rodrigues R = I + K + K^2/(1+c) collapsed to rank 2
        W = Ws["geo"]
        u = np.einsum("nij,nj->ni", W, x)
        c = np.clip((u * y).sum(1), -1.0, 1.0)
        wt = np.einsum("nij,ni->nj", W, y)
        ok = c > -0.999999
        denom = np.where(ok, 1.0 + c, 1.0)[:, None]
        P = np.stack([y, u], axis=2)
        Q = np.stack([((1.0 + 2.0 * c)[:, None] * x - wt) / denom,
                      -(wt + x) / denom], axis=2)
        Q[~ok] = 0.0
        W += P @ Q.transpose(0, 2, 1)
        sums["geo"][t + 1] = np.sqrt(np.maximum(
            2.0 * d - 2.0 * _trace_against_A(W, i2), 0.0)).sum()

    drift = max(np.abs(np.einsum("nji,njk->nik", Ws[m], Ws[m]) - I).max()
                for m in ("pgd", "geo"))
    return np.stack([sums[m] for m in THEORY_C]), drift


def simulate_fast(d, trials, steps, seed=0, pool=None, nchunks=1):
    """Fast simulation; with a pool, trials are split into nchunks chunks."""
    ss = np.random.SeedSequence(seed)
    if pool is None or nchunks <= 1:
        sums, drift = _simulate_chunk((d, trials, steps, ss))
    else:
        nchunks = min(nchunks, trials)
        sizes = [trials // nchunks + (i < trials % nchunks) for i in range(nchunks)]
        tasks = [(d, n, steps, child)
                 for n, child in zip(sizes, ss.spawn(nchunks))]
        results = pool.map(_simulate_chunk, tasks)
        sums = sum(r[0] for r in results)
        drift = max(r[1] for r in results)
    dist = {m: sums[k] / trials for k, m in enumerate(THEORY_C)}
    return dist, drift


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    workers = os.cpu_count()//2
    steps = int(TAU_MAX * D)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        dist, drift = simulate_fast(D, TRIALS, steps, pool=pool, nchunks=workers)
    print(f"d={D} ({TRIALS} trials, {steps} steps, {workers} workers): "
          f"{time.perf_counter() - t0:.1f}s | orthogonality drift {drift:.1e}")

    tau = np.arange(steps + 1) / D
    tau_th = np.linspace(0, TAU_MAX, 200)
    for m, c in THEORY_C.items():
        plt.semilogy(tau, dist[m] / dist[m][0], color=COLORS[m], lw=1.3,
                     label=LABELS[m])
        exponent = r"\tau/2" if c == 1 else rf"{c:g}\tau/2"
        plt.semilogy(tau_th, np.exp(-c * tau_th / 2), "--", color=COLORS[m],
                     lw=1.2, label=rf"theory $e^{{-{exponent}}}$")
    plt.legend(fontsize=9)
    plt.xlabel(r"Rescaled time  $\tau = t/d$")
    plt.ylabel(r"$\|W_t - A\|_F \;/\; \sqrt{2d}$")
    plt.title(rf"$d = {D}$, {TRIALS} trials: $W_0 = I$, $A$ = $\pi/2$ isoclinic")
    plt.tight_layout()
    plt.savefig("large_d_collapse.png", dpi=150)
