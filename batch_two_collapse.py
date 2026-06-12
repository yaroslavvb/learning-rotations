"""Batch-size-2 mirror of large_d_collapse.py: two pairs (x_i, y_i = A x_i) per step.

Updates (columns X = [x1 x2], Y = [y1 y2], both d x 2):
  GD     W <- W + (Y - WX)(X^T X)^{-1} X^T          (block Kaczmarz)
  PGD    polar projection of the GD step onto SO(d)
  Native the smallest rotation R with R(Wx_i) = y_i, i = 1,2 -- the
         trace-maximal orthogonal completion, solved in the <= 4-dim
         subspace span{Wx_1, Wx_2, y_1, y_2} with a Kabsch det correction.

Per-step contraction of E||W - A||_F^2 (P = projector onto a random 2-plane;
E tr(M P) = (2/d) tr M and E||P Omega P||^2 = 2||Omega||^2/(d(d-1))):
  GD     1 - 2/d                  exact, every step
  PGD    1 - 3/d + 1/(d(d-1))     asymptotic
  Native 1 - 4/d + 2/(d(d-1))     asymptotic
"""
import numpy as np

from large_d_collapse import aligned_isoclinic, batched_polar_so, simulate_fast

D, TRIALS, STEPS, SEED = 128, 256, 640, 0
COLORS = {"gd": "tab:blue", "pgd": "tab:orange", "nat": "tab:green"}
LABELS = {"gd": "GD (Unconstrained)", "pgd": "Projected GD", "nat": "Native (2-pair geodesic)"}
THEORY = {"gd": 1 - 2 / D,
          "pgd": 1 - 3 / D + 1 / (D * (D - 1)),
          "nat": 1 - 4 / D + 2 / (D * (D - 1))}
RATE_TEX = {"gd": r"1 - 2/d", "pgd": r"1 - 3/d + \frac{1}{d(d-1)}",
            "nat": r"1 - 4/d + \frac{2}{d(d-1)}"}


def draw_batch(rng, trials, d):
    X = rng.standard_normal((trials, d, 2))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def gd_correction(W, X, Y):
    """(Y - WX)(X^T X)^{-1} X^T, batched; X has unit columns."""
    U = W @ X
    R = Y - U
    g = np.einsum("ti,ti->t", X[:, :, 0], X[:, :, 1])
    den = np.maximum(1 - g * g, 1e-12)
    RG = np.empty_like(R)                       # R @ Ginv, closed-form 2x2 inverse
    RG[:, :, 0] = (R[:, :, 0] - g[:, None] * R[:, :, 1]) / den[:, None]
    RG[:, :, 1] = (R[:, :, 1] - g[:, None] * R[:, :, 0]) / den[:, None]
    return RG @ X.transpose(0, 2, 1)


def native_step(W, X, Y):
    """Smallest rotation R with R(Wx_i) = y_i: solved in span{u1,u2,y1,y2}."""
    T, d = W.shape[0], W.shape[1]
    U = W @ X
    Q = np.linalg.qr(np.concatenate([U, Y], axis=2))[0]          # (T, d, 4)
    us = np.einsum("tdm,tdk->tmk", Q, U)                         # (T, 4, 2)
    ys = np.einsum("tdm,tdk->tmk", Q, Y)

    def split(v):                       # complete QR with positive-diagonal fix
        qf, r = np.linalg.qr(v, mode="complete")
        s = np.sign(np.stack([r[:, 0, 0], r[:, 1, 1]], axis=1))
        s[s == 0] = 1.0
        qf[:, :, :2] *= s[:, None, :]
        return qf[:, :, :2], qf[:, :, 2:]

    qu, qup = split(us)
    qy, qyp = split(ys)
    M2 = np.einsum("tmi,tmj->tij", qup, qyp)                     # (T, 2, 2)
    e2, _, vh2 = np.linalg.svd(M2)

    def assemble(sgn):
        tmp = e2.transpose(0, 2, 1).copy()
        tmp[:, 1, :] *= sgn[:, None]
        Q2 = vh2.transpose(0, 2, 1) @ tmp                        # F diag(1,sgn) E^T
        return qy @ qu.transpose(0, 2, 1) + qyp @ Q2 @ qup.transpose(0, 2, 1)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        det = np.linalg.det(assemble(np.ones(T)))
    Rs = assemble(np.where(det < 0, -1.0, 1.0))                  # det R = +1 (Kabsch)
    return W + Q @ ((Rs - np.eye(4)) @ np.einsum("tdm,tde->tme", Q, W))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    A = aligned_isoclinic(D, np.pi / 2)
    Ws = {m: np.tile(np.eye(D), (TRIALS, 1, 1)) for m in THEORY}
    dist = {m: np.zeros(STEPS + 1) for m in THEORY}
    sq = {m: np.zeros(STEPS + 1) for m in THEORY}
    for m in THEORY:
        dist[m][0], sq[m][0] = np.sqrt(2 * D), 2 * D

    for t in range(STEPS):
        X = draw_batch(rng, TRIALS, D)
        Y = np.einsum("ij,tjk->tik", A, X)
        Ws["gd"] = Ws["gd"] + gd_correction(Ws["gd"], X, Y)
        Ws["pgd"] = batched_polar_so(Ws["pgd"] + gd_correction(Ws["pgd"], X, Y))
        Ws["nat"] = native_step(Ws["nat"], X, Y)
        for m in THEORY:
            e2 = ((Ws[m] - A) ** 2).sum(axis=(1, 2))
            dist[m][t + 1] = np.sqrt(e2).mean()
            sq[m][t + 1] = e2.mean()

    # hygiene + empirical per-step contraction constants c_eff = d (1 - ratio)
    print("orthogonality drift (pgd, nat):",
          max(np.abs(np.einsum("tji,tjk->tik", Ws[m], Ws[m]) - np.eye(D)).max()
              for m in ("pgd", "nat")))
    print("last-step native constraint |W X - Y|:",
          np.linalg.norm(Ws["nat"] @ X - Y, axis=(1, 2)).max())
    lo, hi = 300, 620  # past the nonlinear transient (~250 steps from W_0 = I)
    for m in THEORY:
        slope = np.polyfit(np.arange(lo, hi), np.log(sq[m][lo:hi]), 1)[0]
        print(f"{m:4s} fitted c_eff = {D * (1 - np.exp(slope)):.4f}   "
              f"theory {D * (1 - THEORY[m]):.4f}")

    # figure 1: convergence vs per-step theory (mirrors the main README plot)
    t = np.arange(STEPS + 1)
    plt.figure()
    for m, f in THEORY.items():
        plt.semilogy(t, dist[m] / dist[m][0], color=COLORS[m], lw=1.3,
                     label=LABELS[m])
        plt.semilogy(t, f ** (t / 2), linestyle="none", marker="o",
                     markersize=3.5, markevery=25, color=COLORS[m],
                     label=rf"theory $({RATE_TEX[m]})^{{t/2}}$")
    plt.xlabel("Step $t$  (two pairs per step)")
    plt.ylabel(r"$\|W_t - A\|_F \;/\; \sqrt{2d}$")
    plt.title(rf"$d = {D}$, {TRIALS} trials, batch size 2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/batch_two_convergence.png", dpi=150)

    # figure 2: per-sample comparison against the batch-1 kernels
    d1, _ = simulate_fast(D, TRIALS, 2 * STEPS, seed=1)
    pair = {"gd": "gd", "pgd": "pgd", "nat": "geo"}
    plt.figure()
    for m in THEORY:
        plt.semilogy(2 * t, dist[m] / dist[m][0], color=COLORS[m], lw=1.6,
                     label=f"{LABELS[m]}, batch 2")
        plt.semilogy(np.arange(2 * STEPS + 1), d1[pair[m]] / d1[pair[m]][0],
                     color=COLORS[m], lw=1.0, ls="--", alpha=0.65,
                     label="batch 1")
    plt.xlabel("Samples seen")
    plt.ylabel(r"$\|W - A\|_F \;/\; \sqrt{2d}$")
    plt.title(rf"$d = {D}$: same samples, one at a time vs two at a time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/batch_two_per_sample.png", dpi=150)
    print("figures saved")
