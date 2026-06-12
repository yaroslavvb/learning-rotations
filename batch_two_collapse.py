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


def _rot2_apply(M, a, b, c):
    """Apply the Rodrigues rotation I + K + K^2/(1+c), K = a b^T - b a^T,
    to columns... to a stack of matrices/vectors M (batched on axis 0).
    a, b unit (T, d); c = a.b (T,). M is (T, d, k) or (T, d)."""
    vec = M.ndim == 2
    if vec:
        M = M[:, :, None]
    bM = np.einsum("td,tdk->tk", b, M)
    aM = np.einsum("td,tdk->tk", a, M)
    KM = a[:, :, None] * bM[:, None, :] - b[:, :, None] * aM[:, None, :]
    bK = np.einsum("td,tdk->tk", b, KM)
    aK = np.einsum("td,tdk->tk", a, KM)
    KKM = a[:, :, None] * bK[:, None, :] - b[:, :, None] * aK[:, None, :]
    out = M + KM + KKM / (1 + c)[:, None, None]
    return out[:, :, 0] if vec else out


def native_step_rodrigues2(W, X, Y, tol=1e-9):
    """SVD-free closed form of the batch-2 native update: two Rodrigues
    rotations (align pair 1; align pair 2 inside the orthocomplement of y1)
    followed by one closed-form angle that spends the leftover stabilizer
    freedom on the trace. Exactly equals the constrained-Procrustes optimum."""
    u1, u2o = (W @ X).transpose(2, 0, 1)              # (T, d) each
    y1, y2 = Y.transpose(2, 0, 1)

    # step 1: Rodrigues rotation taking u1 -> y1 (guarded at c1 = -1)
    c1 = np.clip(np.einsum("td,td->t", u1, y1), -1.0, 1.0)
    bad = 1 + c1 < tol
    c1 = np.where(bad, 0.0, c1)                       # bad rows fall back below
    u2 = _rot2_apply(u2o, y1, u1, c1)

    # step 2: Rodrigues rotation in span{p, q} (perp to y1) taking u2 -> y2
    a = np.einsum("td,td->t", y1, y2)
    p = u2 - a[:, None] * y1
    q = y2 - a[:, None] * y1
    pn = np.linalg.norm(p, axis=1)
    skip2 = pn < tol                                  # u2 already in place
    pn = np.where(skip2, 1.0, pn)
    ph, qh = p / pn[:, None], q / pn[:, None]
    c2 = np.clip(np.einsum("td,td->t", ph, qh), -1.0, 1.0)
    bad |= (1 + c2 < tol) & ~skip2
    c2 = np.where(skip2 | bad, 0.0, c2)
    qh = np.where(skip2[:, None], ph, qh)             # makes K2 = 0 on skipped rows

    # step 3: stabilizer plane Z = span{u1,u2}-perp inside the active 4-space
    # (Q acts before R1, so it must fix the ORIGINAL u1, u2)
    f1 = u1
    g = u2o - np.einsum("td,td->t", f1, u2o)[:, None] * f1
    gn = np.linalg.norm(g, axis=1)
    f2 = g / np.where(gn < tol, 1.0, gn)[:, None]

    def gs(v, basis):
        for b in basis:
            v = v - np.einsum("td,td->t", b, v)[:, None] * b
        n = np.linalg.norm(v, axis=1)
        return v / np.where(n < tol, 1.0, n)[:, None], n >= tol

    z1, ok1 = gs(y1.copy(), [f1, f2])
    z2, ok2 = gs(y2.copy(), [f1, f2, z1])
    free = ok1 & ok2                                  # else no in-plane freedom

    Rz1 = _rot2_apply(_rot2_apply(z1, y1, u1, c1), qh, ph, c2)
    Rz2 = _rot2_apply(_rot2_apply(z2, y1, u1, c1), qh, ph, c2)
    alpha = np.einsum("td,td->t", z1, Rz1) + np.einsum("td,td->t", z2, Rz2)
    beta = np.einsum("td,td->t", z1, Rz2) - np.einsum("td,td->t", z2, Rz1)
    theta = np.where(free, np.arctan2(beta, alpha), 0.0)

    # W' = R2 R1 Q W, each factor applied as a rank-2 correction
    ct, st = np.cos(theta), np.sin(theta)
    zW1 = np.einsum("td,tde->te", z1, W)
    zW2 = np.einsum("td,tde->te", z2, W)
    QW = (W + (ct - 1)[:, None, None] * (z1[:, :, None] * zW1[:, None, :]
                                         + z2[:, :, None] * zW2[:, None, :])
          + st[:, None, None] * (z2[:, :, None] * zW1[:, None, :]
                                 - z1[:, :, None] * zW2[:, None, :]))
    Wn = _rot2_apply(_rot2_apply(QW, y1, u1, c1), qh, ph, c2)

    if bad.any():                                     # antipodal guards: rare
        Wn[bad] = native_step(W[bad], X[bad], Y[bad])
    return Wn


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    A = aligned_isoclinic(D, np.pi / 2)

    # self-check: the two-batch Rodrigues closed form equals the subspace-SVD
    # optimum, both from cold start and near convergence
    Xc = draw_batch(rng, 64, D)
    Yc = np.einsum("ij,tjk->tik", A, Xc)
    for Wc in (np.tile(np.eye(D), (64, 1, 1)),
               batched_polar_so(A + 1e-4 * rng.standard_normal((64, D, D)))):
        gap = np.abs(native_step(Wc, Xc, Yc)
                     - native_step_rodrigues2(Wc, Xc, Yc)).max()
        assert gap < 1e-12, gap
    print("rodrigues2 == subspace-SVD optimum (max gap < 1e-12 asserted)")

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
