"""Noisy observations y = Ax + eps, eps ~ N(0, sigma^2 I), batch size one.

Relaxed updates with step size eta:
  GD     W <- W + eta (y - Wx) x^T                  (relaxed Kaczmarz / SGD)
  PGD    polar projection of the relaxed step
  Native rotate by eta * phi toward yhat = y/||y||  (partial geodesic)

Per-step E||W' - A||_F^2 = (1 - r/d) E||W - A||_F^2 + n, floor = n d / r:
  GD     r = 2 eta - eta^2      n = eta^2 sigma^2 d        floor = eta sigma^2 d^2/(2 - eta)
  PGD    r = 2 eta - eta^2/2    n = eta^2 sigma^2 (d-1)/2  floor = eta sigma^2 d(d-1)/(4 - eta)
  Native r = 4 eta - 2 eta^2    n = 2 eta^2 sigma^2 (d-1)  floor = eta sigma^2 d(d-1)/(2 - eta)
"""
import numpy as np

from large_d_collapse import aligned_isoclinic, batched_polar_so

D, TRIALS, SIGMA, STEPS = 32, 64, 1e-3, 6000
COLORS = {"gd": "tab:blue", "pgd": "tab:orange", "nat": "tab:green"}
LABELS = {"gd": "GD (Unconstrained)", "pgd": "Projected GD", "nat": "Partial geodesic"}


def floor(method, eta, d=D, s2=SIGMA**2):
    return {"gd": eta * s2 * d * d / (2 - eta),
            "pgd": eta * s2 * d * (d - 1) / (4 - eta),
            "nat": eta * s2 * d * (d - 1) / (2 - eta)}[method]


def run(method, eta, seed=0):
    rng = np.random.default_rng(seed)
    A = aligned_isoclinic(D, np.pi / 2)
    W = np.tile(np.eye(D), (TRIALS, 1, 1))
    dist = np.zeros(STEPS + 1)
    dist[0] = np.sqrt(2 * D)
    for t in range(STEPS):
        x = rng.standard_normal((TRIALS, D))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        y = x @ A.T + SIGMA * rng.standard_normal((TRIALS, D))
        if method in ("gd", "pgd"):
            Wx = np.einsum("tij,tj->ti", W, x)
            Z = W + eta * (y - Wx)[:, :, None] * x[:, None, :]
            W = Z if method == "gd" else batched_polar_so(Z)
        else:
            yn = y / np.linalg.norm(y, axis=1, keepdims=True)
            u = np.einsum("tij,tj->ti", W, x)
            c = np.clip(np.einsum("ti,ti->t", u, yn), -1.0, 1.0)
            s = np.sqrt(np.maximum(1 - c * c, 1e-24))
            phi = np.arccos(c)
            a1 = (np.sin(eta * phi) / s)[:, None, None]
            a2 = ((1 - np.cos(eta * phi)) / (s * s))[:, None, None]
            uW = np.einsum("ti,tij->tj", u, W)
            yW = np.einsum("ti,tij->tj", yn, W)
            KW = yn[:, :, None] * uW[:, None, :] - u[:, :, None] * yW[:, None, :]
            uK = np.einsum("ti,tij->tj", u, KW)
            yK = np.einsum("ti,tij->tj", yn, KW)
            KKW = yn[:, :, None] * uK[:, None, :] - u[:, :, None] * yK[:, None, :]
            W = W + a1 * KW + a2 * KKW
        dist[t + 1] = np.sqrt(((W - A) ** 2).sum(axis=(1, 2))).mean()
    return dist


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, eta in zip(axes, (1.0, 0.1)):
        for m in COLORS:
            dist = run(m, eta)
            f = floor(m, eta)
            tail = (dist[-STEPS // 4:] ** 2).mean()
            print(f"{m:4s} eta={eta:4.2f}  measured floor {tail:.3e}  "
                  f"predicted {f:.3e}  ratio {tail / f:.3f}")
            ax.semilogy(dist / np.sqrt(2 * D), color=COLORS[m], lw=1.3,
                        label=LABELS[m] if eta == 1.0 else None)
            ax.axhline(np.sqrt(f / (2 * D)), color=COLORS[m], ls=":", lw=1.4)
        ax.set_title(rf"$\eta = {eta:g}$")
        ax.set_xlabel("Step $t$")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel(r"$\|W_t - A\|_F \;/\; \sqrt{2d}$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(rf"$d = {D}$, $\sigma = {SIGMA:g}$: convergence to the predicted "
                 "noise floors (dotted)", y=0.99)
    fig.tight_layout()
    fig.savefig("reports/noise_floors.png", dpi=150)
    print("figure saved")
