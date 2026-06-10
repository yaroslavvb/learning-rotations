"""Animated illustration of learning a rotation from data (d = 3).

Three estimators absorb the same stream of pairs (x, y = A x), one per step.
Each panel shows the current estimate W applied to a reference object (a
trefoil knot, colored along its length) against the target pose A applied to
the same object (gray ghost). Unconstrained gradient descent leaves SO(3),
visibly shearing the object before it fits; GD + projection stays rigid;
the geodesic update rotates straight onto the target and lands first.
A live error trace runs below. Writes learning_rotations.gif beside this file.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

SEED = 4  # chosen so the single-trial final errors land in theory order
STEPS = 14
HOLD0, HOLD1 = 8, 18
FPS = 16
# slow-motion tweening while the action is big, quicker once nearly aligned
SCHEDULE = [(t, (i + 1) / F)
            for t in range(STEPS)
            for F in [6 if t < 6 else 3]
            for i in range(F)]
METHODS = ["gd", "pgd", "geo"]
COLORS = {"gd": "tab:blue", "pgd": "tab:orange", "geo": "tab:green"}
TITLES = {"gd": "1 · gradient descent",
          "pgd": "2 · GD + projection",
          "geo": "3 · geodesic update"}
SUB = {"gd": "any matrix — object deforms",
       "pgd": "snap back to a rotation",
       "geo": "rotate $Wx$ onto $y$ exactly"}


def rot(axis, angle):
    """Rodrigues rotation matrix about a (not necessarily unit) axis."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rot_pow(R, s):
    """Fractional power R^s of a rotation via axis-angle (None if degenerate)."""
    cphi = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    phi = np.arccos(cphi)
    if np.sin(phi) < 1e-8:
        return None
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return rot(axis, s * phi)


def tween(W0, W1, s, rigid):
    if rigid:
        D = rot_pow(W1 @ W0.T, s)
        if D is not None:
            return D @ W0
    return (1 - s) * W0 + s * W1


def step_gd(W, x, y):
    return W + np.outer(y - W @ x, x)


def step_pgd(W, x, y):
    u = W @ x
    c = np.clip(u @ y, -1.0, 1.0)
    beta = np.sqrt(max(1.0 - c * c, 0.0))
    if beta < 1e-9:
        return W.copy()
    w = W.T @ y
    f2, g2 = (w - c * x) / beta, (y - c * u) / beta
    th = np.arctan2(beta, 1.0 + c)  # half the angle from Wx to y
    cs, sn = np.cos(th) - 1.0, np.sin(th)
    return W + np.outer(u, cs * x - sn * f2) + np.outer(g2, sn * x + cs * f2)


def step_geo(W, x, y):
    u = W @ x
    c = np.clip(u @ y, -1.0, 1.0)
    if c < -0.999999:
        return W.copy()
    w = W.T @ y
    q1 = ((1 + 2 * c) * x - w) / (1 + c)
    q2 = -(w + x) / (1 + c)
    return W + np.outer(y, q1) + np.outer(u, q2)


STEP = {"gd": step_gd, "pgd": step_pgd, "geo": step_geo}

# --- data: target rotation, sample stream, per-method iterate histories ---
rng = np.random.default_rng(SEED)
A = rot([0.25, 1.0, 0.4], 2.35)
xs = rng.standard_normal((STEPS, 3))
xs /= np.linalg.norm(xs, axis=1, keepdims=True)
ys = xs @ A.T

hist = {m: [np.eye(3)] for m in METHODS}
for t in range(STEPS):
    for m in METHODS:
        hist[m].append(STEP[m](hist[m][-1], xs[t], ys[t]))
errs = {m: [np.linalg.norm(W - A) for W in hist[m]] for m in METHODS}

# --- reference object: trefoil knot, colored along its length ---
s_par = np.linspace(0, 2 * np.pi, 360)
knot = 0.42 * np.stack([np.sin(s_par) + 2 * np.sin(2 * s_par),
                        np.cos(s_par) - 2 * np.cos(2 * s_par),
                        -np.sin(3 * s_par)], axis=1)
seg_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(knot) - 1))

VIEW = rot([1, 0, 0], -1.1) @ rot([0, 0, 1], 0.55)


def project(points):
    q = points @ VIEW.T
    return q[:, :2], q[:, 2]


def knot_segments(W):
    """Depth-sorted 2-D segments and colors of the knot under W."""
    pts, depth = project(knot @ W.T)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    order = np.argsort((depth[:-1] + depth[1:]) / 2)
    return segs[order], seg_colors[order]


# --- figure ---
fig = plt.figure(figsize=(9.4, 5.4), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[3.1, 1.3])
fig.suptitle("Learning a rotation from data — one pair $(x,\\ y = Ax)$ per step",
             fontsize=12)

panels = {}
for k, m in enumerate(METHODS):
    ax = fig.add_subplot(gs[0, k])
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{TITLES[m]}\n{SUB[m]}", fontsize=10, color=COLORS[m])
    circle = plt.Circle((0, 0), 1.0, fill=False, color="0.85", lw=0.8)
    ax.add_patch(circle)
    ghost_segs, ghost_cols = knot_segments(A)
    ghost_cols = ghost_cols.copy()
    ghost_cols[:, 3] = 0.18
    ax.add_collection(LineCollection(ghost_segs, colors=ghost_cols, lw=2.4))
    moving = LineCollection([], lw=2.0, zorder=3)
    ax.add_collection(moving)
    y_mark, = ax.plot([], [], "*", color="k", ms=11, zorder=5)
    x_mark, = ax.plot([], [], "o", mfc="none", mec="0.55", ms=6, zorder=5)
    wx_mark, = ax.plot([], [], "o", color=COLORS[m], ms=6, zorder=5)
    link, = ax.plot([], [], "--", color="0.55", lw=0.9, zorder=4)
    panels[m] = dict(moving=moving, y=y_mark, x=x_mark, wx=wx_mark, link=link)

ax_err = fig.add_subplot(gs[1, :])
ax_err.set_yscale("log")
ax_err.set_xlim(0, STEPS)
ax_err.set_ylim(min(errs["geo"]) * 0.5, max(errs["gd"]) * 1.6)
ax_err.set_ylabel(r"$\Vert W - A \Vert_F$", fontsize=9)
ax_err.set_xlabel("samples seen", fontsize=9)
ax_err.tick_params(labelsize=8)
ax_err.grid(alpha=0.25)
err_lines, err_dots = {}, {}
for m in METHODS:
    err_lines[m], = ax_err.plot([], [], color=COLORS[m], lw=1.6,
                                label=TITLES[m][4:])
    err_dots[m], = ax_err.plot([], [], "o", color=COLORS[m], ms=4)
ax_err.legend(loc="lower left", fontsize=8, ncol=3, frameon=False)

TOTAL = HOLD0 + len(SCHEDULE) + HOLD1


def draw(frame):
    if frame < HOLD0:
        t, s, sample = 0, 0.0, None
    elif frame < HOLD0 + len(SCHEDULE):
        t, s = SCHEDULE[frame - HOLD0]
        sample = t
    else:
        t, s, sample = STEPS - 1, 1.0, None

    for m in METHODS:
        W = tween(hist[m][t], hist[m][t + 1], s, rigid=m != "gd") if s > 0 \
            else hist[m][0]
        segs, cols = knot_segments(W)
        panels[m]["moving"].set_segments(segs)
        panels[m]["moving"].set_colors(cols)
        if sample is None:
            for key in ("y", "x", "wx", "link"):
                panels[m][key].set_data([], [])
        else:
            x, y = xs[sample], ys[sample]
            (px,), _ = project(x[None])
            (py,), _ = project(y[None])
            (pwx,), _ = project((W @ x)[None])
            panels[m]["x"].set_data([px[0]], [px[1]])
            panels[m]["y"].set_data([py[0]], [py[1]])
            panels[m]["wx"].set_data([pwx[0]], [pwx[1]])
            panels[m]["link"].set_data([pwx[0], py[0]], [pwx[1], py[1]])

    shown = t + 1 if s >= 0.999 else t
    for m in METHODS:
        err_lines[m].set_data(range(shown + 1), errs[m][:shown + 1])
        err_dots[m].set_data([shown], [errs[m][shown]])
    return []


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "learning_rotations.gif")
    anim = FuncAnimation(fig, draw, frames=TOTAL, interval=1000 / FPS)
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=80)
    print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, {TOTAL} frames")
    print({m: f"{errs[m][-1]:.2e}" for m in METHODS})
