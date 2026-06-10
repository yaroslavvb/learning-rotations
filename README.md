# Learning Rotations Online

An unknown rotation $A \in SO(d)$ is observed through a stream of pairs $(x_t, y_t = A x_t)$, with $x_t$ uniform on the unit sphere. The estimate $W$, starting at $W_0 = I$, absorbs one pair per step. The same piece of information — that $W$ should map $x$ to $y$ — can be inserted in three ways, and the amount of rotation structure each update exploits sets its convergence rate.

## Three ways to insert information

### 1. Gradient descent — few assumptions

```math
W \leftarrow W + (y - Wx) x^\top
```

The minimum-change update satisfying $W'x = y$ (a Kaczmarz step): it works for any matrix, uses nothing about rotations, and lets $W$ drift off $SO(d)$. The error obeys the exact identity $W' - A = (W - A)(I - xx^\top)$, so the expected squared error contracts by exactly $1 - 1/d$ per step, from any starting point.

### 2. Gradient descent + projection to the orthogonal group

```math
W \leftarrow \mathrm{polar}_{SO(d)} \left( W + (y - Wx) x^\top \right)
```

The same step, then snap back to the nearest rotation (the polar factor). The projection keeps only the rotational part of the correction, and the combined update has a clean closed form: rotate $W$ in the plane spanned by $Wx$ and $y$ by **half** the angle between them — exactly a half-step of update 3, globally, not just near convergence. Asymptotic contraction: $1 - 3/(2d)$ per step.

### 3. Specialized update for orthogonal matrices

```math
W \leftarrow R W, \qquad R = I + K + \frac{K^2}{1 + c}, \qquad K = y u^\top - u y^\top, \quad u = Wx, \quad c = u \cdot y
```

Rotate $W$ by the geodesic rotation $R$ that carries $Wx$ exactly onto $y$ (defined for $u \neq -y$, an event of probability zero): the full angle, never leaving $SO(d)$. Writing $W = (I + \Omega)A$ with $\Omega$ small and skew-symmetric, one step gives, to first order in $\Omega$, $\Omega' = (I - P) \Omega (I - P)$ with $P = yy^\top$ — the rotation corrects the error two-sidedly, doubling the exponent. Asymptotic contraction: $1 - 2/d$ per step.

## Result

| Update | Structure used | Per-step contraction of $\mathbb{E} \Vert W - A \Vert_F^2$ |
|---|---|---|
| Gradient descent | none | $1 - 1/d$ (exact, every step) |
| GD + projection | $A$ orthogonal | $1 - 3/(2d)$ (asymptotic) |
| Geodesic update | $A$ orthogonal | $1 - 2/d$ (asymptotic) |

The rates sit in ratio **1 : 1.5 : 2** — fully exploiting the constraint doubles the convergence exponent, and projecting after a generic step recovers exactly half of that gain. A constant-factor error reduction costs $O(d)$ samples, so the natural clock is $\tau = t/d$, in which the normalized distance $\Vert W_t - A \Vert_F / \Vert W_0 - A \Vert_F$ approaches $e^{-c\tau/2}$ (equivalently, the squared error goes as $e^{-c\tau}$).

![Convergence of the three updates at d = 128 against theory](large_d_collapse.png)

Simulation at $d = 128$ (256 trials, $A$ a $\pi/2$ isoclinic rotation, $W_0 = I$): gradient descent sits on its theory line at every step; the two constrained methods run parallel to theirs with a small constant offset (about 1.3×) picked up during the early nonlinear transient, while $W$ is still far from $A$.

## Implementation

All three updates are rank-one or rank-two corrections of $W$, so each step costs $O(d^2)$ with no SVD: the projection in update 2 uses its half-angle closed form, and error norms use exact $O(d)$ identities. Independent trials run in parallel across cores; the figure reproduces in about 2 seconds on an Apple M-series laptop.

```sh
python large_d_collapse.py     # simulate + plot
python benchmark_d128.py       # equivalence test vs SVD reference + timings
```
