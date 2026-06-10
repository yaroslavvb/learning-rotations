# Learning Rotations Online

An unknown rotation $A \in SO(d)$ is observed through a stream of pairs $(x_t, y_t = A x_t)$ with $x_t$ uniform on the unit sphere. The estimate $W$, starting at $W_0 = I$, absorbs one pair per step. The same piece of information — "$W$ should map $x$ to $y$" — can be inserted three ways, and how much of the rotation structure the update exploits determines the convergence rate.

## Three ways to insert information

**1. Gradient descent — few assumptions.**

$$W \leftarrow W + (y - Wx)\,x^\top$$

The minimum-Frobenius-change update satisfying $W'x = y$ (a Kaczmarz step). It works for any matrix and uses nothing about rotations: $W$ drifts off $SO(d)$. The error obeys an exact identity $W' - A = (W - A)(I - xx^\top)$, so the expected squared error contracts by exactly $(1 - 1/d)$ per step, from any $W$.

**2. Gradient descent + projection.**

$$W \leftarrow \mathrm{polar}_{SO(d)}\big(W + (y - Wx)\,x^\top\big)$$

Same step, then snap back to the nearest rotation (polar factor). The projection discards the component of the correction normal to the manifold and keeps the tangential half. In closed form this update is the rotation of $W$ in the plane $\mathrm{span}\{Wx, y\}$ by **half** the angle between $Wx$ and $y$ — exactly a half-step of method 3, globally, not just near convergence. Asymptotic contraction: $(1 - \tfrac{3}{2d})$.

**3. Specialized update for orthogonal matrices.**

$$W \leftarrow R\,W, \qquad R = I + K + \frac{K^2}{1+c}, \quad K = y u^\top - u y^\top, \quad u = Wx, \quad c = u \cdot y$$

Rotate $W$ by the geodesic rotation carrying $Wx$ exactly onto $y$ — the full angle, never leaving $SO(d)$. Because a rotation corrects the error on both sides of the plane, the linearized error contracts two-sidedly: $\Omega' = (I-P)\,\Omega\,(I-P)$. Asymptotic contraction: $(1 - 2/d)$.

## Result

| Update | Assumptions used | Contraction of $\mathbb{E}\|W-A\|_F^2$ per step |
|---|---|---|
| Gradient descent | none | $1 - 1/d$ (exact, all $t$) |
| GD + projection | $A$ is orthogonal | $1 - 3/(2d)$ (asymptotic) |
| Geodesic update | $A$ is orthogonal | $1 - 2/d$ (asymptotic) |

The rates sit in ratio **1 : 1.5 : 2**: fully exploiting the constraint doubles the convergence exponent, and projecting after a generic step recovers exactly half of that gain. Each constant-factor error reduction costs $O(d)$ samples, so the natural time axis is $\tau = t/d$, where the curves become $e^{-c\tau/2}$.

![simulation](large_d_collapse.png)

Simulation at $d = 128$ (256 trials, $A$ = $\pi/2$ isoclinic rotation, $W_0 = I$): GD tracks its theory line exactly; the two constrained methods run parallel to theirs with a small constant offset (≈1.3–1.4×) accumulated during the early nonlinear transient, when $W_0 = I$ is far outside the linearized regime.

## Implementation

All three updates are rank-≤2 corrections of $W$, so each step is $O(d^2)$ with no SVD — the polar projection in method 2 uses its half-angle closed form, and error norms use exact $O(d)$ identities. Independent trials run in parallel across cores; the figure reproduces in ~2 s on an Apple M-series laptop:

```
python large_d_collapse.py     # simulate + plot
python benchmark_d128.py       # equivalence test vs SVD reference + timings
```
