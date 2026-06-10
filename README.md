# Learning Rotations Online

An unknown rotation $A \in SO(d)$ is observed through a stream of pairs $(x, y = Ax)$. Customizing update to take into account knowledge of $A$ improves convergence 

![Three estimators learning a rotation from streamed pairs (x, Ax)](docs/learning_rotations.gif)

## Three ways to insert information

### 1. Gradient descent — few assumptions

```math
W \leftarrow W + (y - Wx) x^\top
```

The smallest change to $W$ that makes $Wx = y$ — a Kaczmarz step. It works for any matrix and knows nothing about rotations; $W$ drifts off $SO(d)$. Each step projects the error away from one random direction, hence loss contracts $1 - 1/d$ per step.

### 2. Gradient descent + projection

```math
W \leftarrow \mathrm{polar} \left( W + (y - Wx) x^\top \right)
```

The same step, snapped back to the nearest rotation. The projection keeps only the rotational part of the correction.

### 3. A native update for rotations

```math
W \leftarrow R W, \qquad R = I + K + \frac{K^2}{1 + c}, \qquad K = y u^\top - u y^\top, \quad u = Wx, \quad c = u \cdot y
```

$R$ is the geodesic rotation carrying $Wx$ exactly onto $y$: the full angle, never leaving $SO(d)$.
## Result

| Update | Uses | Contraction of $\mathbb{E} \Vert W - A \Vert_F^2$ |
|---|---|---|
| Gradient descent | nothing | $1 - 1/d$, exact |
| GD + projection | orthogonality | $1 - 3/(2d)$ |
| Geodesic | orthogonality | $1 - 2/d$ |

The rates sit in ratio **1 : 1.5 : 2** — exploiting the constraint doubles the per-step contraction, and projecting recovers exactly half of that gain. After $t$ steps the distance has fallen by $(1 - c/d)^{t/2}$, so each constant factor of error reduction costs $O(d)$ samples:

![Convergence of the three updates at d = 128 against theory](large_d_collapse.png)

At $d = 128$, gradient descent rides its theory line exactly; the constrained methods run parallel to theirs, lifted by a small constant acquired during the early nonlinear transient.

## Implementation

Each update is a correction of rank at most two, so a step costs $O(d^2)$ — no SVD anywhere; the projection in update 2 uses its half-angle closed form. The figure reproduces in about two seconds on a laptop:

```sh
python large_d_collapse.py     # simulate + plot
python benchmark_d128.py       # equivalence vs SVD reference + timings
```
