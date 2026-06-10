# The Geodesic Update, Derived

This note derives, in full detail, the closed form of the geodesic update used in the [main report](../README.md):

```math
W \leftarrow R W, \qquad R = I + K + \frac{K^2}{1 + c}, \qquad K = y u^\top - u y^\top, \quad u = Wx, \quad c = u \cdot y .
```

Throughout, $W \in SO(d)$ is the current estimate, $x$ is a unit input, $y = Ax$ is the observed output of the unknown rotation $A$, and therefore $u = Wx$ and $y$ are both unit vectors: $u$ is where $W$ currently sends $x$, and $y$ is where it should go. We write

```math
c = \cos\varphi = u \cdot y, \qquad s = \sin\varphi = \sqrt{1 - c^2}, \qquad \varphi \in [0, \pi),
```

for the angle $\varphi$ between $u$ and $y$, and assume $c > -1$ (the antipodal case is discussed in Section 7).

The update must repair the prediction — send $u$ to $y$ — while staying on $SO(d)$ and disturbing $W$ as little as possible. Writing the new estimate as $W' = RW$, the condition $W'x = y$ becomes $Ru = y$, so the whole problem reduces to a question about a single rotation:

> Among all $R \in SO(d)$ with $Ru = y$, which one is closest to the identity?

## 1. The smallest rotation taking $u$ to $y$

Every rotation $Q \in SO(d)$ can be written, in some orthonormal basis, as a block-diagonal matrix

```math
Q \cong \mathrm{diag}\left( R(\theta_1), \ldots, R(\theta_m), I_{d - 2m} \right), \qquad \theta_i \in (0, \pi],
```

where each $R(\theta_i)$ is a $2 \times 2$ rotation acting in its own invariant plane and $I_{d-2m}$ fixes the orthogonal complement. This is the standard normal form of a rotation; the $\theta_i$ are its principal angles.

**Lemma.** Every $Q \in SO(d)$ with $Qu = y$ satisfies

```math
\Vert Q - I \Vert_F^2 \geq 4(1 - c),
```

with equality if and only if $Q$ rotates by $\varphi$ in the plane $\mathrm{span} \lbrace u, y \rbrace$ and fixes its orthogonal complement (when $u = y$, read this as $Q = I$).

**Proof.** Two computations in the normal-form basis. First, the size of $Q$: a single block contributes $\Vert R(\theta) - I_2 \Vert_F^2 = (\cos\theta - 1)^2 \cdot 2 + 2\sin^2\theta = 4(1 - \cos\theta)$, so

```math
\Vert Q - I \Vert_F^2 = \sum_{i=1}^m 4(1 - \cos\theta_i).
```

Second, the constraint: decompose $u = u_1 + \cdots + u_m + u_0$ into its components in the invariant planes and the fixed subspace. Then $u^\top Q u = \sum_i \Vert u_i \Vert^2 \cos\theta_i + \Vert u_0 \Vert^2$, and since $Qu = y$ this equals $u^\top y = c$. Subtracting from $1 = \sum_i \Vert u_i \Vert^2 + \Vert u_0 \Vert^2$,

```math
1 - c = \sum_{i=1}^m \Vert u_i \Vert^2 (1 - \cos\theta_i) \leq \sum_{i=1}^m (1 - \cos\theta_i) = \tfrac{1}{4} \Vert Q - I \Vert_F^2 ,
```

which is the bound. Equality forces $\Vert u_i \Vert = 1$ for every plane with $\theta_i > 0$; since the $\Vert u_i \Vert^2$ sum to at most one, only a single plane can rotate, and $u$ must lie entirely inside it. The constraint then reads $1 - c = 1 - \cos\theta$, so $\theta = \varphi$, and $Qu = y$ places $y$ in the same plane — which is therefore $\mathrm{span} \lbrace u, y \rbrace$ — rotated toward $y$. $\blacksquare$

The same minimizer wins under the intrinsic metric: the geodesic distance from $I$ to $Q$ is $\big( 2\sum_i \theta_i^2 \big)^{1/2}$, and for any unit $v$, $v^\top Q v = \sum_i \Vert v_i \Vert^2 \cos\theta_i + \Vert v_0 \Vert^2 \geq \cos \theta_{\max}$, so no vector moves by more than the largest principal angle. Moving $u$ through $\varphi$ thus costs $\sum_i \theta_i^2 \geq \theta_{\max}^2 \geq \varphi^2$, with equality again only for the single-plane rotation by $\varphi$.

So the optimal $R$ is unambiguous: rotate by $\varphi$ in the plane of $u$ and $y$, touch nothing else. It remains to put this rotation into the stated algebraic form.

## 2. An adapted basis

Orthonormalize the plane. Take

```math
e_1 = u, \qquad e_2 = \frac{y - c u}{s},
```

so that $e_1 \cdot e_2 = (c - c)/s = 0$ and $\Vert e_2 \Vert^2 = (1 - 2c^2 + c^2)/s^2 = 1$, while $y = c e_1 + s e_2$. The target rotation acts as

```math
R e_1 = \cos\varphi \: e_1 + \sin\varphi \: e_2, \qquad R e_2 = -\sin\varphi \: e_1 + \cos\varphi \: e_2,
```

and as the identity on the orthogonal complement of $\mathrm{span} \lbrace e_1, e_2 \rbrace$. In particular $Ru = c u + s e_2 = c u + (y - cu) = y$, as required.

## 3. The generator $K$ and its algebra

Everything now follows from the properties of the skew matrix

```math
K = y u^\top - u y^\top .
```

Its action on a vector $v$ is $Kv = (u \cdot v) y - (y \cdot v) u$, so $K$ annihilates everything orthogonal to the plane, and on the basis vectors:

```math
K u = y - c u = s e_2, \qquad K e_2 = \frac{Ky - cKu}{s} = \frac{(cy - u) - c(y - cu)}{s} = -\frac{(1 - c^2) u}{s} = -s e_1 .
```

Dividing by $s$: the normalized generator $G = K/s$ satisfies

```math
G e_1 = e_2, \qquad G e_2 = -e_1, \qquad G v = 0 \text{ for } v \perp e_1, e_2 .
```

That is, $G$ is exactly the $90°$ "advance" operator of the plane — the infinitesimal rotation. Squaring $K$ directly:

```math
K^2 = (y u^\top - u y^\top)(y u^\top - u y^\top)
    = c \: y u^\top - y y^\top - u u^\top + c \: u y^\top
    = c (y u^\top + u y^\top) - (u u^\top + y y^\top),
```

using $u^\top y = c$ and $u^\top u = y^\top y = 1$ on each of the four products. Applying this to $u$ and to $y$:

```math
K^2 u = c(y + cu) - (u + cy) = (c^2 - 1) u = -s^2 u, \qquad K^2 y = c(cy + u) - (cu + y) = -s^2 y .
```

So $K^2$ acts as $-s^2$ times the identity on the plane and as zero off it: $K^2 = -s^2 P$, with $P$ the orthogonal projector onto $\mathrm{span} \lbrace u, y \rbrace$. Multiplying once more,

```math
K^3 = K \cdot K^2 = -s^2 K,
```

the minimal-polynomial relation that makes the exponential below collapse. Equivalently, $G^3 = -G$.

## 4. Exponentiating: Rodrigues' formula

The rotation by $\varphi$ in the plane of $G$ is the exponential $R = \exp(\varphi G)$ — this is precisely the geodesic of $SO(d)$ that starts at $I$ and turns the plane at unit speed. Split the series by parity and use $G^{2k+1} = (-1)^k G$ and $G^{2k+2} = (-1)^k G^2$ (immediate induction from $G^3 = -G$):

```math
\exp(\varphi G)
= I + \sum_{k \geq 0} \frac{(-1)^k \varphi^{2k+1}}{(2k+1)!} \: G + \sum_{k \geq 1} \frac{(-1)^{k-1} \varphi^{2k}}{(2k)!} \: G^2
= I + \sin\varphi \: G + (1 - \cos\varphi) \: G^2 .
```

This is Rodrigues' formula, valid in any dimension for a single-plane generator. Now substitute back $G = K/s$, $\sin\varphi = s$, $\cos\varphi = c$, and use the half-angle-flavored identity

```math
1 - \cos\varphi = \frac{1 - \cos^2\varphi}{1 + \cos\varphi} = \frac{s^2}{1 + c} .
```

```math
R = I + s \cdot \frac{K}{s} + \frac{s^2}{1+c} \cdot \frac{K^2}{s^2}
  = I + K + \frac{K^2}{1 + c} . \qquad \blacksquare
```

Note what happened to the basis: $e_2$ and $s$ have disappeared. The final formula is built only from $u$, $y$, and their inner product — no normalization, no square roots, no case analysis about orientation. The division by $1 + c$ is the only memory of the trigonometry.

## 5. Verification from scratch

The formula can also be checked directly, using only $K^2 = c(yu^\top + uy^\top) - (uu^\top + yy^\top)$ and $K^3 = -s^2 K$.

**It does the job.**

```math
Ru = u + Ku + \frac{K^2 u}{1+c} = u + (y - cu) - \frac{s^2 u}{1 + c} = y + (1 - c) u - (1 - c) u = y,
```

since $s^2/(1+c) = 1 - c$. As a sanity check, $Ry = y + (cy - u) - (1-c) y = 2c \medspace y - u$: rotating $y$ by a further $\varphi$ lands at angle $2\varphi$ from $u$, and indeed $\cos 2\varphi \medspace u + \sin 2\varphi \medspace e_2 = (2c^2 - 1) u + 2cs \medspace e_2 = 2c \medspace y - u$.

**It is orthogonal.** Since $K$ is skew and $K^2$ symmetric, $R^\top = I - K + K^2/(1+c)$, and

```math
R^\top R = I + \left( \frac{2}{1+c} - 1 \right) K^2 + \frac{K^4}{(1+c)^2}
        = I + \frac{1-c}{1+c} K^2 - \frac{s^2 K^2}{(1+c)^2} = I,
```

where the odd terms in $K$ and $K^3$ cancel because $K$ commutes with $K^2$, and $K^4 = K \cdot K^3 = -s^2 K^2$, so the last two terms are equal and opposite because $s^2 = (1-c)(1+c)$.

**It has determinant one.** $R$ fixes the $(d-2)$-dimensional complement pointwise and restricts to a planar rotation (determinant $1$) on $\mathrm{span} \lbrace u, y \rbrace$; alternatively, $R = \exp(\varphi G)$ is connected to the identity through rotations. Either way $\det R = +1$, so $R \in SO(d)$ and $W' = RW \in SO(d)$.

## 6. The update as the code computes it

Right-multiplying the expansion by $W$ and using orthogonality, $u^\top W = (Wx)^\top W = x^\top$, and writing $w = W^\top y$:

```math
KW = y x^\top - u w^\top, \qquad K^2 W = c (y x^\top + u w^\top) - (u x^\top + y w^\top),
```

so the full update collapses to a rank-two correction,

```math
W' = W + y q_1^\top + u q_2^\top, \qquad q_1 = \frac{(1 + 2c) x - w}{1 + c}, \qquad q_2 = -\frac{w + x}{1 + c},
```

obtained by collecting the $y(\cdot)^\top$ and $u(\cdot)^\top$ terms of $KW + K^2W/(1+c)$. This is the $O(d^2)$ form implemented in [large_d_collapse.py](../large_d_collapse.py): two matrix-vector products ($u = Wx$ and $w = W^\top y$) and a rank-two outer-product update — no $d \times d$ matrix multiplication, no SVD.

## 7. Degenerate cases

**Aligned, $c = 1$.** Then $y = u$, $K = uu^\top - uu^\top = 0$, and $R = I$: nothing to fix. The formula is continuous as $c \to 1$.

**Antipodal, $c = -1$.** Then $y = -u$ and the plane is no longer determined: every plane through $u$ contains $y$, every one of them supports a rotation by $\pi$ carrying $u$ to $y$, and the Lemma's minimizer is non-unique. The formula knows this: $K = -uu^\top + uu^\top = 0$ while $1 + c = 0$, a genuine $0/0$. Under the data model the event has probability zero, and the implementation simply skips the update when $c < -1 + 10^{-6}$ (it also clips $c$ to $[-1, 1]$ against floating-point excursions).

## Appendix A. Projected gradient descent is the half-angle rotation

The polar projection of the gradient step turns out to live in the same plane. For $W \in SO(d)$, the GD step factors as $W + (y - Wx) x^\top = W (I + a x^\top)$ with $a = W^\top y - x$, and since polar projection commutes with left multiplication by a rotation, it suffices to project $M = I + a x^\top$. $M$ is the identity off $\mathrm{span} \lbrace x, W^\top y \rbrace$, and in the orthonormal basis $f_1 = x$, $f_2 = (W^\top y - c x)/s$ its $2 \times 2$ block is

```math
B = \begin{pmatrix} c & 0 \\ s & 1 \end{pmatrix},
```

because $M f_1 = f_1 + a = c f_1 + s f_2$ and $M f_2 = f_2$. Reducing to the block is legitimate even when $\det B = c < 0$, where the polar factor of $B$ alone would be a reflection: $M$ has singular values $\sqrt{1+s}$, $\sqrt{1-s}$, and ones, so the smallest is unique, lies in the $B$-block, and is exactly where the $SO(d)$ projection $U \mathrm{diag}(1, \ldots, 1, \det UV^\top) V^\top$ applies its sign correction — the projection stays block-diagonal and restricts, on the block, to the nearest $SO(2)$ matrix to $B$. The nearest $2 \times 2$ rotation to $B$ maximizes $\mathrm{tr}\left( R(\theta)^\top B \right) = (1 + c) \cos\theta + s \sin\theta$, giving

```math
\theta^\star = \mathrm{atan2}(s, 1 + c) = \frac{\varphi}{2}
```

by the half-angle identity $\tan(\varphi/2) = \sin\varphi / (1 + \cos\varphi)$. Conjugating back by $W$ maps the plane $\lbrace f_1, f_2 \rbrace$ to $\lbrace u, y \rbrace$: projected gradient descent rotates in the same plane as the geodesic update, through exactly half the angle — at every step, however far $W$ is from the target. (Re-applying it to the same pair halves the remaining angle each time, approaching the geodesic step only in the limit.) A half-step does not cost half the benefit, though: moving by a fraction $\alpha$ of the full rotation contracts the squared error by $4\alpha - 2\alpha^2$ units of $1/d$, which is $3/2$ at $\alpha = \tfrac{1}{2}$ against $2$ at $\alpha = 1$ — the rate ratio in the report.

## Appendix B. Where the contraction $1 - 2/d$ comes from

Write the misaligned estimate as $W = (I + \Omega) A$ with $\Omega$ small and skew. Then $u = Wx = (I + \Omega) y = y + \Omega y$, and since $y^\top \Omega y = 0$,

```math
c = u \cdot y = 1, \qquad K = y u^\top - u y^\top = y y^\top \Omega^\top - \Omega y y^\top = -(P \Omega + \Omega P), \qquad P = y y^\top,
```

both exactly in this linear parametrization — the only higher-order term in $R$ is $K^2/(1+c) = K^2/2 = O(\Omega^2)$. (The parametrization itself leaves $SO(d)$ at order $\Omega^2$; for a genuine $W = \exp(\Omega) A$ the same identities pick up $O(\Omega^2)$ corrections, which land in the same bucket and do not affect the rate.) Hence $R = I - (P\Omega + \Omega P) + O(\Omega^2)$ and

```math
\Omega' = \Omega - P \Omega - \Omega P + O(\Omega^2) = (I - P) \Omega (I - P) + O(\Omega^2),
```

using $P \Omega P = (y^\top \Omega y) \medspace y y^\top = 0$. The update wipes out the error's row and column components along $y$ simultaneously — the two-sided contraction. Pointwise in $y$,

```math
\Vert (I-P) \Omega (I-P) \Vert_F^2 = \Vert \Omega \Vert_F^2 - 2 \Vert \Omega y \Vert^2
```

(skewness again: for a general matrix the right side carries $- \Vert \Omega^\top y \Vert^2$ in place of the second $- \Vert \Omega y \Vert^2$, plus a $(y^\top \Omega y)^2$ term that vanishes here), and averaging over $y$ uniform on the sphere, $\mathbb{E}\left[ y y^\top \right] = I/d$ gives $\mathbb{E} \Vert \Omega y \Vert^2 = \Vert \Omega \Vert_F^2 / d$. Therefore

```math
\mathbb{E} \Vert \Omega' \Vert_F^2 = \left( 1 - \frac{2}{d} \right) \Vert \Omega \Vert_F^2 + O(\Omega^3),
```

the per-step contraction quoted in the report. The unconstrained GD analysis kills only the column component — one factor of $(I - xx^\top)$, one unit of $1/d$ — which is exactly the gap the rotation structure buys back.
