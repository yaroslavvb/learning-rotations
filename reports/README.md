# Reports

Companion reports to [Learning Rotations Online](https://yaroslavvb.github.io/learning-rotations/), deployed on GitHub Pages:

| Report | What's inside |
|---|---|
| [The trace-maximal orthogonal completion](https://yaroslavvb.github.io/learning-rotations/reports/trace-procrustes.html) | Maximizing $\mathrm{Tr} W$ over orthogonal $W$ with $XW = Y$: a one-SVD closed form, the $O(dn^2)$ compressed solver, benchmarks to $d = 3200$. Follow-up to [A. Kato's answer](https://mathematica.stackexchange.com/a/307371) on Mathematica StackExchange. |
| [Two pairs at a time](https://yaroslavvb.github.io/learning-rotations/reports/batch-two.html) | The batch-size-2 mirror of the main report: block-Kaczmarz GD, projected GD, and the native two-pair update, with per-step contractions $1 - 2/d$, $1 - 3/d$, $1 - 4/d$ and the SVD-free [two-batch Rodrigues formula](https://yaroslavvb.github.io/learning-rotations/reports/batch-two.html#rodrigues-two). |
| [Learning from noisy observations](https://yaroslavvb.github.io/learning-rotations/reports/noise.html) | Additive Gaussian noise $y = Ax + \varepsilon$: the relaxed Kaczmarz, relaxed-projection, and partial-geodesic updates, exact noise floors, the matched-rate equivalence of the two manifold methods, and step-size schedules. |

Each report's simulation lives at the repository root ([trace_procrustes.py](../trace_procrustes.py), [batch_two_collapse.py](../batch_two_collapse.py), [noise_floors.py](../noise_floors.py)) and regenerates its figures with one command.
