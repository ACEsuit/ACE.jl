

# Pair Potential with Environment

To be used either as an environment-dependent pair potential for modelling
PES, or as an environment-dependent bond integral for TB models.

## Specification of the basis

The bond is defined by a vector ``\bar{\bm r}``,and the environment by vectors ``\{ {\bm r}_j \}_{j = 1}^J``. We assume that ``\bar{\bm r}`` is vector between an atom at position ``{\bm 0}`` and ``\bar{\bm r}``. The vectors ``{\bm r}_j`` are therefore relative to ``{\bm 0}``.

Alternatively, we could think of ``\bar{\bm r}`` as describing the bond between two atoms at position ``\pm \frac12 \bar{\bm r}`` and the ``{\bm r}_j`` being distance vectors from ``{\bm 0}`` which is now the bond mid-point.

We think of the potential as being of the form
```math
   V\big( \bar{\bm r}; \{ {\bm r}_j \}_{j} \big)
   = \sum_N \sum_{j_1 < \dots < j_N}
   V_N\big(\bar{\bm r}; \{ {\bm r}_{j_a} \}_{a = 1}^N \big).
```
We now construct a cylindrical coordinate system $(r_j, \theta_j, z_j)$ via
```math
\begin{aligned}
  \bar{r} &= |\bar{\bm r}| \\
  {\bm r}_j &= r_j \cos \theta_j {\bm e}_x + r_j \sin\theta_j {\bm e}_y
               + z_j {\bm e}_z,
\end{aligned}
```
where the orthonormal frame ${\bm e}_x, {\bm e}_y, {\bm e}_z$ are defined by
```math
   {\bm e}_z = \frac{\bar{\bm r}}{\bar r},
```
and is otherwise chosen arbitrarily. The choice of ${\bm e}_x,{\bm e}_y$ are therefore only unique up to a rotation about the ${\bm e}_z$ axis, but since all quantities of interest will be rotation-invariant, this will not affect the results. (hopefully, depends on numerical stability!)

We now rewrite $V_N$ in the form
```math
   V_N = V_N\big(\bar r; \{ {\bm c}_{j_a} \}_{a = 1}^N \big),
   \qquad {\bm c}_j := (r_{j}, \theta_{j}, z_{j})
```
We expand into a polynomial basis,
```math
   V_N \sim \sum_{\bm k, \bm l, \bm m}
   \theta_{\bar{m}, \bm klm}
   \bar{P}_{\bar{m}}(\bar{r}) \times \prod_{a = 1}^N P^r_{k_a}(r_{j_a}) e^{i l_a \theta_{j_a}} P^z_{m_a}(z_{j_a}).
```
and apply the density trick,
```math
\begin{aligned}
   \mathcal{V}_N &:= \sum_{j_1 < \dots < j_N}
   V_N\big(\bar{\bm r}; \{ {\bm r}_{j_a} \}_{a = 1}^N \big)  \\
   &\sim
   \sum_{\bm k, \bm l, \bm m}
   \theta_{\bar{m}, \bm klm}
   \prod_{a = 1}^N
   A_{\bar{m} k_a l_a m_a}, \\
   %
   A_{\bar{m} klm} &=
      \sum_{j = 1}^J \phi_{\bar{m}k l m}(\bar{r}, {\bm c}_j), \\
   \phi_{\bar{m}k l m}(\bar{r}, {\bm c}_j)
      &= \bar{P}_{\bar{m}}(\bar{r}) P^r_{k}(r_j) e^{i l \theta_j} P^z_{m}(z_{j})
\end{aligned}
```
So we can simplify this to
```math
\begin{aligned}
   A_{\bar{m} klm}
   &=
   \bar{P}_{\bar{m}}(\bar{r})
   \sum_{j = 1}^J
   \phi_{k l m}({\bm c}_j) \\
   %
   \phi_{k l m}({\bm c}_j)
      &= P^r_{k}(r_j) e^{i l \theta_j} P^z_{m}(z_{j})
\end{aligned}
```

This suggests the following assembly order
```math
   \mathcal{V}_N
   \sim
   \sum_{\bm klm} A_{\bm klm} \sum_{\bar m} \theta_{\bar{m}\bm klm} \bar{P}_{\bar{m}}(\bar{r}).
```
But this assumes that all $\bar{m}$ have the same list of ${\bm klm}$, which misses a lot of opportunity for sparsification. So if we want to keep the option to sparsify agressively, then we should keep
```math
\mathcal{V}_N
\sim
\sum_{\bar{m} \bm klm} \theta_{\bar{m}\bm klm} \bar{P}_{\bar{m}} A_{\bm klm}
```
To reduce storage one could still store the two arrays
```math
      (\bar{P}_{\bar m})_{\bar m}, \qquad (A_{klm})_{klm}
```
separately rather than its tensor product.
