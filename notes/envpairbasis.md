

# Pair Potential with Environment

To be used either as an environment-dependent pair potential for modelling
PES, or as an environment-dependent bond integral for TB models.

## Specification of the basis

The bond is defined by a vector $\bar{\bf r}$,and the environment by vectors $\{ {\bf r}_j \}_{j = 1}^J$. We assume that $\bar{\bf r}$ is vector between an atom at position ${\bf 0}$ and $\bar{\bf r}$. The vectors ${\bf r}_j$ are therefore relative to ${\bf 0}$.

Alternatively, we could think of $\bar{\bf r}$ as describing the bond between two atoms at position $\pm \frac12 \bar{\bf r}$ and the ${\bf r}_j$ being distance vectors from ${\bf 0}$ which is now the bond mid-point.

We think of the potential as being of the form
$$
   V\big( \bar{\bf r}; \{ {\bf r}_j \}_{j} \big)
   = \sum_N \sum_{j_1 < \dots < j_N}
   V_N\big(\bar{\bf r}; \{ {\bf r}_{j_a} \}_{a = 1}^N \big).
$$
We now construct a cylindrical coordinate system $(r_j, \theta_j, z_j)$ via
$$\begin{align*}
  \bar{r} &= |\bar{\bf r}| \\
  {\bf r}_j &= r_j \cos \theta_j {\bf e}_x + r_j \sin\theta_j {\bf e}_y
               + z_j {\bf e}_z,
\end{align*}$$
where the orthonormal frame ${\bf e}_x, {\bf e}_y, {\bf e}_z$ are defined by
$$
   {\bf e}_z = \frac{\bar{\bf r}}{\bar r},
$$
and is otherwise chosen arbitrarily. The choice of ${\bf e}_x,{\bf e}_y$ are therefore only unique up to a rotation about the ${\bf e}_z$ axis, but since all quantities of interest will be rotation-invariant, this will not affect the results. (hopefully, depends on numerical stability!)

We now rewrite $V_N$ in the form
$$
   V_N = V_N\big(\bar r; \{ {\bf c}_{j_a} \}_{a = 1}^N \big),
   \qquad {\bf c}_j := (r_{j}, \theta_{j}, z_{j})
$$
We expand into a polynomial basis,
$$
   V_N \sim \sum_{\bf k, \bf l, \bf m}
   \theta_{\bar{m}, \bf klm}
   \bar{P}_{\bar{m}}(\bar{r}) \times \prod_{a = 1}^N P^r_{k_a}(r_{j_a}) e^{i l_a \theta_{j_a}} P^z_{m_a}(z_{j_a}).
$$
and apply the density trick,
$$\begin{align*}
   \mathcal{V}_N &:= \sum_{j_1 < \dots < j_N}
   V_N\big(\bar{\bf r}; \{ {\bf r}_{j_a} \}_{a = 1}^N \big)  \\
   &\sim
   \sum_{\bf k, \bf l, \bf m}
   \theta_{\bar{m}, \bf klm}
   \prod_{a = 1}^N
   A_{\bar{m} k_a l_a m_a}, \\
   %
   A_{\bar{m} klm} &=
      \sum_{j = 1}^J \phi_{\bar{m}k l m}(\bar{r}, {\bf c}_j), \\
   \phi_{\bar{m}k l m}(\bar{r}, {\bf c}_j)
      &= \bar{P}_{\bar{m}}(\bar{r}) P^r_{k}(r_j) e^{i l \theta_j} P^z_{m}(z_{j})
\end{align*}$$
So we can simplify this to
$$\begin{align*}
   A_{\bar{m} klm}
   &=
   \bar{P}_{\bar{m}}(\bar{r})
   \sum_{j = 1}^J
   \phi_{k l m}({\bf c}_j) \\
   %
   \phi_{k l m}({\bf c}_j)
      &= P^r_{k}(r_j) e^{i l \theta_j} P^z_{m}(z_{j})
\end{align*}$$

This suggests the following assembly order
$$
   \mathcal{V}_N
   \sim
   \sum_{\bf klm} A_{\bf klm} \sum_{\bar m} \theta_{\bar{m}\bf klm} \bar{P}_{\bar{m}}(\bar{r}).
$$
But this assumes that all $\bar{m}$ have the same list of ${\bf klm}$, which misses a lot of opportunity for sparsification. So if we want to keep the option to sparsify agressively, then we should keep
$$
\mathcal{V}_N
\sim
\sum_{\bar{m} \bf klm} \theta_{\bar{m}\bf klm} \bar{P}_{\bar{m}} A_{\bf klm}
$$
To reduce storage one could still store the two arrays
$$
      (\bar{P}_{\bar m})_{\bar m}, \qquad (A_{klm})_{klm}
$$
separately rather than its tensor product.
