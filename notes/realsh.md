
Suppose we are given an atom-centered potential in the form (ignoring the radial contribution)
$$
  f = \sum_{\bf lm} c_{\bf lm} \prod_{\alpha = 1}^N A_{l_\alpha}^{m_\alpha}
$$
where the coefficients $c_{\bf lm} \in \mathbb{C}$ and
$$
  A_{l}^m = \sum_j Y_l^m({\bf r}_j),
$$
with $Y_l^m$ the spherical harmonics. We can define real spherical harmonics $Y_{lm}$ via
$$
  Y_l^m =
  \begin{cases}
    \frac{1}{\sqrt{2}} \Big(Y_{l |m|} - i Y_{l, -|m|}\Big),
        & m > 0, \\
    Y_{l0},
        & m = 0, \\
    \frac{(-1)^m}{\sqrt{2}} \Big( Y_{l |m|} + i Y_{l, -|m|} \Big),
        & m > 0
    \end{cases}
$$
This is called the Condon-Shortley convention. Other ways to generate a real basis are equally ok. The real SHs also form an orthonormal basis of $L^2$ on the sphere.

This note describes how to convert to a description of $f$ that uses the real spherical harmonic densities,
$$
  f = \sum_{\bf lm} \tilde{c}_{\bf lm} \prod_{\alpha = 1}^N A_{l_\alpha m_\alpha},
$$
where
$$
  A_{lm} = \sum_{j} Y_{lm}({\bf r}_j).
$$


From the relationship connecting $Y_l^m$ and $Y_{lm}$ we can write
$$
  A_l^m =
  \begin{cases}
    \frac{1}{\sqrt{2}} \Big(A_{l |m|} - i A_{l, -|m|}\Big),
        & m > 0, \\
    A_{l0},
        & m = 0, \\
    \frac{(-1)^m}{\sqrt{2}} \Big( A_{l |m|} + i A_{l, -|m|} \Big),
        & m > 0
    \end{cases}
  = \alpha A_{l |m|} + i \beta A_{l, -|m|}
$$
Then for a product we have
$$
  \prod_{j = 0}^N
  A_{l_j}^{m_j}
  = \prod_{j = 0}^N
  \alpha_j A_{l_j |m_j|} + i \beta_j A_{l_j, -|m_j|}
$$
Note that $j = 0$ corresponds to the coefficient in front of the $\prod_{j = 1}^N A_{l_j}^{m_j}$ terms.
Expanding this product we get
$$
  \prod_{j = 0}^N \alpha_j A_{l_j |m_j|}
  - \prod_{j = 0}^{N-2}
  \alpha_j A_{l_j |m_j|} \prod_{j = N-1}^{N-2} \beta_j A_{l_j, -|m_j|}
  \pm \dots
$$
There is probably a  simple enumerate of all the terms arising in this product together with the relevant coefficients. The current implementation is incredibly naive though and just uses `sympy` to compute total expression, then extracts the coefficients in from of each monomial. These coefficients are then written into the new basis. 
