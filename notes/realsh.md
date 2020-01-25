
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
    \frac{1}{\sqrt{2}} \Big(Y_{l |m|} - i Y_{l, -|m|}\Big), &  m < 0,
    Y_{l0}, & m = 0, \\
    \frac{(-1)^m}{\sqrt{2}} \Big( Y_{l |m| + i Y_{l, -|m|} \Big), & m > 0.
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


I will from now on consider only the case $m < 0$, the cases $m \geq 0$ are similar. From the relationship connecting $Y_l^m$ and $Y_{lm}$ we can write
$$
  A_l^m = \frac{1}{\sqrt{2}} \Big( A_{l |m|} - i A_{l, -|m|} \Big)
$$
