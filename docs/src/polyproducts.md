
### Products of Polynomials

### Products of Radial Polynomials

Suppose we have a basis of orthonormal radial polynomials
```math
 (J_n, J_{n'})_w = \delta_{n,n'}
```
where $(\cdot, \cdot)_w$ is any inner product we choose. Product $J_{n_1} J_{n_2}$ are again polynomials and can therefore be expanded in terms of the basis $J_\nu$, i.e.,
```math
   J_{n_1} J_{n_2} = \sum_\nu P_{n_1, n_2}^\nu J_\nu.
```
Because they are orthonormal, the expansion coefficients are simply given by
```math
   P_{n_1, n_2}^\nu = ( J_{n_1} J_{n_2}, J_\nu)_w.
```

#### Implementation

The radial polynomial products are implemented in `src/polynomials/products.jl' in a lazy datastructure.
```julia
struct OrthPolyProdCoeffs
```
If `prodcoeffs::OrthPolyProdCoeffs` then calling
```julia
P = prodcoeffs(n1, n2)
```
returns a vector `P` such that `P[nu]` is the value of $P^{n_1,n_2}_\nu$, allowing indices $\nu \in \mathbb{Z}$. Coefficients outside the actual range will simply be zero. The coefficient vectors are computed lazyly, i.e., will only be precomputed when required but then stored for later use.

The precomputation is done by explicit evaluation of the inner products as described above.

Alternatively we can access $P_{n_1 n_2}^\nu$ with an iterator
```
for (nu, Pnu) in coeffs(n1, n2)
   # ...
end
```
will iterate only over the non-zero coefficients.


### The case of Chebyshev Polynomials (TODO)

The simplest case is that of monomials $J_n = x^n (x - x_l)^{p_l} (x - x_r)^{p_r)$. The product $J_{n_1} J_{n_2}$ can then by written with just five non-zero coefficients $P_{n_1 n_2}^{N}$. But monomials lead to sever numerical instabilities.

The "next-best" option in terms of sparsity of the product coefficients appear to be Chebyshev polynomials.
```math
   2 T_{m}(x) T_n(x) = T_{m+n}(x) + T_{|m-n|}(x)
```
That is if we choose
```math
   J_n = T_{n+1} f_{\rm cut}
```
then
```math
   J_{n_1} J_{n_2} = T_{n_1+1} T_{n_2+1} f_{\rm cut} f_{\rm cut}.
```
For $N_p = p_l + p_r$ we have
```math
   f_{\rm cut} = \sum_{n = 0}^{N_p} T_n,
```
hence
```math
\begin{aligned}
   J_{n_1} J_{n_2}
   &=
   \sum_{n = 0}^{N_p} T_{n_1+1} T_{n_2+1} T_n f_{\rm cut} \\
   &=
   \frac{1}{2} \sum_{n = 0}^{N_p} \big( T_{n+1+n_2+2} + T_{|n_1-n_2|} \big) T_n f_{\rm cut}  \\
   &=
   \frac{1}{2} \sum_{n = 0}^{N_p} \Big(
           T_{n+1+n_2+2 + n} + T_{|n_1+n_2+2 - n|}
         + T_{|n_1-n_2| + n} + T_{| |n_1-n_2| - n |} \Big) f_{\rm cut}.
\end{aligned}
```
This means that there are at most $4 (N_p + 1)$. non-zero coefficients, independently of the chosen maximum degree.

In practise it is quite likely that this makes little difference. We normally choose $p_l = p_r = 2$ which leads to approx. 20 non-zero coefficients.
But this is already close to the maximum degree we allow, hence for a general
basis we will see little gain. An interesting question, maybe, is whether
numerical stability is improved for a Chebyshev basis because of the
explicit analytic recursion.


### Products of spherical harmonics

A product of two spherical harmonics can again be expanded in terms of spherical harmonics,
```math
   Y_{l_1}^{m_1} Y_{l_2}^{m_2}
       = \sum_{\lambda, \mu} P_{l_1 m_1 l_2 m_2}^{\lambda \mu} Y_\lambda^\mu
```
where the "coupling coefficients" are given by
```math
   P_{l_1 m_1 l_2 m_2}^{\lambda \mu}
   =
   \sqrt{\frac{(2l_1+1)(2l_2+1)}{2\pi (2\lambda+1)} }
   C_{l_1 m_1 l_2 m_2}^{LM} C_{l_1 0 l_2 0}^{L0}
```
where $C_{l_1 m_1 l_2 m_2}^{L,M}$ are the Clebsch-Gordan coefficients. These are non-zero only for
```math
\begin{aligned}
   & |l_1 - l_2| \leq L \leq l_1 + l_2, \\
   & M = m_1 + m_2, \\
   & |M| \leq L, |m_i| \leq l_i, \\
   & l_1 + l_2 - L \text{ is even.}
```
The first three of these conditions are the conditions for $C_{l_1 m_1 l_2 m_2}^{LM}$ to be non-zero. The fourth condition follows from the fact that
```math
   C_{l_1 m_1 l_2 m_2}^{L M} =
   (-1)^{l_1 + l_2 - L} C_{l_1 (-m_1) l_2 (-m_2)}^{L (-M)}
```
and hence $C_{l_1 0 l_2 0}^{L0}$ can only be non-zero if
$l_1 + l_2 - L$ is even.

These conditions ensure that the $P_{l_1 m_1 l_2 m_2}^{L M}$ coefficients are
extremely sparse. This is exploited when iterating over all required
$L, M$.


#### Implementation

The coefficients $P_{l_1 m_1 l_2 m_2}^{\lambda \mu}$ are implemented in
the datastructure
```julia
struct SHProdCoeffs
```
To get the coefficients for a specific `l1, m1, l2, m2` we can call
```julia
P = coeffs(l1, m1, l2, m2)
```
where `coeffs::SHProdCoeffs`. To iterate over all non-zero coefficients,
```julia
for (L, M, p) in P
   # ...
end
```


### Products of one-particle basis function

```math
\begin{aligned}
   \phi_{n_1 l_1 m_1} \phi_{n_2 l_2 m_2}
   &=
   R_{n_1} Y_{l_1}^{m_1} R_{n_2} Y_{l_2}^{m_2}  \\
   &=
   \sum_N P_{n_1,n_2}^N R_N \sum_{L, M} P_{l_1m_1l_2m_2}^{LM} Y_{L}^M.
\end{aligned}
```
This means we have
```math
   \phi_{k_1} \phi_{k_2} =
   \sum_K P_{k_1 k_2}^K \phi_k
```
where $k = (n, l, m)$ and
```math
   P_{k_1 k_2}^K = P_{n_1,n_2}^N P_{l_1m_1l_2m_2}^{LM}
```
The tensor product structure can be exploited in code if many such products
have to be computed.


### Appendices


#### Three-term recurrance

If the polynomials satisfy a three-term recurrance then there is an alternative
way to obtain the product coefficients which does not require evaluating the
inner products. This is currently not implemented, but we keep it here as a
record for the future.

Suppose we have a radial polynomial basis satisfying the recursion
```math
\begin{aligned}
   J_1(x) &= A_1 (x - x_l)^{p_l} (x - x_r)^{p_r} \\
   J_2 &= (A_2 x + B_2) J_1(x) \\
   J_{n} &= (A_n x + B_n) J_{n-1}(x) + C_n J_{n-2}(x)
\end{aligned}
```
For the save of brevity, we will write $f_{\rm cut} := (x - x_l)^{p_l} (x - x_r)^{p_r}$, o.e., $J_1 = A_1 f_{\rm cut}$. The functions $J_n$ span the space of all polynomials that are multiples of $f_{\rm cut}$.

Since $f_{\rm cut}$ divides all $J_n$, if we take a product $p := J_{n}(x) J_{n'}(x)$ then $f_{\rm cut}$ also divides $p$ and in particular $p = q f_{\rm cut}$. Since $q$ is itself a polynomials it follows that
```math
 p = \sum_{\nu = 1}^{n+n'+p_l + p_r} P^{n n'}_\nu J_\nu
```
for some coefficients $P^{nn'}_\nu$. We can determine these coefficients recursively as follows.

```math
\begin{aligned}
   \sum_\nu P^{n,n'}_\nu J_\nu &= J_n J_{n'}\\
   %
   &= J_n(x) (A_{n'} x + B_{n'}) J_{n'-1} + C_{n'} J_n(x) J_{n'-2}(x)  \\
   %
   &= \sum_\nu \bigg[ A_{n'} x P^{n,n'-1}_\nu J_\nu
         + B_{n'} P^{n,n'-1}_\nu J_\nu
         + C_{n'} P^{n,n'-2}_\nu J_\nu \bigg] = \dots
\end{aligned}
```
We rewrite
```math
   x J_\nu  = \frac{1}{A_{\nu+1}} J_{\nu+1} - \frac{B_{\nu+1}}{A_{\nu+1}} J_\nu
            - \frac{C_{\nu+1}}{A_{\nu+1}} J_{\nu-1}.
```
and insert this above, and then shift the summation indices, to obtain
```math
\begin{aligned}
   \dots
   &=
   \sum_\nu \bigg[
         \frac{A_{n'} P^{n,n'-1}_\nu}{A_{\nu+1}} J_{\nu+1}
         - \frac{A_{n'} P^{n,n'-1}_\nu B_{\nu+1}}{A_{\nu+1}} J_\nu
         - \frac{A_{n'} P^{n,n'-1}_\nu C_{\nu+1}}{A_{\nu+1}} J_{\nu-1} \\
   & \hspace{2cm}
         + B_{n'} P^{n,n'-1}_\nu J_\nu
         + C_{n'} P^{n,n'-2}_\nu J_\nu \bigg] \\
   &= \sum_\nu \bigg[
         \frac{A_{n'} P^{n,n'-1}_{\nu-1}}{A_{\nu}}
         - \frac{A_{n'} P^{n,n'-1}_\nu B_{\nu+1}}{A_{\nu+1}}
         - \frac{A_{n'} P^{n,n'-1}_{\nu+1} C_{\nu+2}}{A_{\nu+2}}
         + B_{n'} P^{n,n'-1}_\nu
         + C_{n'} P^{n,n'-2}_\nu \bigg]  J_\nu  \\
\end{aligned}
```
Comparing coefficients we finally obtain
```math
      P^{n,n'}_\nu
      =
      \frac{A_{n'} P^{n,n'-1}_{\nu-1}}{A_{\nu}}
      - \frac{A_{n'} P^{n,n'-1}_\nu B_{\nu+1}}{A_{\nu+1}}
      - \frac{A_{n'} P^{n,n'-1}_{\nu+1} C_{\nu+2}}{A_{\nu+2}}
      + B_{n'} P^{n,n'-1}_\nu
      + C_{n'} P^{n,n'-2}_\nu.
```
This calculation is valid verbatim if $n' > 2$. Since $P^{n,n'}_\nu$ is summetric in $n, n'$ we can choose whichever index $n, n'$ is larger and reduce the computation of the $P^{n,n'}_\nu$ to those of smaller $(n, n')$ pairs.
To treat the case $n' = 2$ we have to make the conventions that
```math
      B_1 = C_1 = C_2 = P^{n,0}_\nu = P^{n,n'}_0 = 0.
```
This reduces the problem of precomputing the $P^{n,n'}_\nu$ coefficients to determining just $P^{1,1}_\nu$.

To determing $P^{1,1}_\nu$ (the start of the recursion) we simply use the procedure proposed for the general case, i.e. by evaluating the inner product. If the inner product is not available then one could simply fit to randomly selected data points or suitably chosen interpolation nodes.
