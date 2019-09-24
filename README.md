# SHIPs.jl

An implementation of rotation and permutation invariant function approximation
using a spherical harmonics basis, based on

   Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). doi:10.1103/PhysRevB.99.014104


[![Build Status](https://travis-ci.com/cortner/SHIPs.jl.svg?branch=master)](https://travis-ci.com/cortner/SHIPs.jl)
[![Codecov](https://codecov.io/gh/cortner/SHIPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cortner/SHIPs.jl)


---

## Temporary Documentation of Internals

**this is out of date - will update soon**

### Transformed Jacobi Polynomials

We start from standard Jacobi polynomials `Jn(x)`; our implementation simply follows the description on [Wikipedia](https://en.wikipedia.org/wiki/Jacobi_polynomials). The recursion coefficients are generated using `big` (`BigFloat` and `BigInt`). This may be unnecessary and should be investigated at some point. We plan to move to an arbitrary basis of orthogonal polynomials, which should in fact simplify the code.

The `Jn(x)` only form the starting point. To construct the `r`-basis, we transform them as follows:

- transform a distance r to a transformed distance `t(r)`; this is the "distance-transform" and stored in `TransformedJacobi.trans`
- Then we set `x = -1 + 2*(t-tl)/(tu-tl)` which linearly transforms `t` to `[-1,1]` with 1 always corresponding to the cut-off radius `ru`.
- Then we evaluate the Jacobi-polynomials Jn(x) taken w.r.t to an inner product C(x) = (1-x)^a (1+x)^b.
- This C(x) also acts as a cut-off! That, Pn(x) = sqrt(C(x)) Jn(x) are orthogonal w.r.t. the L2-inner product and for a, b > 0 are zero at the end-points. (Ack: this is an idea due to Markus Bachmayr.)
- Finally, these transformed and cut-off-multiplied polynomials Pn(x) form our basis functions in the `r` variable.

### Spherical Harmonics

We implement standard complex spherical harmonics; our code is a straightforward modification of [SphericalHarmonics.jl](https://github.com/milthorpe/SphericalHarmonics.jl), including fixing some type instabilities for speed. This package follows the same design principle as most of the `SHIPs.jl` code of using buffer arrays for various precomputations and fast computation of multiple basis functions at the same time.

There is another spherical harmonics package which we did not know about when starting to write `SHIPs.jl` but which we should look at to see whether it contains useful ideas [SphericalHarmonics.jl](https://github.com/hofmannmartin/SphericalHarmonics.jl).

### `SHIPBasis`

The basis functions are defined as follows:
```
ϕ_klm(R) = P_k(r) Y_lm(R̂)          #    k, l, m :: Int
A_zklm = ∑_{zj = z} ϕ_klm(R_j)     # z, k, l, m :: Int
Bˢ_zkl = ∑_m C_lm ∏_a A_zₐkₐlₐmₐ   # k, l, m :: Tuple{Int} or Vector{Int}
```
The s-superscript denotes the species of the center-atom. It is only implicit
in that the basis functions are the same for each species, but the coefficents
are of course species-dependent.

The klm values are restriced as follows:
* For every k,l, the m values range through -l:l.
* deg(k, l) <= maxdeg  where maxdeg is a prescribe degree. In practise this
is usually k + wY l <= maxdeg, but a more general form may be implemented
by specifying an `BasisSpec` degree type. See, e.g.,
* `SparseSHIP`
* `HyperbolicCross`

For more information  on how a `SHIPBasis` is constructed and stored, see
`?SHIPBasis`.

### `SHIP` : the fast implementation of a SHIP calculator

While, for training, we use the `SHIPBasis` type, for simulation we convert
this to the `SHIP` type. The idea is to rewrite the site energy as
```
E = ∑_𝐳𝐤𝐥𝐦 c_𝐳𝐤𝐥𝐦 ∏_a A_zₐkₐlₐkₐ
```
and avoid the inner loop over `m` (for given 𝐤,𝐥). The coefficients
`c_𝐤𝐥𝐦` are precomputed in the construction of the `SHIP` type, in particular
no more Clebsch-Gordan evaluations are required after this.

The key point however is that computing the gradients of all basis functions is
much more expensive than computing the gradient of a site energy using the above
identity. Gradients based on the `SHIP` type are very efficient to compute as
follows (pseudo-code)
```
(1) Compute {A_k}, A_𝐳𝐤𝐥𝐦 = ∏ A_zklm
(2) Compute ∂A_𝐳𝐤𝐥𝐦 / ∂A_{zₐkₐlₐmₐ}
(3) For all j: compute ∇_Rj ϕ_{klm}
    (precomputing ∇_rj J_k and ∇_Rj Y_lm is in fact enough)
(4)     Compute ∂A_𝐳𝐤𝐥𝐦 / ∂A_{zₐkₐlₐmₐ} * ∇_Rj ϕ_{kₐlₐmₐ} for zj = zₐ
```

---

---------------------------------------------------------------------------
 ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
 Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
 All rights reserved.
 Contact the author to obtain a license.
---------------------------------------------------------------------------