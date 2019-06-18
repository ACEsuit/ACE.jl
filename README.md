# SHIPs.jl

An implementation of rotation and permutation invariant function approximation
using a spherical harmonics basis, based on

   Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). doi:10.1103/PhysRevB.99.014104


[![Build Status](https://travis-ci.com/cortner/SHIPs.jl.svg?branch=master)](https://travis-ci.com/cortner/SHIPs.jl)
[![Codecov](https://codecov.io/gh/cortner/SHIPs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cortner/SHIPs.jl)

---

work in progress

---

## Developer Documentation

### Transformed Jacobi Polynomials

We start from completely standard Jacobi polynomials Jn(x); our implementation simply follows the description on [Wikipedia](https://en.wikipedia.org/wiki/Jacobi_polynomials). The recursion coefficients are generated using `big` (`BigFloat` and `BigInt`). This may be unnecessary and should be investigated at some point.

However, the `Jn(x)` only form the starting point. To construct the `r`-basis, we transform them as follows:

- transform a distance r to a transformed distance `t(r)`; this is the "distance-transform" and stored in `TransformedJacobi.trans`
- Then we set `x = -1 + 2*(t-tl)/(tu-tl)` which linearly transforms `t` to `[-1,1]` with 1 always corresponding to the cut-off radius `ru`.
- Then we evaluate the Jacobi-polynomials Jn(x) taken w.r.t to an inner product C(x) = (1-x)^a (1+x)^b.
- This C(x) also acts as a cut-off! That, Pn(x) = C(x) Jn(x) are orthogonal w.r.t. the L2-inner product and for a, b > 0 are zero at the end-points. (Ack: this is an idea due to Markus Bachmayr.)
- Finally, these transformed and cut-off-multiplied polynomials Pn(x) form our basis functions in the `r` variable.

### Spherical Harmonics

We implement standard complex spherical harmonics; our code is a straightforward modification of [SphericalHarmonics.jl](https://github.com/milthorpe/SphericalHarmonics.jl), including fixing some type instabilities for speed. This package follows the same design principle as most of the `SHIPs.jl` code of using buffer arrays for various precomputations and fast computation of multiple basis functions at the same time.

There is another spherical harmonics package which we did not know about when starting to write `SHIPs.jl` but which we should look at to see whether it contains useful ideas [SphericalHarmonics.jl](https://github.com/hofmannmartin/SphericalHarmonics.jl).

### `SHIPBasis`

The basis functions are defined as follows:
```
Z_klm(R) = P_k(r) Y_lm(R̂)      # k, l, m :: Int
A_klm = ∑_j Z_klm(R_j)         # k, l, m :: Int
B_kl = ∑_m C_lm ∏_a A_kₐlₐmₐ   # k, l, m :: Tuple{Int} or Vector{Int}
```
The klm values are restriced as follows:
* For every k,l, the m values range through -l:l.
* k + wY l <= D  where D is a prescribe total degree.

For more information  on how a `SHIPBasis` is constructed and stored, see
`?SHIPBasis`.

---------------------------------------------------------------------------
 ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
 Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
 All rights reserved.
 Contact the author to obtain a license.
---------------------------------------------------------------------------