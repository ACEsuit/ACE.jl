## Refactoring of Internals

### Evaluation of the Basis

The basis functions are defined as follows:
```
 ϕ_klm(R) = P_k(r) Y_lm(R̂)          #    k, l, m :: Integer
   A_zklm = ∑_{zⱼ=z} ϕ_klm(R_j)     #          z :: Integer
   A_𝐳𝐤𝐥𝐦 = ∏_a A_zₐkₐlₐmₐ            #   𝐳, 𝐤, 𝐥, 𝐤 :: Tuple{Int} or Vector{Int}
   Bˢⁱ_𝐳𝐤𝐥 = ∑_𝐦 Dⁱ_𝐥𝐦 A_𝐳𝐤𝐥𝐦
```
The s-superscript denotes the species of the center-atom. It is only implicit
in that the basis functions are the same for each species, but the coefficents
are of course species-dependent.

TODO:
 - allow different basis functions for each species

The klm values are restriced as follows:
* For every k,l, the m values range through -l:l.
* deg(k, l) <= maxdeg  where maxdeg is a prescribe degree. In practise this
is usually something like k + wY l <= maxdeg, but something more general is
possible

In practise we therefore proceed as follows:
 1. evaluate all  A_zklm
 2. evaluate all  A_𝐳𝐤𝐥𝐦
 3. evaluate  (Bˢⁱ_𝐳𝐤𝐥)_n = D * (A_𝐳𝐤𝐥𝐦)
    where D is a sparse matrix encoding the Dⁱ_𝐥𝐦 coefficients

The above section makes the actual evaluation of the basis straightforward, but
shifts all the work into the precomputation of the necessary datastructures.

### A_zklm (`Alist.jl -> struct AList`)

Starting from a list of all (𝐳, 𝐤, 𝐥), `Alist` computes a list of all possible
(𝐳, 𝐤, 𝐥, 𝐦) and from those a list of all possible (z, k, l, m). This is stored
in an `AList`, which provides the mapping `i -> zklm` as well as the inverse
mapping `zklm -> i`. The `i -> zklm` mapping is used to compute the `A` vector,
roughly as follows
```
# Alist.jl:precompute_A!
fill!(A, 0)
for (R, Z) in current_neighbourhood
   Φ = evaluate_basis(R)  # {ϕ_klm}
   for i = 1:length(alist)
      zklm = alist[i]
      if zklm.z == Z
         A[i] += Φ[zklm.k, zklm.l, zklm.m]
      end
   end
end
```

### A_𝐳𝐤𝐥𝐦 (`Alist.jl -> struct AAList`)

The second datastructure `struct AAList` computes the products A_𝐳𝐤𝐥𝐦 from the
precomputed scalars A_zklm. To generate A_𝐳𝐤𝐥𝐦 we take a list of all
possible (𝐳, 𝐤, 𝐥), generate a list of all possible (𝐳, 𝐤, 𝐥, 𝐦). These are
stored with references to an `alist::AList`. Say, `aalist::AAList`, and
suppose
```
aalist.i2Aidx[i, :] == [n₁, n₂, ...]
```
then this row of the Matrix specifies the basis function
```
 A_𝐳𝐤𝐥𝐦 = ∏_a A_zₐkₐlₐmₐ
```
where `(zₐ, kₐ, lₐ, mₐ) == alist[nₐ]`. Once `alist::AList` has assembled
the Vector `A`, then the products can be easily assembled into a second
Vector `AA` as follows
```
# Alist.jl:precompute_AA!
fill!(AA, 1)
for i = 1:length(aalist)
   for α = 1:bodyorder_i
      iA = aalist.i2Aidx[i, α]
      AA[i] *= A[iA]
   end
end
```

### Bˢⁱ_𝐳𝐤𝐥 (`basis.jl` -> `evaluate!`)

To define the Bˢⁱ_𝐳𝐤𝐥, we precompute the rotation-coefficients and the assemble
them into a sparse matrix. To be specific, for each species z, we compute
a separate `AList, AAList` and `A2B` matrix. The basis computation can then
simply be achieved as follows:
```
precompute_A!(  A[iz0], params)
precompute_AA!(AA[iz0], params)
B = A2B[iz0] * AA[iz0]
```
Because all look-up is precomputed this is quite fast, and with zero
allocations.


## Temporary Documentation of Internals [OLD VERSION]

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
