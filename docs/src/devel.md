
# Developer Documentation

!!! warning "WARNING"
    This documentation describes what will be implemented on the `rewrite` branch, and not what is currently implemented!


## Summary of types and type hierarchy

The `ACE.jl` package heavily utilizes composition (as opposed to inheritance), which is well aligned with Julia's type system and recommended style. Basis sets and calculators are built from the following two base types:

* `OneParticleBasis` : abstract supertype of a 1-particle basis
* `PIBasis` : concrete implementation of a permutation-invariant basis, employing a `OneParticleBasis` and a specification of all possible correlations
* `SymmetricBasis` : implementation of the "coupling" to achieve O(3) symmetries

TODO: this section could be expanded significantly

## States (Input variables)

Each particle is described by one or more variables, including e.g. its
position, species, etc. The input space ``\mathbb{X}`` is simply the
space in which those variables reside. The input variable must be a subtype `AbstractState` following some strict conventions.

#### Example

The original ACE models interatomic interaction with each state describing one atom in terms of its (relative) position and species. In this case the state could be defined as follow:
```julia
struct AtomState{T} <: AbstractState
   mu::AtomicNumber
   rr::SVector{3, T}
end
```
It is crucial that the properties `mu, rr` are known to the one-particle basis, i.e. when evaluating ``\phi_v(X)`` the one-particle basis ``\phi_v`` must "know" that it can obtain the position by calling `X.rr`.


## One Particle Basis

A one-particle basis is a basis of functions ``\phi_v : \mathbb{X} \to \mathbb{C}`` (or, ``\mathbb{R}``) defined through a subtype of
```julia
abstract type OneParticleBasis end
```
Concrete subtypes must implement the projection of the atom density onto the one-particle basis:
```math
  A_{v}( \{ X_j \}_{j \neq i}; X_i )
   = \sum_{j} \phi_v(X_i, X_j),
```
where including the centre-atom ``X_i`` in the argument allows us to compute relative positions, and incorporate centre-atom information into the basis. For example, this can be used to construct a different radial basis for all species pairs, incorporating information such as atomic radii.
The "standard" evaluation of a single ``\phi_v(X; X_0)`` is of course a special case. In addition, the gradients of individual basis functions, ``\nabla \phi_v(X; X_0)`` must be provided; this gradient may be taken with respect to all continuous variables.

Assuming that `basis1p isa OneParticleBasis`, this is done with the following interface:
```julia
A = alloc_B(basis1p)               # allocate storage for A = [ A_z for iz=1:NZ ]
tmp = alloc_temp(basis1p, args...)    # allocate temporary arrays
evaluate!(A, tmp, basis1p, Xs, X0)    # fill A = [ A_z for iz=1:NZ ]
```

!!! warning "WARNING"
    The gradient interface is not really done yet and needs some design work!

For the gradients the following must be provided:
```julia
dPhi = alloc_dB(basis1p)                     # storage for (∇ϕ_k)_k
tmpd = alloc_temp_d(basis1p, args...)        # temporary storage
evaluate_d!(dPhi, tmpd, basis1p, X, X0)   # fill dPhi with (∇ϕ_k)_k
```
The interface does not require `evaluate_d!(dPhi, tmpd, basis1p, Xs, X0)`.

There is a lot of code duplication in the implementation of `OneParticleBasis`, which we can avoid by a generic implementation of `evaluate!` which loops through all `X in Xs` and then calls
```julia
add_into_A!(A, tmp, basis1p, X, X0)
```
an implementation of `OneParticleBasis` then only needs to overload `add_into_A!` which evaluates all ``\phi_v`` at one state pair `(X, X0)` and adds the basis values into a pre-allocated vector `A`.

The most common situation is that `basis1p` is a product of basis functions acting on different variables. This can be constructed using `Product1pBasis`. For example, a one-particle basis of the kind
```math
   \phi_{\mu n l m}(X) = \delta(\mu_X - \mu) R_n(r_X) Y_l^m({\bm r}_X)
```
can be constructed as
```julia
Bμ = Species1PBasisNeig(species)
Rn = Rn1pBasis(ACE.Utils.radial_basis())
Ylm = Ylm1pBasis(10)
basis1p = Product1pBasis( (Bμ, Rn, Ylm) )
```

To build a `PIBasis` (see below) the `OneParticleBasis` musts also provide methods that specify it:
```
get_spec(basis, i::Integer)   # specification of the ith basis function
get_spec(basis)               # vector containing all basis function specs
```
A basis function is specified as a `NamedTuple`. For instance, in the above
example the values of the ``\mu, n, l, m`` indices would specify ``\phi_{\mu n l m}``. Thus the basis function is specified by
```julia
   b = (μ = ..., n = ..., l = ..., m = ...)
```
This framework is particularly useful if some tuples overlap across different
components of a product one-particle basis. For example, if ``R_n`` depends
also on the species, or if the radial basis is given as ``R_{nl}``.

To let the generic code know which indices are available and what the range
of each index is the one-particle basis must implement `symbols` and `indexrange`.

### Concrete Implementations of One-particle Bases

!!! note "TODO"
    Provide a list of all 1p-basis implementations to build from


## Permutation-Invariant Basis

The permutation-invariant basis is a *concrete* type
```julia
struct PIBasis end
```
which implements the tensor-product like basis functions
```math
   {\bm A}_{\bm z \bm k}^{z_0}
   =
   \prod_{\alpha = 1}^N A_{k_\alpha}^{z_\alpha z_0},
   \qquad \text{where} \quad
   {\bm z} \in \mathbb{Z}^N, {\bm k} \in \mathbb{N}^N
```
as well as the gradients
```math
   \frac{\partial A_{\bm z \bm k}^{z_0}}{\partial {\bm r}_j}
```
The interface for this is as follows:
```julia
alloc_B(pibasis::PIBasis)
alloc_tmp(pibasis::PIBasis)
evaluate!(AA, tmp, pibasis, Rs, Zs, z0)
alloc_dB(pibasis::PIBasis)
alloc_tmp_d(pibasis::PIBasis)
evaluate_d!(dAA, tmp, pibasis, Rs, Zs, z0)
```
where the storage arrays are
* `AA::Vector{<: Number}` : to store any AA_kk^{zz, z0} with z0 fixed, i.e. the AA vector for a single site only. To use a PIBasis as the *actual* basis rather than an auxiliary one should wrap it (see bonds -- TODO!)
* `dAA::Matrix{<: JVec}` with dimension basis-length x number of particles

We don't provide a detailed description here of the implementation, since it is already the final product. But we can summarize the functionality that is provided that can be used to construct further basis sets from it.



## Generating a `OneParticleBasis` and `PIBasis` via `gen_sparse`

!!! note TODO
      explain how the basis sets are generated, and what options there are,
      discuss what a degree is etc.



## Properties and symmetries

A property ``\varphi`` is the output of an ACE model. Each property has
certain symmetries attached to it. For example, an invariant ``\varphi``
satisfies
```math
   \varphi \circ Q = \varphi
```
An equivariant Euclidean vector ``\varphi \in \mathbb{R}^3`` satisfies
```math
   \varphi \circ Q = Q \varphi.
```
An equi-variant spherical vector ``\varphi`` satisfies,
```math
   \varphi \circ Q = D(Q) \varphi.
```
and there are many more options as we move to higher-order tensors.

To model these symmetries we introduce the `SymmetricBasis` in the next
section. To generate it a property must specify what its symmetries are.

All properties must be subtypes of `AbstractProperty`. For example
```math
struct Invariant{T} <: AbstractProperty
   val::T
end
```
The actual value(s) should always be encoded in the field `val` since this
allows for generic implementation of several methods required for properties,
such as arithmetic operations.


!!! note "TODO"
      discuss the interface how the properties specify their symmetry

## The symmetric basis

A key aspect of `ACE.jl` is to treat permutation symmetry *AND* O(3) symmetries.
Given a propert ``\varphi`` which has certain symmetries attached to it we want
to generate a basis ``\mathbf{B}`` which respects these symmetries as well.
In `ACE.jl` this is provided by the `SymmetricBasis` type, which transforms
from the density correlation basis ``\mathbf{A}`` to a symmetry adapted variant
by computing all possible couplings of the spherical harmonics that produce
the desired symmetry.
```math
 {\bm B} = C \cdot {\bm A}.
```
Note this relies on a specific choice of the one-particle basis; see references. The implementation of the ``C`` coefficients in `rotations3d.jl` is based on a numerical SVD as opposed to an analytic SVD.

The `RPIBasis` type stores only two fields: the `PIBasis` and the coefficients ``C``.
