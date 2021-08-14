
# Developer Documentation / Internals 

!!! warning "WARNING"
    This documentation is very much a work in progress


!!! note "TODO"
    need to properly specify the interface for all the main components

## Summary of types and type hierarchy

The `ACE.jl` package heavily utilizes composition (as opposed to inheritance), which is well aligned with Julia's type system and recommended style. Basis sets and calculators are built from the following two base types:

* `OneParticleBasis` : abstract supertype of a 1-particle basis
* `PIBasis` : concrete implementation of a permutation-invariant basis, employing a `OneParticleBasis` and a specification of all possible correlations
* `SymmetricBasis` : implementation of the "coupling" to achieve O(3) symmetries
* `LinearACEModel` : representation of one or more properties in terms of a basis.


## States and Configurations (Inputs)

### States 

Each particle is described by one or more variables, including e.g. its
position, species, spin, charge, etc. The input space ``\mathbb{X}`` is simply the
space in which those variables reside. The input variable must be a subtype `AbstractState` following some strict conventions.

In practise one would most likely use the `State` type which simply wraps a `NamedTuple`. E.g. if we wanted a particle that only has a position we could define this as 
```julia
X = State(rr = rand(SVector{3, Float64}))
```
If we have a particle that has position `rr` and species `Z` attributes, and maybe an invariant feature `u`, then we would define this as 
```julia 
X = State(rr = rand(SVector{3, Float64}), Z = 13, u = rand())
```
The data can be accessed via `.` or `getproperty`.

It is crucial that the attributes/features `rr, Z, u` are known to the one-particle basis, i.e. when evaluating ``\phi_v(X)`` the one-particle basis ``\phi_v`` must "know" that it can obtain the position by calling `X.rr`; more on this below. 

### Configurations

A collection of *states* is a configuration. The supertype for configurations is `AbstractACEConfiguration`. The simples concrete implementatin is `ACEConfiguration` which simply wraps a `Vector{<: AbstractState}`. Configurations must be iterable. Example: 

```julia
cfg = ACEConfig( [ State(rr = randn(SVector{3, Float64})) for _=1:10 ])
```
creates a configuration containing 10 particles having only a position as an attribute.

### DStates 

While a `State` is just a description of an object, a `DState` can be thought of as an element of a vector space which we can manipulate. A classical analogy is that a `State` might be a point while a `DState` a vector pointing to it. E.g. we can add two `DState`s or multipy them with a scalar. This is not allowed for a `State`. This places certain restrictions on what a `DState` might contain. 

The main application of this in ACE.jl is that `DState`s are derivatives of `State`s. 

**TODO:** write a more thorough explanation and how they are used and constructed. 

## One Particle Basis

The one-particle basis is arguably the most important object in defining an ACE model. It is the fundamental building block for reprenting an input, before the generic methods/algorithms for correlatons and symmetrisation take over.

A one-particle basis is a basis of functions ``\phi_v : \mathbb{X} \to \mathbb{C}`` (or, ``\mathbb{R}``) defined through a subtype of
```julia
abstract type OneParticleBasis end
```
Concrete subtypes must implement the projection of the atom density onto the one-particle basis, 
```math
  A_{v}( \{ X_j \}_{j} ) = \sum_{j} \phi_v(X_j).
```
This is done with the following interface:
```julia
A = ...      # allocate storage for A = [ A_z for iz=1:NZ ]
evaluate!(A, basis1p, cfg)    # fill A = [ A_z for iz=1:NZ ]
```
In practise this would more conveniently be called via
```julia 
A = evaluate(basis1p, cfg)
```
with the allocation occuring behind the scenes. Normally, `basis1p` will have an object pool implemented, then the array `A` when no longer needed can be returned to the pool via 
```julia 
release_B!(basis1p, A)
```
For example when 1p-basis evaluate occurs as part of the full ACE model, then all allocations occur at a suitable point in the evaluation chain. The user need not be concerned about this. 

There is a lot of code duplication in the implementation of `OneParticleBasis`, which we can avoid by a generic implementation of `evaluate!` which loops through all `X in Xs` and then calls
```julia
add_into_A!(A, basis1p, X)
# should be equivalent to A[:] += evaluate(basis1p, X)
```
an implementation of `OneParticleBasis` then only needs to overload `add_into_A!` which evaluates all ``\phi_v`` at one state pair `X` and adds the basis values into a pre-allocated vector `A`.


Although in most cases AD and backpropagation will be used to take gradients, for performance reasons it is important to have hand-coded gradients for the 1p-basis implementations. For the gradients w.r.t. a single state the following must be provided:
```julia
dPhi =  .....                     # storage for (∇ϕ_k)_k
evaluate_d!(dPhi, basis1p, X)     # fill dPhi with (∇ϕ_k)_k
```
The interface does not require `evaluate_d!(dPhi, basis1p, cfg)`, which is done behind the scenes (see `oneparticlebasis.jl`). 

### Basis Indexing

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

### Notes

!!! note "Bonds vs Sites"
      The theory allows some extensions that are currently only accessible through an ad hoc "hack", but which may turn out to be the best strategy to implement them anyhow: The most important case is that when modelling e.g. an atomic environment, the one-particle basis may also depend on the center-atom, which could be written as 
      ```math 
         A_v = \sum_{j} \phi_v(X_j, X_i).
      ```
      This is not directly supported. Instead one should simply identify a state ``X_j`` with the state of the bond ``(X_i, X_j)`` and include the attributes of the center-atom in ``X_j``. For example, if ``\phi_v`` depends also on the species of atoms ``i, j``, this might take the form 
      ```julia 
         X_j = State(rr = position[j] - position[i], Z = Z[j], Z0 = Z[i])
      ```


!!! note "TODO"
    It could be worth enabling the possibility to overload `evaluate_d!(dPhi, basis1p, cfg)` for faster (e.g. AVXd) evaluation.

!!! warning "WARNING"
    The gradient interface is not really done yet and may need more design work! The issue remaining is to decide how to manage the situation that gradient with only specific attributes of a state might be required but not w.r.t. the entire state. 



### Product 1p-Basis

The most common situation is that `basis1p` is a product of basis functions acting on different variables. This can be constructed using `Product1pBasis`. For example, a one-particle basis of the kind
```math
   \phi_{\mu n l m}(X) = \delta(\mu_X - \mu) R_n(r_X) Y_l^m({\bm r}_X)
```
can be constructed as
```julia
Bμ = Species1PBasis(species)
Rn = Rn1pBasis(ACE.Utils.radial_basis())
Ylm = Ylm1pBasis(10)
basis1p = Product1pBasis( (Bμ, Rn, Ylm) )
```
Components from which to build a `Product1pBasis` are listed below.


### Concrete Implementations of One-particle Bases

!!! note "TODO"
    Provide a list of all 1p-basis implementations to build from

* `Rn1pBasis`
* `Ylm1pBasis`
* `Scal1pBasis`
* `ACEatoms.jl` provides also a species-1p-basis
* wip: discrete, one-hot, ...

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
