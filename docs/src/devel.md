
# Developer Documentation

!!! warning "WARNING"
    This documentation described what will be implemented on the `rewrite` branch, and not what is currently implemented!

## General Notes

 * Always use `Int` for indexing, never `Int16, Int32`, etc.
 * There is a lot of switching between a species given by an `AtomicNumber` type and the index of that species in a list, given by an `Int`. Functions dispatch on `Int` vs `AtomicNumber` to make sure there is no confusion.

## Types and type hierarchy

The `SHIPs.jl` package heavily utilizes composition (as opposed to inheritance), which is well aligned with Julia's type system and recommended style. Basis sets and calculators are built from the following two base types:

* `OneParticleBasis` : abstract supertype of a 1-particle basis
* `PIBasis` : concrete implementation of a permutation-invariant basis

For example, a rotation-invariant site energy basis set (ACE and extensions) `RPIBasis` is built from a `PIBasis` and the coupling coefficients. The `PIBasis` itself is specified in terms of the `OneParticleBasis`.

## One Particle Basis

A one-particle basis is a basis of functions ``\phi_k : \mathbb{R}^3 \to \mathbb{R}`` defined through a subtype of
```julia
abstract type OneParticleBasis end
```
Concrete subtypes must be able to compute the projection of the atom density onto the one-particle basis:
```math
  A_{k}^{z z_0}( \{ ({\bm r}_j, z_j) \}_{j = 1}^J, z_0 )
   = \sum_{j : z_j = z} \phi_k^{z_j z_0}({\bm r}_j),
```
where ``z_0`` is the atom number of the centre-atom, and ``({\bm r}_j, z_j)`` are relative positions and atom numbers of neighbours.

The "standard" evaluation of a single ``\phi_k({\bm r}; z, z_0)`` is of course a special case. In addition, the gradients of individual basis functions, ``\nabla \phi_k({\bm r}; z, z_0)`` must be provided; this gradient is taken with respect to ``{\bm r}``.

Assuming that `basis isa OneParticleBasis`, this is done with the following interface:
```julia
A = alloc_B(basis)                # allocate storage for A = [ A_z for iz=1:NZ ]
tmp = alloc_temp(basis, args...)        # allocate temporary arrays
evaluate!(A, tmp, basis, Rs, Zs, z0)    # fill A = [ A_z for iz=1:NZ ]
```
For the gradients the following must be provided:
```julia
dPhi = alloc_dB(basis)                     # storage for (∇ϕ_k)_k
tmpd = alloc_temp_d(basis, args...)        # temporary storage
evaluate_d!(dPhi, tmpd, basis, R, z, z0)   # fill dPhi with (∇ϕ_k)_k
```
The interface does not require `evaluate_d!(dPhi, tmpd, basis, Rs, Zs, z0)`.

There is a lot of code duplication in the implementation of `OneParticleBasis`, which we can avoid by a generic implementation of `evaluate!` which loops through all `(R, z) in zip(Rs, Zs)` and then calls
```julia
add_into_A!(A[iz], tmp, basis, R, iz, iz0)
```
an implementation of `OneParticleBasis` then only needs to overload `add_into_A!` which should evaluate ``\phi_k^{z z_0}({\bm r})`` (where `R` represents ``{\bm r}``) and *add* these values into `A[k]`.
For this to work, the type of the 1-particle basis must contain a field `zlist` which implements the interface defined by `JuLIP.Potentials.ZList` and `JuLIP.Potentials.SZList`.

To build a `PIBasis` (see below) the `OneParticleBasis` musts also provide methods that specify it:
```
get_basis_spec(basis::BasicPSH1pBasis, z0::AtomicNumber)
get_basis_spec(basis::BasicPSH1pBasis, z0::AtomicNumber, i::Integer)
```
The first of these should return a `Vector` containing `OnepBasisFcn` objects that specify the list of 1-particle basis functions for a centre atom of species `z0`. The second method should return precisely the ith element of this vector. A concrete `OneParticleBasis` may either simply keep these specifications stored throughout its lifetime, or generate them on the fly, whichever is most convenient.

!!! note "Concrete subtypes of `OneParticleBasis`"
    Concrete subtypes of `OneParticleBasis` are

    * `BasicPSH1PBasis` : implemented and tested
    * `PSH1PBasis` : parameterised version of `BasicPSH1Basis`; under construction
    * `BondEnv1PBasis` : implemented in old code, needs to be ported
    * `Tensor1PBasis` : not yet done

    Should revisit this and maybe add another abstract layer in-between since all of these are really tensor product bases! (Reference relevant sections below)


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

TODO

## Generating a `OneParticleBasis` and `PIBasis` via `gen_sparse`

TODO


## Derived Potentials



## RPI Basis (ACE and Extensions)

The ACE basis (Atomic Cluster Expansion; Drautz 2019) and its modifications and extensions is one of the main user-facing objects provided by `SHIPs.jl`.
It is constructed by reducing a permutation invariant `PIBasis` to a permutation and rotation invariant basis through a single sparse matrix-vector multiplication.
```math
 B = C \cdot {\bm A},
```
where ``B`` is the new RPI basis, ``{\bm A}`` the "inner" PI basis and ``C`` the coupling coefficients that achieve the rotation-invariance. This relies on a specific choice of the one-particle basis. This construction is outlined in (Atomic Cluster Expansion; Drautz 2019) and an extended derivation with full details in (Bachmayr, Drautz, Dusson, Etter, Van der Oort, Csanyi, Ortner, arXiv:19..). The implementation of the ``C`` coefficients in `rpi/rotations3d.jl` is based on a numerical SVD as opposed to an analytic SVD.

The `RPIBasis` type stores only three fields: the `PIBasis`, the coefficients ``C``, and some index management to map the local site basis into a global basis (only needed for multiple species).

TODO: discuss the classes of 1-particle bases that are allowed.



## Bond-Environment Potentials
