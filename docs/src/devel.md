
# Developer Documentation

!!! warning "WARNING"
    This documentation described what will be implemented on the `rewrite` branch, and not what is currently implemented!

## General Notes

 * Always use `Int` for indexing, never `Int16, Int32`, etc.
 *

## Type Hierarchy

### One Particle Basis

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


!!! note "Concrete subtypes of `OneParticleBasis`"
    Concrete subtypes of `OneParticleBasis` are

    * `BasicPSH1PBasis` : implemented and tested
    * `PSH1PBasis` : under construction
    * `BondEnv1PBasis` : implemented in old code, needs to be ported
    * `Tensor1PBasis` : not yet done

    Should revisit this and maybe add another abstract layer in-between since all of these are really tensor product bases! (Reference relevant sections below)


### Permutation-Invariant Basis

The permutation-invariant basis is a *concrete* type
```julia
struct PermInvariantBasis end
```
which implements the tensor-product like basis functions
```math
   {\bm A}_{\bm z \bm k}^{z_0}
   =
   \prod_{\alpha = 1}^N A_{z_\alpha k_\alpha}^{z_0},
   \qquad \text{where} \quad
   {\bm z} \in \mathbb{Z}^N, {\bm k} \in \mathbb{N}^N
```
as well as the gradients
```math
   \frac{\partial A_{\bm z \bm k}^{z_0}}{\partial {\bm r}_j}
```
The interface for this is as follows:
```julia
alloc_B(aabasis::PermInvariantBasis)
alloc_tmp(aabasis::PermInvariantBasis)
evaluate!(AA, tmp, aabasis, Rs, Zs, z0)
alloc_dB(aabasis::PermInvariantBasis)
alloc_tmp_d(aabasis::PermInvariantBasis)
evaluate_d!(dAA, tmp, aabasis, Rs, Zs, z0)
```
where the storage arrays are
* `AA::Vector{<: Number}` of the same length as the basis
* `dAA::Matrix{<: JVec}` with dimension basis-length x number of particles


### Concrete Bases

A concrete basis will be built from `OneParticleBasis` and `PermInvariantBasis` objects.


### Derived Potentials



## Generating a Basis



## Generic Permutation-Invariant Polynomials



## ACE Basis

The ACE basis (Atomic Cluster Expansion; Drautz 2019) is one of the main user-facing objects provided by `SHIPs.jl`.
It is constructed by reducing a permutation invariant `PermInvariantBasis` to a permutation and rotation invariant basis through a single sparse matrix-vector multiplication.
```math
 B = C \cdot {\bm A},
```
where ``B`` is the new RPI basis, ``{\bm A}`` the "inner" PI basis and ``C`` the generalised Clebsch-Gordan coefficients that achieve the rotation-invariance. This relies on a specific choice of the one-particle basis.



## Bond-Environment Potentials
