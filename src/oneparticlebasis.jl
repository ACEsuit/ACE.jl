
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using StaticArrays

using SHIPs.SphericalHarmonics: SHBasis, index_y
using JuLIP.Potentials: SZList

# """
# `Outer1PBasis` : datastructure to convert a list of 1p bases into a single
# multi-species 1p basis. Basically this type takes care of all multi-species
# logic so that the inner radial basis need not worry about that at all.
#
# * `Phi` : list of "inner" 1p bases
# * `zlist` : mapping iz <-> z
# * `firstA` : `firstA[iz0, iz]` stores the first index in the A_zk array for which
#              z = iz. This can be used to iterate over all A entries for which
#              z = zi. (they are sorted by z first)
# """


function evaluate!(A, tmp, basis::OneParticleBasis, Rs, Zs, z0)
   fill!(A, 0)
   iz0 = z2i(basis, z0)
   firstA = basis.firstA
   for (R, Z) in zip(Rs, Zs)
      iz = z2i(basis, Z)
      add_into_A!((@view A[firstA[iz0, iz]:(firstA[iz0, iz+1]-1)]),
                  tmp.tmpPhi, basis.Phi,
                  R, iz, iz0)
   end
   return A
end


function alloc_B end
function alloc_temp end


struct PSH1pBasisFunction
   n::Int
   l::Int
   m::Int
end


@doc raw"""
`struct BasicPSH1pBasis <: OneParticleBasis`

One-particle basis of the form
```math
\phi_{nlm}({\bm r}, z_1, z_0) = J_{n}(r) Y_l^m(\hat{\br r})
```
where ``J_{n}`` denotes a radial basis.
"""
struct BasicPSH1pBasis{T, NZ, TJ <: RadialBasis} <: OneParticleBasis
   J::TJ
   SH::SHBasis{T}
   zlist::SZList{NZ}
   firstA::SMatrix{NZ, NZ, Int}
   spec::Vector{PSH1pBasisFunction}
end

Base.length(basis::BasicPSH1pBasis) = length(basis.spec)

Base.eltype(basis::BasicPSH1pBasis{T}) where T = Complex{T}
# eltype and length should provide automatic allocation of alloc_B, alloc_dB


alloc_temp(basis::BasicPSH1pBasis, args...) =
      ( BJ = alloc_B(basis.J, args...),
        tmpJ = alloc_tmp(basis.J, args...),
        BY = alloc_B(basis.SH, args...),
        tmpY = alloc_tmp(basis.SH, args...),
       )

function add_into_A!(A, tmp, basis::BasicPSH1pBasis, R, iz, iz0)
   # evaluate the r-basis and the R̂-basis for the current neighbour at R
   evaluate!(tmp.J, tmp.tmpJ, basis.J, norm(R))
   evaluate!(tmp.Y, tmp.tmpY, basis.SH, R)
   # add the contributions to the A_zklm
   for (i, nlm) in enumerate(basis.spec)
      A[i] += tmp.J[nlm.n+1] * tmp.Y[index_y(nlm.l, nlm.m)]
   end
   return nothing
end





@doc raw"""
`struct PSH1pBasis <: OneParticleBasis`

Pne-particle basis of the form
```math
\phi_{nlm}({\bm r}) = P_{nl}(r) Y_l^m(\hat{\br r})
```
where ``P_{nl}`` denotes a radial basis given by
```math
P_{nl}(r)^{zz'} = \sum_k p_{nlk}^{zz'} J_k(r)
```
"""
struct PSH1pBasis{T, NZ, TJ <: RadialBasis} <: OneParticleBasis
   J::TJ
   SH::SHBasis{T}
   Pmat::SMatrix{NZ, NZ, Matrix{T}}
   zlist::SZList{NZ}
   firstA::SMatrix{NZ, NZ, Int}
   spec::SMatrix{NZ, NZ, Vector{PSH1pBasisFunction}}
end

Base.length(basis::PSH1pBasis) = length(basis.spec)

Base.eltype(basis::PSH1pBasis{T}) where T = Complex{T}
# eltype and length should provide automatic allocation of alloc_B, alloc_dB


function add_into_A!(A, tmp, basis::PSH1pBasis, R, iz, iz0)
   # evaluate the r-basis and the R̂-basis for the current neighbour at R
   evaluate!(tmp.BJ, tmp.tmpJ, basis.J, norm(R))
   nP = size(basis.Pmat[iz, iz0], 1)
   mul!(@view(tmp.BP[1:nP]), basis.Pmat[iz, iz0], tmp.BJ)
   evaluate!(tmp.BY, tmp.tmpY, basis.SH, R)
   # add the contributions to the A_zklm
   for (i, nlm) in enumerate(basis.spec)
      # TODO: indexing into P should be via (n, l)
      #       how to implement this properly for general nl lists?
      A[i] += tmp.BP[nlm.n+1] * tmp.BY[index_y(nlm.l, nlm.m)]
   end
   return nothing
end
