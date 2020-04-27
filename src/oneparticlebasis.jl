
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using StaticArrays

using SHIPs.SphericalHarmonics: SHBasis, index_y
using JuLIP.Potentials: ZList, SZList



function evaluate!(A, tmp, basis::OneParticleBasis, Rs, Zs::AbstractVector, z0)
   fill!.(A, 0)
   iz0 = z2i(basis, z0)
   for (R, Z) in zip(Rs, Zs)
      iz = z2i(basis, Z)
      add_into_A!(A[iz], tmp.tmpPhi, basis, R, iz, iz0)
   end
   return A
end

function evaluate!(A, tmp, basis::OneParticleBasis, Rs, Zs::Number, z0)
   fill!.(A, 0)
   iz0, iz = z2i(basis, z0), z2i(basis, z)
   add_into_A!(A[iz], tmp.tmpPhi, basis, R, iz, iz0)
   return A
end


alloc_B(basis::OneParticleBasis, args...) =
      [ zeros(eltype(basis), length(basis, iz))   for iz = 1:numz(basis) ]



@doc raw"""
`struct BasicPSH1pBasis <: OneParticleBasis`

One-particle basis of the form
```math
\phi_{nlm}({\bm r}, z_1, z_0) = J_{n}(r) Y_l^m(\hat{\br r})
```
where ``J_{n}`` denotes a radial basis.
"""
struct BasicPSH1pBasis{T, NZ, TJ <: ScalarBasis{T}} <: OneParticleBasis{T}
   J::TJ
   SH::SHBasis{T}
   zlist::SZList{NZ}
   spec::Vector{PSH1pBasisFunction}
end


function BasicPSH1pBasis(J::ScalarBasis{T};
                         species = :X,
                         D::AbstractDegree = SparsePSHDegree()
                ) where {T}
   # find out what the largest degree is that we can allow:
   maxdeg = maximum(D(PSH1pBasisFunction(n, 0, 0)) for n = 1:length(J))

   # generate the `spec::Vector{PSH1pBasisFunction}` using length(J)
   specnl = gensparse(2, maxdeg;
                      tup2b = t -> PSH1pBasisFunction(t[1]+1, t[2], 0),
                      degfun = t -> D(t),
                      ordered = false)
   # add the m-parameters
   spec = [ PSH1pBasisFunction(b.n, b.l, b.m)
            for b in specnl for m = -b.l:b.l ]
   # now get the maximum L-degree to generate the SH basis
   maxL = maximum(b.l for b in spec)
   SH = SHBasis(maxL, T)

   # construct the basis
   return BasicPSH1pBasis(J, SH, ZList(species; static=true), spec)
end

Base.length(basis::BasicPSH1pBasis) = length(basis.spec)
Base.length(basis::BasicPSH1pBasis, iz::Integer) = length(basis.spec)

Base.eltype(basis::BasicPSH1pBasis{T}) where T = Complex{T}
reltype(basis::BasicPSH1pBasis{T}) where T = T
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

One-particle basis of the form
```math
\phi_{nlm}({\bm r}) = P_{nl}(r) Y_l^m(\hat{\br r})
```
where ``P_{nl}`` denotes a radial basis given by
```math
P_{nl}(r)^{zz'} = \sum_k p_{nlk}^{zz'} J_k(r)
```
"""
struct PSH1pBasis{T, NZ, TJ <: ScalarBasis{T}} <: OneParticleBasis{T}
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
