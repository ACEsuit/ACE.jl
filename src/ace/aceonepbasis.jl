
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------




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
   spec::SMatrix{NZ, NZ, Vector{PSH1pBasisFcn}}
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
      A[i] += tmp.BP[nlm.n] * tmp.BY[index_y(nlm.l, nlm.m)]
   end
   return nothing
end
