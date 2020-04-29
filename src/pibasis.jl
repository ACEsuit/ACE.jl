
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



"""
`PIBasisFcn{N, TOP}` : represents a single multivariate basis function
in terms of 1-particle pasis functions in each coordinate direction. Crucially,
this function will be interpreted as a *permutation invariant* basis function!
"""
struct PIBasisFcn{N, TOP <: OneParticleBasisFcn}
   oneps::NTuple{N, TOP}
end

order(b::PIBasisFcn{N}) where {N} = N


mutable struct InnerPIBasis <: IPBasis
   order::Vector{Int}            # order (length) of ith basis function
   iAA2iA::Matrix{Int}           # where in A can we find the ith basis function
   b2iAA::Dict{PIBasisFcn, Int}  # inverse mapping PIBasisFcn -> iAA
end

Base.length(basis::InnerPIBasis) = length(basis.order)

mutable struct PermInvariantBasis{BOP, NZ} <: IPBasis
   basis1p::BOP                           # a one-particle basis
   zlist::SZList{NZ}
   inner::NTuple{NZ, InnerPIBasis}
   AAindices::NTuple{NZ, UnitRange{Int}}
end

alloc_B(basis::PermInvariantBasis, args...) =
      zeros(eltype(basis.basis1p), length(basis))

alloc_B_z0(basis::PermInvariantBasis, iz0::Integer) =
      zeros(eltype(basis.basis1p), length(basis.inner[iz0]))

function evaluate!(AA, tmp, basis::PermInvariantBasis, Rs, Zs, z0)
   # compute the (multi-variate) density projection
   A = evaluate!(tmp.A, tmp.tmp_basis1p, basis.basis1p, Rs, Zs, z0)
   # now evaluate the correct inner basis (this one doesn't know about z0)
   iz0 = z2i(basis, z0)
   AA_z0 = @view AA[basis.firstAAidx[iz0]:(basis.firstAAidx[iz0+1]-1)]
   evaluate!(AA_z0, tmp, basis.inner[iz0], A)
   return AA
end


function evaluate!(AA, tmp, basis::InnerPIBasis, A)
   fill!(AA, 1)
   for i = 1:length(basis)
      for α = 1:basis.order[i]
         iA = basis.i2Aidx[i, α]
         AA[i] *= A[iA]
      end
   end
   return AA
end
