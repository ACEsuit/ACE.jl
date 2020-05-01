
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------




"""
`mutable struct InnerPIBasis` : this type is just an auxilary type to
make the implementation of `PermInvariantBasis` clearer. It implements the
permutation-invariant basis for a single centre-atom species. The main
type `PermInvariantBasis` then stores `NZ` objects of type `InnerPIBasis`
and "dispatches" the work accordingly.
"""
mutable struct InnerPIBasis <: IPBasis
   orders::Vector{Int}           # order (length) of ith basis function
   iAA2iA::Matrix{Int}           # where in A can we find the ith basis function
   b2iAA::Dict{PIBasisFcn, Int}  # mapping PIBasisFcn -> iAA =  index in AA[z0]
   b2iA::Dict{Any, Int}          # mapping from 1-p basis fcn to index in A[z0]
   AAindices::UnitRange{Int}     # where in AA does AA[z0] fit?
end

Base.length(basis::InnerPIBasis) = length(basis.order)

function InnerPIBasis(Aspec, AAspec, AAindices, z0)
   len = length(AAspec)
   maxorder = maximum(order, AAspec)

   # construct the b2iA mapping
   b2iA = Dict{Any, Int}()
   for (iA, b) in enumerate(Aspec)
      if haskey(b2iA, b)
         @show b
         error("b2iA already has the key b")
      end
      b2iA[b] = iA
   end
   # construct the b2iAA mapping
   b2iAA = Dict{PIBasisFcn, Int}()
   for (iAA, b) in enumerate(AAspec)
      if haskey(b2iAA, b)
         @show b
         error("b2iAA already has the key b")
      end
      b2iAA[b] = iAA
   end

   # allocate the two main arrays used for evaluation ...
   orders = zeros(Int, len)
   iAA2iA = zeros(Int, len, maxorder)
   # ... and fill them up with the cross-indices
   for (iAA, b) in enumerate(AAspec)
      @assert b2iAA[b] == iAA
      @assert b.z0 == z0
      orders[iAA] = order(b)
      for α = 1:order(b)
         iAA2iA[iAA, α] = b2iA[ b.oneps[α] ]
      end
   end

   # put it all together ...
   return InnerPIBasis(orders, iAA2iA, b2iAA, b2iA, AAindices)
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





"""
`mutable struct PermInvariantBasis:` implementation of a permutation-invariant
basis based on the density projection trick.

The standard constructor is
```
PermInvariantBasis(basis1p, N, D, maxdeg)
```
* `basis1p` : a one-particle basis
* `N` : maximum interaction order
* `D` : an abstract degee specification, e.g., SparsePSHDegree
* `maxdeg` : the maximum polynomial degree as measured by `D`

Note the species list will be taken from `basis1p`
"""
mutable struct PermInvariantBasis{BOP, NZ} <: IPBasis
   basis1p::BOP                           # a one-particle basis
   zlist::SZList{NZ}
   inner::NTuple{NZ, InnerPIBasis}
end



function PermInvariantBasis(basis1p::OneParticleBasis,
                            N::Integer,
                            D::AbstractDegree,
                            maxdeg::Real)
   zlist = basis1p.zlist
   inner = InnerPIBasis[]
   idx = 0
   # now for each iz0 (i.e. for each z0) construct an "inner basis".
   for iz0 = 1:numz(basis1p)
      z0 = i2z(basis1p, iz0)
      # get a list of 1-p basis function
      Aspec_z0, AAspec_z0 = get_PI_spec(basis1p, N, D, maxdeg, z0)
      AAindices = (idx+1):(idx+length(AAspec_z0))
      push!(inner, InnerPIBasis(Aspec_z0, AAspec_z0, AAindices, z0))
      idx += length(AAspec_z0)
   end
   return PermInvariantBasis(basis1p, zlist, tuple(inner...))
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
   AA_z0 = @view AA[basis.inner[iz0].AAindices]
   evaluate!(AA_z0, tmp, basis.inner[iz0], A)
   return AA
end
