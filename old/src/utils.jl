
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


module Utils

using StaticArrays
using LinearAlgebra: norm
using JuLIP: JVecF
using ACE: TransformedJacobi,  inv_transform,
            PolyTransform, PolyCutoff1s, SparseSHIP, SHIPBasis,
            TransformedPolys

import ACE
import ACE: rand_sphere, rand_radial
import Base: rand
import JuLIP: evaluate
import JuLIP.MLIPs: IPBasis


_get_ll(KL, νz) = getfield.(KL[νz.ν], :l)


# Some utilities to analyze the basis
# ------------------------------------

"""
quickly  generate a basis for testing and analysing
"""
function TestBasis(N, deg; rnn=1.0, rcut = 2.5, rin = 0.5, kwargs... )
   trans = PolyTransform(2, rnn)
   fcut = PolyCutoff1s(2, rin, rcut)
   spec = SparseSHIP(N, deg)
   return SHIPBasis(spec, trans, fcut; kwargs...)
end

"""
find all basis functions with prescribed ll tuple

TODO: generalize so we can also prescribe kk, or both.
"""

function findall_basis(basis; N = nothing, ll = nothing,
                              kwargs...)
   if N != nothing &&  ll != nothing
      error("findall_basis: only tell me N or ll but not both")
   end
   if N != nothing
      return findall_basis_N(basis, N; kwargs...)
   end
   if ll != nothing
      return findall_basis_ll(basis, ll; kwargs...)
   end
   error("findall_basis: I need either an N or an ll argument")
end


function findall_basis_N(basis, N; verbose=true)::Vector{Int} 
   verbose && @warn("This code assumes there is only one species. (not checked!)")
   I1 = findall(b -> length(b[end]) == N, basis.bgrps[1])
   if isempty(I1); return Int[]; end
   verbose && @info("There are $(length(I1)) basis groups with N = $N:")
   Ib = mapreduce(i -> (basis.firstb[1][i]+1):basis.firstb[1][i+1],
                  vcat, I1)
   verbose && @info("  ... and $(length(Ib)) basis functions.")
   return Ib
end


function findall_basis_ll(basis, ll)
   ll = SVector(ll...)
   @warn("This code assumes there is only one species. (not checked!)")
   @info(" ll = $(ll)")
   @info("Get the purely rotation-invariant basis:")
   CA = ACE.Rotations.Rot3DCoeffs()
   Brot = ACE.Rotations.basis(CA, ll)
   @info("   ... there are $(size(Brot, 2)) rotation-invariance basis functions")

   @info("Find the RPI basis")
   I1 = findall(b -> b[end] == ll, basis.bgrps[1])
   @info("   ... there are $(length(I1)) basis groups with this `ll`:")
   println("grp-idx        kk              ll         length    B-idx")
   for i in I1
      b = basis.bgrps[1][i]
      len = basis.firstb[1][i+1] - basis.firstb[1][i]
      bidx = (basis.firstb[1][i]+1):basis.firstb[1][i+1]
      println(" $(i)   |  $(Int64.(b[2]))  |  $(Int64.(b[3]))  |  $(len) | $(bidx)")
   end
   return nothing
end

"""
create a sub-basis for easier evaluation
"""
struct SubBasis{TB} <: IPBasis
   B::TB
   Ib::Vector{Int}
end

SubBasis(B, Ib::AbstractVector) =
   SubBasis(B, convert(Vector{Int}, collect(Ib)))

(B::SubBasis)(Rs) =
   evaluate(B.B, Rs, zeros(Int16,length(Rs)), 0)[B.Ib]


end
