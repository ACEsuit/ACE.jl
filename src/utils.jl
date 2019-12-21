
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Utils

using LinearAlgebra: norm
using JuLIP: JVecF
using PoSH: TransformedJacobi,  inv_transform,
            PolyTransform, PolyCutoff1s, SparseSHIP, SHIPBasis

import PoSH
import Base: rand

function rand_sphere()
   R = randn(JVecF)
   return R / norm(R)
end

function rand_radial(J::TransformedJacobi)
   # uniform sample from [tl, tu]
   x = J.tl + rand() * (J.tu - J.tl)
   # transform back
   return inv_transform(J.trans, x)
end

rand_radial(J::TransformedJacobi, N::Integer) = [ rand_radial(J) for _=1:N ]

rand(J::TransformedJacobi) = rand_radial(J) *  rand_sphere()

rand(J::TransformedJacobi, N::Integer) =  [ rand(J) for _ = 1:N ]


_get_ll(KL, νz) = getfield.(KL[νz.ν], :l)


# Some utilities to analyze the basis
# ------------------------------------

"""
quickly  generate a basis for testing and analysing
"""
function TestBasis(N, deg; rnn=1.0, rcut = 2.5, rin = 0.5, )
   trans = PolyTransform(2, rnn)
   fcut = PolyCutoff1s(2, rin, rcut)
   spec = SparseSHIP(N, deg)
   return SHIPBasis(spec, trans, fcut)
end

"""
find all basis functions with prescribed ll tuple

TODO: generalize so we can also prescribe kk, or both.
"""
function findall_basis(basis=nothing; ll = nothing)
   ll = SVector(ll...)
   @warn("This code assumes there is only one species. (not checked!)")
   @info(" ll = $(ll)")
   @info("Get the purely rotation-invariant basis:")
   CA = PoSH.Rotations.CoeffArray()
   Brot = PoSH.Rotations.basis(CA, ll)
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
struct SubBasis{TB}
   B::TB
   Ib::Vector{Int}
end

SubBasis(B, Ib::AbstractVector) =
   SubBasis(B, convert(Vector{Int}, collect(Ib)))

(B::SubBasis)(Rs) =
   PoSH.eval_basis(B.B, Rs, zeros(Int16,length(Rs)), 0)[B.Ib]


end
