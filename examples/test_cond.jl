
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


@info("Testing conditioning of non-orth 3B SHIP Basis")

using Test
using SHIPs, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, SHIPs.JacobiPolys
using SHIPs: TransformedJacobi, transform, transform_d, eval_basis!,
             alloc_B, alloc_temp

##

get_IN(N) = collect((shpB.idx_Bll[N][1]+1):(shpB.idx_Bll[N][end]+shpB.len_Bll[N][end]))

# function barrier
gramian(N, shpB, Nsamples=100_000) =
   gramian(N, get_IN(N), alloc_temp(shpB), alloc_B(shpB), shpB, Nsamples)

function gramian(N, IN, tmp, B, shpB, Nsamples = 100_000)
   Zs = zeros(Int16, N)
   lenB = length(IN)
   G = zeros(Float64, lenB, lenB)
   for n = 1:Nsamples
      Rs = SHIPs.Utils.rand(shpB.J, N)
      eval_basis!(B, tmp, shpB, Rs, Zs, 0)
      for j = 1:lenB
         Bj = B[IN[j]]'
         @simd for i = 1:lenB
            @inbounds G[i,j] += Bj * B[IN[i]]
         end
      end
   end
   return G / Nsamples
end

##

rl, ru = 0.5, 3.0
fcut =  PolyCutoff2s(2, rl, ru)
trans = PolyTransform(2, 1.0)
shpB = SHIPBasis( SparseSHIP(4, 10), trans, fcut )

for N = 1:4
   GN = gramian(N, shpB, 10_000)
   @show N, cond(GN)
end
