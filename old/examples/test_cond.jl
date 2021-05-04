
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


@info("Testing conditioning of non-orth 3B SHIP Basis")

using Test
using ACE, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, ACE.JacobiPolys
using ACE: TransformedJacobi, transform, transform_d,
             alloc_B, alloc_temp

import JuLIP: evaluate!
##

get_IN(N) = collect((aceB.idx_Bll[N][1]+1):(aceB.idx_Bll[N][end]+aceB.len_Bll[N][end]))

# function barrier
gramian(N, aceB, Nsamples=100_000; normalise=false) =
   gramian(N, get_IN(N), alloc_temp(aceB), alloc_B(aceB), aceB, Nsamples, normalise)

function gramian(N, IN, tmp, B, aceB, Nsamples = 100_000, normalise = false)
   Zs = zeros(Int16, N)
   lenB = length(IN)
   G = zeros(Float64, lenB, lenB)
   for n = 1:Nsamples
      Rs = ACE.rand_vec(aceB.J, N)
      evaluate!(B, tmp, aceB, Rs, Zs, 0)
      for j = 1:lenB
         Bj = B[IN[j]]'
         @simd for i = 1:lenB
            @inbounds G[i,j] += Bj * B[IN[i]]
         end
      end
   end
   if normalise
      g = diag(G)
      for i = 1:lenB, j = 1:lenB
         G[i,j] /= sqrt(g[i]*g[j])
      end
   end
   return G / Nsamples
end

##
Nmax = 4
Nsamples = 100_000
rl, ru = 0.5, 3.0
fcut =  PolyCutoff2s(2, rl, ru)
trans = PolyTransform(2, 1.0)
aceB = SHIPBasis( SparseSHIP(Nmax, 10), trans, fcut )

@info("Conditions numbers of gramians")
for N = 1:Nmax
   GN = gramian(N, aceB, Nsamples, normalise=false)
   @show N, cond(GN)
end

@info("Conditions numbers of normalised gramians")
for N = 1:Nmax
   GN = gramian(N, aceB, Nsamples, normalise=true)
   @show N, cond(GN)
end
