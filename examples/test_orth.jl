
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------



using Test
using SHIPs, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, SHIPs.JacobiPolys
using SHIPs: TransformedJacobi, transform, transform_d,
             alloc_B, alloc_temp

using JuLIP: evaluate!

##

@noinline get_IN(N) = collect((shpB.idx_Bll[N][1]+1):(shpB.idx_Bll[N][end]+shpB.len_Bll[N][end]))

# function barrier
@noinline gramian(N, shpB, Nsamples=100_000; normalise=false) =
   gramian(N, get_IN(N), shpB, Nsamples; normalise=normalise)

@noinline gramian(N, IN, shpB, Nsamples; normalise=false) =
   gramian(N, IN, alloc_temp(shpB), alloc_B(shpB), shpB, Nsamples, normalise)

function gramian(N, IN, tmp, B, shpB, Nsamples = 100_000, normalise = false)
   Zs = zeros(Int16, N)
   lenB = length(IN)
   G = zeros(Float64, lenB, lenB)
   for n = 1:Nsamples
      Rs = SHIPs.Utils.rand(shpB.J, N)
      evaluate!(B, tmp, shpB, Rs, Zs, 0)
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
Nsamples = 1_000
rl, ru = 0.5, 3.0
fcut =  PolyCutoff2s(2, rl, ru)
trans = PolyTransform(2, 1.0)
shpB = SHIPBasis( SparseSHIP(6, 15), trans, fcut )

ctr = 0
nnz = 0
for N = 1:Nmax
   for νz in shpB.NuZ[N]
      kk, ll = SHIPs._kl(νz.ν, νz.izz, shpB.KL)
      for mm in SHIPs._mrange(ll)
         global ctr += 1
         global nnz += size( shpB.rotcoefs[N][ll], 2 )
      end
   end
end
length(shpB)
ctr
nnz

function fill_an_array!(A)
   for n = 1:length(A)
      A[n] = rand()
   end
   return A
end

A = zeros(100_000);
(@elapsed fill_an_array!(A)) * 1000



@show shpB.idx_Bll[end]
@show shpB.len_Bll[end]
@show length(shpB)

G = gramian(4, shpB, 10_000)
rank(G)
svdG = svd(G)
svdG.U

@info("Conditions numbers of gramians")
for N = 1:Nmax
   GN = gramian(N, shpB, Nsamples, normalise=false)
   @show N, cond(GN)
end

@info("Conditions numbers of normalised gramians")
for N = 1:Nmax
   GN = gramian(N, shpB, Nsamples, normalise=true)
   @show N, cond(GN)
end
