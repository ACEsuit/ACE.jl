
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



using Test
using ACE, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, ACE.JacobiPolys
using ACE: TransformedJacobi, transform, transform_d,
             alloc_B, alloc_temp

using JuLIP: evaluate!

##

@noinline get_IN(N) = collect((aceB.idx_Bll[N][1]+1):(aceB.idx_Bll[N][end]+aceB.len_Bll[N][end]))

# function barrier
@noinline gramian(N, aceB, Nsamples=100_000; normalise=false) =
   gramian(N, get_IN(N), aceB, Nsamples; normalise=normalise)

@noinline gramian(N, IN, aceB, Nsamples; normalise=false) =
   gramian(N, IN, alloc_temp(aceB), alloc_B(aceB), aceB, Nsamples, normalise)

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
Nsamples = 1_000
rl, ru = 0.5, 3.0
fcut =  PolyCutoff2s(2, rl, ru)
trans = PolyTransform(2, 1.0)
aceB = SHIPBasis( SparseSHIP(6, 15), trans, fcut )

ctr = 0
nnz = 0
for N = 1:Nmax
   for νz in aceB.NuZ[N]
      kk, ll = ACE._kl(νz.ν, νz.izz, aceB.KL)
      for mm in ACE._mrange(ll)
         global ctr += 1
         global nnz += size( aceB.rotcoefs[N][ll], 2 )
      end
   end
end
length(aceB)
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



@show aceB.idx_Bll[end]
@show aceB.len_Bll[end]
@show length(aceB)

G = gramian(4, aceB, 10_000)
rank(G)
svdG = svd(G)
svdG.U

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
