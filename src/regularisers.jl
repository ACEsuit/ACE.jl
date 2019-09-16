
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Regularisers

using SHIPs: SHIPBasis, _get_NuZ_N, nfcalls, nspecies, _first_B_idx, _kl, Tνz
using LinearAlgebra: Diagonal

function _get_kls!(ww, ship::SHIPBasis{BO, T}, ::Val{N}, iz0,
                   NuZ_N::Vector{Tνz{N}},
                   weightfcn) where {BO, T, N}
   @assert N <= BO
   ZKL = ship.KL
   # compute the zeroth (not first!) index of the N-body subset of the SHIPBasis
   idx0 = _first_B_idx(ship, N, iz0)
   # loop over N-body basis functions
   # A has already been filled in the outer eval_basis!
   for (idx, νz) in enumerate(NuZ_N)
      ν = νz.ν
      izz = νz.izz
      kk, ll = _kl(ν, izz, ZKL)
      # @show kk, ll, weightfcn(kk, ll)
      ww[idx0+idx] = weightfcn(kk, ll)
   end
   return ww
end

function _get_ww(ship::SHIPBasis{BO, T}, weightfcn) where {BO, T}
   ww = zeros(length(ship))
   for iz = 1:nspecies(ship)
      nfcalls(Val(BO), valN -> _get_kls!(ww, ship, valN, iz,
                                         _get_NuZ_N(ship, valN, iz),
                                         weightfcn))
   end
   return ww
end

diffweight(kk, ll, diff) = sqrt(sum(kk.^(2*diff)) + sum(ll.^(2*diff)))

function diagonal_regulariser(shp::SHIPBasis;
                              diff = 0,
                              weightfcn = (kk, ll) -> diffweight(kk, ll, diff))
   return Diagonal(_get_ww(shp, weightfcn))
end


end
