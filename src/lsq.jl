
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# TODO
#   - create SumPot in JuLIP
#   - create a SumSHIPBasis or simply Vector{ShipBasis}?
#   - we may need an additional ONE function, corresponding to index
#     in the J_k basis set. Then J_1 is the first non-constant function
#     and we just need to disallow all zeros (that's the OneBody)

"""
A baby regression module for some quick testing. This is primarily
experimental and will eventually move back into IPFitting.
"""
module Lsq

using JuLIP, LinearAlgebra
using IPFitting: Dat, tfor, observations, observation, eval_obs, vec_obs
using IPFitting.DB: set_matrows!, matrows
using SHIPs: SHIPBasis, SHIP
using JuLIP.MLIPs: IPBasis



function _alloc_lsq_matrix(configs, basis)
   nrows = 0
   for (okey, d, _) in observations(configs)
      len = length(observation(d, okey))
      set_matrows!(d, okey, collect(nrows .+ (1:len)))
      nrows += len
   end
   # allocate and return the matrix
   return zeros(Float64, nrows, length(basis))
end

# eval_obs(::Val{:E}, shipB::IPBasis, at::Atoms) =
#          reshape(energy(shipB, at), (1, length(shipB)))

# function eval_obs(::Val{:F}, shipB::IPBasis, at::Atoms)
#    Fs = forces(shipB, at)
#    vecFs = zeros(3*length(at), length(shipB))
#    for iB = 1:length(shipB)
#       vecFs[:, iB] = vec_obs(Val{:F}(), Fs[iB])
#    end
#    return vecFs
# end

# TODO: rewrite with tfor loop
function lsq_system(configs::Vector{Dat},
                    shipB::IPBasis, cfgweights, obsweights)
   A = _alloc_lsq_matrix(configs, shipB)
   Y = zeros(Float64, size(A, 1))
   for (okey, d, icfg) in observations(configs)
      if !haskey(obsweights, okey); continue; end
      if !haskey(cfgweights, d.configtype); continue; end
      # print(".")
      w = obsweights[okey] * cfgweights[d.configtype]
      irows = matrows(d, okey)
      Y[irows] = w * observation(d, okey)
      A[irows, :] = w * vec_obs(okey, eval_obs(okey, shipB, d.at))
      if any(isnan, Y[irows])
         @show icfg, okey
         @error("found NaNs in Y")
         throw(nothing)
      end
      if any(isnan, A[irows, :])
         @show icfg, okey
         @show eval_obs(okey, shipB, d.at)
         @show cell(d.at)
         @show positions(d.at)
         @error("found NaNs in A")
         throw(nothing)
      end
   end
   return A, Y
end

function lsqfit(configs::Vector{Dat},
                shipB::IPBasis;
                configweights = nothing,
                obsweights = nothing,
                verbose = false)
   A, Y = lsq_system(configs, shipB, configweights, obsweights)
   qrA = qr(A)
   if verbose; (@show cond(qrA.R)); end
   if verbose; (@show size(A)); end
   coeffs = qr(A) \ Y

   # return SHIP(shipB, coeffs),
   return norm(A * coeffs - Y) / norm(Y)
end


end
