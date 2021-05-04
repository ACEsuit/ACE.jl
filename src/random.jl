
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Random

# TODO: rename rand_radial -> rand???

import LinearAlgebra: norm
import ACE: ScalarBasis, ZList, rand_radial, scaling

using Random: shuffle
using JuLIP: JVecF, AbstractCalculator, rnn, chemical_symbol, bulk, rattle!,
             fltype, rfltype
using JuLIP.MLIPs: combine
using JuLIP.Potentials: zlist, ZList, SZList
using StaticArrays: @SMatrix

export rand_nhd, rand_config, rand_sym, randcoeffs, randcombine

# -------------------------------------------
#   random neighbourhoods and  configurations

# TODO: JVecF is hard-coded
function rand_sphere()
   R = randn(JVecF)
   return R / norm(R)
end

# TODO: this could be rand(PSH1BasisFcn) ...
#       rand_sphere = rand(SphericalHarmonics)
rand_vec(J::ScalarBasis) where T = rand_radial(J) *  rand_sphere()
rand_vec(J::ScalarBasis, N::Integer) = [ rand_vec(J) for _ = 1:N ]

# -> rand_config?
function rand_nhd(Nat, J::ScalarBasis, species = :X)
   zlist = ZList(species)
   Rs = [ rand_vec(J) for _ = 1:Nat ]
   Zs = [ rand(zlist.list) for _ = 1:Nat ]
   z0 = rand(zlist.list)
   return Rs, Zs, z0
end


rand_config(species; kwargs...) =
      rand_config(ZList(species); kwargs...)

rand_config(V::AbstractCalculator; kwargs...) =
      rand_config(zlist(V); kwargs...)

function rand_config(zlist::Union{ZList, SZList};
                     absrattle = 0.0, relrattle = 0.2, repeat = 3,
                     kwargs...)
   # start with the longest rnn
   rnns = rnn.(zlist.list)
   sym = chemical_symbol( zlist.list[findmax(rnns)[2]] )
   at = bulk(sym; kwargs...) * repeat
   for n = 1:length(at)
      at.Z[n] = rand(zlist.list)
   end
   rattle!(at, maximum(rnns) * relrattle + absrattle)
   return at
end


# --------------------------------------------------------------
# random operations on neighbourhoods, mostly for testing

function rand_perm(Rs, Zs)
   @assert length(Rs) == length(Zs)
   p = shuffle(1:length(Rs))
   return Rs[p], Zs[p]
end

function rand_rot(Rs, Zs)
   @assert length(Rs) == length(Zs)
   K = (@SMatrix rand(3,3)) .- 0.5
   K = K - K'
   Q = exp(K)
   return [ Q * R for R in Rs ], Zs
end

rand_refl(Rs, Zs) = (-1) .* Rs, Zs

rand_sym(Rs, Zs) = rand_refl(rand_rot(rand_perm(Rs, Zs)...)...)


# -------------------------------------------
#     random potentials

# TODO: we have an issue with eltypes that needs to be fixed!!!

function randcoeffs(basis; diff = 2)
   ww = scaling(basis, diff)
   c = 2 * (rand(rfltype(basis), length(basis)) .- 0.5) ./ ww
   return c / norm(c)
end

randcombine(basis; diff = 2) =
   combine(basis, randcoeffs(basis; diff = diff))

# # move to utility???
# function rand(::Type{ACE.RPI.RPIBasis}; kwargs...)
#
# end

end
