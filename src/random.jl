
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


module Random

# TODO: rename rand_radial -> rand???

import LinearAlgebra: norm
import SHIPs: ScalarBasis, ZList, rand_radial, scaling

using Random: shuffle
using JuLIP: JVecF
using StaticArrays: @SMatrix

export rand_nhd, rand_sym, randcoeffs, randcombine

# -------------------------------------------
#   random configurations
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
   c = 2 * (rand(real(eltype(basis)), length(basis)) .- 0.5) ./ ww
   return c / norm(c)
end

randcombine(basis; diff = 2) =
   combine(basis, randcoeffs(basis; diff = diff))

# # move to utility???
# function rand(::Type{SHIPs.RPI.RPIBasis}; kwargs...)
#
# end

end
