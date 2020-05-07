

import JuLIP.Potentials: z2i

import JuLIP: alloc_temp, alloc_temp_d,
              cutoff,
              evaluate, evaluate_d,
              evaluate!, evaluate_d!,
              SitePotential

import JuLIP.MLIPs: IPBasis, alloc_B, alloc_dB
using JuLIP.Potentials: ZList, z2i, i2z

using LinearAlgebra: mul

using Random: shuffle

import Base: Dict, convert, ==

# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function fcut end
function fcut_d end

"""
`VecOrTup = Union{AbstractVector, Tuple}`
"""
const VecOrTup = Union{AbstractVector, Tuple}


abstract type ScalarBasis{T} <: IPBasis end

abstract type OneParticleBasis{T} <: IPBasis end

abstract type OnepBasisFcn end

# ------------------------------------------------------------
#  Abstract polynomial degree business

"""
`AbstractDegree` : object specifying a degree can be called via
`degree(D, arg)` or via `D(arg)`
"""
abstract type AbstractDegree end

(D::AbstractDegree)(args...) = degree(D, args...)

"""
`function degree(D::AbstractDegree, arg)` : compute some notion of degree of
the `arg` argument.
"""
function degree end




# ----------------------------------------------------------------------

# TODO: put type information to the random stuff

# Some methods for generating random samples
function rand_radial end

function rand_sphere()
   R = randn(JVecF)
   return R / norm(R)
end

rand_vec(J::ScalarBasis) where T = rand_radial(J) *  rand_sphere()
rand_vec(J::ScalarBasis, N::Integer) = [ rand_vec(J) for _ = 1:N ]

function rand_nhd(Nat, J::ScalarBasis, species = :X)
   zlist = ZList(species)
   Rs = [ rand_vec(J) for _ = 1:Nat ]
   Zs = [ rand(zlist.list) for _ = 1:Nat ]
   z0 = rand(zlist.list)
   return Rs, Zs, z0
end

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
