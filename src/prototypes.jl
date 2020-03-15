
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


# We use IntS for all index integer types for all SHIPs.jl basis types
# The `S` originates from the previous name of the package, `SHIPs.jl`
const IntS = Int32

import JuLIP.Potentials: z2i

import JuLIP: energy, forces, virial, alloc_temp, alloc_temp_d, cutoff,
              evaluate, evaluate_d,
              energy!, forces!, virial!, evaluate!, evaluate_d!

import JuLIP.MLIPs: IPBasis, alloc_B, alloc_dB

import Base: Dict, convert, ==

# prototypes for space transforms and cutoffs
function transform end
function transform_d end
function fcut end
function fcut_d end

function rand_radial end

function rand_sphere()
   R = randn(JVecF)
   return R / norm(R)
end

rand_vec(J) = rand_radial(J) *  rand_sphere()
rand_vec(J, N::Integer) = [ rand_vec(J) for _ = 1:N ]
