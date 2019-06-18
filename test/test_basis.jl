
# --------------------------------------------------------------------------
# ACE.jl and SHIPs.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# All rights reserved.
# --------------------------------------------------------------------------


using SHIPs, JuLIP, BenchmarkTools, LinearAlgebra
using SHIPs: eval_basis!, eval_basis

trans = PolyTransform(2, 1.0)
ship = SHIPBasis(3, 15, 2.0, trans, 2, 0.5, 3.0)

length(ship.Nu)

Rs = 1.0 .+ 0.3 * (rand(JVecF, 12) .- 0.5)
@btime SHIPs.precompute_A!($ship, $Rs)
SHIPs.length_B(ship)
length(ship.A)

B = SHIPs.alloc_B(ship)
@btime SHIPs.eval_basis!($B, $ship, $Rs)
